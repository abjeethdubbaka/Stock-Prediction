import os
import pandas as pd
import json
import logging
import requests
import time
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from src.config_loader import CONFIG
from nltk.sentiment import SentimentIntensityAnalyzer
from src.utils import ensure_directory_exists
from alpha_vantage.timeseries import TimeSeries

logger = logging.getLogger('StockPrediction')


def load_and_preprocess_data(stock_ticker, include_sentiment, sentiment_dir, api_key, period):
    raw_file_path = f"{CONFIG['store_or_read_stock_data']}/{stock_ticker}.csv"
    cache_dir = f"{CONFIG['store_or_read_stock_data']}/cache"
    ensure_directory_exists(cache_dir)
    today = datetime.now().strftime('%Y-%m-%d')
    live_cache_path = f"{cache_dir}/{stock_ticker}_live_{today}.csv"
    logger.info(f"Loading data for {stock_ticker} for period {period}")
    print(f"Checking file: {raw_file_path}, period: {period}, live cache: {live_cache_path}")
    data = None
    sentiment_score = 0

    # API keys
    alpha_vantage_key = "9GWBFNMFX3MNIKD3"  # Replace with your key from https://www.alphavantage.co/
    polygon_key = "9K2Z2CA5BR18HZT4FZUCP6JRJPNW787R3U"  # Replace with your key from https://polygon.io/ (optional)

    if period == '1D':
        if os.path.exists(live_cache_path):
            try:
                data = pd.read_csv(live_cache_path, index_col=0, parse_dates=True)
                logger.info(f"Loaded cached live data from {live_cache_path}")
                print(f"Loaded cached live data: {live_cache_path}, shape: {data.shape}")
            except Exception as e:
                logger.error(f"Error loading cached live data: {e}")
                print(f"Error loading cached live data: {e}")
        if data is None or data.empty:
            data = download_live_data(stock_ticker, live_cache_path, alpha_vantage_key, polygon_key)
        if data is None or data.empty:
            logger.warning(f"No live data available for {stock_ticker}. Falling back to historical data.")
            print(f"No live data available. Falling back to historical data.")
            data = load_historical_data(stock_ticker, raw_file_path, alpha_vantage_key, polygon_key)
    else:
        data = load_historical_data(stock_ticker, raw_file_path, alpha_vantage_key, polygon_key)

    if data is None or data.empty:
        logger.warning(f"No data available for {stock_ticker}. Skipping.")
        print(f"No data available for {stock_ticker}")
        return None, sentiment_score

    # Check data age
    latest_date = data.index.max().date() if isinstance(data.index.max(), pd.Timestamp) else pd.to_datetime(
        data.index.max()).date()
    if latest_date < pd.to_datetime('today').date() - pd.Timedelta(days=5):
        print(f"WARNING: Data is outdated (latest: {latest_date}). Predictions may be less accurate.")

    required_indicators = ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'Volatility', 'Avg_Return', 'Volume_SMA', 'Volume']
    missing_indicators = [ind for ind in required_indicators if ind not in data.columns]
    if missing_indicators:
        logger.info(f"Adding missing indicators for {stock_ticker}: {missing_indicators}")
        print(f"Adding indicators: {missing_indicators}")
        try:
            data = add_technical_indicators(data)
            print(f"Data after adding indicators: {data.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error adding indicators for {stock_ticker}: {e}")
            print(f"Error adding indicators: {e}")
            return None, sentiment_score

    data.to_csv(raw_file_path)
    logger.info(f"Updated data saved to {raw_file_path}.")
    print(f"Data saved to {raw_file_path}")

    sentiment_file = f"{sentiment_dir}/{stock_ticker}_sentiment.json"
    ensure_directory_exists(sentiment_dir)
    print(f"Sentiment file: {sentiment_file}")

    if include_sentiment:
        if 'Sentiment' not in data.columns:
            data['Sentiment'] = 0.0

        today = pd.Timestamp(datetime.now()).normalize()
        previous_business_day = pd.bdate_range(end=today, periods=2)[-2].date()
        previous_business_day = pd.Timestamp(previous_business_day)
        print(f"Previous business day: {previous_business_day}")

        sentiment_data = {}
        if os.path.exists(sentiment_file):
            with open(sentiment_file, "r") as f:
                sentiment_data = json.load(f)
            print(f"Loaded sentiment data: {sentiment_data}")

        if str(previous_business_day) in sentiment_data:
            sentiment_score = sentiment_data[str(previous_business_day)]
            logger.info(f"Using cached sentiment score ({sentiment_score}) for {previous_business_day}.")
            print(f"Cached sentiment score: {sentiment_score}")
        else:
            company_name, industry_keywords = get_stock_info(stock_ticker, alpha_vantage_key)
            print(f"Company: {company_name}, Keywords: {industry_keywords}")
            sentiment_score = get_market_sentiment(stock_ticker, sentiment_dir, api_key, company_name,
                                                   industry_keywords)
            sentiment_data[str(previous_business_day)] = sentiment_score
            with open(sentiment_file, "w") as f:
                json.dump(sentiment_data, f, indent=4)
            logger.info(f"Sentiment score ({sentiment_score}) saved for {previous_business_day}.")
            print(f"New sentiment score: {sentiment_score}")

        if previous_business_day in data.index:
            data.loc[previous_business_day, "Sentiment"] = sentiment_score
            data.to_csv(raw_file_path)
            print(f"Sentiment updated in data and saved")
        else:
            logger.warning(f"Warning: Previous business day ({previous_business_day}) not in stock data.")
            print(f"Warning: Previous business day not in data")

    data = data.dropna()
    print(f"Final data shape after dropna: {data.shape}")

    return data, sentiment_score


def load_historical_data(stock_ticker, raw_file_path, alpha_vantage_key, polygon_key):
    cache_dir = f"{CONFIG['store_or_read_stock_data']}/cache"
    today = datetime.now().strftime('%Y-%m-%d')
    hist_cache_path = f"{cache_dir}/{stock_ticker}_hist_{today}.csv"
    ensure_directory_exists(cache_dir)

    # Reuse cached data if less than 14 days old
    if os.path.exists(hist_cache_path):
        cache_mtime = datetime.fromtimestamp(os.path.getmtime(hist_cache_path))
        if (datetime.now() - cache_mtime).days <= 14:
            try:
                data = pd.read_csv(hist_cache_path, index_col=0, parse_dates=True)
                logger.info(f"Loaded recent cached historical data from {hist_cache_path}")
                print(f"Loaded recent cached historical data: {hist_cache_path}, shape: {data.shape}")
                return data
            except Exception as e:
                logger.error(f"Error loading cached historical data: {e}")
                print(f"Error loading cached historical data: {e}")

    # Try loading local file
    if os.path.exists(raw_file_path):
        try:
            data = pd.read_csv(raw_file_path, index_col=0, parse_dates=True)
            logger.info(f"Data loaded from {raw_file_path}.")
            print(f"Loaded historical data shape: {data.shape}")
            print(f"Columns: {data.columns.tolist()}")
            print(f"Latest date: {data.index.max().date()}")

            if 'Close' not in data.columns or not pd.api.types.is_numeric_dtype(data['Close']):
                raise ValueError("Invalid or missing 'Close' column in the dataset.")

            latest_date = data.index.max().date() if isinstance(data.index.max(), pd.Timestamp) else pd.to_datetime(
                data.index.max()).date()
            today = pd.Timestamp(datetime.now()).date()
            if latest_date < today - pd.Timedelta(days=5):
                print(f"Warning: Historical data is outdated (latest: {latest_date}). Attempting to update.")
                new_data = download_stock_data(stock_ticker,
                                               start_date=(pd.Timestamp(latest_date) + BDay(1)).strftime('%Y-%m-%d'),
                                               cache_path=hist_cache_path, api_key=alpha_vantage_key,
                                               polygon_key=polygon_key)
                if new_data is not None and not new_data.empty:
                    new_data = add_technical_indicators(new_data)
                    data = pd.concat([data, new_data]).drop_duplicates()
                    data.to_csv(hist_cache_path)
                    logger.info(f"Cached updated historical data to {hist_cache_path}")
                    print(f"Cached updated historical data to {hist_cache_path}")
                    print(f"Updated historical data shape: {data.shape}")
                return data
            return data
        except Exception as e:
            logger.error(f"Error loading historical data for {stock_ticker}: {e}")
            print(f"Error loading historical data: {e}")

    # Download data if local file is missing or empty
    print(f"No local data found for {stock_ticker}. Downloading historical data.")
    data = download_stock_data(stock_ticker, cache_path=raw_file_path, api_key=alpha_vantage_key,
                               polygon_key=polygon_key)
    if data is not None and not data.empty:
        data = add_technical_indicators(data)
        data.to_csv(raw_file_path)
        logger.info(f"Initial data saved to {raw_file_path}")
        print(f"Initial data saved to {raw_file_path}")
    return data


def download_live_data(stock_ticker, cache_path, alpha_vantage_key, polygon_key):
    logger.info(f"Downloading live data for {stock_ticker} from Alpha Vantage.")
    print(f"Downloading live data for {stock_ticker}")
    max_retries = 5
    ts = TimeSeries(key=alpha_vantage_key, output_format='pandas')
    for attempt in range(max_retries):
        try:
            interval = CONFIG.get("live_data_interval", "5minute").replace("min", "minute")
            data, _ = ts.get_intraday(symbol=stock_ticker, interval=interval, outputsize='compact')
            if data.empty:
                logger.info(f"No live data found for {stock_ticker}.")
                print(f"No live data found for {stock_ticker}")
                return pd.DataFrame()

            # Rename columns to match expected format
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data.index = pd.to_datetime(data.index)
            print(f"Live data shape: {data.shape}")
            print(f"Live data columns: {data.columns.tolist()}")
            print(f"Latest timestamp: {data.index.max()}")
            data = add_technical_indicators(data)
            data.to_csv(cache_path)
            logger.info(f"Cached live data to {cache_path}")
            print(f"Cached live data to {cache_path}")
            time.sleep(12)  # Respect Alpha Vantage rate limit (5 calls/min)
            return data
        except Exception as e:
            logger.error(f"Error downloading live data for {stock_ticker}: {e}")
            print(f"Error downloading live data (attempt {attempt + 1}): {e}")
            if "invalid api call" in str(e).lower():
                print(
                    "Invalid API key or parameters. Please verify your Alpha Vantage API key at https://www.alphavantage.co/.")
                break
            if "rate limit" in str(e).lower() or "thank you for using alpha vantage" in str(e).lower():
                time.sleep(12 * (2 ** attempt))
            else:
                break

    # Fallback to Polygon.io
    logger.info(f"Alpha Vantage failed. Attempting Polygon.io for {stock_ticker}.")
    print(f"Attempting Polygon.io for live data")
    try:
        interval = CONFIG.get("live_data_interval", "5minute").replace("min", "minute")
        url = f"https://api.polygon.io/v2/aggs/ticker/{stock_ticker}/range/5/minute/{datetime.now().strftime('%Y-%m-%d')}/{datetime.now().strftime('%Y-%m-%d')}?apiKey={polygon_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get('results', [])
        if not data:
            print(f"No live data found for {stock_ticker} on Polygon.io")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df = df.set_index('t')
        df.columns = ['Volume', 'Volume Weighted', 'Open', 'Close', 'High', 'Low', 'Timestamp', 'Transactions']
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"Polygon.io live data shape: {df.shape}")
        df = add_technical_indicators(df)
        df.to_csv(cache_path)
        print(f"Cached live data to {cache_path}")
        time.sleep(12)
        return df
    except Exception as e:
        logger.error(f"Polygon.io failed for {stock_ticker}: {e}")
        print(f"Polygon.io error: {e}")
        return pd.DataFrame()


def download_stock_data(stock_ticker, start_date=CONFIG["start_date"],
                        end_date=pd.to_datetime("today").strftime('%Y-%m-%d'), cache_path=None, api_key=None,
                        polygon_key=None):
    logger.info(f"Downloading historical data for {stock_ticker} from Alpha Vantage.")
    print(f"Downloading historical {stock_ticker} from {start_date} to {end_date}")
    max_retries = 5
    ts = TimeSeries(key=api_key, output_format='pandas')
    for attempt in range(max_retries):
        try:
            data, _ = ts.get_daily_adjusted(symbol=stock_ticker, outputsize='full')
            if data.empty:
                logger.info(f"No historical data found for {stock_ticker}.")
                print(f"No historical data found for {stock_ticker}")
                return pd.DataFrame()

            # Rename columns to match expected format
            data.columns = ['Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume', 'Dividend Amount',
                            'Split Coefficient']
            data.index = pd.to_datetime(data.index)
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]  # Keep only required columns
            # Filter by date range
            data = data.loc[start_date:end_date]
            print(f"Downloaded historical data shape: {data.shape}")
            print(f"Columns: {data.columns.tolist()}")

            save_path = cache_path if cache_path else f"{CONFIG['store_or_read_stock_data']}/{stock_ticker}.csv"
            data.to_csv(save_path)
            logger.info(f"Data for {stock_ticker} saved to {save_path}.")
            print(f"Data saved to {save_path}")
            time.sleep(12)  # Respect Alpha Vantage rate limit
            return data
        except Exception as e:
            logger.error(f"Error downloading historical data for {stock_ticker}: {e}")
            print(f"Error downloading historical data (attempt {attempt + 1}): {e}")
            if "invalid api call" in str(e).lower():
                print(
                    "Invalid API key or parameters. Please verify your Alpha Vantage API key at https://www.alphavantage.co/.")
                break
            if "rate limit" in str(e).lower() or "thank you for using alpha vantage" in str(e).lower():
                time.sleep(12 * (2 ** attempt))
            else:
                break

    # Fallback to Polygon.io
    logger.info(f"Alpha Vantage failed. Attempting Polygon.io for {stock_ticker}.")
    print(f"Attempting Polygon.io for historical data")
    try:
        start = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        end = pd.to_datetime(end_date).strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/ticker/{stock_ticker}/range/1/day/{start}/{end}?apiKey={polygon_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get('results', [])
        if not data:
            print(f"No historical data found for {stock_ticker} on Polygon.io")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df = df.set_index('t')
        df.columns = ['Volume', 'Volume Weighted', 'Open', 'Close', 'High', 'Low', 'Timestamp', 'Transactions']
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"Polygon.io historical data shape: {df.shape}")
        save_path = cache_path if cache_path else f"{CONFIG['store_or_read_stock_data']}/{stock_ticker}.csv"
        df.to_csv(save_path)
        print(f"Data saved to {save_path}")
        time.sleep(12)
        return df
    except Exception as e:
        logger.error(f"Polygon.io failed for {stock_ticker}: {e}")
        print(f"Polygon.io error: {e}")
        return pd.DataFrame()


def get_stock_info(ticker, api_key):
    logger.info(f"Retrieving stock info for {ticker}")
    print(f"Retrieving stock info for {ticker}")
    try:
        stock_info = {
            'AAPL': ('Apple Inc.', ['Technology', 'Consumer Electronics']),
            # Add more tickers as needed
        }
        company_name, industry_keywords = stock_info.get(ticker, ('Unknown Company', ['Unknown Industry']))
        print(f"Stock info - Name: {company_name}, Industry: {industry_keywords}")
        return company_name, industry_keywords
    except Exception as e:
        logger.error(f"Error retrieving stock info for {ticker}: {e}")
        print(f"Error retrieving stock info: {e}")
        return 'Unknown Company', ['Unknown Industry']


def get_market_sentiment(stock_ticker, directory, news_api_key, company_name, industry_keywords):
    today_date = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(directory, f"{stock_ticker}_sentiment.json")
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(2)).strftime('%Y-%m-%d')
    query_keywords = construct_query(stock_ticker, company_name, industry_keywords)
    logger.info(f"Fetching market sentiment for {stock_ticker}...using {query_keywords}")
    print(f"Sentiment query: {query_keywords}")

    sentiment_score_news_api = 0
    try:
        url = (
            f'https://newsapi.org/v2/everything?q={query_keywords}&'
            f'from={start_date}&to={end_date}&sortBy=publishedAt&apiKey={news_api_key}'
        )
        logger.info(url)
        print(f"News API URL: {url}")
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        print(f"Retrieved {len(articles)} articles")
        sentiment_score_news_api = analyze_sentiment(articles)
        print(f"Sentiment score from articles: {sentiment_score_news_api}")
    except Exception as e:
        logger.info(f"Error fetching sentiment for {stock_ticker} from News API: {e}")
        print(f"Error fetching sentiment: {e}")

    sentiment_data = {
        "date": today_date,
        "stock_ticker": stock_ticker,
        "sentiment_score": sentiment_score_news_api
    }
    save_sentiment_to_file(sentiment_data, file_path)
    return sentiment_score_news_api


def save_sentiment_to_file(sentiment_data, file_path):
    with open(file_path, 'w') as f:
        json.dump(sentiment_data, f, indent=4)
    logger.info(f"Sentiment data saved to {file_path}.")
    print(f"Sentiment saved to {file_path}")


def analyze_sentiment(articles):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = 0
    if articles:
        for article in articles[:5]:
            text = article.get('title', '') + " " + article.get('description', '')
            score = analyzer.polarity_scores(text)['compound']
            print(f"Article sentiment: {score}")
            sentiment_score += score
        sentiment_score /= len(articles[:5])
    print(f"Average sentiment score: {sentiment_score}")
    return sentiment_score


def construct_query(stock_ticker, company_name, industry_keywords):
    query_parts = [stock_ticker, company_name]
    query_parts.extend(industry_keywords)
    query = " OR ".join(query_parts)
    financial_focus = '("financial news" OR "earnings report" OR "market analysis")'
    full_query = f'{query} AND {financial_focus}'
    print(f"Constructed query: {full_query}")
    return full_query


def add_technical_indicators(data):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in data.columns]
        print(f"Columns before indicators: {data.columns.tolist()}")

        if 'Close' not in data.columns or 'Volume' not in data.columns:
            raise ValueError("Missing 'Close' or 'Volume' column")

        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
        data = data.dropna(subset=['Close', 'Volume'])
        print(f"Data shape after numeric conversion: {data.shape}")

        if len(data) >= 10 and 'SMA_10' not in data.columns:
            data['SMA_10'] = data['Close'].rolling(window=10).mean()
        if len(data) >= 50 and 'SMA_50' not in data.columns:
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
        print(f"SMA_10 last: {data['SMA_10'].iloc[-1] if 'SMA_10' in data.columns else 'N/A'}")
        print(f"SMA_50 last: {data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns else 'N/A'}")

        if len(data) >= 14 and 'RSI' not in data.columns:
            delta = data['Close'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))
        print(f"RSI last: {data['RSI'].iloc[-1] if 'RSI' in data.columns else 'N/A'}")

        if len(data) >= 26 and 'MACD' not in data.columns:
            short_ema = data['Close'].ewm(span=12, min_periods=12).mean()
            long_ema = data['Close'].ewm(span=26, min_periods=26).mean()
            data['MACD'] = short_ema - long_ema
        print(f"MACD last: {data['MACD'].iloc[-1] if 'MACD' in data.columns else 'N/A'}")

        if len(data) >= 30 and 'Volatility' not in data.columns:
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=30).std() * (252 ** 0.5)
        print(f"Volatility last: {data['Volatility'].iloc[-1] if 'Volatility' in data.columns else 'N/A'}")

        if len(data) >= 30 and 'Avg_Return' not in data.columns:
            data['Avg_Return'] = data['Returns'].rolling(window=30).mean()
        print(f"Avg_Return last: {data['Avg_Return'].iloc[-1] if 'Avg_Return' in data.columns else 'N/A'}")

        if len(data) >= 30 and 'Volume_SMA' not in data.columns:
            data['Volume_SMA'] = data['Volume'].rolling(window=30).mean()
        print(f"Volume_SMA last: {data['Volume_SMA'].iloc[-1] if 'Volume_SMA' in data.columns else 'N/A'}")

        print(f"Columns after indicators: {data.columns.tolist()}")
        return data

    except Exception as e:
        logger.error(f"Error adding indicators: {e}")
        print(f"Error in add_technical_indicators: {e}")
        return None
