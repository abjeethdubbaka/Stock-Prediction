import os
import pandas as pd
import yfinance as yf
import json
import logging
import requests
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from src.config_loader import CONFIG
from nltk.sentiment import SentimentIntensityAnalyzer
from src.utils import ensure_directory_exists

logger = logging.getLogger('StockPrediction')


def load_and_preprocess_data(stock_ticker, include_sentiment, sentiment_dir, api_key):
    """Load stock data from CSV or download it if missing or outdated."""

    raw_file_path = f"{CONFIG['store_or_read_stock_data']}/{stock_ticker}.csv"
    logger.info(f"Loading data for {stock_ticker} from file.")
    data = None
    sentiment_score = 0

    # Check if the file exists and if it's outdated
    if os.path.exists(raw_file_path):
        try:
            # Load existing data
            data = pd.read_csv(raw_file_path, index_col=0, parse_dates=True)
            logger.info(f"Data loaded from {raw_file_path}.")

            # Ensure columns match expectations
            if 'Close' not in data.columns or not pd.api.types.is_numeric_dtype(data['Close']):
                raise ValueError("Invalid or missing 'Close' column in the dataset.")

            # Determine the latest available date in the dataset
            latest_date = data.index.max()
            if isinstance(latest_date, pd.Timestamp):
                latest_date = latest_date.date()
            else:
                latest_date = pd.to_datetime(latest_date).date()

            # Determine the last market day
            today = pd.Timestamp(datetime.now()).date()
            last_market_day = pd.bdate_range(end=today, periods=2)[-2].date()

            logger.info(f"Latest date in data: {latest_date}")
            logger.info(f"Last market day: {last_market_day}")

            # Fetch new data if the latest available date is before the last market day
            if latest_date < last_market_day:
                logger.info(f"Data for {stock_ticker} is not current. Fetching new data...")
                new_data = download_stock_data(stock_ticker,
                                               start_date=(pd.Timestamp(latest_date) + BDay(1)).strftime('%Y-%m-%d'))
                if new_data is not None and not new_data.empty:
                    # Add technical indicators for new rows only
                    new_data = add_technical_indicators(new_data)
                    logger.info('new_data', new_data)
                    data = pd.concat([data, new_data]).drop_duplicates()
                    logger.info(f"Data updated with new rows for {stock_ticker}.")
        except Exception as e:
            logger.error(f"Error loading or updating data for {stock_ticker}: {e}")
            data = download_stock_data(stock_ticker)
    else:
        logger.info(f"No existing data found for {stock_ticker}. Fetching new data...")
        data = download_stock_data(stock_ticker)
        logger.info(data.index.date)
        if data is not None and not data.empty:
            data = add_technical_indicators(data)

    if data is None or data.empty:
        logger.warning(f"No data available for {stock_ticker}. Skipping.")
        return None, sentiment_score

    # Check and add missing indicators
    required_indicators = ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'Volatility', 'Avg_Return', 'Volume_SMA', "Volume"]
    missing_indicators = [ind for ind in required_indicators if ind not in data.columns]
    if missing_indicators:
        logger.info(f"Adding missing indicators for {stock_ticker}: {missing_indicators}")
        try:
            data = add_technical_indicators(data)
        except Exception as e:
            logger.error(f"Error adding indicators for {stock_ticker}: {e}")
            return None, sentiment_score

    # Save updated data only if new rows or indicators are added
    data.to_csv(raw_file_path)
    logger.info(f"Updated data saved to {raw_file_path}.")

    # Process sentiment if needed
    sentiment_file = f"{sentiment_dir}/{stock_ticker}_sentiment.json"
    ensure_directory_exists(sentiment_dir)

    if include_sentiment:
        if 'Sentiment' not in data.columns:
            data['Sentiment'] = 0.0

        today = pd.Timestamp(datetime.now()).normalize()
        previous_business_day = pd.bdate_range(end=today, periods=2)[-2].date()
        previous_business_day = pd.Timestamp(previous_business_day)

        sentiment_data = {}
        if os.path.exists(sentiment_file):
            with open(sentiment_file, "r") as f:
                sentiment_data = json.load(f)

        if str(previous_business_day) in sentiment_data:
            sentiment_score = sentiment_data[str(previous_business_day)]
            logger.info(f"Using cached sentiment score ({sentiment_score}) for {previous_business_day}.")
        else:
            company_name, industry_keywords = get_stock_info(stock_ticker)
            sentiment_score = get_market_sentiment(stock_ticker, sentiment_dir, api_key, company_name,
                                                   industry_keywords)
            sentiment_data[str(previous_business_day)] = sentiment_score
            with open(sentiment_file, "w") as f:
                json.dump(sentiment_data, f, indent=4)
            logger.info(f"Sentiment score ({sentiment_score}) saved for {previous_business_day}.")

        if previous_business_day in data.index:
            data.loc[previous_business_day, "Sentiment"] = sentiment_score
            data.to_csv(raw_file_path)
        else:
            logger.warning(f"Warning: Previous business day ({previous_business_day}) not in stock data. ")

    # Clean data and drop NaNs
    data = data.dropna()

    return data, sentiment_score


def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    # Get company name and industry
    company_name = info.get('longName', 'Unknown Company')
    industry = info.get('industry', 'Unknown Industry')

    # Generate industry keywords by splitting the industry string into words
    industry_keywords = industry.split() if industry != 'Unknown Industry' else []

    return company_name, industry_keywords


def download_stock_data(stock_ticker, start_date=CONFIG["start_date"],
                        end_date=pd.to_datetime("today").strftime('%Y-%m-%d')):
    """Download stock data from Yahoo Finance."""
    logger.info(f"Downloading data for {stock_ticker} from Yahoo Finance.")
    try:
        stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
    except Exception as e:
        logger.error(f"Error downloading data for {stock_ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

    if stock_data.empty:
        logger.info(f"No data found for {stock_ticker}.")
        return pd.DataFrame()  # Return an empty DataFrame if no data is found

    # Flatten multi-index columns, if present
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = [col[0] if isinstance(col, tuple) else col for col in stock_data.columns]
        logger.info(f"Flattened columns: {stock_data.columns.tolist()}")

    # Save the downloaded data
    save_path = f"{CONFIG['store_or_read_stock_data']}/{stock_ticker}.csv"
    stock_data.to_csv(save_path)
    logger.info(f"Data for {stock_ticker} saved to {save_path}.")
    return stock_data


def get_market_sentiment(stock_ticker, directory, news_api_key, company_name, industry_keywords):
    """Fetch market sentiment from News API and save it locally."""
    today_date = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(directory, f"{stock_ticker}_sentiment.json")
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(2)).strftime('%Y-%m-%d')
    # Fetch sentiment using News API
    query_keywords = construct_query(stock_ticker, company_name, industry_keywords)

    logger.info(f"Fetching market sentiment for {stock_ticker}...using {query_keywords}")

    # 1. Fetch sentiment from News API
    sentiment_score_news_api = 0  # Default sentiment if no news found from News API
    try:
        url = (
            f'https://newsapi.org/v2/everything?q={query_keywords}&'
            f'from={start_date}&to={end_date}&sortBy=publishedAt&apiKey={news_api_key}'
        )
        logger.info(url)
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        sentiment_score_news_api = analyze_sentiment(articles)
    except Exception as e:
        logger.info(f"Error fetching sentiment for {stock_ticker} from News API: {e}")

    # Save sentiment data locally
    sentiment_data = {
        "date": today_date,
        "stock_ticker": stock_ticker,
        "sentiment_score": sentiment_score_news_api
    }
    save_sentiment_to_file(sentiment_data, file_path)

    return sentiment_score_news_api


def save_sentiment_to_file(sentiment_data, file_path):
    """Save sentiment data to a file."""
    with open(file_path, 'w') as f:
        json.dump(sentiment_data, f, indent=4)
    logger.info(f"Sentiment data saved to {file_path}.")


def analyze_sentiment(articles):
    """Analyze sentiment of the articles using VADER SentimentIntensityAnalyzer."""
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = 0
    if articles:
        for article in articles[:5]:  # Limit to the top 5 articles
            text = article.get('title', '') + " " + article.get('description', '')
            sentiment_score += analyzer.polarity_scores(text)['compound']
        sentiment_score /= len(articles[:5])  # Average sentiment score
    return sentiment_score


def construct_query(stock_ticker, company_name, industry_keywords):
    """
    Construct a meaningful query for fetching market sentiment.
    """
    # Base query using stock ticker and company name
    query_parts = [stock_ticker, company_name]

    # Add industry-specific keywords
    query_parts.extend(industry_keywords)

    # Combine with OR logical operator
    query = " OR ".join(query_parts)
    financial_focus = '("financial news" OR "earnings report" OR "market analysis")'
    return f'{query} AND {financial_focus}'


def add_technical_indicators(data):
    """Add technical indicators and historical performance metrics to the stock data."""
    try:
        # Flatten multi-level columns if they exist
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in data.columns]

        # Ensure required columns exist
        if 'Close' not in data.columns:
            raise ValueError("The 'Close' column is missing from the data.")
        if 'Volume' not in data.columns:
            raise ValueError("The 'Volume' column is missing from the data.")

        # Ensure 'Close' and 'Volume' are numeric
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
        data = data.dropna(subset=['Close', 'Volume'])

        # Add SMA (if not present)
        if 'SMA_10' not in data.columns:
            data['SMA_10'] = data['Close'].rolling(window=10).mean()
        if 'SMA_50' not in data.columns:
            data['SMA_50'] = data['Close'].rolling(window=50).mean()

        # RSI
        if 'RSI' not in data.columns:
            delta = data['Close'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        if 'MACD' not in data.columns:
            short_ema = data['Close'].ewm(span=12, min_periods=12).mean()
            long_ema = data['Close'].ewm(span=26, min_periods=26).mean()
            data['MACD'] = short_ema - long_ema

        # Volatility
        if 'Volatility' not in data.columns:
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=30).std() * (252 ** 0.5)

        # Avg Return
        if 'Avg_Return' not in data.columns:
            data['Avg_Return'] = data['Returns'].rolling(window=30).mean()

        # Volume SMA
        if 'Volume_SMA' not in data.columns:
            data['Volume_SMA'] = data['Volume'].rolling(window=30).mean()

        return data

    except Exception as e:
        logger.error(f"Error adding indicators: {e}")
        return None  # Return None if any error occurs
