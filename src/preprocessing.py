import os
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from src.config_loader import CONFIG
import requests
from nltk.sentiment import SentimentIntensityAnalyzer


def load_and_preprocess_data(stock_ticker, sentiment_):
    """Load stock data from CSV or download it if missing or outdated."""

    raw_file_path = f"{CONFIG['store_or_read_stock_data']}/{stock_ticker}.csv"
    print(f"Loading data for {stock_ticker} from file.")
    data = None
    sentiment_score = 0

    # Check if the file exists and if it's outdated
    if os.path.exists(raw_file_path):
        try:
            # Load existing data
            data = pd.read_csv(raw_file_path, index_col=0, parse_dates=True)
            print(f"Data loaded from {raw_file_path}.")

            # Ensure columns match expectations
            if 'Close' not in data.columns or not pd.api.types.is_numeric_dtype(data['Close']):
                raise ValueError("Invalid or missing 'Close' column in the dataset.")

            # Determine the latest available date in the dataset
            latest_date = data.index.max()
            if isinstance(latest_date, pd.Timestamp):
                latest_date = latest_date.date()
            elif isinstance(latest_date, str):
                latest_date = pd.to_datetime(latest_date).date()
            else:
                raise ValueError(f"Unexpected type for latest_date: {type(latest_date)}")

            # Determine the last market day
            today = pd.Timestamp(datetime.now()).date()
            last_market_day = pd.bdate_range(end=today, periods=1)[0].date()

            print(f"Latest date in data: {latest_date}")
            print(f"Last market day: {last_market_day}")

            # Fetch new data if the latest available date is before the last market day
            if latest_date < last_market_day:
                print(f"Data for {stock_ticker} is not current. Fetching new data...")
                new_data = download_stock_data(stock_ticker,
                                               start_date=(latest_date + timedelta(days=1)).strftime('%Y-%m-%d'))
                if new_data is not None:
                    data = pd.concat([data, new_data]).drop_duplicates()
                    print(f"Data updated with new rows for {stock_ticker}.")
        except Exception as e:
            print(f"Error loading or updating data for {stock_ticker}: {e}")
            data = download_stock_data(stock_ticker)
    else:
        print(f"No existing data found for {stock_ticker}. Fetching new data...")
        data = download_stock_data(stock_ticker)

    if data is None or data.empty:
        print(f"No data available for {stock_ticker}. Skipping.")
        return None, sentiment_score

    # Check and add missing indicators
    required_indicators = ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'Volatility', 'Avg_Return', 'Volume_SMA', "Volume"]
    missing_indicators = [ind for ind in required_indicators if ind not in data.columns]
    if missing_indicators:
        print(f"Adding missing indicators for {stock_ticker}: {missing_indicators}")
        try:
            data = add_technical_indicators(data)
        except Exception as e:
            print(f"Error adding indicators for {stock_ticker}: {e}")
            return None, sentiment_score

    # Save updated data only if new rows or indicators are added
    if missing_indicators or (latest_date and latest_date < last_market_day):
        data.to_csv(raw_file_path)
        print(f"Updated data saved to {raw_file_path}.")

    # Update sentiment scores if enabled
    if sentiment_ and ('Sentiment' not in data.columns or data['Sentiment'].iloc[-1] == 0):
        sentiment_score = fetch_sentiment_for_today(stock_ticker, "your_newsapi_key")
        data['Sentiment'] = sentiment_score
        print(f"Sentiment scores added for {stock_ticker}.")
        data.to_csv(raw_file_path)
        print(f"Updated data with sentiment saved to {raw_file_path}.")

    # Clean the data by removing rows with missing values
    data = data.dropna()

    return data, sentiment_score


def download_stock_data(stock_ticker, start_date='2020-01-01', end_date=pd.to_datetime("today").strftime('%Y-%m-%d')):
    """Download stock data from Yahoo Finance."""
    print(f"Downloading data for {stock_ticker} from Yahoo Finance.")
    try:
        stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
    except Exception as e:
        print(f"Error downloading data for {stock_ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

    if stock_data.empty:
        print(f"No data found for {stock_ticker}.")
        return pd.DataFrame()  # Return an empty DataFrame if no data is found

    # Flatten multi-index columns, if present
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = [col[0] if isinstance(col, tuple) else col for col in stock_data.columns]
        print(f"Flattened columns: {stock_data.columns.tolist()}")

    # Save the downloaded data
    save_path = f"E:/Stock Price Prediction System/data/{stock_ticker}.csv"
    stock_data.to_csv(save_path)
    print(f"Data for {stock_ticker} saved to {save_path}.")
    return stock_data


def fetch_sentiment_for_today(stock_ticker, newsapi_key):
    """Fetch the sentiment score for the current day from NewsAPI."""
    # Fetch market sentiment
    sentiment_score = get_news_sentiment(stock_ticker, datetime.today(), newsapi_key)
    return sentiment_score


def get_news_sentiment(stock_ticker, date, api_key):
    """Fetch sentiment from news API for a specific stock ticker."""
    print(f"Fetching market sentiment for {stock_ticker} on {date}...")
    sentiment_score = 0
    try:
        start_date = date.strftime('%Y-%m-%d')
        end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')

        newsapi_url = f'https://newsapi.org/v2/everything?q={stock_ticker}&from={start_date}&to={end_date}&apiKey={api_key}'
        response = requests.get(newsapi_url)
        response.raise_for_status()
        articles = response.json().get('articles', [])

        # Perform sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = [analyzer.polarity_scores(article['title'])['compound'] for article in articles if
                            'title' in article]
        sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        print(f"Sentiment score for {stock_ticker} on {date}: {sentiment_score}")
    except Exception as e:
        print(f"Error fetching sentiment for {stock_ticker}: {e}")

    return sentiment_score


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
        print(f"Error adding indicators: {e}")
        return None  # Return None if any error occurs


def update_data(existing_data, stock_ticker, config):
    """Update outdated data"""
    last_date = existing_data.index.max().date()
    today = datetime.now().date()

    if last_date < today:
        new_data = download_stock_data(
            stock_ticker,
            start=(last_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            config=config
        )
        if not new_data.empty:
            return pd.concat([existing_data, new_data]).drop_duplicates()
    return existing_data
