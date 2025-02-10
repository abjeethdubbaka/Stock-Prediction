from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from src.config_loader import CONFIG
import logging

logger = logging.getLogger('StockPrediction')


def train_model(data, model_choice, sentiment, period='1D'):
    """
    Train a model to predict the best buy (min price) and sell (max price) prices,
    including sentiment as a feature if available.
    """
    # Define base features
    features = CONFIG["features"]
    if sentiment and period == '1D':
        features.append('Sentiment')  # Add sentiment as a feature if available

    # Ensure all required features are in the dataset
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise KeyError(f"The following features are missing from the dataset: {missing_features}")

    # Select features for training
    X = data[features]

    # Define targets for buy and sell prices
    if period == '1D':
        y_buy = data['Close'].shift(-1).rolling(window=2).min()  # Min price in the next day
        y_sell = data['Close'].shift(-1).rolling(window=2).max()  # Max price in the next day
    elif period == '1W':
        y_buy = data['Close'].shift(-5).rolling(window=5).min()  # Min price in the next week
        y_sell = data['Close'].shift(-5).rolling(window=5).max()  # Max price in the next week
    elif period == '1M':
        y_buy = data['Close'].shift(-22).rolling(window=22).min()  # Min price in the next month
        y_sell = data['Close'].shift(-22).rolling(window=22).max()  # Max price in the next month
    else:
        logger.error("Invalid period. Choose from '1D', '1W', or '1M'.")
        raise ValueError("Invalid period. Choose from '1D', '1W', or '1M'.")

    # Drop NaNs from targets and align features
    y_buy = y_buy.dropna()
    y_sell = y_sell.dropna()
    X = X.iloc[:len(y_buy)]  # Align features with the targets

    # Split data into training and test sets
    X_train, X_test, y_train_buy, y_test_buy = train_test_split(X, y_buy, test_size=0.2, random_state=42)
    _, _, y_train_sell, y_test_sell = train_test_split(X, y_sell, test_size=0.2, random_state=42)

    # Define models (e.g., RandomForestRegressor, LinearRegression)
    if model_choice == "Random Forest":
        model_buy = RandomForestRegressor(n_estimators=100, random_state=42)
        model_sell = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model_buy = LinearRegression()
        model_sell = LinearRegression()

    # Train models for buy and sell prices
    model_buy.fit(X_train, y_train_buy)
    model_sell.fit(X_train, y_train_sell)

    # Evaluate models' accuracy
    accuracy_buy = model_buy.score(X_test, y_test_buy) * 100
    accuracy_sell = model_sell.score(X_test, y_test_sell) * 100

    return model_buy, model_sell, accuracy_buy, accuracy_sell, features
