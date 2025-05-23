from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from src.config_loader import CONFIG
import logging
import pandas as pd

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
        # For intraday, predict min/max within the next hour (12 * 5min intervals)
        y_buy = data['Close'].shift(-12).rolling(window=12).min() if CONFIG.get("live_data_interval") else data['Close'].shift(-1).rolling(window=2).min()
        y_sell = data['Close'].shift(-12).rolling(window=12).max() if CONFIG.get("live_data_interval") else data['Close'].shift(-1).rolling(window=2).max()
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
    X = X.loc[y_buy.index]  # Align features with the targets

    # Split data into training and test sets
    X_train, X_test, y_train_buy, y_test_buy = train_test_split(X, y_buy, test_size=0.2, random_state=42)
    _, _, y_train_sell, y_test_sell = train_test_split(X, y_sell, test_size=0.2, random_state=42)

    # Define models
    if model_choice == "Random Forest":
        model_buy = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
        model_sell = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
    else:
        model_buy = LinearRegression()
        model_sell = LinearRegression()

    # Train models
    model_buy.fit(X_train, y_train_buy)
    model_sell.fit(X_train, y_train_sell)

    # Evaluate with cross-validation
    cv_scores_buy = cross_val_score(model_buy, X, y_buy, cv=5, scoring='r2')
    cv_scores_sell = cross_val_score(model_sell, X, y_sell, cv=5, scoring='r2')
    accuracy_buy = cv_scores_buy.mean() * 100
    accuracy_sell = cv_scores_sell.mean() * 100

    logger.info(f"Cross-validated accuracy - Buy: {accuracy_buy:.2f}%, Sell: {accuracy_sell:.2f}%")
    print(f"Cross-validated accuracy - Buy: {accuracy_buy:.2f}%, Sell: {accuracy_sell:.2f}%")

    return model_buy, model_sell, accuracy_buy, accuracy_sell, features