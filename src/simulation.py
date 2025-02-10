import pandas as pd
import logging

logger = logging.getLogger('StockPrediction')


def simulate_future_prices(model_buy, model_sell, data, feature_columns, period='1W'):
    """
    Simulate future prices dynamically using trained models for buy and sell, considering only market open days.
    Enforces that the sell day must occur after the buy day.
    """
    # Determine the number of future trading days to simulate
    num_days = 5 if period == '1W' else 22  # 5 business days for 1 week, 22 for 1 month

    # Generate future business days starting from the last available date in the dataset
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_days, freq='B')

    logger.info(f"Simulating future prices for {num_days} market days starting from {last_date}...")

    predicted_buy_prices = []
    predicted_sell_prices = []

    # Start with the last known data point
    last_data_point = data.iloc[-1].copy()

    for i, future_date in enumerate(future_dates):
        # Prepare the features dynamically for the model
        future_data = pd.DataFrame([last_data_point[feature_columns]], columns=feature_columns)

        # Predict buy and sell prices
        predicted_buy_price = model_buy.predict(future_data)[0]
        predicted_sell_price = model_sell.predict(future_data)[0]

        predicted_buy_prices.append(predicted_buy_price)
        predicted_sell_prices.append(predicted_sell_price)

        # Log the predictions
        logger.info(f"Day {i + 1} ({future_date.strftime('%Y-%m-%d')}):")
        logger.info(f"  Predicted Buy Price: {predicted_buy_price:.2f}")
        logger.info(f"  Predicted Sell Price: {predicted_sell_price:.2f}")

        # Update the last data point for the next iteration
        prev_close = last_data_point['Close']
        last_data_point['Close'] = predicted_buy_price  # Assume predicted buy price as the next close
        last_data_point['SMA_10'] = (last_data_point['SMA_10'] * 9 + predicted_buy_price) / 10  # Update SMA
        last_data_point['SMA_50'] = (last_data_point['SMA_50'] * 49 + predicted_buy_price) / 50  # Update SMA

        # Update RSI
        delta = predicted_buy_price - prev_close
        gain = max(delta, 0)
        loss = max(-delta, 0)
        last_data_point['RSI'] = 100 - (100 / (1 + (gain / loss if loss != 0 else 1)))

    # Find the best buy/sell day ensuring sell day is after buy day
    best_buy_idx = 0
    best_sell_idx = 0
    max_profit = 0

    for buy_idx in range(len(predicted_buy_prices) - 1):
        for sell_idx in range(buy_idx + 1, len(predicted_sell_prices)):
            profit = predicted_sell_prices[sell_idx] - predicted_buy_prices[buy_idx]
            if profit > max_profit:
                max_profit = profit
                best_buy_idx = buy_idx
                best_sell_idx = sell_idx

    best_buy_day = future_dates[best_buy_idx]
    best_sell_day = future_dates[best_sell_idx]

    logger.info(f"\nBest Buy Day: {best_buy_day.strftime('%Y-%m-%d')} with Price: {predicted_buy_prices[best_buy_idx]:.2f}")
    logger.info(f"Best Sell Day: {best_sell_day.strftime('%Y-%m-%d')} with Price: {predicted_sell_prices[best_sell_idx]:.2f}")

    return best_buy_day, best_sell_day, predicted_buy_prices, predicted_sell_prices
