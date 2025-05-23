import pandas as pd
import logging

logger = logging.getLogger('StockPrediction')

def simulate_future_prices(model_buy, model_sell, data, feature_columns, period='1W'):
    """
    Simulate future prices dynamically using trained models for buy and sell, considering only market open days.
    Enforces that the sell day must occur after the buy day.
    """
    # Determine the number of future trading days or intervals to simulate
    if period == '1D':
        num_days = 1  # Single 5-minute interval for intraday
        freq = '5min' if CONFIG.get("live_data_interval") else 'B'
    else:
        num_days = 5 if period == '1W' else 22  # 5 business days for 1 week, 22 for 1 month
        freq = 'B'

    # Generate future dates starting from the last available date
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + (pd.Timedelta(minutes=5) if period == '1D' and CONFIG.get("live_data_interval") else pd.Timedelta(days=1)),
                                periods=num_days, freq=freq)

    logger.info(f"Simulating future prices for {num_days} {'intervals' if period == '1D' else 'market days'} starting from {last_date}...")
    print(f"Simulating for {num_days} {'intervals' if period == '1D' else 'days'}, freq: {freq}, from {last_date}")

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
        logger.info(f"{'Interval' if period == '1D' else 'Day'} {i + 1} ({future_date.strftime('%Y-%m-%d %H:%M:%S' if period == '1D' else '%Y-%m-%d')}):")
        logger.info(f"  Predicted Buy Price: {predicted_buy_price:.2f}")
        logger.info(f"  Predicted Sell Price: {predicted_sell_price:.2f}")
        print(f"{'Interval' if period == '1D' else 'Day'} {i + 1}: Buy={predicted_buy_price:.2f}, Sell={predicted_sell_price:.2f}")

        # Update the last data point for the next iteration
        prev_close = last_data_point['Close']
        last_data_point['Close'] = predicted_buy_price  # Assume predicted buy price as the next close
        last_data_point['SMA_10'] = (last_data_point['SMA_10'] * 9 + predicted_buy_price) / 10 if 'SMA_10' in last_data_point else last_data_point['SMA_10']
        last_data_point['SMA_50'] = (last_data_point['SMA_50'] * 49 + predicted_buy_price) / 50 if 'SMA_50' in last_data_point else last_data_point['SMA_50']

        # Update RSI
        delta = predicted_buy_price - prev_close
        gain = max(delta, 0)
        loss = max(-delta, 0)
        last_data_point['RSI'] = 100 - (100 / (1 + (gain / loss if loss != 0 else 1))) if 'RSI' in last_data_point else last_data_point['RSI']

    # For 1D, use the single prediction
    if period == '1D':
        best_buy_idx = best_sell_idx = 0
    else:
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

    logger.info(f"\nBest Buy {'Interval' if period == '1D' else 'Day'}: {best_buy_day.strftime('%Y-%m-%d %H:%M:%S' if period == '1D' else '%Y-%m-%d')} with Price: {predicted_buy_prices[best_buy_idx]:.2f}")
    logger.info(f"Best Sell {'Interval' if period == '1D' else 'Day'}: {best_sell_day.strftime('%Y-%m-%d %H:%M:%S' if period == '1D' else '%Y-%m-%d')} with Price: {predicted_sell_prices[best_sell_idx]:.2f}")
    print(f"Best Buy: {best_buy_day}, Price: {predicted_buy_prices[best_buy_idx]:.2f}")
    print(f"Best Sell: {best_sell_day}, Price: {predicted_sell_prices[best_sell_idx]:.2f}")

    return best_buy_day, best_sell_day, predicted_buy_prices, predicted_sell_prices