import os
import logging
from datetime import datetime
from pandas.tseries.offsets import BDay
from src.config_loader import CONFIG
from src.preprocessing import load_and_preprocess_data
from src.model import train_model
from src.simulation import simulate_future_prices
from src.utils import ensure_directory_exists, save_predictions_to_file

logger = logging.getLogger('StockPrediction')


def process_stocks_for_predictions(stock_tickers):
    """
        Process multiple stocks and generate predictions.
        """

    results_dir = CONFIG["results_directory"]
    ensure_directory_exists(results_dir)
    model_type = CONFIG["model_type"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_file = os.path.join(results_dir, f"predictions_{timestamp}.txt")
    sentiment_ = CONFIG["sentiment_analysis"]
    buy_adjustment = CONFIG["adjustment_factors"]["buy"]
    sell_adjustment = CONFIG["adjustment_factors"]["sell"]
    sentiment_dir = CONFIG["news_info"]
    news_api_key = CONFIG["news_api_key"]
    for stock_ticker in stock_tickers:
        try:
            logger.info(f"\nProcessing {stock_ticker}")
            # Save earnings data to CSV
            data, sentiment_score = load_and_preprocess_data(stock_ticker, sentiment_, sentiment_dir, news_api_key)
            logger.info(f"Sentiment score for {stock_ticker}: {sentiment_score}")
            if data is None:
                logger.info(f"Skipping {stock_ticker}: No data available.")
                continue

            logger.info('Start train and test Model')
            # Train models for different prediction periods
            model_buy_1D, model_sell_1D, accuracy_buy_1D, accuracy_sell_1D, feature_columns = \
                train_model(data, model_type,
                            sentiment_,
                            period='1D')
            model_buy_1W, model_sell_1W, accuracy_buy_1W, accuracy_sell_1W, _ = train_model(data, model_type,
                                                                                            sentiment_,
                                                                                            period='1W')
            model_buy_1M, model_sell_1M, accuracy_buy_1M, accuracy_sell_1M, _ = train_model(data, model_type,
                                                                                            sentiment_,
                                                                                            period='1M')

            logger.info("\nSimulating for 1 Week...")
            logger.info(model_buy_1W)
            best_buy_day_1W, best_sell_day_1W, predicted_buy_prices_1W, predicted_sell_prices_1W = \
                simulate_future_prices(
                    model_buy_1W, model_sell_1W, data, feature_columns, period='1W'
                )

            logger.info("\nSimulating for 1 Month...")
            best_buy_day_1M, best_sell_day_1M, predicted_buy_prices_1M, predicted_sell_prices_1M = \
                simulate_future_prices(
                    model_buy_1M, model_sell_1M, data, feature_columns, period='1M'
                )

            # Perform Next Day Prediction
            logger.info("\nPerforming Next Day Prediction...")

            # Calculate the next business day
            # Get the next business day
            next_business_day = data.index[-1] + BDay(1)
            # Extract the last data point for prediction
            next_day_data = data[feature_columns].iloc[-1:]
            next_day_buy = model_buy_1D.predict(next_day_data)[0]
            next_day_sell = model_sell_1D.predict(next_day_data)[0]

            # Adjust predictions for thresholds
            adjusted_buy_next_day = next_day_buy * buy_adjustment  # Adjust for 2% lower buy threshold
            adjusted_sell_next_day = next_day_sell * sell_adjustment  # Adjust for 2% higher sell threshold

            # logger.info predictions
            logger.info(f"Next Business Day: {next_business_day.strftime('%Y-%m-%d')}")
            logger.info(f"Next Day Predicted Buy Price (Actual): {next_day_buy: .2f}")
            logger.info(f"Next Day Predicted Sell Price (Actual): {next_day_sell: .2f}")
            logger.info(f"Next Day Predicted Buy Price (Adjusted): {adjusted_buy_next_day: .2f}")
            logger.info(f"Next Day Predicted Sell Price (Adjusted): {adjusted_sell_next_day: .2f}")

            # Save next day predictions to file
            with open(result_file, 'a') as f:
                # Next Day Predictions
                next_day_header = f"\n{stock_ticker} - Next Day Prediction"
                next_day_prediction = (
                    f"Prediction Date: {next_business_day.strftime('%Y-%m-%d')}\n"
                    f"Actual Predicted Buy Price: {next_day_buy: .2f}\n"
                    f"Actual Predicted Sell Price: {next_day_sell: .2f}\n"
                    f"Adjusted Buy Price: {adjusted_buy_next_day: .2f}\n"
                    f"Adjusted Sell Price: {adjusted_sell_next_day: .2f}\n"
                    f"Accuracy (Buy): {accuracy_buy_1D: .2f}%\n"
                    f"Accuracy (Sell): {accuracy_sell_1D: .2f}%\n"
                    f"Sentiment Score: {sentiment_score: .2f}\n"
                )
                logger.info(next_day_header)
                logger.info(next_day_prediction)
                f.write(next_day_header + "\n" + next_day_prediction + "\n" + "-" * 40 + "\n")

                # Weekly Predictions
                logger.info(f"\n{stock_ticker} - 1 Week Prediction: ")
                logger.info(f"Best Buy Date: {best_buy_day_1W.strftime('%Y-%m-%d')}")
                logger.info(f"Best Sell Date: {best_sell_day_1W.strftime('%Y-%m-%d')}")
                save_predictions_to_file(
                    stock_ticker,
                    best_buy_day_1W.strftime("%Y-%m-%d"),
                    best_sell_day_1W.strftime("%Y-%m-%d"),
                    predicted_buy_prices_1W,
                    predicted_sell_prices_1W,
                    accuracy_buy_1W,
                    accuracy_sell_1W,
                    sentiment_score,
                    datetime.now().strftime("%Y-%m-%d"),
                    "1 Week",
                    result_file
                )

                # Monthly Predictions
                logger.info(f"\n{stock_ticker} - 1 Month Prediction: ")
                logger.info(f"Best Buy Date: {best_buy_day_1M.strftime('%Y-%m-%d')}")
                logger.info(f"Best Sell Date: {best_sell_day_1M.strftime('%Y-%m-%d')}")
                save_predictions_to_file(
                    stock_ticker,
                    best_buy_day_1M.strftime("%Y-%m-%d"),
                    best_sell_day_1M.strftime("%Y-%m-%d"),
                    predicted_buy_prices_1M,
                    predicted_sell_prices_1M,
                    accuracy_buy_1M,
                    accuracy_sell_1M,
                    sentiment_score,
                    datetime.now().strftime("%Y-%m-%d"),
                    "1 Month",
                    result_file
                )
        except Exception as e:
            logger.error(f"Error processing {stock_ticker}: {e}")

    logger.info(f"All predictions have been saved to {result_file}.")
    logger.info(f"******************-----------*******************.")
    logger.info(f"******************-----------*******************.")



