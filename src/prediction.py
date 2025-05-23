import os
import logging
from datetime import datetime
from pandas.tseries.offsets import BDay
from src.config_loader import CONFIG
from src.preprocessing import load_and_preprocess_data
from src.model import train_model
from src.simulation import simulate_future_prices
from src.utils import ensure_directory_exists

logger = logging.getLogger('StockPrediction')


def process_stocks_for_predictions(stock_tickers, period='1W', return_results=False):
    results = []
    config = CONFIG
    print(f"CONFIG: {config}")

    for stock_ticker in stock_tickers:
        try:
            logger.info(f"\nProcessing {stock_ticker} for period {period}")
            print(f"Processing ticker: {stock_ticker}, period: {period}")
            data, sentiment_score = load_and_preprocess_data(
                stock_ticker,
                config["sentiment_analysis"],
                config["news_info"],
                config["news_api_key"],
                period
            )
            print(f"Data shape: {data.shape if data is not None else 'None'}")
            print(f"Sentiment score: {sentiment_score}")
            print(f"Latest date: {data.index.max() if data is not None else 'N/A'}")

            if data is None or data.empty:
                logger.info(f"Skipping {stock_ticker}: No data available.")
                print(f"Skipping {stock_ticker}: No data available.")
                continue

            # Train models
            model_buy_1D, model_sell_1D, _, _, features = train_model(
                data,
                config["model_type"],
                config["sentiment_analysis"],
                '1D'
            ) if period == '1D' else (None, None, None, None, None)
            model_buy, model_sell, acc_buy, acc_sell, features = train_model(
                data,
                config["model_type"],
                config["sentiment_analysis"],
                period
            )
            print(f"Features used: {features}")
            print(f"Buy model accuracy ({period}): {acc_buy}")
            print(f"Sell model accuracy ({period}): {acc_sell}")

            # Simulate future prices
            best_buy_day, best_sell_day, buy_prices, sell_prices = simulate_future_prices(
                model_buy, model_sell, data, features, period
            )
            print(f"Best buy day: {best_buy_day}")
            print(f"Best sell day: {best_sell_day}")
            print(f"Buy prices: {buy_prices}")
            print(f"Sell prices: {sell_prices}")

            # Next day prediction (only for 1D)
            next_day_pred = _get_next_day_prediction(data, features, model_buy_1D, model_sell_1D,
                                                     config) if period == '1D' else {}
            print(f"Next day prediction: {next_day_pred}")

            # Format results
            result = {
                "ticker": stock_ticker,
                "period": period,
                "best_buy_day": best_buy_day.strftime("%Y-%m-%d %H:%M:%S" if period == '1D' else "%Y-%m-%d"),
                "best_sell_day": best_sell_day.strftime("%Y-%m-%d %H:%M:%S" if period == '1D' else "%Y-%m-%d"),
                "buy_prices": [round(float(p), 2) for p in buy_prices],
                "sell_prices": [round(float(p), 2) for p in sell_prices],
                "accuracy_buy": round(acc_buy, 2),
                "accuracy_sell": round(acc_sell, 2),
                "sentiment_score": round(sentiment_score, 2),
                "dates": [f"{'Interval' if period == '1D' else 'Day'} {i + 1}" for i in range(len(buy_prices))],
                "next_day_prediction": next_day_pred
            }
            print(f"Formatted result: {result}")

            if not return_results:
                _save_to_file(result, config)
                print(f"Results saved for {stock_ticker}")

            results.append(result)

        except Exception as e:
            logger.error(f"Error processing {stock_ticker}: {e}")
            print(f"Error processing {stock_ticker}: {e}")
            if return_results:
                results.append({"error": str(e), "ticker": stock_ticker})

    return results if return_results else None


def _get_next_day_prediction(data, features, model_buy, model_sell, config):
    next_day_data = data[features].iloc[-1:]
    print(f"Next day input data: {next_day_data}")
    next_day_buy = model_buy.predict(next_day_data)[0]
    next_day_sell = model_sell.predict(next_day_data)[0]
    print(f"Raw predictions - Buy: {next_day_buy}, Sell: {next_day_sell}")

    return {
        "date": (data.index[-1] + pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S") if config.get(
            "live_data_interval") and config["live_data_interval"] in ['1m', '5m', '15m'] else (
                    data.index[-1] + BDay(1)).strftime("%Y-%m-%d"),
        "actual_buy": round(float(next_day_buy), 2),
        "actual_sell": round(float(next_day_sell), 2),
        "adjusted_buy": round(float(next_day_buy) * config["adjustment_factors"]["buy"], 2),
        "adjusted_sell": round(float(next_day_sell) * config["adjustment_factors"]["sell"], 2)
    }


def _save_to_file(result, config):
    ensure_directory_exists(config["results_directory"])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_file = os.path.join(config["results_directory"], f"predictions_{timestamp}.txt")
    print(f"Saving to file: {result_file}")

    with open(result_file, 'a') as f:
        f.write(f"\n{result['ticker']} - {result['period']} Prediction\n")
        f.write(f"Best Buy Date: {result['best_buy_day']}\n")
        f.write(f"Best Sell Date: {result['best_sell_day']}\n")
        f.write(f"Predicted Buy Prices: {result['buy_prices']}\n")
        f.write(f"Predicted Sell Prices: {result['sell_prices']}\n")
        f.write(f"Accuracy (Buy): {result['accuracy_buy']}%\n")
        f.write(f"Accuracy (Sell): {result['accuracy_sell']}%\n")
        f.write(f"Sentiment Score: {result['sentiment_score']}\n")
        if result['next_day_prediction']:
            f.write("Next Day Prediction:\n")
            f.write(f"  Date: {result['next_day_prediction']['date']}\n")
            f.write(f"  Actual Buy: {result['next_day_prediction']['actual_buy']}\n")
            f.write(f"  Actual Sell: {result['next_day_prediction']['actual_sell']}\n")
            f.write(f"  Adjusted Buy: {result['next_day_prediction']['adjusted_buy']}\n")
            f.write(f"  Adjusted Sell: {result['next_day_prediction']['adjusted_sell']}\n")
        f.write("-" * 40 + "\n")
