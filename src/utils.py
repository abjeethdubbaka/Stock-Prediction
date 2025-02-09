import os


def ensure_directory_exists(directory):
    """
    Ensure that the given directory exists.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_predictions_to_file(stock_ticker, best_buy_day, best_sell_day, buy_prices, sell_prices, acc_buy, acc_sell,
                             sentiment_score, date, period, result_file):
    """
    Save the predictions for the given stock to a result file.
    """
    prediction_output = (
            f"\n{stock_ticker} - {period} Prediction\n"
            f"Best Buy Date: {best_buy_day}\n"
            f"Best Sell Date: {best_sell_day}\n"
            f"Predicted Buy Price: {buy_prices[0]:.2f}\n"
            f"Predicted Sell Price: {sell_prices[0]:.2f}\n"
            f"Accuracy for Buy: {acc_buy:.2f}%\n"
            f"Accuracy for Sell: {acc_sell:.2f}%\n"
            f"Sentiment Score: {sentiment_score:.2f}\n"
            f"Prediction Date: {date}\n"
            + "-" * 40 + "\n"
    )
    with open(result_file, 'a') as f:
        f.write(prediction_output)
    print(prediction_output)  # Also display on the console
