import logging

from src.prediction import process_stocks_for_predictions
from src.logger import setup_logging

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger('StockPrediction')
    logger.info("Starting stock prediction process")
    stock_tickers = ["AAPL"]
    process_stocks_for_predictions(stock_tickers)

