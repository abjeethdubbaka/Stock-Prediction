import time
import logging
from src.prediction import process_stocks_for_predictions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    # Test predictions for multiple periods
    stock_tickers = ['AAPL']
    periods = ['1D', '1W', '1M']

    for period in periods:
        print(f"\nTesting period: {period}")
        results = None
        for attempt in range(5):
            try:
                results = process_stocks_for_predictions(stock_tickers, period=period, return_results=True)
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {period}: {e}")
                time.sleep(5 * (2 ** attempt))  # Exponential backoff
        if results:
            print(f"\nFinal Results for {period}:")
            for result in results:
                print(result)
        else:
            print(f"Failed to generate predictions for {period} after retries.")
        time.sleep(10)  # Delay between periods to avoid rate limits