from models import db, Trade
from datetime import datetime
from app import app

app = create_app()

with app.app_context():
    print("Checking trade table...")
    trades = Trade.query.all()
    if not trades:
        print("No trades found.")
    else:
        for trade in trades:
            print(f"Trade ID: {trade.id}")
            print(f"Symbol: {trade.symbol}")
            print(f"Entry Date: {trade.entry_date}")
            print(f"Exit Date: {trade.exit_date}")
            print("-" * 40)