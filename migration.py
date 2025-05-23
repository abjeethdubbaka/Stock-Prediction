from app import app, db
from models import Trade

with app.app_context():
    try:
        # Delete all records from the trade table
        num_deleted = db.session.query(Trade).delete()
        db.session.commit()
        print(f"Deleted {num_deleted} records from the trade table.")
    except Exception as e:
        print(f"Error deleting records: {str(e)}")
        db.session.rollback()