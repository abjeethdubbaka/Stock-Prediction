from app import app, db
import sqlite3


def check_schema():
    with app.app_context():
        # Connect to SQLite database
        conn = sqlite3.connect('instance/users.db')
        cursor = conn.cursor()

        # Get schema for all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            print(f"\nSchema for table '{table_name}':")
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  Column: {col[1]}, Type: {col[2]}, Not Null: {col[3]}, Primary Key: {col[5]}")

        conn.close()


if __name__ == "__main__":
    check_schema()
