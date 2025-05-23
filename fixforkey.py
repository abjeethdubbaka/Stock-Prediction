import sqlite3
import os

def fix_foreign_keys(db_path):
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Step 1: Check activities table data
        print("Checking data in 'activities' table")
        cursor.execute("SELECT * FROM activities LIMIT 5")
        rows = cursor.fetchall()
        if rows:
            print("Sample data in 'activities':", rows)
        else:
            print("No data found in 'activities'")

        # Step 2: Drop activities table (since activity table is used by models.py)
        print("Dropping 'activities' table")
        cursor.execute("DROP TABLE IF EXISTS activities")

        # Step 3: Verify remaining tables
        print("\nVerifying schemas:")
        tables = ['user', 'trade', 'activity']
        for table in tables:
            print(f"\nSchema for table '{table}':")
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  Column: {col[1]}, Type: {col[2]}, Not Null: {col[3]}, Primary Key: {col[5]}")

        # Step 4: Check foreign keys
        print("\nChecking foreign key integrity:")
        cursor.execute("PRAGMA foreign_key_check")
        fk_issues = cursor.fetchall()
        if fk_issues:
            print("Foreign key issues found:", fk_issues)
        else:
            print("No foreign key issues found.")

        # Commit changes
        conn.commit()
        print("\nForeign key fix completed successfully.")

    except sqlite3.Error as e:
        print(f"Error occurred: {e}")
        conn.rollback()

    finally:
        conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), 'instance', 'users.db')
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
    else:
        fix_foreign_keys(db_path)