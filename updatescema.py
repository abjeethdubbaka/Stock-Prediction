import sqlite3
import os

def update_schema(db_path):
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Step 1: Rename user table
        print("Renaming 'user' table to 'user_old'")
        cursor.execute("ALTER TABLE user RENAME TO user_old")

        # Step 2: Create new user table
        print("Creating new 'user' table")
        cursor.execute("""
            CREATE TABLE user (
                id INTEGER PRIMARY KEY,
                username VARCHAR(80) NOT NULL UNIQUE,
                password_hash VARCHAR(120) NOT NULL
            )
        """)

        # Step 3: Migrate data from user_old to user
        print("Migrating data from 'user_old' to 'user'")
        cursor.execute("""
            INSERT INTO user (id, username, password_hash)
            SELECT id, username, password FROM user_old
        """)

        # Step 4: Drop old user table
        print("Dropping 'user_old' table")
        cursor.execute("DROP TABLE user_old")

        # Step 5: Drop unused tables
        for table in ['users', 'trades', 'stocks']:
            print(f"Dropping '{table}' table")
            cursor.execute(f"DROP TABLE IF EXISTS {table}")

        # Step 6: Verify schemas
        print("\nVerifying schemas:")
        tables = ['user', 'trade', 'activity']
        for table in tables:
            print(f"\nSchema for table '{table}':")
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  Column: {col[1]}, Type: {col[2]}, Not Null: {col[3]}, Primary Key: {col[5]}")

        # Step 7: Check foreign keys
        print("\nChecking foreign key integrity:")
        cursor.execute("PRAGMA foreign_key_check")
        fk_issues = cursor.fetchall()
        if fk_issues:
            print("Foreign key issues found:", fk_issues)
        else:
            print("No foreign key issues found.")

        # Commit changes
        conn.commit()
        print("\nSchema update completed successfully.")

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
        update_schema(db_path)