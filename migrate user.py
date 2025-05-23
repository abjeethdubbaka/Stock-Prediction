from app import app, db
from models import User
from sqlalchemy import inspect, text
from werkzeug.security import generate_password_hash


with app.app_context():
    print("Checking user table schema...")
    try:
        # Inspect the user table
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('user')]

        # Add password column if not exists
        if 'password' not in columns:
            with db.engine.connect() as connection:
                connection.execute(text("ALTER TABLE user ADD COLUMN password TEXT NOT NULL DEFAULT 'default'"))
                connection.commit()
            print("Added password column to user table.")
        else:
            print("Password column already exists.")

        # Update existing users with a default hashed password
        users = User.query.all()
        updated_count = 0
        default_password = generate_password_hash("default_password")
        for user in users:
            if not user.password:
                user.password = default_password
                updated_count += 1
        db.session.commit()
        print(f"Updated {updated_count} users with default password.")
        if updated_count > 0:
            print("Users with default password should reset their password via /profile.")
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        db.session.rollback()