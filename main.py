from flask import Flask
from flask_login import LoginManager
from auth import auth_bp
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

# Import user loader after app creation to avoid circular imports
from models import load_user
login_manager.user_loader(load_user)

# Register blueprints
app.register_blueprint(auth_bp)

# # Import routes after app creation
from main import routes

if __name__ == "__main__":
    app.run(debug=True)