from flask import Flask, render_template, url_for
from flask_login import LoginManager
from models import db, User  # Import db from models
import os
from pathlib import Path
from datetime import timedelta

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                                                    'instance', 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Session configuration
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Session lasts 7 days
app.config['SESSION_REFRESH_EACH_REQUEST'] = True  # Automatically extends session

# Initialize extensions
db.init_app(app)  # Initialize db with app
login_manager = LoginManager(app)
login_manager.login_view = 'auth.login'


@login_manager.user_loader
def load_user(user_id):
    with app.app_context():  # Add app context
        return db.session.get(User, int(user_id))


# Create tables
with app.app_context():
    # Ensure instance directory exists
    instance_path = Path('instance')
    instance_path.mkdir(exist_ok=True)

    # Create all tables
    db.create_all()

    # Verify tables were created
    # print("Tables created:", db.engine.table_names())

# Register blueprints
from auth import auth_bp
from routes import main_bp

app.register_blueprint(auth_bp)
app.register_blueprint(main_bp)


# @app.errorhandler(404)
# def not_found_error(error):
#     return render_template('404.html', home_url=url_for('main.home')), 404


if __name__ == '__main__':
    app.run(debug=True)
