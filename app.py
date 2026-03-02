import os
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps

from config import get_config
from utils.logger import setup_logger

load_dotenv()

app = Flask(__name__)
config_class = get_config(os.environ.get('FLASK_ENV', 'development'))
app.config.from_object(config_class)

logger = setup_logger(__name__)
db = SQLAlchemy(app)

UPLOAD_FOLDER = app.config.get('UPLOAD_FOLDER', 'uploads')
DATASET_FOLDER = app.config.get('DATASET_FOLDER', 'datasets')
VECTOR_DB_FOLDER = app.config.get('VECTOR_DB_FOLDER', 'vector_db')

for folder in [UPLOAD_FOLDER, DATASET_FOLDER, VECTOR_DB_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username} ({self.role})>'

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    restaurant = db.Column(db.String(200), index=True, nullable=False)
    reviewer = db.Column(db.String(200), default="anonymous")
    text = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Float, nullable=True)
    sentiment = db.Column(db.String(50), index=True)
    score = db.Column(db.Float)
    keywords = db.Column(db.String(500))
    categories = db.Column(db.String(500))
    source_file = db.Column(db.String(100), index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f'<Review {self.id}: {self.restaurant}>'

with app.app_context():
    db.create_all()
    logger.info("Database tables created/verified")

# Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        user = db.session.get(User, session['user_id'])
        if not user or not user.is_active:
            session.clear()
            flash('Please login to continue.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def manager_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        user = db.session.get(User, session['user_id'])
        if not user or not user.is_active or user.role != 'manager':
            session.clear()
            flash('Manager access required.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ========== ROUTES ==========

@app.route("/")
def index():
    """Redirect to appropriate dashboard or login"""
    if 'user_id' in session:
        user = db.session.get(User, session['user_id'])
        if user and user.is_active:
            if user.role == 'manager':
                return redirect('/manager/dashboard')
            else:
                return redirect('/user/dashboard')
        else:
            # Invalid session - clear it
            session.clear()
    return redirect(url_for('login'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == "POST":
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Username and password are required.', 'danger')
            return render_template('login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            if not user.is_active:
                flash('Your account has been deactivated. Please contact support.', 'danger')
                return render_template('login.html')
            
            # Login successful
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            session.permanent = True
            
            # Update last login
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            flash(f'Welcome back, {user.username}!', 'success')
            logger.info(f"User logged in: {username} ({user.role})")
            
            # Redirect based on role
            if user.role == 'manager':
                return redirect('/manager/dashboard')
            else:
                return redirect('/user/dashboard')
        else:
            flash('Invalid username or password.', 'danger')
            logger.warning(f"Failed login attempt for username: {username}")
    
    return render_template('login.html')

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == "POST":
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        role = request.form.get('role', 'user')
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required.', 'danger')
            return render_template('signup.html')
        
        if len(username) < 3:
            flash('Username must be at least 3 characters long.', 'danger')
            return render_template('signup.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('signup.html')
        
        if role not in ['user', 'manager']:
            flash('Invalid role selected.', 'danger')
            return render_template('signup.html')
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose another.', 'danger')
            return render_template('signup.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please use another email.', 'danger')
            return render_template('signup.html')
        
        # Create new user
        try:
            new_user = User(username=username, email=email, role=role)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            
            flash(f'Account created successfully! Welcome {username}. Please login.', 'success')
            logger.info(f"New user registered: {username} ({role})")
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating user: {e}", exc_info=True)
            flash('An error occurred. Please try again.', 'danger')
    
    return render_template('signup.html')

@app.route("/logout")
def logout():
    username = session.get('username', 'User')
    session.clear()
    flash(f'Goodbye {username}! You have been logged out successfully.', 'info')
    logger.info(f"User logged out: {username}")
    return redirect(url_for('login'))

# Import route modules
from manager import register_manager_routes
from user import register_user_routes

# Register routes
register_manager_routes(app, db, User, Review, manager_required, login_required)
register_user_routes(app, db, User, Review, login_required)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
