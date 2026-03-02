"""Configuration management for Smart Restaurant System."""
import os
from datetime import timedelta
from urllib.parse import quote_plus


def _normalize_database_url(url):
    if not url:
        return None
    if url.startswith('mysql://'):
        return url.replace('mysql://', 'mysql+pymysql://', 1)
    return url


def _mysql_uri_from_env(prefix=''):
    host = os.environ.get(f'{prefix}MYSQL_HOST') or os.environ.get('MYSQL_HOST')
    port = os.environ.get(f'{prefix}MYSQL_PORT') or os.environ.get('MYSQL_PORT') or '3306'
    user = os.environ.get(f'{prefix}MYSQL_USER') or os.environ.get('MYSQL_USER')
    password = os.environ.get(f'{prefix}MYSQL_PASSWORD') or os.environ.get('MYSQL_PASSWORD') or ''
    database = os.environ.get(f'{prefix}MYSQL_DATABASE') or os.environ.get('MYSQL_DATABASE')

    if not host or not user or not database:
        return None

    encoded_password = quote_plus(password)
    return f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{database}?charset=utf8mb4"


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-change-in-production-2026'
    DEBUG = False
    TESTING = False
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_POOL_SIZE = 10
    SQLALCHEMY_POOL_RECYCLE = 3600
    
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    DATASET_FOLDER = os.environ.get('DATASET_FOLDER') or 'datasets'
    VECTOR_DB_FOLDER = os.environ.get('VECTOR_DB_FOLDER') or 'vector_db'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'csv', 'txt'}
    
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = None
    
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    MAX_SCRAPE_REVIEWS = int(os.environ.get('MAX_SCRAPE_REVIEWS', 50))
    SCRAPE_TIMEOUT = int(os.environ.get('SCRAPE_TIMEOUT', 30))
    
    MIN_REVIEW_LENGTH = 10
    MAX_KEYWORDS = 8
    TOP_K_RAG = 3
    
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'app.log')
    
    GOOGLE_PLACES_API_KEY = os.environ.get('GOOGLE_PLACES_API_KEY', '')


class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = _normalize_database_url(os.environ.get('DEV_DATABASE_URL')) or\
        _mysql_uri_from_env('DEV_') or\
        _mysql_uri_from_env() or\
        'sqlite:///reviews.db'
    SQLALCHEMY_ECHO = False
    SESSION_COOKIE_SECURE = False
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = _normalize_database_url(os.environ.get('DATABASE_URL')) or\
        _normalize_database_url(os.environ.get('PROD_DATABASE_URL')) or\
        _mysql_uri_from_env('PROD_') or\
        _mysql_uri_from_env() or\
        'sqlite:///reviews.db'
    
    SESSION_COOKIE_SECURE = True
    WTF_CSRF_ENABLED = True
    
    SQLALCHEMY_POOL_SIZE = 20
    CACHE_TYPE = 'simple'


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    
    return config.get(env, config['default'])
