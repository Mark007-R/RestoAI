# RestoAI - Smart Restaurant System

Comprehensive dual-system AI platform for restaurant management and customer experience with sentiment analysis, RAG chat, advanced analytics, and AI-powered booking.

**Status**: Production-Ready | **Python**: 3.8+ | **Framework**: Flask 3.0 | **License**: MIT

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Configuration](#configuration)
- [System Architecture](#system-architecture)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Visualizations & Output Images](#visualizations--output-images)
- [Usage Guide](#usage-guide)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

RestoAI is a full-featured restaurant intelligence platform with two distinct systems:

- **Manager System**: Analytics dashboard for restaurant owners with review analysis, sentiment tracking, complaint categorization, and AI chat assistant
- **User System**: Customer-facing platform with restaurant discovery, detailed reviews, AI-powered booking, and personalized recommendations

## Key Features

### Manager System Features

- **Advanced Sentiment Analysis**: VADER-based sentiment scoring with compound scores and keyword extraction
- **Intelligent Complaint Categorization**: 8-category automatic classification (Service, Food Quality, Hygiene, Price, Delivery, Portion, Ambience, Variety)
- **RAG-Powered Chat Assistant**: FAISS vector database with semantic search using Sentence-BERT (384-dim embeddings)
- **Comprehensive Visualizations**: 9+ interactive chart types (sentiment distribution, category trends, rating analysis, temporal patterns)
- **AI-Generated Recommendations**: Data-driven actionable insights for business improvement
- **Multi-Source Data Integration**: Support for Zomato, Mumbai Aires, Google Reviews CSV formats
- **Web Scraping**: Automated fallback scraping when local data is insufficient
- **Quality Scoring System**: 0-100 quality metrics with intelligent deduplication
- **Consolidated Vector Store**: Single FAISS index for all restaurants with per-restaurant filtering
- **Image Integration**: Google Places API, Unsplash, and web scraping for restaurant images

### User System Features

- **Restaurant Discovery**: Browse comprehensive restaurant catalog with ratings, cuisines, and pricing
- **AI-Powered Auto-Booking**: Intelligent booking system for both dine-in and home delivery
- **Restaurant Details**: Detailed pages with reviews, sample menus, ratings, and booking options
- **Advanced Search & Filtering**: Search by name, cuisine, location, price range, and ratings
- **Review Access**: View real customer reviews with sentiment scores
- **Dynamic Menu Display**: Context-aware menu generation based on cuisine type

### Authentication & Security

- **Role-Based Access Control**: Separate dashboards and permissions for Users and Managers
- **Secure Authentication**: Password hashing with PBKDF2-HMAC-SHA256
- **Session Management**: Persistent sessions with configurable expiry (24-hour default)
- **CSRF Protection**: Flask-WTF integration for form security
- **Input Validation**: Comprehensive validators for restaurant names, ratings, file uploads, and text content

## Screenshots

### Authentication & Dashboard

#### Login Page
![Login Page](images/login.png)
*Secure authentication with role-based access control*

#### Signup Page
![Signup Page](images/signup.png)
*User registration with role selection (User/Manager)*

**Note**: Add login and signup screenshots to the images folder with these filenames.

### Manager System

#### Manager Dashboard
![Manager Dashboard](images/Home_Page.png)
*Central hub for restaurant analytics with available restaurants and data sources*

![Manager Dashboard Alternative View](images/Home_Page_2.png)
*Manager dashboard with restaurant selection and analysis options*

#### Analysis Results Page
![Analysis Results](images/Analysis_Result.png)
*Comprehensive analytics dashboard with sentiment analysis and visualizations*

#### Visual Analytics Dashboard
![Visual Analytics Dashboard](images/Visual_Analytics_Dashboard.png)
*Interactive charts showing sentiment distribution, complaint categories, and rating analysis*

![Visual Analytics Dashboard - Detailed View](images/Visual_Analytics_Dashboard_2.png)
*Detailed visualization with keyword analysis and sentiment scores*

#### RAG Chat Interface
![RAG Chat Assistant](images/AI_Chat_Assistant.png)
*AI-powered chat assistant with semantic search and source citations*

![RAG Chat Assistant - Conversation](images/AI_Chat_Assistant_2.png)
*Chat interface showing contextual responses from review analysis*

#### AI Recommendations
![Recommendations](images/Recommendation_List.png)
*Data-driven actionable insights for business improvement*

### User System

#### User Dashboard
![User Dashboard](images/user_dashboard.png)
*Restaurant catalog with ratings, cuisines, and pricing information*

#### Restaurant Details
![Restaurant Details](images/restaurant_details.png)
*Detailed restaurant page with reviews, menu, and booking options*

#### AI Booking Interface
![AI Booking](images/booking_interface.png)
*Intelligent booking system for dine-in and home delivery*

#### Booking Confirmation
![Booking Confirmation](images/booking_confirmation.png)
**Note**: Add user system screenshots to the images folder with these filenames.

### Additional Features

#### Search Functionality
![Search Results](images/search_results.png)
*Real-time restaurant search with autocomplete*

#### Mobile Responsive Design
![Mobile View](images/mobile_view.png)
*Responsive layout for mobile and tablet devices*

**Note**: Add additional feature screenshots to the images folder with these filenames.
![Mobile View](images/mobile_view.png)
*Responsive layout for mobile and tablet devices*

---

**Note**: Screenshot images are stored in the `images/` directory. To update screenshots, replace the corresponding image files while maintaining the same filenames.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) Google Places API key for restaurant images

### Setup Instructions

```bash
# Clone repository
git clone <repo-url>
cd Smart_Restaurant_System

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your configuration (SECRET_KEY, DATABASE_URL, etc.)

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Create required directories
mkdir -p datasets manager_system/uploads manager_system/vector_db manager_system/cache

# Add CSV files to datasets/ folder
# Supported formats: mumbaires.csv, zomato.csv, zomato2.csv, Resreviews.csv, reviews.csv

# Initialize database
python app.py
# Database tables will be created automatically on first run

# Visit http://localhost:5000
```

### First-Time Setup

1. **Create Admin Account**: Visit `/signup` and create a manager account
2. **Upload Data**: Place CSV files in the `datasets/` folder
3. **Access Manager Dashboard**: Login and navigate to manager dashboard for analytics
4. **Create User Account**: Create a regular user account to test user features

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

#### Core Configuration
```bash
# Flask Configuration
FLASK_ENV=development          # development|production|testing
FLASK_DEBUG=True               # Enable debug mode
SECRET_KEY=your-secret-key     # Strong random key for sessions

# Database Configuration
DATABASE_URL=sqlite:///instance/reviews.db    # SQLite (default)
# For MySQL:
# MYSQL_HOST=localhost
# MYSQL_PORT=3306
# MYSQL_USER=root
# MYSQL_PASSWORD=password
# MYSQL_DATABASE=restoai

# File Storage
UPLOAD_FOLDER=uploads           # User uploads (relative to manager_system/)
DATASET_FOLDER=datasets         # CSV datasets (relative to project root)
VECTOR_DB_FOLDER=vector_db      # FAISS indexes (relative to manager_system/)
MAX_CONTENT_LENGTH=16777216     # 16MB file upload limit

# External APIs (Optional)
GOOGLE_PLACES_API_KEY=your-api-key    # For restaurant images
```

#### Security Configuration
```bash
# Session Security
SESSION_COOKIE_SECURE=True           # HTTPS only in production
SESSION_COOKIE_HTTPONLY=True         # Prevent JavaScript access
SESSION_COOKIE_SAMESITE=Lax          # CSRF protection
PERMANENT_SESSION_LIFETIME=86400     # 24 hours

# CSRF Protection
WTF_CSRF_ENABLED=True                # Enable CSRF tokens
```

**Security Best Practices**:
- Never commit `.env` file to version control
- Use strong SECRET_KEY (32+ random characters)
- Enable HTTPS in production
- Regularly rotate API keys
- Use environment-specific configurations

## System Architecture

### Dual-System Design

```
┌─────────────────────────────────────────────────────────┐
│                     RestoAI Platform                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────┐         ┌──────────────────┐    │
│  │  Manager System  │         │   User System    │    │
│  ├──────────────────┤         ├──────────────────┤    │
│  │ • Analytics      │         │ • Browse         │    │
│  │ • Sentiment      │         │ • Search         │    │
│  │ • RAG Chat       │         │ • AI Booking     │    │
│  │ • Visualizations │         │ • Reviews        │    │
│  │ • Scraping       │         │ • Details        │    │
│  └──────────────────┘         └──────────────────┘    │
│           │                            │               │
│           └────────────┬───────────────┘               │
│                        │                               │
│              ┌─────────▼──────────┐                   │
│              │   Shared Layer     │                   │
│              ├────────────────────┤                   │
│              │ • Authentication   │                   │
│              │ • Database (SQL)   │                   │
│              │ • Restaurant Search│                   │
│              │ • Review Model     │                   │
│              └────────────────────┘                   │
│                        │                               │
│              ┌─────────▼──────────┐                   │
│              │   Data Layer       │                   │
│              ├────────────────────┤                   │
│              │ • CSV Datasets     │                   │
│              │ • FAISS Vector DB  │                   │
│              │ • SQLite/MySQL     │                   │
│              │ • Cache System     │                   │
│              └────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

### Key Components

1. **Flask Application (`app.py`)**: 
   - Main entry point
   - Authentication & authorization
   - Role-based routing
   - Database models (User, Review)
   - Session management

2. **Manager System** (`manager_system/`):
   - `manager.py`: Analytics routes and business logic
   - `analyzer.py`: Sentiment analysis, keyword extraction, categorization
   - `rag_chat.py`: RAG implementation with FAISS
   - `scraper.py`: Data loading and web scraping
   - `config.py`: Environment configuration

3. **User System** (`user_system/`):
   - `user.py`: User routes, booking, restaurant details
   - User preferences and dietary restrictions
   - Booking history management

4. **Shared Components** (`shared/`):
   - `restaurant_search.py`: Search and filtering utilities

5. **Utilities** (`manager_system/utils/`):
   - `validators.py`: Input validation
   - `helpers.py`: Data processing utilities
   - `logger.py`: Logging configuration
   - `cache.py`: Caching decorators

## Project Structure

```
Smart_Restaurant_System/
│
├── app.py                          # Main Flask application & auth
├── requirements.txt                # Dependencies (27 packages)
├── LICENSE                         # MIT License
├── Readme.md                       # This file
│
├── datasets/                       # CSV data files
│   ├── mumbaires.csv              # Mumbai Aires restaurant reviews
│   ├── Resreviews.csv             # Restaurant reviews dataset
│   ├── reviews.csv                # Google-style reviews
│   ├── zomato.csv                 # Zomato dataset 1
│   └── zomato2.csv                # Zomato dataset 2
│
├── templates/                      # Root-level templates (auth)
│   ├── base.html                  # Base template
│   ├── login.html                 # Login page
│   ├── signup.html                # Signup page
│   ├── user_dashboard.html        # User dashboard
│   └── restaurant_details.html    # Restaurant detail page
│
├── manager_system/                 # Manager subsystem
│   ├── __init__.py                # Package initialization
│   ├── manager.py                 # Manager routes & controllers
│   ├── analyzer.py                # Sentiment analysis engine
│   ├── scraper.py                 # Web scraping & data loading
│   ├── rag_chat.py                # RAG chat with FAISS
│   ├── config.py                  # Configuration management
│   │
│   ├── templates/                 # Manager templates
│   │   ├── base.html             # Manager base template
│   │   ├── index.html            # Manager dashboard
│   │   ├── chat.html             # RAG chat interface
│   │   ├── recommendations.html   # AI recommendations
│   │   └── results.html          # Analysis results
│   │
│   ├── static/                    # Static assets
│   │   └── style.css             # Styling
│   │
│   ├── utils/                     # Utility modules
│   │   ├── __init__.py
│   │   ├── validators.py         # Input validation
│   │   ├── helpers.py            # Helper functions
│   │   ├── logger.py             # Logging setup
│   │   └── cache.py              # Caching utilities
│   │
│   ├── vector_db/                 # FAISS vector storage
│   │   ├── all_restaurants.faiss # Consolidated vector index
│   │   └── *_metadata.pkl        # Restaurant metadata
│   │
│   ├── uploads/                   # User file uploads
│   ├── cache/                     # Cache storage
│   └── instance/                  # Instance-specific files
│
├── user_system/                    # User subsystem
│   ├── user.py                    # User routes & booking logic
│   └── __pycache__/
│
├── shared/                         # Shared utilities
│   ├── restaurant_search.py       # Search & filter functions
│   └── __pycache__/
│
├── instance/                       # Instance folder (database)
│   └── reviews.db                 # SQLite database
│
└── images/                         # Documentation images
```


## API Endpoints

### Authentication Endpoints

| Method | Endpoint | Description | Access | Parameters |
|--------|----------|-------------|--------|------------|
| GET/POST | `/login` | User login | Public | username, password |
| GET/POST | `/signup` | User registration | Public | username, email, password, confirm_password, role |
| GET | `/logout` | User logout | Authenticated | - |
| GET | `/` | Root redirect | Public | Redirects to appropriate dashboard |

### Manager Endpoints

| Method | Endpoint | Description | Access | Parameters |
|--------|----------|-------------|--------|------------|
| GET | `/manager/dashboard` | Manager analytics dashboard | Manager | - |
| GET/POST | `/analyze` | Analyze restaurant reviews | Manager | restaurant_name, datafile, try_scrape |
| GET | `/results` | View analysis results | Manager | restaurant_name |
| GET | `/recommendations` | Get AI recommendations | Manager | restaurant_name |
| GET/POST | `/chat` | RAG chat interface | Manager | question (JSON) |
| GET | `/search_restaurants` | Search restaurants | Manager | q (query string) |
| GET | `/api/image-status` | Check image API status | Manager | - |
| POST | `/api/reset-google-api` | Reset Google API | Manager | - |
| GET | `/api/vector-cache-status` | Vector cache info | Manager | - |
| DELETE | `/api/clear-vector-cache` | Clear vector cache | Manager | - |

### User Endpoints

| Method | Endpoint | Description | Access | Parameters |
|--------|----------|-------------|--------|------------|
| GET | `/user/dashboard` | User restaurant catalog | User | - |
| POST | `/user/auto-booking` | AI-powered booking | User | restaurant_name, service_mode, party_size, preferred_time, delivery_address |
| GET | `/user/restaurant/<name>` | Restaurant details | User | name (URL parameter) |

### Request/Response Examples

#### Login Request
```json
POST /login
Content-Type: application/x-www-form-urlencoded

username=john_doe&password=securepass123
```

#### Analyze Restaurant Request
```json
POST /analyze
Content-Type: application/x-www-form-urlencoded

restaurant_name=Bombay+Dreams&datafile=zomato.csv&try_scrape=on
```

#### RAG Chat Request
```json
POST /chat
Content-Type: application/json

{
  "question": "What do customers say about the food quality?"
}
```

#### RAG Chat Response
```json
{
  "answer": "Based on the reviews, customers generally praise the food quality, mentioning fresh ingredients and authentic flavors. However, some reviews note inconsistency in seasoning.",
  "sources": [
    "The food was absolutely delicious with fresh ingredients",
    "Authentic taste but sometimes the spice level varies"
  ]
}
```

#### Auto-Booking Request
```json
POST /user/auto-booking
Content-Type: application/x-www-form-urlencoded

restaurant_name=Saffron+Bistro&service_mode=restaurant&party_size=4&preferred_time=19:00
```

## Technologies

### Backend Stack
- **Flask 3.0.0**: Web framework
- **SQLAlchemy 2.0.23**: ORM and database management
- **Flask-SQLAlchemy 3.1.1**: Flask-SQLAlchemy integration
- **Flask-WTF 1.2.1**: CSRF protection and form handling
- **Werkzeug 3.0.1**: Security utilities (password hashing)
- **python-dotenv 1.0.0**: Environment variable management
- **python-decouple 3.8**: Configuration management

### NLP & AI
- **sentence-transformers 2.2.2**: Sentence embeddings (all-MiniLM-L6-v2)
- **transformers 4.35.2**: Transformer models
- **torch 2.1.1**: PyTorch backend
- **faiss-cpu 1.7.4**: Vector similarity search
- **vaderSentiment 3.3.2**: Sentiment analysis
- **nltk 3.8.1**: Natural language processing
- **scikit-learn 1.3.2**: Machine learning utilities

### Data Processing
- **pandas 2.1.4**: Data manipulation
- **numpy 1.26.2**: Numerical computing

### Visualization
- **matplotlib 3.8.2**: Plotting and charts
- **seaborn 0.13.0**: Statistical visualizations

### Web Scraping
- **requests 2.31.0**: HTTP library
- **beautifulsoup4 4.12.2**: HTML parsing
- **lxml 4.9.3**: XML/HTML processing
- **aiohttp 3.9.1**: Async HTTP client

### Database Support
- **PyMySQL 1.1.0**: MySQL connector (optional)

### Utilities
- **tqdm 4.66.1**: Progress bars
- **pillow 10.1.0**: Image processing
- **flask-limiter 3.5.0**: Rate limiting
- **colorlog 6.8.0**: Colored logging

**Total Dependencies**: 27 packages

### Model Information
- **Embedding Model**: `all-MiniLM-L6-v2` (Sentence-BERT)
- **Embedding Dimension**: 384
- **Sentiment Analyzer**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Vector Store**: FAISS (Facebook AI Similarity Search)

## Visualizations & Output Images

The Manager System generates comprehensive visual analytics for restaurant analysis. All charts are dynamically generated as base64-encoded PNG images and embedded directly in the results page.

### Generated Visualizations

#### 1. **Sentiment Distribution Chart**
- **Type**: Pie Chart
- **Purpose**: Shows breakdown of Positive, Negative, and Neutral reviews
- **Colors**: Green (Positive), Red (Negative), Yellow (Neutral)
- **Output**: Percentage distribution with counts

#### 2. **Sentiment Score Distribution**
- **Type**: Histogram
- **Purpose**: Distribution of VADER compound sentiment scores (-1 to +1)
- **Features**: KDE (Kernel Density Estimation) overlay
- **Bins**: 30 bins for detailed distribution

#### 3. **Complaint Category Analysis**
- **Type**: Horizontal Bar Chart
- **Purpose**: Frequency of each complaint category
- **Categories**: Service, Food Quality, Hygiene, Price, Delivery, Portion, Ambience, Variety
- **Output**: Count and percentage for each category

#### 4. **Rating Distribution**
- **Type**: Bar Chart
- **Purpose**: Distribution of star ratings (1-5 stars)
- **Features**: Shows review count per rating level
- **Analysis**: Helps identify rating patterns

#### 5. **Keyword Cloud Analysis**
- **Type**: Word Frequency Chart
- **Purpose**: Top 20 most common keywords from reviews
- **Features**: Stopword filtered, minimum 3 characters
- **Visualization**: Horizontal bar chart with frequency counts

#### 6. **Sentiment by Rating**
- **Type**: Box Plot
- **Purpose**: Correlation between star ratings and sentiment scores
- **Features**: Shows sentiment score distribution for each rating level
- **Insights**: Identifies rating-sentiment consistency

#### 7. **Review Quality Score**
- **Type**: Gauge/Score Display
- **Purpose**: Overall quality metric (0-100)
- **Calculation**: Based on review count, rating distribution, sentiment balance
- **Thresholds**: 
  - 80-100: Excellent
  - 60-79: Good
  - 40-59: Average
  - 20-39: Below Average
  - 0-19: Poor

#### 8. **Temporal Trends** (if date available)
- **Type**: Line Chart
- **Purpose**: Sentiment trends over time
- **Features**: Shows sentiment score evolution
- **Analysis**: Identifies improvement or decline patterns

#### 9. **Category Comparison**
- **Type**: Stacked Bar Chart
- **Purpose**: Complaint categories by sentiment
- **Features**: Shows which categories correlate with negative sentiment
- **Insights**: Prioritizes improvement areas

### Image Sources

#### Restaurant Images
The system fetches restaurant images from multiple sources with intelligent fallback:

1. **Google Places API** (Primary)
   - High-quality, verified restaurant photos
   - Requires API key configuration
   - Auto-disabled after 10 consecutive failures

2. **Unsplash API** (Secondary)
   - Professional stock photos
   - Restaurant and food imagery
   - No API key required

3. **Web Scraping** (Tertiary)
   - Bing Image Search fallback
   - Extracts restaurant-related images
   - User-agent rotation for reliability

4. **Placeholder Images** (Fallback)
   - LoremFlickr placeholder service
   - Restaurant/food themed images
   - Deterministic based on restaurant name

### Chart Styling

All visualizations use consistent styling:
- **Background**: Dark theme (`#0f172a` - slate-900)
- **Figure Size**: 10x6 inches (optimized for web display)
- **DPI**: 120 (high resolution)
- **Format**: PNG with base64 encoding
- **Color Palette**: Seaborn default with custom colors for sentiment
- **Font**: Default matplotlib fonts with adequate sizing

### Accessing Visualizations

After analyzing a restaurant:
```
/results?restaurant_name=YourRestaurant
```

The results page displays:
- Restaurant header with image
- Summary statistics cards
- 9+ interactive charts
- AI-generated insights
- Recommendations link

### Exporting Images

Images can be saved by:
1. Right-click on any chart → "Save image as..."
2. Use browser's print function → "Save as PDF"
3. Screenshot individual charts

## Usage Guide

### For Restaurant Managers

#### 1. Access Manager Dashboard
```
1. Login with manager credentials at /login
2. Navigate to Manager Dashboard
3. View available restaurants and analysis options
```

#### 2. Analyze Restaurant Reviews
```
1. Select restaurant from dropdown or enter name
2. Choose data source (CSV file)
3. Enable web scraping for additional data (optional)
4. Click "Analyze"
5. View comprehensive analytics:
   - Sentiment distribution
   - Complaint categories
   - Rating trends
   - Quality score
   - Keywords and insights
```

#### 3. Use RAG Chat Assistant
```
1. Navigate to Chat interface
2. Ask questions about reviews:
   - "What are the main complaints?"
   - "How is the food quality?"
   - "What do customers say about service?"
3. Receive AI-generated answers with source citations
```

#### 4. Generate Recommendations
```
1. Click "Get Recommendations" after analysis
2. Review AI-generated actionable insights
3. Export or implement suggestions
```

### For Customers (Users)

#### 1. Browse Restaurants
```
1. Login with user credentials
2. Access User Dashboard
3. Browse restaurant catalog with:
   - Ratings
   - Cuisines
   - Price levels
   - Locations
```

#### 2. View Restaurant Details
```
1. Click on any restaurant
2. See detailed information:
   - Customer reviews
   - Sample menu
   - Ratings and statistics
   - Location and contact
```

#### 3. Make AI Booking
```
1. Select restaurant
2. Choose service mode:
   - Restaurant Table (Dine-in)
   - Home Delivery
3. Specify:
   - Party size (1-20)
   - Preferred time
   - Delivery address (for delivery mode)
4. Submit for instant AI booking confirmation
```

#### 4. Search and Filter
```
1. Use search bar for restaurant names
2. Filter by:
   - Cuisine type
   - Price range
   - Ratings
   - Location
```

### Data Management

#### Supported CSV Formats

**Mumbai Aires Format** (`mumbaires.csv`):
```csv
Restaurant Name,Review Text,Reviewer Rating,Rating,Address,Price Level
```

**Resreviews Format** (`Resreviews.csv`):
```csv
Restaurant,Review,Rating,Reviewer
```

**Reviews Format** (`reviews.csv`):
```csv
business_name,text,rating,author_name
```

**Zomato Format** (`zomato.csv`, `zomato2.csv`):
```csv
name,reviews_list,rating,cuisine,location
```

#### Adding New Data
```bash
# Place CSV files in datasets/ folder
cp your_reviews.csv datasets/

# Restart application to reload catalog
python app.py
```

## Troubleshooting

### Common Issues

#### 1. Dependencies Not Installing
```bash
# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# For FAISS issues:
pip install faiss-cpu  # CPU version
# OR
pip install faiss-gpu  # GPU version (requires CUDA)
```

#### 2. Module Import Errors
```bash
# Sentence-transformers not found
pip install sentence-transformers

# Test installation
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# NLTK data missing
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

#### 3. Database Issues
```bash
# Database locked error
rm instance/reviews.db
python app.py

# Migration needed
flask db upgrade

# Reset database
rm instance/reviews.db && python app.py
```

#### 4. Reviews Not Found
- **Check CSV files**: Verify files exist in `datasets/` folder
- **Restaurant name**: Check exact spelling and case
- **Enable debug**: Set `FLASK_DEBUG=True` in `.env`
- **Check logs**: Review console output for errors

#### 5. Scraping Issues
```bash
# Check internet connection
ping google.com

# Use local datasets if scraping blocked
# Disable scraping in analyze form

# Update user agent in scraper.py if needed
```

#### 6. Vector Database Issues
```bash
# Clear vector cache
DELETE /api/clear-vector-cache

# Manual cleanup
rm manager_system/vector_db/*.faiss
rm manager_system/vector_db/*.pkl

# Rebuild vectors
# Re-analyze restaurants
```

#### 7. Login/Session Issues
```bash
# Clear browser cookies
# Check SECRET_KEY in .env
# Verify session configuration

# Test with new incognito window
```

#### 8. Permission Errors
```bash
# Windows: Run as Administrator
# Linux/Mac: Check folder permissions
chmod -R 755 datasets/ manager_system/uploads/ manager_system/vector_db/
```

### Performance Optimization

#### 1. Large Datasets
- Use pagination for review display
- Limit analysis to recent reviews
- Enable caching for frequently accessed data

#### 2. FAISS Performance
- Use GPU version for faster similarity search
- Batch process embeddings
- Monitor vector DB size

#### 3. Memory Usage
- Limit concurrent analysis jobs
- Clear cache periodically
- Use streaming for large CSV files

### Debugging Tips

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check database records
from app import db, Review
with app.app_context():
    print(Review.query.count())
    print(Review.query.filter_by(restaurant='YourRestaurant').all())

# Test sentiment analysis
from manager_system.analyzer import analyze_text_and_keywords
label, score, keywords = analyze_text_and_keywords("Great food!")
print(f"{label}: {score}, Keywords: {keywords}")
```



## Contributing

We welcome contributions to RestoAI! Please follow these guidelines:

### Development Workflow

1. **Fork the Repository**
```bash
git clone https://github.com/your-username/Smart_Restaurant_System.git
cd Smart_Restaurant_System
```

2. **Create Feature Branch**
```bash
git checkout -b feature/your-feature-name
# OR
git checkout -b fix/bug-description
```

3. **Setup Development Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists
```

4. **Make Changes**
- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Update tests if applicable
- Update documentation

5. **Test Locally**
```bash
# Run application
python app.py

# Test specific features
# - Create test accounts (user and manager)
# - Test all CRUD operations
# - Verify authentication flows
# - Test edge cases

# Run linting (if configured)
flake8 . --max-line-length=120
black . --check
```

6. **Commit Changes**
```bash
git add .
git commit -m "feat: add new feature description"

# Commit message conventions:
# feat: New feature
# fix: Bug fix
# docs: Documentation changes
# refactor: Code refactoring
# test: Test updates
# chore: Maintenance tasks
```

7. **Push and Create Pull Request**
```bash
git push origin feature/your-feature-name
# Create PR on GitHub
```

### Code Standards

#### Python Style
- Follow PEP 8
- Use type hints where applicable
- Maximum line length: 120 characters
- Use descriptive variable names

#### Documentation
- Add docstrings to all functions
```python
def analyze_reviews(restaurant_name: str, reviews: List[dict]) -> dict:
    """
    Analyze restaurant reviews with sentiment analysis.
    
    Args:
        restaurant_name: Name of the restaurant
        reviews: List of review dictionaries
        
    Returns:
        Dictionary containing analysis results
        
    Raises:
        ValueError: If restaurant_name is empty
    """
    pass
```

#### Security
- Validate all user inputs
- Use parameterized queries
- Never commit `.env` files
- Store secrets in environment variables
- Sanitize file uploads

#### Testing
- Test authentication flows
- Verify role-based access control
- Test edge cases and error handling
- Validate data processing pipelines

### Areas for Contribution

#### Features
- [ ] Email notifications for bookings
- [ ] Advanced filtering options
- [ ] Export reports to PDF/Excel
- [ ] Real-time chat support
- [ ] Mobile responsive design improvements
- [ ] Restaurant owner dashboard
- [ ] Payment integration
- [ ] Reservation confirmation system
- [ ] Multi-language support
- [ ] Dark mode theme

#### Improvements
- [ ] Performance optimization for large datasets
- [ ] Better error handling and user feedback
- [ ] Enhanced visualization options
- [ ] Improved caching strategies
- [ ] API documentation (Swagger/OpenAPI)
- [ ] Unit and integration tests
- [ ] CI/CD pipeline setup
- [ ] Docker containerization

#### Bug Fixes
- Report bugs via GitHub Issues
- Include steps to reproduce
- Provide error messages and logs
- Specify environment details

### Pull Request Checklist

- [ ] Code follows PEP 8 style guidelines
- [ ] All functions have docstrings
- [ ] Changes are tested locally
- [ ] No sensitive data committed
- [ ] Documentation updated (if needed)
- [ ] No breaking changes (or clearly documented)
- [ ] PR description is clear and detailed

### Getting Help

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for security issues

## License

MIT License

Copyright (c) 2026 RestoAI Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

See [LICENSE](LICENSE) file for full details.

## Authors & Acknowledgments

**Lead Developer**: Anthony

**Contributors**: See [CONTRIBUTORS.md](CONTRIBUTORS.md) for full list

**Special Thanks**:
- VADER Sentiment Analysis team
- Sentence-Transformers developers
- FAISS team at Facebook AI Research
- Flask and SQLAlchemy communities

## Project Status

**Current Version**: 1.0.0
**Status**: Production Ready
**Last Updated**: March 2026

### Roadmap

**Version 1.1.0** (Q2 2026):
- Mobile app integration
- Advanced analytics dashboard
- Email notification system
- Payment gateway integration

**Version 1.2.0** (Q3 2026):
- Multi-language support
- Restaurant owner portal
- Enhanced recommendation engine
- Real-time booking system

**Version 2.0.0** (Q4 2026):
- Microservices architecture
- GraphQL API
- Advanced ML models
- Scalability improvements

---

For issues, feature requests, and contributions, please visit the [GitHub repository](https://github.com/your-username/Smart_Restaurant_System).

**Happy Coding! 🍽️🤖**