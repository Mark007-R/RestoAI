# RestoAI - Smart Restaurant System

Comprehensive dual-system AI platform for restaurant management and customer experience with sentiment analysis, RAG chat, advanced analytics, and AI-powered booking.

**Status**: Production-Ready | **Python**: 3.8+ | **Framework**: Flask 3.0 | **License**: MIT

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

### Manager System

#### Manager Dashboard
![Manager Dashboard](images/Home_Page.png)
![Manager Dashboard Alternative View](images/Home_Page_2.png)

#### Analysis Results Page
![Analysis Results](images/Analysis_Result.png)
#### Visual Analytics Dashboard
![Visual Analytics Dashboard](images/Visual_Analytics_Dashboard.png)
![Visual Analytics Dashboard - Detailed View](images/Visual_Analytics_Dashboard_2.png)

#### AI Recommendations
![Recommendations](images/Recommendation_List.png)

#### RAG Chat Interface
![RAG Chat Assistant](images/AI_Chat_Assistant.png)
![RAG Chat Assistant - Conversation](images/AI_Chat_Assistant_2.png)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) Google Places API key for restaurant images

### Setup Instructions

```bash
git clone <repo-url>
cd Smart_Restaurant_System

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

cp .env

python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

mkdir -p datasets manager_system/uploads manager_system/vector_db manager_system/cache

python app.py
```

### First-Time Setup

1. **Create Admin Account**: Visit `/signup` and create a manager account
2. **Upload Data**: Place CSV files in the `datasets/` folder
3. **Access Manager Dashboard**: Login and navigate to manager dashboard for analytics
4. **Create User Account**: Create a regular user account to test user features

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
