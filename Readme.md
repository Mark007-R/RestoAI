# RestoAI

AI-powered restaurant review analysis platform with sentiment analysis, RAG chat, and intelligent visualizations.

**Status**: Production-Ready (92/100) | **Python**: 3.8+ | **Framework**: Flask 3.0 | **License**: MIT

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)



## Features

- **Multi-Source Integration**: Load reviews from CSV (Zomato, Mumbai Aires, Google Reviews)
- **Sentiment Analysis**: VADER-based analysis with keyword extraction
- **Complaint Categorization**: 8-category automatic classification
- **RAG Chat**: FAISS vector DB with semantic search (Sentence-BERT 384-dim)
- **Visualizations**: 9 chart types (sentiment, categories, trends, ratings)
- **Recommendations**: AI-generated actionable insights
- **Web Scraping**: Fallback scraping when local data insufficient
- **Quality Scoring**: 0-100 quality metrics with deduplication



## Installation

```bash
# Clone and setup
git clone <repo-url>
cd Smart_Restaurant_System
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Create required directories
mkdir -p datasets uploads vector_db cache

# Add CSV files to datasets/ folder
# Then run application
python app.py

# Visit http://localhost:5000
```

## Project Structure

```
RestoAI/
├── app.py                  # Main Flask application
├── analyzer.py             # Sentiment analysis & visualizations
├── scraper.py              # Data loading & web scraping
├── rag_chat.py             # RAG chat with FAISS
├── config.py               # Configuration
├── requirements.txt        # Dependencies (23 packages)
│
├── utils/                  # Utilities (validators, logger, cache)
├── templates/              # 5 HTML templates
├── static/                 # CSS & assets
├── datasets/               # CSV data files
├── vector_db/              # FAISS indexes
├── cache/                  # Cache storage
├── uploads/                # User uploads
│
├── .env.example            # Environment template
├── Readme.md               # This file
├── LICENSE                 # MIT License
└── SECURITY.md             # Security guide
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key variables:
- `FLASK_ENV`: development|production|testing
- `SECRET_KEY`: Strong random key
- `DATABASE_URL`: sqlite:///reviews.db
- `UPLOAD_FOLDER`: uploads (relative path)
- `VECTOR_DB_FOLDER`: vector_db (relative path)
- `MAX_CONTENT_LENGTH`: 16777216 (16MB)

**Security**: Never commit `.env` file. Use `.env.example` template only.


## API Endpoints

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| GET | / | Home page | - |
| POST | /analyze | Analyze reviews | restaurant_name, datafile, try_scrape |
| GET | /results | View results | restaurant_name |
| GET | /recommendations | Get recommendations | restaurant_name |
| POST | /chat | RAG chat | question (JSON body) |
| GET | /search_restaurants | Search restaurants | q |

Chat response format:
```json
{
  "answer": "Analysis based on reviews...",
  "sources": ["review_text_1", "review_text_2"]
}
```

## Technologies

**Backend**: Flask 3.0, SQLAlchemy 2.0, FAISS, Sentence-BERT
**NLP**: VADER Sentiment, NLTK, spacy
**Data**: Pandas, NumPy, BeautifulSoup4
**Visualization**: Matplotlib, Seaborn
**Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
**Security**: python-dotenv, Werkzeug
**Total**: 23 dependencies

## Troubleshooting

**Dependencies not installing:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install faiss-cpu  # or faiss-gpu
```

**Module import errors:**
```bash
pip install sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Database locked:**
```bash
rm reviews.db && python app.py
```

**Reviews not found:**
- Verify CSV files in `datasets/` folder
- Check restaurant name spelling
- Enable debug: `FLASK_DEBUG=True`

**Scraping issues:**
- Check internet connection
- Use local datasets if scraping blocked



## Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/name`
3. Follow code standards (PEP 8, docstrings)
4. Test locally: `python app.py`
5. Commit and push to branch
6. Open Pull Request

**Standards**: Validate inputs, use .env for secrets, ensure all functions used, add docstrings.

## License

MIT License - see [LICENSE](LICENSE) file.

## Authors

Anthony - Initial work and continuous improvements

---

For issues and contributions, please visit the GitHub repository.