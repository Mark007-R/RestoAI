import os
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from analyzer import (analyze_text_and_keywords, categorize_complaints, 
                     summarize_reviews_for_recommendations, generate_visualizations)
from scraper import scrape_generic_reviews, scrape_zomato_placeholder
from rag_chat import RAGChat
import pandas as pd
from datetime import datetime

from config import get_config
from utils.logger import setup_logger
from utils.validators import (
    validate_restaurant_name, validate_filename, sanitize_string,
    validate_file_size, validate_rating, validate_text_length
)
from utils.helpers import (
    extract_reviews_from_zomato_reviews_list, safe_read_csv,
    clean_value, is_valid_review_text, deduplicate_reviews,
    extract_value_safely
)
from utils.cache import memoize

# Initialize Flask app with configuration
app = Flask(__name__)
config_class = get_config(os.environ.get('FLASK_ENV', 'development'))
app.config.from_object(config_class)

logger = setup_logger(__name__, log_file='logs/app.log')

UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
DATASET_FOLDER = app.config['DATASET_FOLDER']
ALLOWED_EXT = app.config['ALLOWED_EXTENSIONS']

db = SQLAlchemy(app)

for folder in [UPLOAD_FOLDER, DATASET_FOLDER, 'logs', app.config.get('VECTOR_DB_FOLDER', 'vector_db')]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        logger.info(f"Created directory: {folder}")

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
    
    __table_args__ = (
        db.Index('idx_restaurant_created', 'restaurant', 'created_at'),
        db.Index('idx_restaurant_sentiment', 'restaurant', 'sentiment'),
    )
    
    def __repr__(self):
        return f'<Review {self.id}: {self.restaurant}>'

with app.app_context():
    db.create_all()
    logger.info("Database tables created/verified")

rag_instance = None
current_indexed_restaurant = None

def build_or_get_rag(reviews_texts, docs_texts=None):
    rag = RAGChat()
    combined = list(reviews_texts)
    if docs_texts:
        combined.extend(docs_texts)
    if combined:
        rag.index_documents(combined)
    return rag

def fetch_image_from_google_places(restaurant_name, api_key):
    """Fetch image from Google Places API"""
    try:
        import requests
        search_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
        params = {
            'input': restaurant_name,
            'inputtype': 'textquery',
            'fields': 'photos,name,place_id',
            'key': api_key
        }
        
        response = requests.get(search_url, params=params, timeout=5)
        data = response.json()
        
        if data.get('status') == 'OK' and data.get('candidates'):
            place = data['candidates'][0]
            if 'photos' in place and len(place['photos']) > 0:
                photo_reference = place['photos'][0]['photo_reference']
                photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=800&photo_reference={photo_reference}&key={api_key}"
                logger.info(f"✓ Found Google Places image for {restaurant_name}")
                return photo_url
        
        logger.debug(f"No Google Places image found for {restaurant_name}")
        return None
    except Exception as e:
        logger.debug(f"Google Places API error for {restaurant_name}: {e}")
        return None


def fetch_image_from_web_search(restaurant_name):
    """Fetch image by web scraping - Old fallback method"""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Try searching on restaurant review sites
        search_urls = [
            f"https://www.google.com/search?tbm=isch&q={restaurant_name}+restaurant",
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # More reliable: use DuckDuckGo/Bing image search approach
        try:
            search_url = f"https://www.bing.com/images/search?q={restaurant_name}+restaurant"
            response = requests.get(search_url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Looking for image tags
            img_tags = soup.find_all('img', {'class': 'mimg'})
            if img_tags:
                for img in img_tags:
                    src = img.get('src') or img.get('data-src')
                    if src and 'http' in src and '.jpg' in src.lower() or '.png' in src.lower():
                        logger.info(f"✓ Found web image for {restaurant_name}")
                        return src
        except Exception as e:
            logger.debug(f"Web scraping attempt failed: {e}")
        
        return None
    except Exception as e:
        logger.debug(f"Web search error for {restaurant_name}: {e}")
        return None


def fetch_image_from_unsplash(restaurant_name):
    """Fetch image from Unsplash with multiple query attempts"""
    try:
        import requests
        
        # Try multiple search queries with decreasing specificity
        search_queries = [
            f"restaurant {restaurant_name}",
            f"{restaurant_name} food",
            "restaurant dining",
            "restaurant interior",
            "restaurant food",
            "restaurant ambiance"
        ]
        
        for query in search_queries:
            try:
                response = requests.get(
                    "https://api.unsplash.com/search/photos",
                    params={
                        'query': query,
                        'per_page': 1,
                        'w': 800,
                        'h': 600
                    },
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('results'):
                        image_url = data['results'][0]['urls']['regular']
                        logger.info(f"✓ Found Unsplash image for {restaurant_name} using '{query}'")
                        return image_url
            except Exception as e:
                logger.debug(f"Unsplash search failed for '{query}': {e}")
                continue
        
        return None
    except Exception as e:
        logger.debug(f"Unsplash error for {restaurant_name}: {e}")
        return None


def generate_placeholder_image(restaurant_name):
    """Generate a consistent placeholder image URL based on restaurant name"""
    hash_value = sum(ord(c) * (i + 1) for i, c in enumerate(restaurant_name))
    unique_id = hash_value % 10000
    return f"https://source.unsplash.com/800x600/?restaurant,food&sig={unique_id}"


@memoize
def get_restaurant_image(restaurant_name):
    """
    Fetch restaurant image with multiple fallback strategies:
    1. Google Places API (if API key available)
    2. Web scraping from search engines
    3. Unsplash API with multiple query attempts
    4. Placeholder image as final fallback
    """
    api_key = app.config.get('GOOGLE_PLACES_API_KEY')
    
    # Strategy 1: Try Google Places API first
    if api_key:
        image_url = fetch_image_from_google_places(restaurant_name, api_key)
        if image_url:
            return image_url
        logger.warning(f"Google Places API failed for {restaurant_name}, trying fallback methods...")
    else:
        logger.debug("Google Places API key not configured, using fallback methods")
    
    # Strategy 2: Try web scraping (old method)
    image_url = fetch_image_from_web_search(restaurant_name)
    if image_url:
        return image_url
    
    # Strategy 3: Try Unsplash API
    image_url = fetch_image_from_unsplash(restaurant_name)
    if image_url:
        return image_url
    
    # Strategy 4: Generate placeholder image
    logger.warning(f"All image fetching methods failed for {restaurant_name}, using placeholder")
    return generate_placeholder_image(restaurant_name)

def allowed_file(filename):
    is_valid, error = validate_filename(filename, ALLOWED_EXT)
    if not is_valid:
        logger.warning(f"Invalid filename: {filename} - {error}")
        return False
    return True

def process_mumbaires_csv(filepath, restaurant_filter=None):
    try:
        df = safe_read_csv(filepath)
        if df is None:
            logger.error(f"Failed to read CSV: {filepath}")
            return [], []
        
        reviews = []
        restaurants_data = []
        
        for _, row in df.iterrows():
            restaurant_name = clean_value(row.get('Restaurant Name', ''))
            review_text = clean_value(row.get('Review Text', ''))
            
            if not restaurant_name:
                continue
                
            if restaurant_filter:
                if restaurant_filter.lower() not in restaurant_name.lower():
                    continue
            
            if is_valid_review_text(review_text):
                reviews.append({
                    'text': review_text,
                    'restaurant': restaurant_name,
                    'rating': extract_value_safely(row, 'Reviewer Rating'),
                    'source': 'mumbaires'
                })
            
            if not restaurant_filter:
                restaurants_data.append({
                    'name': restaurant_name,
                    'rating': extract_value_safely(row, 'Rating', 'N/A'),
                    'address': extract_value_safely(row, 'Address', 'Address not available'),
                    'price_level': extract_value_safely(row, 'Price Level', ''),
                })
        
        logger.info(f"Processed mumbaires.csv: {len(reviews)} reviews, {len(restaurants_data)} restaurants")
        return reviews, restaurants_data
    except Exception as e:
        logger.error(f"Error processing mumbaires.csv: {e}", exc_info=True)
        return [], []

def process_resreviews_csv(filepath, restaurant_filter=None):
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        
        reviews = []
        restaurants_data = []
        
        for _, row in df.iterrows():
            restaurant_name = str(row.get('Restaurant', '')).strip()
            review_text = str(row.get('Review', '')).strip()
            
            if pd.isna(restaurant_name) or restaurant_name.lower() == 'nan':
                continue
                
            if restaurant_filter:
                if restaurant_filter.lower() not in restaurant_name.lower():
                    continue
            
            if review_text and review_text.lower() != 'nan' and len(review_text) > 10:
                reviews.append({
                    'text': review_text,
                    'restaurant': restaurant_name,
                    'rating': row.get('Rating', None),
                    'reviewer': row.get('Reviewer', 'anonymous'),
                    'source': 'resreviews'
                })
            
            if not restaurant_filter:
                restaurants_data.append({
                    'name': restaurant_name,
                    'rating': row.get('Rating', 'N/A'),
                })
        
        return reviews, restaurants_data
    except Exception as e:
        print(f"Error processing Resreviews.csv: {e}")
        return [], []

def process_reviews_csv(filepath, restaurant_filter=None):
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        
        reviews = []
        restaurants_data = []
        
        for _, row in df.iterrows():
            restaurant_name = str(row.get('business_name', '')).strip()
            review_text = str(row.get('text', '')).strip()
            
            if pd.isna(restaurant_name) or restaurant_name.lower() == 'nan':
                continue
                
            if restaurant_filter:
                if restaurant_filter.lower() not in restaurant_name.lower():
                    continue
            
            if review_text and review_text.lower() != 'nan' and len(review_text) > 10:
                reviews.append({
                    'text': review_text,
                    'restaurant': restaurant_name,
                    'rating': row.get('rating', None),
                    'reviewer': row.get('author_name', 'anonymous'),
                    'source': 'reviews'
                })
            
            if not restaurant_filter:
                restaurants_data.append({
                    'name': restaurant_name,
                    'rating': row.get('rating', 'N/A'),
                })
        
        return reviews, restaurants_data
    except Exception as e:
        print(f"Error processing reviews.csv: {e}")
        return [], []

def process_zomato_csv(filepath, restaurant_filter=None):
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        
        reviews = []
        restaurants_data = []
        
        for _, row in df.iterrows():
            restaurant_name = str(row.get('name', '')).strip()
            reviews_list = row.get('reviews_list', '')
            
            if pd.isna(restaurant_name) or restaurant_name.lower() == 'nan':
                continue
                
            if restaurant_filter:
                if restaurant_filter.lower() not in restaurant_name.lower():
                    continue
            
            extracted_reviews = extract_reviews_from_zomato_reviews_list(reviews_list)
            
            for review_text in extracted_reviews:
                if len(review_text) > 10:
                    reviews.append({
                        'text': review_text,
                        'restaurant': restaurant_name,
                        'rating': row.get('rate', None),
                        'source': 'zomato'
                    })
            
            if not restaurant_filter:
                restaurants_data.append({
                    'name': restaurant_name,
                    'rating': row.get('rate', 'N/A'),
                    'address': row.get('address', 'Address not available'),
                    'location': row.get('location', ''),
                    'cuisines': row.get('cuisines', ''),
                    'cost': row.get('approx_cost(for two people)', ''),
                })
        
        return reviews, restaurants_data
    except Exception as e:
        print(f"Error processing zomato.csv: {e}")
        return [], []

def process_zomato2_csv(filepath, restaurant_filter=None):
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        
        reviews = []
        restaurants_data = []
        
        restaurant_groups = df.groupby('Restaurant_Name')
        
        for restaurant_name, group in restaurant_groups:
            if pd.isna(restaurant_name) or str(restaurant_name).lower() == 'nan':
                continue
                
            if restaurant_filter:
                if restaurant_filter.lower() not in str(restaurant_name).lower():
                    continue
            
            for _, row in group.iterrows():
                item_name = str(row.get('Item_Name', '')).strip()
                if item_name and item_name.lower() != 'nan':
                    review_text = f"Tried {item_name}"
                    if row.get('Best_Seller', False):
                        review_text += " - Best Seller"
                    
                    reviews.append({
                        'text': review_text,
                        'restaurant': restaurant_name,
                        'rating': row.get('Average_Rating', None),
                        'source': 'zomato2'
                    })
            
            if not restaurant_filter:
                avg_row = group.iloc[0]
                restaurants_data.append({
                    'name': restaurant_name,
                    'rating': avg_row.get('Avg_Rating_Restaurant', 'N/A'),
                    'location': avg_row.get('Place_Name', ''),
                    'city': avg_row.get('City', ''),
                    'cuisine': avg_row.get('Cuisine', ''),
                })
        
        return reviews, restaurants_data
    except Exception as e:
        print(f"Error processing zomato2.csv: {e}")
        return [], []

def process_all_datasets(restaurant_filter=None):
    all_reviews = []
    all_restaurants = []
    
    dataset_processors = {
        "mumbaires.csv": process_mumbaires_csv,
        "Resreviews.csv": process_resreviews_csv,
        "reviews.csv": process_reviews_csv,
        "zomato.csv": process_zomato_csv,
        "zomato2.csv": process_zomato2_csv,
    }
    
    for filename, processor in dataset_processors.items():
        filepath = os.path.join(DATASET_FOLDER, filename)
        if os.path.exists(filepath):
            try:
                reviews, restaurants = processor(filepath, restaurant_filter)
                all_reviews.extend(reviews)
                all_restaurants.extend(restaurants)
                if restaurant_filter and reviews:
                    print(f"✓ Found {len(reviews)} reviews in {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return all_reviews, all_restaurants

@app.route("/", methods=["GET"])
def index():
    try:
        _, restaurants_data = process_all_datasets(restaurant_filter=None)
        
        unique_restaurants = {}
        for r in restaurants_data:
            name = r['name']
            if name not in unique_restaurants:
                r['photo'] = get_restaurant_image(name)
                unique_restaurants[name] = r
        
        restaurants = list(unique_restaurants.values())
        
        if not restaurants:
            logger.warning("No restaurant data found in datasets folder")
            flash("No restaurant data found in datasets folder.", "warning")
        else:
            logger.info(f"Loaded {len(restaurants)} restaurants for index page")
        
        return render_template("index.html", restaurants=restaurants[:50])
    except Exception as e:
        logger.error(f"Error loading index page: {e}", exc_info=True)
        flash("An error occurred while loading restaurants. Please try again.", "error")
        return render_template("index.html", restaurants=[])

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    try:
        if request.method == "GET":
            restaurant = request.args.get("restaurant", "").strip()
        else:
            restaurant = request.form.get("restaurant_name", "").strip()
        
        # Validate restaurant name
        restaurant = sanitize_string(restaurant, max_length=200)
        is_valid, error_msg = validate_restaurant_name(restaurant)
        if not is_valid:
            flash(error_msg or "Please provide a valid restaurant name.", "error")
            logger.warning(f"Invalid restaurant name: {restaurant}")
            return redirect(url_for("index"))
        
        logger.info(f"Starting analysis for restaurant: {restaurant}")

        reviews_data = []
        from_uploaded_docs = []
        source = "database"

        # Handle file upload
        if request.method == "POST" and "datafile" in request.files:
            f = request.files["datafile"]
            if f and f.filename and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                
                try:
                    f.save(save_path)
                    
                    # Validate file size
                    is_valid_size, size_error = validate_file_size(save_path, max_size_mb=16)
                    if not is_valid_size:
                        os.remove(save_path)
                        flash(size_error, "error")
                        logger.warning(f"File size validation failed: {size_error}")
                        return redirect(url_for("index"))
                    
                    logger.info(f"File uploaded successfully: {filename}")
                    
                    if filename.lower().endswith(".csv"):
                        for processor in [process_mumbaires_csv, process_resreviews_csv, 
                                        process_reviews_csv, process_zomato_csv, process_zomato2_csv]:
                            try:
                                reviews, _ = processor(save_path, restaurant)
                                if reviews:
                                    reviews_data.extend(reviews)
                                    source = "csv_upload"
                                    from_uploaded_docs.append(save_path)
                                    break
                            except Exception as proc_error:
                                logger.debug(f"Processor {processor.__name__} failed: {proc_error}")
                                continue
                        
                        if reviews_data:
                            flash(f"✓ Loaded {len(reviews_data)} reviews from uploaded CSV", "success")
                            logger.info(f"Loaded {len(reviews_data)} reviews from CSV upload")
                    else:
                        with open(save_path, "r", encoding="utf-8", errors="ignore") as fh:
                            lines = [l.strip() for l in fh.readlines() if l.strip()]
                            reviews_data = [{'text': line, 'restaurant': restaurant, 
                                           'source': 'txt_upload'} for line in lines 
                                           if is_valid_review_text(line)]
                            source = "txt_upload"
                            from_uploaded_docs.append(save_path)
                            flash(f"✓ Loaded {len(reviews_data)} reviews from text file", "success")
                            logger.info(f"Loaded {len(reviews_data)} reviews from text upload")
                except Exception as upload_error:
                    logger.error(f"Error processing uploaded file: {upload_error}", exc_info=True)
                    flash("Error processing uploaded file. Please try again.", "error")
                    if os.path.exists(save_path):
                        os.remove(save_path)

        # Try to load from local datasets
        if not reviews_data:
            reviews_data, _ = process_all_datasets(restaurant_filter=restaurant)
            if reviews_data:
                source = "local_dataset"
                flash(f"✓ Found {len(reviews_data)} reviews for '{restaurant}' across all datasets", "success")
                logger.info(f"Loaded {len(reviews_data)} reviews from local datasets")

        # Try to load from database
        if not reviews_data:
            db_reviews = Review.query.filter(Review.restaurant.ilike(f"%{restaurant}%")).all()
            if db_reviews:
                reviews_data = [{'text': r.text, 'restaurant': r.restaurant, 
                               'rating': r.rating, 'source': 'database'} for r in db_reviews]
                source = "database"
                flash(f"✓ Found {len(reviews_data)} reviews in database", "info")
                logger.info(f"Loaded {len(reviews_data)} reviews from database")

        # Try scraping if requested
        if not reviews_data and request.method == "POST":
            try_scrape = request.form.get("try_scrape") == "on"
            if try_scrape:
                try:
                    logger.info(f"Attempting to scrape reviews for {restaurant}")
                    scraped = scrape_generic_reviews(restaurant, max_reviews=20)
                    if len(scraped) < 5:
                        scraped += scrape_zomato_placeholder(restaurant, max_reviews=10)
                    
                    if scraped:
                        reviews_data = [{'text': item['review'] if isinstance(item, dict) else item,
                                       'restaurant': restaurant, 'source': 'scraping'} for item in scraped]
                        source = "scraping"
                        flash(f"✓ Scraped {len(reviews_data)} reviews", "success")
                        logger.info(f"Scraped {len(reviews_data)} reviews")
                except Exception as scrape_error:
                    logger.error(f"Scraping error: {scrape_error}", exc_info=True)
                    flash("Scraping failed. Please try another method.", "warning")

        if not reviews_data:
            flash(f"❌ No reviews found for '{restaurant}'. Try uploading a CSV or enabling scraping.", "error")
            logger.warning(f"No reviews found for restaurant: {restaurant}")
            return redirect(url_for("index"))

        # Deduplicate reviews before processing
        original_count = len(reviews_data)
        reviews_data = deduplicate_reviews(reviews_data)
        if original_count > len(reviews_data):
            logger.info(f"Removed {original_count - len(reviews_data)} duplicate reviews")
            flash(f"Removed {original_count - len(reviews_data)} duplicate reviews", "info")

        stored_count = 0
        reviews_texts = []
        
        try:
            for review in reviews_data:
                review_text = review['text']
                reviews_texts.append(review_text)
                
                # Check if review already exists
                exists = Review.query.filter(
                    Review.restaurant.ilike(f"%{restaurant}%"), 
                    Review.text == review_text
                ).first()
                
                if exists:
                    continue
                
                sentiment, score, keywords = analyze_text_and_keywords(review_text)
                categories = categorize_complaints(review_text)
                
                r = Review(
                    restaurant=restaurant,
                    reviewer=review.get('reviewer', 'anonymous'),
                    text=review_text,
                    rating=review.get('rating'),
                    sentiment=sentiment,
                    score=score,
                    keywords=",".join(keywords),
                    categories=",".join(categories),
                    source_file=review.get('source', source)
                )
                db.session.add(r)
                stored_count += 1
            
            if stored_count > 0:
                db.session.commit()
                flash(f"✓ Added {stored_count} new reviews to database", "success")
                logger.info(f"Stored {stored_count} new reviews in database")
        except Exception as db_error:
            db.session.rollback()
            logger.error(f"Database error while storing reviews: {db_error}", exc_info=True)
            flash("Error storing reviews in database, but analysis will continue.", "warning")

        # Prepare documents for RAG
        docs_texts = []
        for p in from_uploaded_docs:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    docs_texts.append(fh.read())
            except Exception as doc_error:
                logger.warning(f"Could not read document {p}: {doc_error}")

        # Initialize RAG
        global rag_instance, current_indexed_restaurant
        try:
            rag_instance = build_or_get_rag(reviews_texts, docs_texts)
            current_indexed_restaurant = restaurant
            logger.info(f"RAG instance initialized for {restaurant}")
        except Exception as rag_error:
            logger.error(f"Error building RAG: {rag_error}", exc_info=True)
            flash("Note: Chat functionality may not be available.", "warning")

        flash(f"✓ Analysis complete! Processed {len(reviews_texts)} reviews.", "success")
        logger.info(f"Analysis complete for {restaurant}: {len(reviews_texts)} reviews")
        return redirect(url_for("results", restaurant_name=restaurant))
    
    except Exception as e:
        logger.error(f"Unexpected error in analyze route: {e}", exc_info=True)
        flash("An unexpected error occurred. Please try again.", "error")
        return redirect(url_for("index"))

@app.route("/results")
def results():
    restaurant = request.args.get("restaurant_name", "")
    if not restaurant:
        flash("Missing restaurant name", "error")
        return redirect(url_for("index"))
    
    reviews = Review.query.filter(
        Review.restaurant.ilike(f"%{restaurant}%")
    ).order_by(Review.created_at.desc()).all()
    
    if not reviews:
        flash(f"No reviews found for '{restaurant}'", "warning")
        return redirect(url_for("index"))
    
    viz_images = generate_visualizations(reviews)
    
    category_counts = {}
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    source_counts = {}
    
    for r in reviews:
        cats = [c.strip() for c in (r.categories or "").split(",") if c.strip()]
        for c in cats:
            category_counts[c] = category_counts.get(c, 0) + 1
        
        if r.sentiment:
            sentiment_counts[r.sentiment] = sentiment_counts.get(r.sentiment, 0) + 1
        
        source = r.source_file or "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1
    
    total = len(reviews)
    sentiment_percentages = {k: round((v/total)*100, 1) for k, v in sentiment_counts.items()}
    
    return render_template("results.html",
                         restaurant=restaurant,
                         reviews=reviews,
                         total_reviews=total,
                         category_counts=category_counts,
                         sentiment_counts=sentiment_counts,
                         sentiment_percentages=sentiment_percentages,
                         source_counts=source_counts,
                         visualizations=viz_images)

@app.route("/recommendations")
def recommendations():
    restaurant = request.args.get("restaurant_name", "")
    if not restaurant:
        flash("Missing restaurant name", "error")
        return redirect(url_for("index"))
    
    reviews = Review.query.filter(Review.restaurant.ilike(f"%{restaurant}%")).all()
    if not reviews:
        flash(f"No reviews found for '{restaurant}'", "warning")
        return redirect(url_for("index"))
    
    recs, counts, sentiments = summarize_reviews_for_recommendations(reviews)
    
    return render_template("recommendations.html",
                         restaurant=restaurant,
                         recommendations=recs,
                         category_counts=counts,
                         sentiment_counts=sentiments)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    global rag_instance, current_indexed_restaurant
    
    if request.method == "GET":
        return render_template("chat.html", 
                             restaurant=current_indexed_restaurant or "")
    
    q = request.form.get("question", "").strip()
    if not q:
        return jsonify({"error": "No question provided"}), 400
    
    if rag_instance is None:
        return jsonify({"error": "No indexed documents. Please run analysis first."}), 400
    
    answer, sources = rag_instance.answer_query(q, top_k=3)
    return jsonify({"answer": answer, "sources": sources})

@app.route("/search_restaurants")
def search_restaurants():
    query = request.args.get("q", "").strip().lower()
    if not query:
        return jsonify([])
    
    _, restaurants_data = process_all_datasets(restaurant_filter=None)
    
    matches = [r for r in restaurants_data if query in r['name'].lower()]
    
    unique = {}
    for r in matches:
        if r['name'] not in unique:
            unique[r['name']] = r
    
    return jsonify(list(unique.values())[:10])

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
