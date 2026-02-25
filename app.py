import os
import io
import requests
import random
import json
import ast
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from analyzer import (analyze_text_and_keywords, categorize_complaints, 
                     summarize_reviews_for_recommendations, generate_visualizations)
from scraper import scrape_generic_reviews, scrape_zomato_placeholder
from rag_chat import RAGChat
import pandas as pd
from datetime import datetime

UPLOAD_FOLDER = "uploads"
DATASET_FOLDER = "datasets"
ALLOWED_EXT = {"csv", "txt"}

app = Flask(__name__)
app.secret_key = "dev-secret-change-in-production"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///reviews.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

db = SQLAlchemy(app)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER, exist_ok=True)

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    restaurant = db.Column(db.String(200), index=True, nullable=False)
    reviewer = db.Column(db.String(200), default="anonymous")
    text = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Float, nullable=True)
    sentiment = db.Column(db.String(50))
    score = db.Column(db.Float)
    keywords = db.Column(db.String(500))
    categories = db.Column(db.String(500))
    source_file = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

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

def get_restaurant_image(restaurant_name):
    hash_value = sum(ord(c) * (i + 1) for i, c in enumerate(restaurant_name))
    unique_id = hash_value % 10000
    
    
    return f"https://loremflickr.com/800/600/food,restaurant/all?random={unique_id}"

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def extract_reviews_from_zomato_reviews_list(reviews_list_str):
    reviews = []
    if pd.isna(reviews_list_str) or not reviews_list_str:
        return reviews
    
    try:
        parsed = ast.literal_eval(reviews_list_str)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, tuple) and len(item) >= 2:
                    review_text = item[0]
                    if review_text and isinstance(review_text, str):
                        reviews.append(review_text.strip())
        return reviews
    except:
        try:
            cleaned = str(reviews_list_str).replace('[', '').replace(']', '')
            parts = cleaned.split('(')
            for part in parts:
                if ')' in part:
                    text = part.split(')')[0].strip()
                    if text and len(text) > 10:
                        reviews.append(text.strip('"').strip("'"))
        except:
            pass
    
    return reviews

def process_mumbaires_csv(filepath, restaurant_filter=None):
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        
        reviews = []
        restaurants_data = []
        
        for _, row in df.iterrows():
            restaurant_name = str(row.get('Restaurant Name', '')).strip()
            review_text = str(row.get('Review Text', '')).strip()
            
            if pd.isna(restaurant_name) or restaurant_name.lower() == 'nan':
                continue
                
            if restaurant_filter:
                if restaurant_filter.lower() not in restaurant_name.lower():
                    continue
            
            if review_text and review_text.lower() != 'nan' and len(review_text) > 10:
                reviews.append({
                    'text': review_text,
                    'restaurant': restaurant_name,
                    'rating': row.get('Reviewer Rating', None),
                    'source': 'mumbaires'
                })
            
            if not restaurant_filter:
                restaurants_data.append({
                    'name': restaurant_name,
                    'rating': row.get('Rating', 'N/A'),
                    'address': row.get('Address', 'Address not available'),
                    'price_level': row.get('Price Level', ''),
                })
        
        return reviews, restaurants_data
    except Exception as e:
        print(f"Error processing mumbaires.csv: {e}")
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
    _, restaurants_data = process_all_datasets(restaurant_filter=None)
    
    unique_restaurants = {}
    for r in restaurants_data:
        name = r['name']
        if name not in unique_restaurants:
            r['photo'] = get_restaurant_image(name)
            unique_restaurants[name] = r
    
    restaurants = list(unique_restaurants.values())
    
    if not restaurants:
        flash("No restaurant data found in datasets folder.", "warning")
    
    return render_template("index.html", restaurants=restaurants[:50])

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "GET":
        restaurant = request.args.get("restaurant", "").strip()
    else:
        restaurant = request.form.get("restaurant_name", "").strip()
    
    if not restaurant:
        flash("Please provide a restaurant name.", "error")
        return redirect(url_for("index"))

    reviews_data = []
    from_uploaded_docs = []
    source = "database"

    if request.method == "POST" and "datafile" in request.files:
        f = request.files["datafile"]
        if f and f.filename and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            f.save(save_path)
            
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
                    except:
                        continue
                
                if reviews_data:
                    flash(f"✓ Loaded {len(reviews_data)} reviews from uploaded CSV", "success")
            else:
                with open(save_path, "r", encoding="utf-8", errors="ignore") as fh:
                    lines = [l.strip() for l in fh.readlines() if l.strip()]
                    reviews_data = [{'text': line, 'restaurant': restaurant, 
                                   'source': 'txt_upload'} for line in lines]
                    source = "txt_upload"
                    from_uploaded_docs.append(save_path)
                    flash(f"✓ Loaded {len(reviews_data)} reviews from text file", "success")

    if not reviews_data:
        reviews_data, _ = process_all_datasets(restaurant_filter=restaurant)
        if reviews_data:
            source = "local_dataset"
            flash(f"✓ Found {len(reviews_data)} reviews for '{restaurant}' across all datasets", "success")

    if not reviews_data:
        db_reviews = Review.query.filter(Review.restaurant.ilike(f"%{restaurant}%")).all()
        if db_reviews:
            reviews_data = [{'text': r.text, 'restaurant': r.restaurant, 
                           'rating': r.rating, 'source': 'database'} for r in db_reviews]
            source = "database"
            flash(f"✓ Found {len(reviews_data)} reviews in database", "info")

    if not reviews_data and request.method == "POST":
        try_scrape = request.form.get("try_scrape") == "on"
        if try_scrape:
            scraped = scrape_generic_reviews(restaurant, max_reviews=20)
            if len(scraped) < 5:
                scraped += scrape_zomato_placeholder(restaurant, max_reviews=10)
            
            if scraped:
                reviews_data = [{'text': item['review'] if isinstance(item, dict) else item,
                               'restaurant': restaurant, 'source': 'scraping'} for item in scraped]
                source = "scraping"
                flash(f"✓ Scraped {len(reviews_data)} reviews", "success")

    if not reviews_data:
        flash(f"❌ No reviews found for '{restaurant}'. Try uploading a CSV or enabling scraping.", "error")
        return redirect(url_for("index"))

    stored_count = 0
    reviews_texts = []
    
    for review in reviews_data:
        review_text = review['text']
        reviews_texts.append(review_text)
        
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

    docs_texts = []
    for p in from_uploaded_docs:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                docs_texts.append(fh.read())
        except Exception:
            pass

    global rag_instance, current_indexed_restaurant
    rag_instance = build_or_get_rag(reviews_texts, docs_texts)
    current_indexed_restaurant = restaurant

    flash(f"✓ Analysis complete! Processed {len(reviews_texts)} reviews.", "success")
    return redirect(url_for("results", restaurant_name=restaurant))

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
