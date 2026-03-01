import re
import io
import base64
import ast
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from nltk.corpus import stopwords
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

nltk.download('stopwords', quiet=True)

analyzer = SentimentIntensityAnalyzer()
STOPWORDS = set(stopwords.words('english'))

CATEGORY_KEYWORDS = {
    "service": ["service", "wait", "waiter", "staff", "server", "attitude", "rude", "slow", 
                "unfriendly", "impolite", "ignored", "attention", "waiting"],
    "food_quality": ["cold", "burnt", "undercooked", "bland", "taste", "flavour", "flavor", 
                     "spoiled", "stale", "overcooked", "raw", "soggy", "hard", "dry", "greasy"],
    "hygiene": ["dirty", "hygiene", "clean", "unclean", "flies", "smell", "smelly", 
                "sanitation", "filthy", "unhygienic", "cockroach", "insects"],
    "price": ["expensive", "price", "cost", "overpriced", "costly", "pricey", "value", "money"],
    "delivery": ["delivery", "late", "packaging", "missing", "driver", "delayed", "order", 
                 "arrived", "cold food", "damaged"],
    "portion": ["small", "portion", "quantity", "size", "less", "tiny", "inadequate"],
    "ambience": ["ambience", "music", "noisy", "crowded", "lighting", "atmosphere", 
                 "decor", "seating", "comfort", "space"],
    "variety": ["menu", "options", "variety", "limited", "choices", "selection"],
}

DATASET_FOLDER = "datasets"

def extract_keywords(text, top_k=8):
    text = text.lower()
    words = re.findall(r"\b[a-z]{3,}\b", text)
    words = [w for w in words if w not in STOPWORDS and len(w) > 3]
    c = Counter(words)
    return [w for w, _ in c.most_common(top_k)]

def analyze_text_and_keywords(text):
    vs = analyzer.polarity_scores(text)
    compound = vs.get("compound", 0.0)
    
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    
    keywords = extract_keywords(text, top_k=8)
    return label, compound, keywords

def categorize_complaints(text):
    text_l = text.lower()
    cats = []
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in text_l:
                cats.append(cat)
                break
    return list(dict.fromkeys(cats))

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='#0f172a')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

def get_restaurant_info(restaurant_name):
    info = {
        'name': restaurant_name,
        'rating': None,
        'address': None,
        'cuisines': None,
        'cost': None,
        'location': None,
        'description': None
    }
    
    dataset_files = {
        "mumbaires.csv": {
            "name_col": "Restaurant Name",
            "rating_col": "Rating",
            "address_col": "Address",
            "price_col": "Price Level"
        },
        "Resreviews.csv": {
            "name_col": "Restaurant",
            "rating_col": "Rating"
        },
        "reviews.csv": {
            "name_col": "business_name",
            "rating_col": "rating"
        },
        "zomato.csv": {
            "name_col": "name",
            "rating_col": "rate",
            "address_col": "address",
            "location_col": "location",
            "cuisines_col": "cuisines",
            "cost_col": "approx_cost(for two people)"
        },
        "zomato2.csv": {
            "name_col": "Restaurant_Name",
            "rating_col": "Avg_Rating_Restaurant",
            "location_col": "Place_Name",
            "city_col": "City",
            "cuisine_col": "Cuisine"
        }
    }
    
    for fname, config in dataset_files.items():
        fpath = os.path.join(DATASET_FOLDER, fname)
        if not os.path.exists(fpath):
            continue
        
        try:
            df = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            
            name_col = config["name_col"]
            if name_col not in df.columns:
                continue
            
            df_filtered = df[df[name_col].astype(str).str.contains(
                restaurant_name, case=False, na=False, regex=False
            )]
            
            if not df_filtered.empty:
                row = df_filtered.iloc[0]
                
                if 'rating_col' in config and config['rating_col'] in df.columns:
                    rating = row.get(config['rating_col'])
                    if pd.notna(rating):
                        info['rating'] = rating
                
                if 'address_col' in config and config['address_col'] in df.columns:
                    address = row.get(config['address_col'])
                    if pd.notna(address):
                        info['address'] = address
                
                if 'cuisines_col' in config and config['cuisines_col'] in df.columns:
                    cuisines = row.get(config['cuisines_col'])
                    if pd.notna(cuisines):
                        info['cuisines'] = cuisines
                
                if 'cost_col' in config and config['cost_col'] in df.columns:
                    cost = row.get(config['cost_col'])
                    if pd.notna(cost):
                        info['cost'] = cost
                
                if 'location_col' in config and config['location_col'] in df.columns:
                    location = row.get(config['location_col'])
                    if pd.notna(location):
                        info['location'] = location
                
                if 'cuisine_col' in config and config['cuisine_col'] in df.columns:
                    cuisine = row.get(config['cuisine_col'])
                    if pd.notna(cuisine):
                        info['cuisines'] = cuisine
                
                if info['rating'] or info['address']:
                    break
        
        except Exception as e:
            print(f"Error reading {fname}: {e}")
    
    return info

def generate_visualizations(reviews):
    images = {}
    plt.style.use('dark_background')
    
    if not reviews:
        return images
    
    restaurant_name = reviews[0].restaurant
    restaurant_info = get_restaurant_info(restaurant_name)
    
    kpi_metrics = {
        'total_reviews': len(reviews),
        'avg_rating': 0,
        'positive_pct': 0,
        'negative_pct': 0,
        'avg_cost': 'N/A',
        'online_order': 'N/A',
        'table_booking': 'N/A',
        'address': restaurant_info.get('address') or 'N/A',
        'location': 'N/A',
        'cuisines': restaurant_info.get('cuisines') or 'N/A'
    }
    
    rating_list = []
    for r in reviews:
        if not hasattr(r, 'rating') or r.rating in (None, ''):
            continue
        try:
            rating_val = float(r.rating)
        except (TypeError, ValueError):
            continue
        if not np.isnan(rating_val):
            rating_list.append(rating_val)
    sentiment_list = [r.sentiment for r in reviews if hasattr(r, 'sentiment') and r.sentiment]
    
    if rating_list:
        kpi_metrics['avg_rating'] = round(np.mean(rating_list), 2)
    if sentiment_list:
        pos_count = sentiment_list.count('Positive')
        neg_count = sentiment_list.count('Negative')
        total = len(sentiment_list)
        kpi_metrics['positive_pct'] = round((pos_count / total) * 100, 1)
        kpi_metrics['negative_pct'] = round((neg_count / total) * 100, 1)
    
    csv_files_for_kpis = ['zomato.csv', 'mumbaires.csv', 'Resreviews.csv']
    for csv_file in csv_files_for_kpis:
        try:
            csv_path = os.path.join(DATASET_FOLDER, csv_file)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
                df.columns = df.columns.str.strip()
                
                name_col = None
                for col in ['name', 'Restaurant Name', 'business_name', 'restaurant_name']:
                    if col in df.columns:
                        name_col = col
                        break
                
                if name_col:
                    df_rest = df[df[name_col].astype(str).str.contains(restaurant_name, case=False, na=False)]
                    if not df_rest.empty:
                        row = df_rest.iloc[0]
                        
                        if kpi_metrics['avg_cost'] == 'N/A':
                            for cost_col in ['cost', 'approx_cost(for two people)', 'Cost', 'price']:
                                if cost_col in df_rest.columns and pd.notna(row[cost_col]):
                                    kpi_metrics['avg_cost'] = str(row[cost_col])
                                    break
                        
                        if kpi_metrics['online_order'] == 'N/A' and 'online_order' in df_rest.columns:
                            if pd.notna(row['online_order']):
                                kpi_metrics['online_order'] = str(row['online_order'])
                        
                        if kpi_metrics['table_booking'] == 'N/A' and 'book_table' in df_rest.columns:
                            if pd.notna(row['book_table']):
                                kpi_metrics['table_booking'] = str(row['book_table'])
                        
                        if kpi_metrics['location'] == 'N/A':
                            for loc_col in ['location', 'Location', 'address', 'Address', 'area']:
                                if loc_col in df_rest.columns and pd.notna(row[loc_col]):
                                    kpi_metrics['location'] = str(row[loc_col])
                                    break
        except Exception as e:
            print(f"Error loading {csv_file} for KPIs: {e}")
    
    images['kpi_metrics'] = kpi_metrics
    images['restaurant_info'] = restaurant_info
    
    ratings_by_category = {}
    avg_ratings = []
    
    csv_files_for_ratings = ['reviews.csv', 'Resreviews.csv', 'Yelpreviws.csv']
    for csv_file in csv_files_for_ratings:
        try:
            csv_path = os.path.join(DATASET_FOLDER, csv_file)
            if os.path.exists(csv_path):
                df_reviews = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
                df_reviews.columns = df_reviews.columns.str.strip()
                
                name_col = None
                for col in ['business_name', 'Restaurant Name', 'name', 'restaurant_name', 'Business Name']:
                    if col in df_reviews.columns:
                        name_col = col
                        break
                
                if name_col:
                    df_rest = df_reviews[df_reviews[name_col].astype(str).str.contains(
                        restaurant_name, case=False, na=False, regex=False
                    )]
                    
                    if not df_rest.empty:
                        if 'rating_category' in df_rest.columns and 'rating' in df_rest.columns:
                            for cat in df_rest['rating_category'].unique():
                                if pd.notna(cat):
                                    cat_data = df_rest[df_rest['rating_category'] == cat]['rating']
                                    if cat not in ratings_by_category:
                                        ratings_by_category[cat] = []
                                    ratings_by_category[cat].extend(list(cat_data.dropna()))
                            
                            avg_ratings.extend(df_rest['rating'].dropna().tolist())
                        
                        elif 'rating' in df_rest.columns:
                            avg_ratings.extend(df_rest['rating'].dropna().tolist())
                        elif 'Rating' in df_rest.columns:
                            avg_ratings.extend(df_rest['Rating'].dropna().tolist())
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    categories = [c.replace('_', ' ').title() for c in ratings_by_category.keys()]
    avg_scores = [np.mean(ratings_by_category[c]) if ratings_by_category[c] else 0
                  for c in ratings_by_category.keys()]
    if categories:
        sorted_pairs = sorted(zip(categories, avg_scores), key=lambda x: x[1])
        categories_sorted, scores_sorted = zip(*sorted_pairs)
        y_pos = np.arange(len(categories_sorted))
        colors_cat = plt.cm.RdYlGn(np.array(scores_sorted) / 5.0)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.hlines(y_pos, 0, scores_sorted, color='#334155', linewidth=4, alpha=0.7)
        ax.scatter(scores_sorted, y_pos, s=180, c=colors_cat, edgecolor='#e2e8f0',
                   linewidth=1.2, zorder=3)

        ax.set_title("⭐ Ratings by Category", fontsize=20, fontweight='bold',
                    pad=25, color='#e2e8f0')
        ax.set_xlabel("Average Rating", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories_sorted, color='#e2e8f0')
        ax.set_xlim(0, 5.5)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_facecolor('#0f172a')

        for y, score in zip(y_pos, scores_sorted):
            ax.text(score + 0.08, y, f'{score:.2f}', va='center', fontsize=11,
                    fontweight='bold', color='#e2e8f0')

        plt.tight_layout()
        images['ratings_by_category'] = plot_to_base64(fig)
    
    all_menu_items = []
    csv_files_for_menu = ['zomato2.csv', 'zomato.csv', 'mumbaires.csv']
    
    for csv_file in csv_files_for_menu:
        try:
            csv_path = os.path.join(DATASET_FOLDER, csv_file)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
                df.columns = df.columns.str.strip()
                
                name_col = None
                for col in ['Restaurant_Name', 'name', 'Restaurant Name', 'restaurant_name']:
                    if col in df.columns:
                        name_col = col
                        break
                
                if name_col:
                    df_items = df[df[name_col].astype(str).str.contains(
                        restaurant_name, case=False, na=False, regex=False
                    )]
                    
                    if not df_items.empty:
                        item_col = None
                        for col in ['Item_Name', 'dish_name', 'menu_item', 'item', 'dish']:
                            if col in df_items.columns:
                                item_col = col
                                break
                        
                        rating_col = None
                        for col in ['Average_Rating', 'rating', 'item_rating', 'rate']:
                            if col in df_items.columns:
                                rating_col = col
                                break
                        
                        votes_col = None
                        for col in ['Votes', 'votes', 'popularity', 'count', 'orders']:
                            if col in df_items.columns:
                                votes_col = col
                                break
                        
                        bestseller_col = None
                        for col in ['Best_Seller', 'bestseller', 'popular', 'best_seller']:
                            if col in df_items.columns:
                                bestseller_col = col
                                break
                        
                        if item_col and (rating_col or votes_col):
                            for _, row in df_items.iterrows():
                                item_data = {
                                    'name': str(row[item_col]) if pd.notna(row[item_col]) else 'Unknown',
                                    'rating': pd.to_numeric(row[rating_col], errors='coerce') if rating_col else 0,
                                    'votes': pd.to_numeric(row[votes_col], errors='coerce') if votes_col else 0,
                                    'bestseller': str(row[bestseller_col]) if bestseller_col and pd.notna(row[bestseller_col]) else 'No'
                                }
                                all_menu_items.append(item_data)
        except Exception as e:
            print(f"Error reading menu from {csv_file}: {e}")
    
    if all_menu_items:
        menu_df = pd.DataFrame(all_menu_items)
        menu_df = menu_df[menu_df['votes'] > 0]
        if not menu_df.empty:
            top_items = menu_df.nlargest(8, 'votes')
            
            fig, ax = plt.subplots(figsize=(14, 7))
            items = [str(item)[:30] for item in top_items['name']]
            ratings = top_items['rating'].fillna(0)
            votes = top_items['votes'].fillna(0)
            colors_items = ['#fbbf24' if str(bs).upper() == 'BESTSELLER' or str(bs).upper() == 'YES' else '#14b8a6' 
                           for bs in top_items['bestseller']]
            
            bars = ax.bar(items, votes, color=colors_items, edgecolor='#0ea5e9', 
                         linewidth=2, alpha=0.85)
            
            ax.set_title("🏆 Top Menu Items by Popularity (All Sources)", fontsize=20, fontweight='bold', 
                        pad=25, color='#e2e8f0')
            ax.set_ylabel("Number of Votes", fontsize=14, fontweight='bold', color='#e2e8f0')
            ax.set_facecolor('#0f172a')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            plt.xticks(rotation=45, ha='right', fontsize=10)
            
            for i, (bar, rating) in enumerate(zip(bars, ratings)):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(votes)*0.02,
                           f'★{rating:.1f}', ha='center', va='bottom', fontsize=10, 
                           fontweight='bold', color='#fbbf24')
            
            plt.tight_layout()
            images['top_items'] = plot_to_base64(fig)
    
    counts = {}
    sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
    sentiment_scores = []
    category_sentiment_map = {}
    
    for r in reviews:
        cats = [c.strip() for c in (r.categories or "").split(",") if c.strip()]
        sent = r.sentiment if hasattr(r, "sentiment") and r.sentiment else "Neutral"
        
        for cat in cats:
            if cat not in category_sentiment_map:
                category_sentiment_map[cat] = {"Positive": 0, "Negative": 0, "Neutral": 0}
            category_sentiment_map[cat][sent] += 1
            counts[cat] = counts.get(cat, 0) + 1
        
        if hasattr(r, "sentiment") and r.sentiment:
            sentiments[r.sentiment] = sentiments.get(r.sentiment, 0) + 1
        
        if hasattr(r, "score") and r.score is not None:
            sentiment_scores.append(r.score)
    
    if any(sentiments.values()):
        total = sum(sentiments.values())
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = ['Positive', 'Negative', 'Neutral']
        values = [sentiments.get(c, 0) for c in categories]
        colors_sent = ['#22c55e', '#ef4444', '#64748b']
        
        bars = ax.bar(categories, values, color=colors_sent, edgecolor='#14b8a6', 
                     linewidth=2.5, alpha=0.8, width=0.6)
        
        ax.set_title("💯 Sentiment Analysis", fontsize=20, fontweight='bold', 
                    pad=25, color='#e2e8f0')
        ax.set_ylabel("Number of Reviews", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.set_facecolor('#0f172a')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   f'{int(val)}\n({val/total*100:.1f}%)',
                   ha='center', va='bottom', fontsize=12, fontweight='bold', 
                   color='#e2e8f0')
        
        plt.tight_layout()
        images['sentiment'] = plot_to_base64(fig)
    
    if category_sentiment_map:
        problem_severity = {}
        for cat, sentiments_dict in category_sentiment_map.items():
            negative_count = sentiments_dict.get("Negative", 0)
            total_count = sum(sentiments_dict.values())
            problem_severity[cat] = {
                'negative_count': negative_count,
                'total_count': total_count,
                'neg_percentage': (negative_count / total_count * 100) if total_count > 0 else 0
            }
        
        sorted_problems = sorted(problem_severity.items(), 
                                key=lambda x: x[1]['neg_percentage'], reverse=True)
        
        if sorted_problems:
            fig, ax = plt.subplots(figsize=(14, 7))
            categories = [cat for cat, _ in sorted_problems[:10]]
            neg_percentages = [data['neg_percentage'] for _, data in sorted_problems[:10]]
            colors_severity = ['#ef4444' if p > 50 else '#f97316' if p > 30 else '#fbbf24' 
                              for p in neg_percentages]
            
            y_pos = np.arange(len(categories))
            ax.hlines(y_pos, 0, neg_percentages, color='#334155', linewidth=4, alpha=0.7)
            ax.scatter(neg_percentages, y_pos, s=220, c=colors_severity,
                       edgecolor='#e2e8f0', linewidth=1.2, zorder=3)
            
            ax.set_title("🚨 Problem Areas by Negative Sentiment", 
                        fontsize=20, fontweight='bold', pad=25, color='#e2e8f0')
            ax.set_xlabel("% Negative Mentions", fontsize=14, fontweight='bold', color='#e2e8f0')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories, color='#e2e8f0')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_facecolor('#0f172a')
            
            for y, p in zip(y_pos, neg_percentages):
                ax.text(p + 1.2, y, f'{p:.1f}%', va='center', 
                       fontsize=11, fontweight='bold', color='#e2e8f0')
            
            plt.tight_layout()
            images['problem_areas'] = plot_to_base64(fig)
    
    if sentiment_scores:
        fig, ax = plt.subplots(figsize=(13, 6))
        sorted_scores = np.array(sorted(sentiment_scores))
        window = max(3, len(sorted_scores) // 10)
        moving_avg = pd.Series(sorted_scores).rolling(window=window, center=True).mean()
        
        ax.scatter(range(len(sorted_scores)), sorted_scores, alpha=0.4, s=60, 
                  color='#14b8a6', edgecolor='#0ea5e9', linewidth=0.8, label='Individual Reviews')
        ax.plot(range(len(sorted_scores)), moving_avg, color='#f59e0b', linewidth=3, 
               label=f'Trend (MA-{window})', alpha=0.9)
        ax.axhline(y=0, color='#64748b', linestyle='--', linewidth=2, alpha=0.6, label='Neutral')
        
        ax.fill_between(range(len(sorted_scores)), 0, sorted_scores, 
                       where=(sorted_scores >= 0), alpha=0.2, color='#22c55e')
        ax.fill_between(range(len(sorted_scores)), 0, sorted_scores, 
                       where=(sorted_scores < 0), alpha=0.2, color='#ef4444')
        
        ax.set_title("📈 Sentiment Trend in Reviews", fontsize=20, fontweight='bold', 
                    pad=25, color='#e2e8f0')
        ax.set_xlabel("Review Index", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.set_ylabel("Sentiment Score", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.legend(fontsize=11, loc='best', framealpha=0.95)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_facecolor('#0f172a')
        
        plt.tight_layout()
        images['sentiment_trend'] = plot_to_base64(fig)
    
    all_keywords = []
    for r in reviews:
        if hasattr(r, 'keywords') and r.keywords:
            all_keywords.extend([k.strip() for k in r.keywords.split(',') if k.strip()])
    
    if all_keywords:
        keyword_counts = Counter(all_keywords)
        top_keywords = dict(keyword_counts.most_common(12))
        
        keywords_list = list(top_keywords.keys())
        counts_list = list(top_keywords.values())
        sorted_pairs = sorted(zip(keywords_list, counts_list), key=lambda x: x[1])
        keywords_sorted, counts_sorted = zip(*sorted_pairs)
        y_pos = np.arange(len(keywords_sorted))
        sizes = np.array(counts_sorted) * 80
        colors = plt.cm.YlOrRd(np.array(counts_sorted) / max(counts_sorted))

        fig, ax = plt.subplots(figsize=(13, 8))
        ax.hlines(y_pos, 0, counts_sorted, color='#334155', linewidth=3, alpha=0.7)
        ax.scatter(counts_sorted, y_pos, s=sizes, c=colors, alpha=0.85,
                   edgecolor='#e2e8f0', linewidth=1.2, zorder=3)
        
        ax.set_title("🎯 Top Customer Concerns & Keywords", fontsize=20, fontweight='bold', 
                    pad=25, color='#e2e8f0')
        ax.set_xlabel("Frequency", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(keywords_sorted, color='#e2e8f0')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_facecolor('#0f172a')

        for y, count in zip(y_pos, counts_sorted):
            ax.text(count + 0.3, y, str(count), va='center', fontsize=11,
                    fontweight='bold', color='#e2e8f0')
        
        plt.tight_layout()
        images['keywords'] = plot_to_base64(fig)
    
    if rating_list and len(rating_list) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        kde_enabled = len(set(rating_list)) > 1
        histplot_kwargs = dict(
            bins=[0, 1, 2, 3, 4, 5, 6],
            kde=kde_enabled,
            color='#14b8a6',
            edgecolor='#0ea5e9',
            linewidth=1.5,
            ax=ax
        )
        if kde_enabled:
            histplot_kwargs['line_kws'] = {'color': '#f59e0b', 'linewidth': 2}
        sns.histplot(rating_list, **histplot_kwargs)
        ax.set_title("⭐ Rating Distribution", fontsize=20, fontweight='bold', 
                    pad=25, color='#e2e8f0')
        ax.set_xlabel("Rating", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.set_ylabel("Frequency", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.axvline(x=kpi_metrics['avg_rating'], color='#f59e0b', linestyle='--', 
                  linewidth=2.5, label=f"Avg: {kpi_metrics['avg_rating']}")
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#0f172a')
        plt.tight_layout()
        images['rating_histogram'] = plot_to_base64(fig)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(x=rating_list, orient='h', inner='quartile',
                       color='#0ea5e9', linewidth=1.5, ax=ax)
        mean_val = np.mean(rating_list)
        ax.scatter([mean_val], [0], color='#f59e0b', s=90, zorder=3, label='Mean')
        ax.set_title("📊 Rating Spread Analysis", fontsize=20, fontweight='bold', 
                    pad=25, color='#e2e8f0')
        ax.set_xlabel("Rating", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.set_yticks([])
        ax.legend(fontsize=12)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_facecolor('#0f172a')
        plt.tight_layout()
        images['rating_boxplot'] = plot_to_base64(fig)
    
    all_cost_rating_data = []
    all_online_order_data = []
    all_table_booking_data = []
    
    csv_files_for_features = ['zomato.csv', 'mumbaires.csv', 'Resreviews.csv', 'reviews.csv']
    
    for csv_file in csv_files_for_features:
        try:
            csv_path = os.path.join(DATASET_FOLDER, csv_file)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
                df.columns = df.columns.str.strip()
                
                name_col = None
                for col in ['name', 'Restaurant Name', 'business_name', 'restaurant_name']:
                    if col in df.columns:
                        name_col = col
                        break
                
                rating_col = None
                for col in ['rate', 'rating', 'aggregate_rating', 'Rating']:
                    if col in df.columns:
                        rating_col = col
                        break
                
                if name_col and rating_col:
                    df_rest = df[df[name_col].astype(str).str.contains(restaurant_name, case=False, na=False)]
                    
                    if not df_rest.empty:
                        cost_col = None
                        for col in ['cost', 'approx_cost(for two people)', 'Cost', 'price']:
                            if col in df.columns:
                                cost_col = col
                                break
                        
                        if cost_col:
                            for _, row in df_rest.iterrows():
                                cost_val = pd.to_numeric(str(row[cost_col]).replace(',', '').replace('₹', ''), errors='coerce')
                                rate_val = pd.to_numeric(str(row[rating_col]).replace('/5', ''), errors='coerce')
                                if pd.notna(cost_val) and pd.notna(rate_val) and cost_val > 0:
                                    all_cost_rating_data.append({'cost': cost_val, 'rating': rate_val})
                        
                        if 'online_order' in df.columns:
                            for _, row in df_rest.iterrows():
                                online_val = str(row['online_order']).strip()
                                rate_val = pd.to_numeric(str(row[rating_col]).replace('/5', ''), errors='coerce')
                                if pd.notna(rate_val) and online_val in ['Yes', 'No']:
                                    all_online_order_data.append({'online_order': online_val, 'rating': rate_val})
                        
                        if 'book_table' in df.columns:
                            for _, row in df_rest.iterrows():
                                table_val = str(row['book_table']).strip()
                                rate_val = pd.to_numeric(str(row[rating_col]).replace('/5', ''), errors='coerce')
                                if pd.notna(rate_val) and table_val in ['Yes', 'No']:
                                    all_table_booking_data.append({'book_table': table_val, 'rating': rate_val})
        except Exception as e:
            print(f"Error reading features from {csv_file}: {e}")
    
    if all_online_order_data:
        try:
            online_df = pd.DataFrame(all_online_order_data)
            if not online_df.empty and 'online_order' in online_df.columns:
                online_groups = online_df.groupby('online_order')['rating'].mean()
                if len(online_groups) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#22c55e' if idx == 'Yes' else '#ef4444' for idx in online_groups.index]
                    bars = ax.bar(online_groups.index, online_groups.values, 
                                 color=colors, edgecolor='#14b8a6', linewidth=2, alpha=0.85)
                    ax.set_title("🛒 Online Order Availability vs Avg Rating (All Sources)", 
                               fontsize=18, fontweight='bold', pad=25, color='#e2e8f0')
                    ax.set_ylabel("Average Rating", fontsize=14, fontweight='bold', color='#e2e8f0')
                    ax.set_ylim(0, 5.5)
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    ax.set_facecolor('#0f172a')
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{height:.2f}', ha='center', va='bottom', 
                               fontsize=12, fontweight='bold', color='#e2e8f0')
                    plt.tight_layout()
                    images['online_order_rating'] = plot_to_base64(fig)
        except Exception as e:
            print(f"Error creating online order chart: {e}")
    
    if all_table_booking_data:
        try:
            table_df = pd.DataFrame(all_table_booking_data)
            if not table_df.empty and 'book_table' in table_df.columns:
                table_groups = table_df.groupby('book_table')['rating'].mean()
                if len(table_groups) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#22c55e' if idx == 'Yes' else '#ef4444' for idx in table_groups.index]
                    bars = ax.bar(table_groups.index, table_groups.values, 
                                 color=colors, edgecolor='#14b8a6', linewidth=2, alpha=0.85)
                    ax.set_title("📅 Table Booking Availability vs Avg Rating (All Sources)", 
                               fontsize=18, fontweight='bold', pad=25, color='#e2e8f0')
                    ax.set_ylabel("Average Rating", fontsize=14, fontweight='bold', color='#e2e8f0')
                    ax.set_ylim(0, 5.5)
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    ax.set_facecolor('#0f172a')
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{height:.2f}', ha='center', va='bottom', 
                               fontsize=12, fontweight='bold', color='#e2e8f0')
                    plt.tight_layout()
                    images['table_booking_rating'] = plot_to_base64(fig)
        except Exception as e:
            print(f"Error creating table booking chart: {e}")
    
    if all_cost_rating_data:
        try:
            cost_df = pd.DataFrame(all_cost_rating_data)
            if not cost_df.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.scatter(cost_df['cost'], cost_df['rating'], 
                          s=100, alpha=0.7, c='#14b8a6', edgecolor='#0ea5e9', linewidth=2)
                ax.set_title("💰 Cost vs Rating Analysis (All Sources)", fontsize=20, 
                           fontweight='bold', pad=25, color='#e2e8f0')
                ax.set_xlabel("Cost for Two", fontsize=14, fontweight='bold', color='#e2e8f0')
                ax.set_ylabel("Rating", fontsize=14, fontweight='bold', color='#e2e8f0')
                ax.grid(alpha=0.3, linestyle='--')
                ax.set_facecolor('#0f172a')
                plt.tight_layout()
                images['cost_vs_rating'] = plot_to_base64(fig)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.hist(cost_df['cost'], bins=10, color='#22c55e', 
                       edgecolor='#14b8a6', linewidth=2, alpha=0.8)
                ax.set_title("💵 Cost Distribution (All Sources)", fontsize=20, fontweight='bold', 
                           pad=25, color='#e2e8f0')
                ax.set_xlabel("Cost for Two", fontsize=14, fontweight='bold', color='#e2e8f0')
                ax.set_ylabel("Frequency", fontsize=14, fontweight='bold', color='#e2e8f0')
                ax.axvline(x=cost_df['cost'].mean(), color='#f59e0b', 
                         linestyle='--', linewidth=2.5, 
                          label=f"Avg: ₹{cost_df['cost'].mean():.0f}")
                ax.legend(fontsize=12)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.set_facecolor('#0f172a')
                plt.tight_layout()
                images['cost_distribution'] = plot_to_base64(fig)
        except Exception as e:
            print(f"Error creating cost charts: {e}")
    
    try:
        cuisine_list = []
        for fname in ['zomato.csv', 'mumbaires.csv', 'Resreviews.csv', 'Yelpreviws.csv', 'reviews.csv']:
            fpath = os.path.join(DATASET_FOLDER, fname)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip")
                df.columns = df.columns.str.strip()
                
                cuisine_col = None
                for col in ['cuisines', 'Cousines', 'cuisine', 'food_type', 'category']:
                    if col in df.columns:
                        cuisine_col = col
                        break
                
                name_col = None
                for col in ['name', 'Restaurant Name', 'business_name', 'restaurant_name']:
                    if col in df.columns:
                        name_col = col
                        break
                
                if cuisine_col and name_col:
                    df_rest = df[df[name_col].astype(str).str.contains(restaurant_name, case=False, na=False)]
                    for _, row in df_rest.iterrows():
                        cuisines_str = str(row[cuisine_col])
                        if cuisines_str and cuisines_str != 'nan':
                            cuisine_list.extend([c.strip() for c in cuisines_str.split(',') if c.strip()])
        
        if cuisine_list:
            cuisine_counts = Counter(cuisine_list)
            top_cuisines = dict(cuisine_counts.most_common(10))
            
            fig, ax = plt.subplots(figsize=(12, 7))
            cuisines = list(top_cuisines.keys())
            counts = list(top_cuisines.values())
            colors = ['#14b8a6', '#22c55e', '#0ea5e9', '#06b6d4', '#10b981', '#84cc16', '#eab308', '#f59e0b', '#ec4899', '#a855f7'][:len(cuisines)]
            bars = ax.barh(cuisines, counts, color=colors, edgecolor='#0ea5e9', 
                          linewidth=2, alpha=0.85)
            ax.set_title("🍽️ Cuisine Type Distribution (All Sources)", fontsize=20, fontweight='bold', 
                       pad=25, color='#e2e8f0')
            ax.set_xlabel("Frequency", fontsize=14, fontweight='bold', color='#e2e8f0')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_facecolor('#0f172a')
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(count + 0.2, i, str(count), va='center', 
                       fontsize=11, fontweight='bold', color='#e2e8f0')
            plt.tight_layout()
            images['cuisine_distribution'] = plot_to_base64(fig)
    except Exception as e:
        print(f"Error creating cuisine charts: {e}")
    
    try:
        location_branches = []
        for fname in ['zomato.csv', 'mumbaires.csv', 'Resreviews.csv', 'reviews.csv', 'Yelpreviws.csv']:
            fpath = os.path.join(DATASET_FOLDER, fname)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip")
                df.columns = df.columns.str.strip()
                name_col = 'name' if 'name' in df.columns else 'Restaurant Name' if 'Restaurant Name' in df.columns else None
                loc_col = 'location' if 'location' in df.columns else 'Address' if 'Address' in df.columns else None
                
                if name_col and loc_col:
                    df_rest = df[df[name_col].astype(str).str.contains(restaurant_name, case=False, na=False)]
                    for _, row in df_rest.iterrows():
                        loc = str(row[loc_col])
                        if loc and loc != 'nan':
                            location_branches.append(loc)
        
        if len(location_branches) > 1:
            branch_counts = Counter(location_branches)
            locations = list(branch_counts.keys())
            counts = list(branch_counts.values())
            colors = sns.color_palette("viridis", len(locations))

            fig, ax = plt.subplots(figsize=(10, 8))
            wedges, _ = ax.pie(counts, colors=colors, startangle=90,
                               wedgeprops={'width': 0.4, 'edgecolor': '#0f172a'})
            ax.add_artist(plt.Circle((0, 0), 0.55, fc='#0f172a'))
            ax.set_title("📍 Branch Distribution by Location", fontsize=20,
                       fontweight='bold', pad=25, color='#e2e8f0')
            ax.legend(wedges, [f"{loc} ({cnt})" for loc, cnt in zip(locations, counts)],
                      loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, frameon=False)
            fig.patch.set_facecolor('#0f172a')
            plt.tight_layout()
            images['branch_distribution'] = plot_to_base64(fig)
    except Exception as e:
        print(f"Error creating branch chart: {e}")
    
    review_lengths = [len(r.text) if hasattr(r, 'text') and r.text else 0 for r in reviews]
    review_lengths = [l for l in review_lengths if l > 0]
    
    if len(review_lengths) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        kde_enabled = len(set(review_lengths)) > 1
        histplot_kwargs = dict(
            bins=20,
            kde=kde_enabled,
            color='#0ea5e9',
            edgecolor='#14b8a6',
            linewidth=1.5,
            ax=ax
        )
        if kde_enabled:
            histplot_kwargs['line_kws'] = {'color': '#f59e0b', 'linewidth': 2}
        sns.histplot(review_lengths, **histplot_kwargs)
        p25, p50, p75 = np.percentile(review_lengths, [25, 50, 75])
        ax.axvline(x=p25, color='#94a3b8', linestyle='--', linewidth=1.5, label='25th')
        ax.axvline(x=p50, color='#f59e0b', linestyle='--', linewidth=2, label='Median')
        ax.axvline(x=p75, color='#22c55e', linestyle='--', linewidth=1.5, label='75th')
        ax.set_title("📝 Review Length Distribution", fontsize=20, fontweight='bold', 
                    pad=25, color='#e2e8f0')
        ax.set_xlabel("Character Count", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.set_ylabel("Frequency", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#0f172a')
        plt.tight_layout()
        images['review_length_dist'] = plot_to_base64(fig)
        
        if rating_list and len(rating_list) == len(review_lengths):
            fig, ax = plt.subplots(figsize=(12, 6))
            hb = ax.hexbin(review_lengths, rating_list, gridsize=25, cmap='viridis',
                           mincnt=1, linewidths=0.2, edgecolors='#0f172a')
            ax.set_title("📊 Review Length vs Rating", fontsize=20, fontweight='bold', 
                       pad=25, color='#e2e8f0')
            ax.set_xlabel("Review Length (characters)", fontsize=14, fontweight='bold', color='#e2e8f0')
            ax.set_ylabel("Rating", fontsize=14, fontweight='bold', color='#e2e8f0')
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_facecolor('#0f172a')
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label("Review Density", color='#e2e8f0')
            cb.ax.yaxis.set_tick_params(color='#e2e8f0')
            for label in cb.ax.get_yticklabels():
                label.set_color('#e2e8f0')
            plt.tight_layout()
            images['length_vs_rating'] = plot_to_base64(fig)
    
    category_counts = {}
    for r in reviews:
        if hasattr(r, 'categories') and r.categories:
            cats = [c.strip() for c in r.categories.split(',') if c.strip()]
            for cat in cats:
                category_counts[cat] = category_counts.get(cat, 0) + 1
    
    if category_counts:
        fig, ax = plt.subplots(figsize=(12, 7))
        sorted_cats = sorted(category_counts.items(), key=lambda x: x[1])
        cats = [c[0].replace('_', ' ').title() for c in sorted_cats]
        vals = [c[1] for c in sorted_cats]
        y_pos = np.arange(len(cats))
        colors = plt.cm.OrRd(np.array(vals) / max(vals))

        ax.hlines(y_pos, 0, vals, color='#334155', linewidth=3, alpha=0.7)
        ax.scatter(vals, y_pos, s=200, c=colors, edgecolor='#e2e8f0',
                   linewidth=1.2, zorder=3)
        ax.set_title("🔍 Complaint Category Frequency", fontsize=20, fontweight='bold', 
                   pad=25, color='#e2e8f0')
        ax.set_xlabel("Frequency", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cats, color='#e2e8f0')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_facecolor('#0f172a')
        for y, val in zip(y_pos, vals):
            ax.text(val + 0.5, y, str(val), va='center', fontsize=11, 
                   fontweight='bold', color='#e2e8f0')
        plt.tight_layout()
        images['category_frequency'] = plot_to_base64(fig)
    
    try:
        corr_data = []
        for r in reviews:
            row_data = {}
            if hasattr(r, 'rating') and r.rating:
                row_data['rating'] = r.rating
            if hasattr(r, 'score') and r.score is not None:
                row_data['sentiment_score'] = r.score
            if hasattr(r, 'text') and r.text:
                row_data['review_length'] = len(r.text)
            if row_data:
                corr_data.append(row_data)
        
        if len(corr_data) > 2:
            df_corr = pd.DataFrame(corr_data)
            correlation_matrix = df_corr.corr()
            
            if not correlation_matrix.empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, linewidths=2, linecolor='#0f172a', 
                           cbar_kws={'label': 'Correlation'}, ax=ax,
                           annot_kws={'color': '#e2e8f0', 'fontweight': 'bold'})
                ax.set_title("🔗 Feature Correlation Heatmap", fontsize=20, 
                           fontweight='bold', pad=25, color='#e2e8f0')
                ax.set_xticklabels(ax.get_xticklabels(), color='#e2e8f0')
                ax.set_yticklabels(ax.get_yticklabels(), color='#e2e8f0')
                ax.set_facecolor('#0f172a')
                fig.patch.set_facecolor('#0f172a')
                plt.tight_layout()
                images['correlation_heatmap'] = plot_to_base64(fig)
    except Exception as e:
        print(f"Error creating correlation heatmap: {e}")
    
    all_rated_items = []
    for csv_file in ['zomato2.csv', 'zomato.csv', 'mumbaires.csv']:
        try:
            csv_path = os.path.join(DATASET_FOLDER, csv_file)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
                df.columns = df.columns.str.strip()
                
                name_col = None
                for col in ['Restaurant_Name', 'name', 'Restaurant Name', 'restaurant_name']:
                    if col in df.columns:
                        name_col = col
                        break
                
                if name_col:
                    df_items = df[df[name_col].astype(str).str.contains(
                        restaurant_name, case=False, na=False, regex=False
                    )]
                    
                    if not df_items.empty:
                        item_col = None
                        for col in ['Item_Name', 'dish_name', 'menu_item', 'item', 'dish']:
                            if col in df_items.columns:
                                item_col = col
                                break
                        
                        rating_col = None
                        for col in ['Average_Rating', 'rating', 'item_rating', 'rate']:
                            if col in df_items.columns:
                                rating_col = col
                                break
                        
                        bestseller_col = None
                        for col in ['Best_Seller', 'bestseller', 'popular', 'best_seller']:
                            if col in df_items.columns:
                                bestseller_col = col
                                break
                        
                        if item_col and rating_col:
                            for _, row in df_items.iterrows():
                                rating_val = pd.to_numeric(row[rating_col], errors='coerce')
                                if pd.notna(rating_val) and rating_val > 0:
                                    all_rated_items.append({
                                        'name': str(row[item_col]),
                                        'rating': rating_val,
                                        'bestseller': str(row[bestseller_col]) if bestseller_col and pd.notna(row[bestseller_col]) else 'No'
                                    })
        except Exception as e:
            print(f"Error reading rated items from {csv_file}: {e}")
    
    if all_rated_items:
        items_df = pd.DataFrame(all_rated_items)
        if not items_df.empty:
            top_rated = items_df.nlargest(5, 'rating')
            if not top_rated.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                items = [str(item)[:30] for item in top_rated['name']]
                ratings = top_rated['rating']
                colors = ['#fbbf24', '#f59e0b', '#fb923c', '#f97316', '#ea580c'][:len(items)]
                bars = ax.barh(items, ratings, color=colors, edgecolor='#14b8a6', 
                              linewidth=2, alpha=0.85)
                ax.set_title("⭐ Top 5 Highest Rated Menu Items (All Sources)", fontsize=20, 
                           fontweight='bold', pad=25, color='#e2e8f0')
                ax.set_xlabel("Average Rating", fontsize=14, fontweight='bold', color='#e2e8f0')
                ax.invert_yaxis()
                ax.set_xlim(0, 5.5)
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                ax.set_facecolor('#0f172a')
                for i, (bar, rating) in enumerate(zip(bars, ratings)):
                    ax.text(rating + 0.1, i, f'{rating:.1f}', va='center', 
                           fontsize=11, fontweight='bold', color='#e2e8f0')
                plt.tight_layout()
                images['top_rated_items'] = plot_to_base64(fig)
            
            bestseller_count = sum(1 for item in all_rated_items 
                                  if str(item['bestseller']).upper() in ['BESTSELLER', 'YES'])
            total_items = len(all_rated_items)
            
            if total_items > 0 and bestseller_count > 0:
                fig, ax = plt.subplots(figsize=(8, 8))
                sizes = [bestseller_count, total_items - bestseller_count]
                labels = [f'Best Sellers\n({bestseller_count})', 
                         f'Regular Items\n({total_items - bestseller_count})']
                colors = ['#fbbf24', '#64748b']
                explode = (0.1, 0)
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                      startangle=90, explode=explode, textprops={'fontsize': 12, 'color': '#e2e8f0', 'fontweight': 'bold'})
                ax.set_title("🏆 Best Seller Distribution (All Sources)", fontsize=20, 
                           fontweight='bold', pad=25, color='#e2e8f0')
                fig.patch.set_facecolor('#0f172a')
                plt.tight_layout()
                images['bestseller_distribution'] = plot_to_base64(fig)
    
    return images

def extract_reviews_from_zomato_list(reviews_list_str):
    reviews = []
    if pd.isna(reviews_list_str) or not reviews_list_str:
        return reviews
    
    try:
        parsed = ast.literal_eval(reviews_list_str)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, tuple) and len(item) >= 2:
                    review_text = item[0]
                    if review_text and isinstance(review_text, str) and len(review_text) > 10:
                        reviews.append(review_text.strip())
        return reviews
    except Exception:
        try:
            cleaned = str(reviews_list_str).replace('[', '').replace(']', '')
            parts = cleaned.split('(')
            for part in parts:
                if ')' in part:
                    text = part.split(')')[0].strip()
                    if text and len(text) > 10:
                        reviews.append(text.strip('"').strip("'"))
        except Exception:
            pass
    
    return reviews


def get_reviews_from_datasets(restaurant_name):
    all_reviews = []
    
    dataset_files = {
        "mumbaires.csv": {
            "name_col": "Restaurant Name",
            "review_col": "Review Text",
            "rating_col": "Reviewer Rating"
        },
        "Resreviews.csv": {
            "name_col": "Restaurant",
            "review_col": "Review",
            "rating_col": "Rating"
        },
        "reviews.csv": {
            "name_col": "business_name",
            "review_col": "text",
            "rating_col": "rating"
        },
        "zomato.csv": {
            "name_col": "name",
            "review_col": "reviews_list",
            "rating_col": "rate"
        },
        "zomato2.csv": {
            "name_col": "Restaurant_Name",
            "review_col": None,
            "rating_col": "Average_Rating"
        }
    }
    
    for fname, config in dataset_files.items():
        fpath = os.path.join(DATASET_FOLDER, fname)
        if not os.path.exists(fpath):
            continue
        
        try:
            df = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            
            name_col = config["name_col"]
            if name_col not in df.columns:
                continue
            
            df_filtered = df[df[name_col].astype(str).str.contains(
                restaurant_name, case=False, na=False, regex=False
            )]
            
            if df_filtered.empty:
                continue
            
            if fname == "zomato.csv":
                for _, row in df_filtered.iterrows():
                    reviews_list = row.get(config["review_col"], "")
                    extracted = extract_reviews_from_zomato_list(reviews_list)
                    for review_text in extracted:
                        all_reviews.append({
                            'text': review_text,
                            'rating': row.get(config["rating_col"]),
                            'source': fname
                        })
            
            elif fname == "zomato2.csv":
                for _, row in df_filtered.iterrows():
                    item = row.get('Item_Name', '')
                    if item and str(item).lower() != 'nan':
                        review_text = f"Tried {item}"
                        if row.get('Best_Seller', False):
                            review_text += " - highly popular item"
                        all_reviews.append({
                            'text': review_text,
                            'rating': row.get(config["rating_col"]),
                            'source': fname
                        })
            
            else:
                review_col = config["review_col"]
                if review_col and review_col in df_filtered.columns:
                    for _, row in df_filtered.iterrows():
                        review_text = str(row.get(review_col, '')).strip()
                        if review_text and review_text.lower() != 'nan' and len(review_text) > 10:
                            all_reviews.append({
                                'text': review_text,
                                'rating': row.get(config["rating_col"]),
                                'source': fname
                            })
        
        except Exception as e:
            print(f"Error reading {fname}: {e}")
    
    return all_reviews


def summarize_reviews_for_recommendations(reviews, restaurant_name=None):
    counts = {}
    sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
    total_reviews = len(reviews)
    
    for r in reviews:
        for cat in (r.categories or "").split(","):
            if cat.strip():
                counts[cat.strip()] = counts.get(cat.strip(), 0) + 1
        
        if hasattr(r, "sentiment") and r.sentiment:
            sentiments[r.sentiment] = sentiments.get(r.sentiment, 0) + 1
    
    if restaurant_name:
        dataset_reviews = get_reviews_from_datasets(restaurant_name)
        total_reviews += len(dataset_reviews)
        
        for review_dict in dataset_reviews:
            text = review_dict['text']
            sent, score, _ = analyze_text_and_keywords(text)
            sentiments[sent] = sentiments.get(sent, 0) + 1
            
            for cat in categorize_complaints(text):
                counts[cat] = counts.get(cat, 0) + 1
    
    category_percentages = {}
    if total_reviews > 0:
        for cat, count in counts.items():
            category_percentages[cat] = round((count / total_reviews) * 100, 1)
    
    recs = []
    priority_recs = []
    standard_recs = []
    
    if category_percentages.get("service", 0) > 20:
        priority_recs.append({
            "category": "service",
            "priority": "HIGH",
            "recommendation": "🚨 URGENT: Train staff on customer service and reduce wait times",
            "percentage": category_percentages["service"]
        })
    elif counts.get("service", 0) > 0:
        standard_recs.append({
            "category": "service",
            "priority": "MEDIUM",
            "recommendation": "🧑‍🍳 Improve staff training and response efficiency",
            "percentage": category_percentages.get("service", 0)
        })
    
    if category_percentages.get("food_quality", 0) > 20:
        priority_recs.append({
            "category": "food_quality",
            "priority": "HIGH",
            "recommendation": "🚨 CRITICAL: Implement strict quality control in kitchen operations",
            "percentage": category_percentages["food_quality"]
        })
    elif counts.get("food_quality", 0) > 0:
        standard_recs.append({
            "category": "food_quality",
            "priority": "MEDIUM",
            "recommendation": "🍽️ Enhance food quality with regular kitchen audits",
            "percentage": category_percentages.get("food_quality", 0)
        })
    
    if category_percentages.get("hygiene", 0) > 15:
        priority_recs.append({
            "category": "hygiene",
            "priority": "HIGH",
            "recommendation": "🚨 CRITICAL: Immediate deep cleaning and hygiene protocol enforcement",
            "percentage": category_percentages["hygiene"]
        })
    elif counts.get("hygiene", 0) > 0:
        standard_recs.append({
            "category": "hygiene",
            "priority": "MEDIUM",
            "recommendation": "🧼 Improve cleanliness standards and visible hygiene practices",
            "percentage": category_percentages.get("hygiene", 0)
        })
    
    if counts.get("price", 0) > 0:
        standard_recs.append({
            "category": "price",
            "priority": "MEDIUM",
            "recommendation": "💰 Review pricing strategy and introduce value deals/combos",
            "percentage": category_percentages.get("price", 0)
        })
    
    if counts.get("delivery", 0) > 0:
        standard_recs.append({
            "category": "delivery",
            "priority": "MEDIUM",
            "recommendation": "🚚 Optimize delivery logistics and packaging quality",
            "percentage": category_percentages.get("delivery", 0)
        })
    
    if counts.get("portion", 0) > 0:
        standard_recs.append({
            "category": "portion",
            "priority": "LOW",
            "recommendation": "🍛 Standardize portion sizes to meet customer expectations",
            "percentage": category_percentages.get("portion", 0)
        })
    
    if counts.get("ambience", 0) > 0:
        standard_recs.append({
            "category": "ambience",
            "priority": "LOW",
            "recommendation": "🎶 Enhance ambience with better decor, lighting, and seating",
            "percentage": category_percentages.get("ambience", 0)
        })
    
    if counts.get("variety", 0) > 0:
        standard_recs.append({
            "category": "variety",
            "priority": "LOW",
            "recommendation": "📋 Expand menu variety and introduce seasonal specials",
            "percentage": category_percentages.get("variety", 0)
        })
    
    recs = priority_recs + standard_recs
    
    if not recs:
        positive_percentage = (sentiments.get("Positive", 0) / total_reviews * 100) if total_reviews > 0 else 0
        recs.append({
            "category": "overall",
            "priority": "INFO",
            "recommendation": f"✅ Overall sentiment is positive ({positive_percentage:.1f}%); maintain quality consistency",
            "percentage": positive_percentage
        })
    
    if total_reviews > 0:
        neg_percentage = (sentiments.get("Negative", 0) / total_reviews) * 100
        pos_percentage = (sentiments.get("Positive", 0) / total_reviews) * 100
        
        if neg_percentage > 40:
            recs.insert(0, {
                "category": "sentiment",
                "priority": "CRITICAL",
                "recommendation": f"⚠️ ALERT: {neg_percentage:.1f}% negative reviews - immediate action required!",
                "percentage": neg_percentage
            })
        elif pos_percentage > 70:
            recs.append({
                "category": "sentiment",
                "priority": "INFO",
                "recommendation": f"🌟 Excellent performance with {pos_percentage:.1f}% positive reviews!",
                "percentage": pos_percentage
            })
    
    return recs, counts, sentiments