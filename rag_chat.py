import os
import ast
import logging
import pandas as pd
import numpy as np
import textwrap
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from collections import Counter
import re
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Install with: pip install faiss-cpu")

class RAGChat:
    def __init__(self, data_folder="datasets", vector_db_folder="vector_db"):
        self.model = None
        self.doc_texts = []
        self.doc_metadata = []
        self.faiss_index = None
        self.embedding_dimension = 384
        self.data_folder = data_folder
        self.vector_db_folder = vector_db_folder
        self.loaded = False
        self.current_restaurant = None
        
        if not os.path.exists(vector_db_folder):
            os.makedirs(vector_db_folder, exist_ok=True)
        
        self._initialize_model()

    def _initialize_model(self):
        try:
            print("🔄 Loading embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            print(f"✅ Model loaded! Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("Install sentence-transformers: pip install sentence-transformers")
            self.model = None

    def _get_vector_db_path(self, restaurant_name):
        safe_name = re.sub(r'[^\w\s-]', '', restaurant_name).strip().replace(' ', '_')
        return {
            'index': os.path.join(self.vector_db_folder, f"{safe_name}.faiss"),
            'metadata': os.path.join(self.vector_db_folder, f"{safe_name}_metadata.pkl")
        }

    def _save_vector_db(self, restaurant_name):
        if not FAISS_AVAILABLE or self.faiss_index is None:
            return False
        
        try:
            paths = self._get_vector_db_path(restaurant_name)
            
            faiss.write_index(self.faiss_index, paths['index'])
            
            metadata = {
                'doc_texts': self.doc_texts,
                'doc_metadata': self.doc_metadata,
                'embedding_dimension': self.embedding_dimension,
                'timestamp': datetime.now(),
                'total_vectors': self.faiss_index.ntotal
            }
            with open(paths['metadata'], 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"💾 Saved vector DB: {self.faiss_index.ntotal} vectors for '{restaurant_name}'")
            return True
        except Exception as e:
            print(f"⚠️  Error saving vector DB: {e}")
            return False

    def _load_vector_db(self, restaurant_name):
        if not FAISS_AVAILABLE:
            return False
        
        try:
            paths = self._get_vector_db_path(restaurant_name)
            
            if not os.path.exists(paths['index']) or not os.path.exists(paths['metadata']):
                return False
            
            self.faiss_index = faiss.read_index(paths['index'])
            
            with open(paths['metadata'], 'rb') as f:
                metadata = pickle.load(f)
            
            self.doc_texts = metadata['doc_texts']
            self.doc_metadata = metadata['doc_metadata']
            self.embedding_dimension = metadata['embedding_dimension']
            
            print(f"✅ Loaded vector DB: {self.faiss_index.ntotal} vectors for '{restaurant_name}'")
            return True
        except Exception as e:
            print(f"⚠️  Error loading vector DB: {e}")
            return False

    def extract_reviews_from_zomato_list(self, reviews_list_str):
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
        except (ValueError, SyntaxError) as e:
            logger.debug(f"Failed to parse reviews via ast.literal_eval: {e}")
            try:
                cleaned = str(reviews_list_str).replace('[', '').replace(']', '')
                parts = cleaned.split('(')
                for part in parts:
                    if ')' in part:
                        text = part.split(')')[0].strip()
                        if text and len(text) > 10:
                            reviews.append(text.strip('"').strip("'"))
            except (ValueError, IndexError, AttributeError) as e:
                logger.debug(f"Fallback review extraction failed: {e}")
        
        return reviews

    def load_mumbaires_csv(self, filepath, restaurant_name=None):
        reviews = []
        try:
            df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            
            if restaurant_name:
                df = df[df['Restaurant Name'].astype(str).str.contains(
                    restaurant_name, case=False, na=False, regex=False
                )]
            
            for _, row in df.iterrows():
                review_text = str(row.get('Review Text', '')).strip()
                restaurant = str(row.get('Restaurant Name', '')).strip()
                
                if review_text and review_text.lower() != 'nan' and len(review_text) > 20:
                    reviews.append({
                        'text': review_text,
                        'restaurant': restaurant,
                        'rating': row.get('Reviewer Rating'),
                        'source': 'mumbaires.csv'
                    })
        except Exception as e:
            print(f"Error loading mumbaires.csv: {e}")
        
        return reviews

    def load_resreviews_csv(self, filepath, restaurant_name=None):
        reviews = []
        try:
            df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            
            if restaurant_name:
                df = df[df['Restaurant'].astype(str).str.contains(
                    restaurant_name, case=False, na=False, regex=False
                )]
            
            for _, row in df.iterrows():
                review_text = str(row.get('Review', '')).strip()
                restaurant = str(row.get('Restaurant', '')).strip()
                
                if review_text and review_text.lower() != 'nan' and len(review_text) > 20:
                    reviews.append({
                        'text': review_text,
                        'restaurant': restaurant,
                        'rating': row.get('Rating'),
                        'reviewer': row.get('Reviewer', 'anonymous'),
                        'source': 'Resreviews.csv'
                    })
        except Exception as e:
            print(f"Error loading Resreviews.csv: {e}")
        
        return reviews

    def load_reviews_csv(self, filepath, restaurant_name=None):
        reviews = []
        try:
            df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            
            if restaurant_name:
                df = df[df['business_name'].astype(str).str.contains(
                    restaurant_name, case=False, na=False, regex=False
                )]
            
            for _, row in df.iterrows():
                review_text = str(row.get('text', '')).strip()
                restaurant = str(row.get('business_name', '')).strip()
                
                if review_text and review_text.lower() != 'nan' and len(review_text) > 20:
                    reviews.append({
                        'text': review_text,
                        'restaurant': restaurant,
                        'rating': row.get('rating'),
                        'reviewer': row.get('author_name', 'anonymous'),
                        'source': 'reviews.csv'
                    })
        except Exception as e:
            print(f"Error loading reviews.csv: {e}")
        
        return reviews

    def load_zomato_csv(self, filepath, restaurant_name=None):
        reviews = []
        try:
            df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            
            if restaurant_name:
                df = df[df['name'].astype(str).str.contains(
                    restaurant_name, case=False, na=False, regex=False
                )]
            
            for _, row in df.iterrows():
                restaurant = str(row.get('name', '')).strip()
                reviews_list = row.get('reviews_list', '')
                
                extracted_reviews = self.extract_reviews_from_zomato_list(reviews_list)
                
                for review_text in extracted_reviews:
                    reviews.append({
                        'text': review_text,
                        'restaurant': restaurant,
                        'rating': row.get('rate'),
                        'location': row.get('location'),
                        'cuisines': row.get('cuisines'),
                        'source': 'zomato.csv'
                    })
        except Exception as e:
            print(f"Error loading zomato.csv: {e}")
        
        return reviews

    def load_zomato2_csv(self, filepath, restaurant_name=None):
        reviews = []
        try:
            df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            
            if restaurant_name:
                df = df[df['Restaurant_Name'].astype(str).str.contains(
                    restaurant_name, case=False, na=False, regex=False
                )]
            
            restaurant_groups = df.groupby('Restaurant_Name')
            
            for restaurant, group in restaurant_groups:
                for _, row in group.iterrows():
                    item = str(row.get('Item_Name', '')).strip()
                    if item and item.lower() != 'nan':
                        review_text = f"Tried {item}"
                        if row.get('Best_Seller', False):
                            review_text += " - This is a best seller"
                        if row.get('Votes', 0) > 0:
                            review_text += f" with {row.get('Votes')} votes"
                        
                        reviews.append({
                            'text': review_text,
                            'restaurant': restaurant,
                            'rating': row.get('Average_Rating'),
                            'cuisine': row.get('Cuisine'),
                            'city': row.get('City'),
                            'source': 'zomato2.csv'
                        })
        except Exception as e:
            print(f"Error loading zomato2.csv: {e}")
        
        return reviews

    def load_csv_data(self, restaurant_name=None):
        all_reviews = []
        
        if not os.path.exists(self.data_folder):
            print(f"Dataset folder '{self.data_folder}' not found.")
            return []

        dataset_loaders = {
            "mumbaires.csv": self.load_mumbaires_csv,
            "Resreviews.csv": self.load_resreviews_csv,
            "reviews.csv": self.load_reviews_csv,
            "zomato.csv": self.load_zomato_csv,
            "zomato2.csv": self.load_zomato2_csv,
        }
        
        for filename, loader_func in dataset_loaders.items():
            filepath = os.path.join(self.data_folder, filename)
            if os.path.exists(filepath):
                try:
                    reviews = loader_func(filepath, restaurant_name)
                    all_reviews.extend(reviews)
                    if restaurant_name and reviews:
                        print(f"✓ Loaded {len(reviews)} reviews from {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        self.loaded = True
        self.current_restaurant = restaurant_name
        return all_reviews

    def index_documents(self, texts, metadata=None):
        if self.model is None:
            print("❌ Embedding model not loaded. Cannot create vectors.")
            return
        
        if not FAISS_AVAILABLE:
            print("❌ FAISS not available. Install with: pip install faiss-cpu")
            return
        
        valid_indices = [i for i, t in enumerate(texts) if t and len(str(t)) > 20]
        self.doc_texts = [texts[i] for i in valid_indices]
        
        if metadata:
            self.doc_metadata = [metadata[i] for i in valid_indices]
        else:
            self.doc_metadata = [{}] * len(self.doc_texts)
        
        if not self.doc_texts:
            print("No valid documents to index.")
            return
        
        print(f"🔄 Creating embeddings for {len(self.doc_texts)} documents...")
        
        embeddings = self.model.encode(
            self.doc_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
        
        self.faiss_index.add(embeddings.astype('float32'))
        
        print(f"✅ Indexed {self.faiss_index.ntotal} documents in FAISS")
        print(f"✅ Vector dimension: {self.embedding_dimension}")
        
        if self.current_restaurant:
            self._save_vector_db(self.current_restaurant)

    def semantic_search(self, query, top_k=5):
        if self.faiss_index is None or self.model is None:
            return [], []
        
        try:
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype('float32')
            
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            results = []
            score_list = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.doc_texts):
                    results.append({
                        'text': self.doc_texts[idx],
                        'metadata': self.doc_metadata[idx],
                        'score': float(score)
                    })
                    score_list.append(float(score))
            
            return results, score_list
            
        except Exception as e:
            print(f"Error during semantic search: {e}")
            return [], []

    def answer_query(self, query, restaurant_name=None, top_k=5):
        if not self.loaded or (restaurant_name and restaurant_name != self.current_restaurant):
            print(f"📥 Loading reviews for '{restaurant_name}'...")
            
            if restaurant_name and self._load_vector_db(restaurant_name):
                self.loaded = True
                self.current_restaurant = restaurant_name
            else:
                reviews = self.load_csv_data(restaurant_name)
                
                if reviews:
                    texts = [r['text'] for r in reviews]
                    self.index_documents(texts, metadata=reviews)
                else:
                    return "❌ No reviews found for this restaurant in the datasets.", []

        if self.faiss_index is None or self.model is None:
            return "❌ No indexed reviews available. Please analyze the restaurant first.", []

        retrieved_docs, scores = self.semantic_search(query, top_k=top_k)
        
        if not retrieved_docs:
            web_answer = self._search_google_fallback(query, restaurant_name)
            return web_answer, []
        
        answer = self._generate_answer(query, retrieved_docs, scores, restaurant_name)
        source_texts = [doc['text'] for doc in retrieved_docs]
        
        return answer, source_texts

    def _generate_answer(self, query, retrieved_docs, scores, restaurant_name):
        query_lower = query.lower()
        
        intents = {
            'quality': ['quality', 'taste', 'food', 'delicious', 'flavor'],
            'service': ['service', 'staff', 'waiter', 'wait', 'server'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value'],
            'hygiene': ['clean', 'hygiene', 'dirty', 'sanitize'],
            'ambience': ['ambience', 'atmosphere', 'decor', 'vibe'],
            'recommend': ['recommend', 'suggest', 'best', 'should', 'worth']
        }
        
        detected_intent = None
        for intent, keywords in intents.items():
            if any(kw in query_lower for kw in keywords):
                detected_intent = intent
                break
        
        restaurant_phrase = f"**{restaurant_name}**" if restaurant_name else "this restaurant"
        answer = f"💬 Based on {len(retrieved_docs)} relevant reviews about {restaurant_phrase}:\n\n"
        
        answer += "**📋 Most Relevant Reviews (by FAISS similarity):**\n"
        for i, (doc, score) in enumerate(zip(retrieved_docs, scores), 1):
            text = doc['text']
            meta = doc['metadata']
            
            snippet = textwrap.shorten(text, width=180, placeholder="...")
            rating = meta.get('rating', 'N/A')
            source = meta.get('source', 'unknown')
            relevance = int(score * 100)
            
            answer += f"\n**{i}.** {snippet}"
            answer += f"\n   📊 Similarity: {relevance}% | ⭐ Rating: {rating} | 📁 Source: {source}\n"
        
        answer += "\n**🎯 AI Summary:**\n"
        summary = self._synthesize_intelligent_answer(
            query, retrieved_docs, detected_intent
        )
        answer += summary
        
        return answer

    def _synthesize_intelligent_answer(self, query, retrieved_docs, intent):
        all_text = " ".join([doc['text'].lower() for doc in retrieved_docs])
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'delicious', 'perfect', 
                         'wonderful', 'fantastic', 'love', 'best', 'awesome', 'outstanding']
        negative_words = ['bad', 'poor', 'terrible', 'horrible', 'awful', 'worst', 
                         'disappointing', 'waste', 'avoid', 'never', 'disgusting', 'pathetic']
        
        pos_count = sum(all_text.count(word) for word in positive_words)
        neg_count = sum(all_text.count(word) for word in negative_words)
        
        key_terms = self._extract_key_terms([doc['text'] for doc in retrieved_docs])
        
        ratings = [doc['metadata'].get('rating') for doc in retrieved_docs 
                  if doc['metadata'].get('rating')]
        avg_rating = None
        if ratings:
            valid_ratings = [float(r) for r in ratings if str(r).replace('.','').isdigit()]
            if valid_ratings:
                avg_rating = sum(valid_ratings) / len(valid_ratings)
        
        summary = ""
        
        if intent == 'quality':
            if pos_count > neg_count * 1.5:
                summary = "✅ **Food quality is highly praised** by customers. "
            elif pos_count > neg_count:
                summary = "👍 **Generally good food quality** with some positive mentions. "
            else:
                summary = "⚠️ **Mixed reviews about food quality** - check specifics. "
            
            if avg_rating:
                summary += f"Average rating: **{avg_rating:.1f}/5**. "
            summary += f"\n   Key mentions: {', '.join(key_terms[:5])}"
        
        elif intent == 'service':
            if 'slow' in all_text or 'long wait' in all_text:
                summary = "⏰ **Service speed is a concern** mentioned by multiple customers. "
            elif 'rude' in all_text or 'unfriendly' in all_text:
                summary = "😐 **Staff attitude needs improvement** according to reviews. "
            elif pos_count > neg_count:
                summary = "👏 **Service is appreciated** by most customers. "
            else:
                summary = "⚠️ **Service quality varies** - mixed experiences reported. "
        
        elif intent == 'price':
            if 'expensive' in all_text or 'overpriced' in all_text:
                summary = "💸 **Prices are on the higher side** as per customer feedback. "
            elif 'value' in all_text and pos_count > neg_count:
                summary = "💰 **Good value for money** mentioned by customers. "
            else:
                summary = "💵 **Pricing is considered reasonable** by most reviewers. "
        
        elif intent == 'hygiene':
            if neg_count > pos_count:
                summary = "🚨 **Hygiene concerns raised** - cleanliness needs attention. "
            else:
                summary = "✨ **Cleanliness standards are maintained** well. "
        
        elif intent == 'ambience':
            if pos_count > neg_count:
                summary = "🏪 **Good ambience and atmosphere** appreciated by visitors. "
            else:
                summary = "🪑 **Ambience feedback is mixed** - personal preference varies. "
        
        elif intent == 'recommend':
            sentiment_ratio = pos_count / (neg_count + 1)
            
            if sentiment_ratio > 2.0 and (avg_rating is None or avg_rating >= 4.0):
                summary = "🌟 **HIGHLY RECOMMENDED!** Strong positive feedback across reviews. "
            elif sentiment_ratio > 1.2:
                summary = "✅ **Generally Recommended** with mostly positive experiences. "
            elif sentiment_ratio > 0.8:
                summary = "⚖️ **Mixed Reviews** - read details before deciding. "
            else:
                summary = "⚠️ **Caution Advised** - significant negative feedback present. "
            
            if avg_rating:
                summary += f"\n   Overall Rating: **{avg_rating:.1f}/5**"
        
        else:
            if pos_count > neg_count * 1.5:
                summary = "✅ **Overall positive sentiment** in reviews. "
            elif pos_count > neg_count:
                summary = "👍 **Mostly positive feedback** with some concerns. "
            else:
                summary = "⚠️ **Mixed or negative feedback** - proceed with caution. "
            
            summary += f"\n   Frequently mentioned: {', '.join(key_terms[:6])}"
        
        return summary

    def _extract_key_terms(self, texts):
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'been', 'be',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 
                    'they', 'very', 'really', 'also', 'just', 'like', 'good', 'bad'}
        
        words = []
        for text in texts:
            tokens = re.findall(r"\b[a-z]{4,}\b", text.lower())
            words.extend([t for t in tokens if t not in stopwords])
        
        counter = Counter(words)
        return [word for word, _ in counter.most_common(12)]

    def _search_google_fallback(self, query, restaurant_name=None):
        try:
            search_query = f"{restaurant_name} {query}" if restaurant_name else query
            url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            
            snippets = []
            for div in soup.select("div.BNeawe.s3v9rd.AP7Wnd")[:5]:
                text = div.get_text().strip()
                if text and len(text) > 30:
                    snippets.append(text)
            
            if not snippets:
                return (f"❌ I couldn't find relevant information in the reviews or online "
                       f"about '{query}' for {restaurant_name or 'this restaurant'}.")
            
            result = "🌐 **No matches in local reviews. Here's what I found online:**\n\n"
            for i, snippet in enumerate(snippets, 1):
                shortened = textwrap.shorten(snippet, width=200, placeholder="...")
                result += f"{i}. {shortened}\n\n"
            
            return result
            
        except Exception as e:
            return (f"❌ Couldn't find information in local reviews. "
                   f"Error searching online: {str(e)}")