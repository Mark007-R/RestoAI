"""Helper utilities for data processing."""
import pandas as pd
import ast
import logging
from typing import List, Dict, Optional, Tuple
import hashlib

logger = logging.getLogger(__name__)


def extract_reviews_from_zomato_reviews_list(reviews_list_str) -> List[str]:
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


def safe_read_csv(filepath: str, encoding: str = "utf-8") -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(filepath, encoding=encoding, on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        return df
    except UnicodeDecodeError:
        for alt_encoding in ['latin1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, encoding=alt_encoding, on_bad_lines="skip")
                df.columns = df.columns.str.strip()
                return df
            except (FileNotFoundError, PermissionError, ValueError) as e:
                logger.debug(f"Failed to read CSV with {alt_encoding} encoding: {e}")
                continue
    except (FileNotFoundError, PermissionError, ValueError) as e:
        logger.error(f"Error reading CSV {filepath}: {e}")
    
    return None


def clean_value(value, default: str = "") -> str:
    if pd.isna(value) or value is None:
        return default
    
    value_str = str(value).strip()
    
    if value_str.lower() in ['nan', 'none', 'null', '']:
        return default
    
    return value_str


def is_valid_review_text(text: str, min_length: int = 10) -> bool:
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    
    if len(text) < min_length:
        return False
    
    if text.lower() in ['nan', 'none', 'null']:
        return False
    
    return True


def deduplicate_reviews(reviews: List[Dict]) -> List[Dict]:
    seen_hashes = set()
    unique_reviews = []
    
    for review in reviews:
        text = review.get('text', '')
        text_hash = hashlib.md5(text.lower().strip().encode()).hexdigest()
        
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_reviews.append(review)
    
    return unique_reviews


def normalize_rating(rating_value) -> Optional[float]:
    if pd.isna(rating_value) or rating_value is None:
        return None
    
    try:
        rating_str = str(rating_value).strip()
        
        if '/5' in rating_str:
            rating_str = rating_str.split('/')[0]
        
        rating_str = rating_str.replace('out of 5', '').strip()
        
        rating_float = float(rating_str)
        
        if rating_float <= 5:
            return round(rating_float, 2)
        elif rating_float <= 10:
            return round(rating_float / 2, 2)
        elif rating_float <= 100:
            return round(rating_float / 20, 2)
        
        return None
    except (ValueError, TypeError):
        return None


def generate_stable_hash(text: str, length: int = 8) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def filter_by_restaurant(df: pd.DataFrame, restaurant_name: str, name_column: str) -> pd.DataFrame:
    if name_column not in df.columns:
        return pd.DataFrame()
    
    return df[df[name_column].astype(str).str.contains(
        restaurant_name, case=False, na=False, regex=False
    )]


def extract_value_safely(row, column_name: str, default=None):
    try:
        value = row.get(column_name, default)
        if pd.isna(value):
            return default
        return value
    except (AttributeError, TypeError, KeyError) as e:
        logger.debug(f"Error extracting value from {column_name}: {e}")
        return default


def batch_process(items: List, batch_size: int = 100):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
