import time
import random
import re
import requests
from urllib.parse import quote_plus, urlparse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

HEADERS_LIST = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"},
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0"},
]

MIN_REVIEW_LENGTH = 30
MAX_REVIEW_LENGTH = 2000
MIN_WORD_COUNT = 5

def get_headers():
    return random.choice(HEADERS_LIST)

def smart_delay(min_delay=0.5, max_delay=2.0):
    delay = min_delay + random.random() * (max_delay - min_delay)
    time.sleep(delay)

def clean_text(txt):
    if not txt:
        return ""
    
    txt = re.sub(r'\s+', ' ', str(txt)).strip()
    
    txt = re.sub(r'[^\x20-\x7E\n\t]', '', txt)
    
    txt = re.sub(r'([.!?])\1+', r'\1', txt)
    
    txt = re.sub(r'\s+([.,!?;:])', r'\1', txt)
    txt = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1 \2', txt)
    
    return txt.strip()

def is_valid_review(text, min_length=MIN_REVIEW_LENGTH, max_length=MAX_REVIEW_LENGTH):
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    
    if len(text) < min_length or len(text) > max_length:
        return False
    
    words = text.split()
    if len(words) < MIN_WORD_COUNT:
        return False
    
    if not re.search(r'[a-zA-Z]', text):
        return False
    
    invalid_patterns = [
        r'^(copyright|privacy|terms|conditions|policy|disclaimer)',
        r'^(login|sign up|register|subscribe|newsletter)',
        r'^(home|about|contact|menu|reservation|book|order)',
        r'^\d+$',
        r'^[^a-zA-Z]+$',
        r'^(photo|image|video|gallery)',
        r'^(facebook|twitter|instagram|social)',
        r'(cookie|gdpr|accept|decline)\s*(policy|notice)',
        r'^\s*[\d\W]+\s*$',
    ]
    
    text_lower = text.lower()
    for pattern in invalid_patterns:
        if re.search(pattern, text_lower):
            return False
    
    if not re.search(r'[.!?]', text):
        if len(words) < 10:
            return False
    
    letter_count = sum(c.isalpha() for c in text)
    if letter_count / len(text) < 0.5:
        return False
    
    return True

def calculate_review_quality_score(text):
    if not text:
        return 0
    
    score = 50
    
    length = len(text)
    if 100 <= length <= 500:
        score += 20
    elif 50 <= length < 100:
        score += 10
    elif length > 500:
        score += 10
    
    sentences = len(re.findall(r'[.!?]+', text))
    if sentences >= 2:
        score += 15
    
    words = text.lower().split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    score += int(unique_ratio * 15)
    
    food_keywords = ['food', 'dish', 'menu', 'taste', 'delicious', 'flavor', 
                     'service', 'staff', 'ambience', 'restaurant', 'meal', 
                     'ordered', 'tried', 'recommend', 'experience']
    keyword_count = sum(1 for word in food_keywords if word in text.lower())
    score += min(keyword_count * 2, 10)
    
    return min(score, 100)

def scrape_single_url(url, restaurant_name, timeout=10):
    results = []
    
    try:
        smart_delay(0.3, 0.8)
        response = requests.get(url, headers=get_headers(), timeout=timeout)
        
        if response.status_code != 200:
            return results
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        potential_reviews = []
        
        for p in soup.find_all("p"):
            text = clean_text(p.get_text())
            if is_valid_review(text, min_length=50):
                potential_reviews.append(text)
        
        review_divs = soup.find_all("div", class_=re.compile(r'review|comment|feedback|testimonial', re.I))
        for div in review_divs:
            text = clean_text(div.get_text())
            if is_valid_review(text, min_length=50):
                potential_reviews.append(text)
        
        for text in potential_reviews:
            if any(word in text.lower() for word in 
                  ['food', 'dish', 'menu', 'service', 'taste', 'delicious', 
                   'restaurant', 'meal', 'ordered', 'ambience', 'staff']):
                
                quality_score = calculate_review_quality_score(text)
                
                if quality_score >= 50:
                    results.append({
                        "source": urlparse(url).netloc,
                        "review": text,
                        "restaurant": restaurant_name,
                        "quality_score": quality_score,
                        "url": url
                    })
        
    except Exception as e:
        print(f"  ⚠️  Error scraping {urlparse(url).netloc}: {str(e)[:50]}")
    
    return results

def scrape_generic_reviews(restaurant_name, max_reviews=20):
    results = []
    seen = set()
    
    try:
        query = f"{restaurant_name} restaurant customer reviews Mumbai"
        url = "https://html.duckduckgo.com/html/"
        
        print(f"🌐 Web scraping: Searching for '{restaurant_name}'...")
        
        response = requests.post(url, data={"q": query}, headers=get_headers(), timeout=10)
        if response.status_code != 200:
            print(f"⚠️  Web search failed with status {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        links = [a.get("href") for a in soup.find_all("a", href=True)]
        
        review_links = [
            l for l in links 
            if any(x in l.lower() for x in ["review", "tripadvisor", "zomato", "yelp", 
                                           "dineout", "blog", "food", "restaurant"])
            and not any(x in l.lower() for x in ["facebook", "twitter", "instagram", "ad"])
        ][:12]
        
        print(f"📍 Found {len(review_links)} potential sources")

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_url = {
                executor.submit(scrape_single_url, link, restaurant_name): link 
                for link in review_links
            }
            
            for i, future in enumerate(as_completed(future_to_url), 1):
                try:
                    url_results = future.result()
                    
                    for review_dict in url_results:
                        text = review_dict['review']
                        normalized = re.sub(r'\W+', '', text.lower())
                        
                        if normalized not in seen:
                            seen.add(normalized)
                            results.append(review_dict)
                            
                            if len(results) >= max_reviews:
                                break
                    
                    print(f"  ✓ Processed {i}/{len(review_links)} sources ({len(results)} reviews found)")
                    
                except Exception as e:
                    print(f"  ⚠️  Error: {str(e)[:50]}")
                
                if len(results) >= max_reviews:
                    break
                    
    except Exception as e:
        print(f"❌ Generic scraping error: {e}")
        return []
    
    results.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
    
    print(f"✅ Scraped {len(results)} quality reviews from web")
    return results[:max_reviews]

def scrape_zomato_placeholder(restaurant_name, max_reviews=10):
    results = []
    seen = set()
    
    try:
        print(f"🔍 Attempting Zomato search for '{restaurant_name}'...")
        
        base = "https://www.zomato.com"
        search_url = base + "/mumbai/search?q=" + quote_plus(restaurant_name)
        
        response = requests.get(search_url, headers=get_headers(), timeout=10)
        
        if response.status_code != 200:
            print(f"⚠️  Zomato search returned status {response.status_code}")
            return results

        soup = BeautifulSoup(response.text, "html.parser")
        
        possible_blocks = soup.find_all(["p", "div", "span"], text=True)
        
        for tag in possible_blocks:
            if len(results) >= max_reviews:
                break
                
            text = clean_text(tag.get_text())
            
            if not is_valid_review(text, min_length=50):
                continue
            
            if re.search(r"\b(food|service|taste|ambience|price|delivery|dish|menu|recommend)\b", text, re.I):
                normalized = re.sub(r'\W+', '', text.lower())
                
                if normalized not in seen:
                    seen.add(normalized)
                    quality_score = calculate_review_quality_score(text)
                    
                    results.append({
                        "source": "zomato.com",
                        "review": text,
                        "restaurant": restaurant_name,
                        "quality_score": quality_score
                    })
        
        if results:
            print(f"✅ Found {len(results)} potential reviews on Zomato")
        else:
            print("⚠️  No reviews extracted from Zomato (likely dynamic content)")
        
    except Exception as e:
        print(f"⚠️  Zomato scraping error: {e}")
    
    return results

