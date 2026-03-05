"""
User Management Module for RestoAI
Handles user authentication, profile management, and preferences
"""

from datetime import datetime, timedelta
import hashlib
import secrets
import os
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import pandas as pd
from flask import render_template, request, session, flash, redirect, url_for


class UserRole(Enum):
    """User role enumeration"""
    CUSTOMER = "customer"
    ADMIN = "admin"
    RESTAURANT_OWNER = "restaurant_owner"


class DietaryRestriction(Enum):
    """Dietary restrictions enumeration"""
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    GLUTEN_FREE = "gluten_free"
    LACTOSE_FREE = "lactose_free"
    HALAL = "halal"
    KOSHER = "kosher"
    NUT_ALLERGY = "nut_allergy"
    SHELLFISH_ALLERGY = "shellfish_allergy"
    NONE = "none"


class CuisinePreference(Enum):
    """Cuisine type preferences"""
    ITALIAN = "italian"
    ASIAN = "asian"
    INDIAN = "indian"
    MEXICAN = "mexican"
    THAI = "thai"
    CHINESE = "chinese"
    JAPANESE = "japanese"
    AMERICAN = "american"
    MEDITERRANEAN = "mediterranean"
    FAST_FOOD = "fast_food"
    DESSERTS = "desserts"
    SEAFOOD = "seafood"
    VEGETARIAN = "vegetarian"
    FUSION = "fusion"


@dataclass
class UserPreferences:
    """User food and restaurant preferences"""
    budget_min: float = 500.0  # Minimum budget in local currency
    budget_max: float = 5000.0  # Maximum budget
    preferred_cuisines: List[str] = field(default_factory=lambda: ["indian", "chinese"])
    dietary_restrictions: List[str] = field(default_factory=lambda: [])
    preferred_locations: List[str] = field(default_factory=list)
    ambiance_preferences: List[str] = field(default_factory=lambda: ["casual", "fine_dining"])
    dietary_allergies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert preferences to dictionary"""
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict) -> 'UserPreferences':
        """Create preferences from dictionary"""
        return UserPreferences(**{k: v for k, v in data.items() if k in UserPreferences.__dataclass_fields__})


@dataclass
class UserBookingHistory:
    """Track user booking history"""
    booking_id: str
    restaurant_id: str
    restaurant_name: str
    booking_date: datetime
    party_size: int
    booking_time: str
    status: str  # "confirmed", "completed", "cancelled"
    rating: Optional[float] = None
    review: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['booking_date'] = self.booking_date.isoformat()
        return data


@dataclass
class User:
    """Main User Model for RestoAI"""
    user_id: str
    email: str
    username: str
    password_hash: str
    phone_number: str
    first_name: str
    last_name: str
    role: str = UserRole.CUSTOMER.value
    preferences: UserPreferences = field(default_factory=UserPreferences)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    is_verified: bool = False
    profile_image_url: Optional[str] = None
    booking_history: List[UserBookingHistory] = field(default_factory=list)
    saved_restaurants: List[str] = field(default_factory=list)  # Restaurant IDs
    favorite_restaurants: List[str] = field(default_factory=list)  # Restaurant IDs
    dietary_info: Dict[str, Any] = field(default_factory=dict)
    notification_preferences: Dict[str, bool] = field(default_factory=lambda: {
        "email_notifications": True,
        "sms_notifications": False,
        "booking_reminders": True,
        "new_restaurants": False,
        "personalized_recommendations": True
    })
    
    def to_dict(self, include_password: bool = False) -> Dict[str, Any]:
        """Convert user to dictionary"""
        user_dict = {
            'user_id': self.user_id,
            'email': self.email,
            'username': self.username,
            'phone_number': self.phone_number,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'role': self.role,
            'preferences': self.preferences.to_dict(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'profile_image_url': self.profile_image_url,
            'booking_history': [bh.to_dict() for bh in self.booking_history],
            'saved_restaurants': self.saved_restaurants,
            'favorite_restaurants': self.favorite_restaurants,
            'dietary_info': self.dietary_info,
            'notification_preferences': self.notification_preferences
        }
        
        if include_password:
            user_dict['password_hash'] = self.password_hash
        
        return user_dict
    
    @staticmethod
    def from_dict(data: Dict) -> 'User':
        """Create user from dictionary"""
        # Handle datetime fields
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        updated_at = data.get('updated_at')
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        last_login = data.get('last_login')
        if isinstance(last_login, str):
            last_login = datetime.fromisoformat(last_login)
        
        # Handle preferences
        pref_data = data.get('preferences', {})
        preferences = UserPreferences.from_dict(pref_data) if pref_data else UserPreferences()
        
        # Handle booking history
        booking_history = []
        for bh in data.get('booking_history', []):
            if isinstance(bh.get('booking_date'), str):
                bh['booking_date'] = datetime.fromisoformat(bh['booking_date'])
            booking_history.append(UserBookingHistory(**bh))
        
        return User(
            user_id=data.get('user_id'),
            email=data.get('email'),
            username=data.get('username'),
            password_hash=data.get('password_hash'),
            phone_number=data.get('phone_number'),
            first_name=data.get('first_name'),
            last_name=data.get('last_name'),
            role=data.get('role', UserRole.CUSTOMER.value),
            preferences=preferences,
            created_at=created_at or datetime.now(),
            updated_at=updated_at or datetime.now(),
            last_login=last_login,
            is_active=data.get('is_active', True),
            is_verified=data.get('is_verified', False),
            profile_image_url=data.get('profile_image_url'),
            booking_history=booking_history,
            saved_restaurants=data.get('saved_restaurants', []),
            favorite_restaurants=data.get('favorite_restaurants', []),
            dietary_info=data.get('dietary_info', {}),
            notification_preferences=data.get('notification_preferences', {})
        )


class UserManager:
    """Manager class for user operations"""
    
    def __init__(self):
        """Initialize user manager"""
        self.users: Dict[str, User] = {}  # In-memory storage (replace with DB)
        self.email_index: Dict[str, str] = {}  # Email to user_id mapping
        self.username_index: Dict[str, str] = {}  # Username to user_id mapping
        self.sessions: Dict[str, Dict[str, Any]] = {}  # Session management
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256"""
        salt = secrets.token_hex(32)
        pwd_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}${pwd_hash.hex()}"
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, stored_hash = password_hash.split('$')
            pwd_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return pwd_hash.hex() == stored_hash
        except Exception:
            return False
    
    @staticmethod
    def generate_user_id() -> str:
        """Generate unique user ID"""
        return f"USR_{secrets.token_hex(8).upper()}"
    
    def create_user(
        self,
        email: str,
        username: str,
        password: str,
        first_name: str,
        last_name: str,
        phone_number: str,
        preferences: Optional[UserPreferences] = None
    ) -> tuple[bool, str, Optional[User]]:
        """
        Create a new user
        
        Args:
            email: User email
            username: Username
            password: User password (will be hashed)
            first_name: First name
            last_name: Last name
            phone_number: Phone number
            preferences: User preferences (optional)
        
        Returns:
            Tuple of (success, message, user_object)
        """
        # Validate inputs
        if not email or '@' not in email:
            return False, "Invalid email format", None
        
        if len(username) < 3:
            return False, "Username must be at least 3 characters", None
        
        if len(password) < 8:
            return False, "Password must be at least 8 characters", None
        
        if email.lower() in self.email_index:
            return False, "Email already registered", None
        
        if username.lower() in self.username_index:
            return False, "Username already taken", None
        
        # Create user
        user_id = self.generate_user_id()
        password_hash = self.hash_password(password)
        
        user = User(
            user_id=user_id,
            email=email.lower(),
            username=username.lower(),
            password_hash=password_hash,
            phone_number=phone_number,
            first_name=first_name,
            last_name=last_name,
            preferences=preferences or UserPreferences()
        )
        
        # Store user
        self.users[user_id] = user
        self.email_index[email.lower()] = user_id
        self.username_index[username.lower()] = user_id
        
        return True, "User created successfully", user
    
    def authenticate_user(self, email: str, password: str) -> tuple[bool, str, Optional[User]]:
        """
        Authenticate user with email and password
        
        Args:
            email: User email
            password: User password
        
        Returns:
            Tuple of (success, message, user_object)
        """
        user_id = self.email_index.get(email.lower())
        
        if not user_id:
            return False, "User not found", None
        
        user = self.users[user_id]
        
        if not user.is_active:
            return False, "User account is inactive", None
        
        if not self.verify_password(password, user.password_hash):
            return False, "Invalid password", None
        
        # Update last login
        user.last_login = datetime.now()
        
        return True, "Authentication successful", user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by user_id"""
        return self.users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        user_id = self.email_index.get(email.lower())
        return self.users.get(user_id) if user_id else None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        user_id = self.username_index.get(username.lower())
        return self.users.get(user_id) if user_id else None
    
    def update_user_preferences(self, user_id: str, preferences: UserPreferences) -> bool:
        """Update user preferences"""
        user = self.get_user(user_id)
        if user:
            user.preferences = preferences
            user.updated_at = datetime.now()
            return True
        return False
    
    def add_booking_history(
        self,
        user_id: str,
        booking_id: str,
        restaurant_id: str,
        restaurant_name: str,
        booking_date: datetime,
        party_size: int,
        booking_time: str,
        status: str
    ) -> bool:
        """Add booking to user history"""
        user = self.get_user(user_id)
        if user:
            booking = UserBookingHistory(
                booking_id=booking_id,
                restaurant_id=restaurant_id,
                restaurant_name=restaurant_name,
                booking_date=booking_date,
                party_size=party_size,
                booking_time=booking_time,
                status=status
            )
            user.booking_history.append(booking)
            user.updated_at = datetime.now()
            return True
        return False
    
    def add_to_favorites(self, user_id: str, restaurant_id: str) -> bool:
        """Add restaurant to user's favorites"""
        user = self.get_user(user_id)
        if user and restaurant_id not in user.favorite_restaurants:
            user.favorite_restaurants.append(restaurant_id)
            user.updated_at = datetime.now()
            return True
        return False
    
    def remove_from_favorites(self, user_id: str, restaurant_id: str) -> bool:
        """Remove restaurant from user's favorites"""
        user = self.get_user(user_id)
        if user and restaurant_id in user.favorite_restaurants:
            user.favorite_restaurants.remove(restaurant_id)
            user.updated_at = datetime.now()
            return True
        return False
    
    def save_restaurant(self, user_id: str, restaurant_id: str) -> bool:
        """Save restaurant for later"""
        user = self.get_user(user_id)
        if user and restaurant_id not in user.saved_restaurants:
            user.saved_restaurants.append(restaurant_id)
            user.updated_at = datetime.now()
            return True
        return False
    
    def unsave_restaurant(self, user_id: str, restaurant_id: str) -> bool:
        """Unsave restaurant"""
        user = self.get_user(user_id)
        if user and restaurant_id in user.saved_restaurants:
            user.saved_restaurants.remove(restaurant_id)
            user.updated_at = datetime.now()
            return True
        return False
    
    def create_session(self, user_id: str, expiry_hours: int = 24) -> str:
        """Create user session"""
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(hours=expiry_hours)
        }
        return session_id
    
    def verify_session(self, session_id: str) -> Optional[str]:
        """Verify session and return user_id if valid"""
        session = self.sessions.get(session_id)
        if session and session['expires_at'] > datetime.now():
            return session['user_id']
        return None
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def save_user_to_file(self, user_id: str, filepath: str) -> bool:
        """Save user data to JSON file"""
        user = self.get_user(user_id)
        if user:
            with open(filepath, 'w') as f:
                json.dump(user.to_dict(), f, indent=2)
            return True
        return False
    
    def load_user_from_file(self, filepath: str) -> Optional[User]:
        """Load user data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            user = User.from_dict(data)
            self.users[user.user_id] = user
            self.email_index[user.email] = user.user_id
            self.username_index[user.username] = user.user_id
            return user
        except Exception as e:
            print(f"Error loading user: {e}")
            return None


logger = logging.getLogger(__name__)


def _safe_float(value):
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None

    if "/" in text:
        text = text.split("/")[0].strip()

    filtered = "".join(ch for ch in text if ch.isdigit() or ch in {".", "-"})
    if not filtered:
        return None

    try:
        return float(filtered)
    except ValueError:
        return None


def _placeholder_image(restaurant_name):
    lock = (sum(ord(ch) for ch in (restaurant_name or "restaurant")) % 10000) + 1
    return f"https://loremflickr.com/800/600/restaurant,food,dining?lock={lock}"


def _load_user_restaurant_catalog(dataset_folder, Review):
    dataset_files = ["mumbaires.csv", "Resreviews.csv", "reviews.csv", "zomato.csv", "zomato2.csv"]

    catalog = {}

    def ensure_entry(name):
        key = (name or "").strip()
        if not key:
            return None

        if key not in catalog:
            catalog[key] = {
                "name": key,
                "ratings": [],
                "cuisines": set(),
                "prices": [],
                "review_count": 0,
                "sample_review": "",
            }
        return catalog[key]

    def add_dataset_row(row):
        name = ""
        for k in ["Restaurant Name", "Restaurant", "business_name", "name", "Restaurant_Name"]:
            value = row.get(k)
            if pd.notna(value) and str(value).strip():
                name = str(value).strip()
                break

        entry = ensure_entry(name)
        if not entry:
            return

        for k in ["Rating", "Reviewer Rating", "rating", "rate", "Avg_Rating_Restaurant", "Average_Rating"]:
            score = _safe_float(row.get(k))
            if score is not None:
                entry["ratings"].append(score)
                break

        for k in ["cuisines", "Cuisine", "Item_Offered"]:
            value = row.get(k)
            if pd.notna(value):
                cuisine_text = str(value).strip()
                if cuisine_text and cuisine_text.lower() != "nan":
                    for cuisine in cuisine_text.replace("/", ",").split(","):
                        cleaned = cuisine.strip()
                        if cleaned:
                            entry["cuisines"].add(cleaned)

        for k in ["approx_cost(for two people)", "Price Level", "cost", "Average_Cost", "Cost_for_Two"]:
            value = row.get(k)
            if pd.notna(value):
                price_text = str(value).strip()
                if price_text and price_text.lower() != "nan":
                    if k == "Price Level":
                        entry["prices"].append(price_text)
                    else:
                        numeric = _safe_float(price_text)
                        entry["prices"].append(f"₹{int(numeric)} for two" if numeric is not None else price_text)
                    break

        review_text = ""
        for k in ["Review Text", "Review", "text", "reviews_list", "Item_Name"]:
            value = row.get(k)
            if pd.notna(value):
                candidate = str(value).strip()
                if candidate and candidate.lower() != "nan":
                    review_text = candidate
                    break

        if review_text:
            entry["review_count"] += 1
            if not entry["sample_review"]:
                entry["sample_review"] = review_text[:180]

    for filename in dataset_files:
        file_path = os.path.join(dataset_folder, filename)
        if not os.path.exists(file_path):
            continue

        try:
            df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            for _, row in df.iterrows():
                add_dataset_row(row)
        except Exception as exc:
            logger.warning("Failed reading dataset %s: %s", filename, exc)

    db_reviews = Review.query.order_by(Review.created_at.desc()).all()
    for review in db_reviews:
        entry = ensure_entry(review.restaurant)
        if not entry:
            continue

        entry["review_count"] += 1
        score = _safe_float(review.rating)
        if score is not None:
            entry["ratings"].append(score)
        if review.text and not entry["sample_review"]:
            entry["sample_review"] = review.text[:180]

    restaurants = []
    total_reviews = 0
    for data in catalog.values():
        rating_display = "N/A"
        if data["ratings"]:
            rating_display = f"{(sum(data['ratings']) / len(data['ratings'])):.1f}"

        cuisines_display = ", ".join(sorted(data["cuisines"])) if data["cuisines"] else "Not specified"
        price_display = data["prices"][0] if data["prices"] else "Not available"
        review_count = data["review_count"]
        total_reviews += review_count

        restaurants.append(
            {
                "name": data["name"],
                "rating": rating_display,
                "cuisines": cuisines_display,
                "price": price_display,
                "review_count": review_count,
                "sample_review": data["sample_review"] or "No review text available yet.",
                "photo": _placeholder_image(data["name"]),
            }
        )

    restaurants.sort(key=lambda item: (item["review_count"], item["name"]), reverse=True)
    return restaurants[:60], total_reviews


def _generate_ai_booking_summary(restaurant_name, service_mode, party_size, preferred_time, delivery_address):
    booking_id = f"BK_{secrets.token_hex(4).upper()}"

    if service_mode == "home_delivery":
        eta = min(65, 30 + party_size * 4)
        return {
            "booking_id": booking_id,
            "restaurant": restaurant_name,
            "mode": "Home Delivery",
            "summary": f"AI confirmed delivery from {restaurant_name}. Estimated arrival in {eta} minutes.",
            "details": f"Delivery to: {delivery_address or 'saved address'}",
        }

    slot = preferred_time or "8:00 PM"
    return {
        "booking_id": booking_id,
        "restaurant": restaurant_name,
        "mode": "Restaurant Table",
        "summary": f"AI reserved a table for {party_size} at {restaurant_name} around {slot}.",
        "details": "Please arrive 10 minutes early for smooth check-in.",
    }


def register_user_routes(app, Review, login_required):
    """Register user-specific routes for dashboard and automated booking."""

    @app.route("/user/dashboard")
    @login_required
    def user_dashboard():
        restaurants, total_reviews = _load_user_restaurant_catalog(app.config["DATASET_FOLDER"], Review)
        last_booking = session.pop("last_user_booking", None)

        return render_template(
            "user_dashboard.html",
            restaurants=restaurants,
            total_restaurants=len(restaurants),
            total_reviews=total_reviews,
            last_booking=last_booking,
        )

    @app.route("/user/auto-booking", methods=["POST"])
    @login_required
    def user_auto_booking():
        restaurant_name = request.form.get("restaurant_name", "").strip()
        service_mode = request.form.get("service_mode", "restaurant").strip().lower()
        preferred_time = request.form.get("preferred_time", "").strip()
        delivery_address = request.form.get("delivery_address", "").strip()

        try:
            party_size = int(request.form.get("party_size", "2"))
        except ValueError:
            party_size = 2

        if not restaurant_name:
            flash("Restaurant is required for auto booking.", "warning")
            return redirect(url_for("user_dashboard"))

        if service_mode not in {"restaurant", "home_delivery"}:
            flash("Invalid service mode selected.", "warning")
            return redirect(url_for("user_dashboard"))

        if party_size < 1 or party_size > 20:
            flash("Party size must be between 1 and 20.", "warning")
            return redirect(url_for("user_dashboard"))

        booking_result = _generate_ai_booking_summary(
            restaurant_name=restaurant_name,
            service_mode=service_mode,
            party_size=party_size,
            preferred_time=preferred_time,
            delivery_address=delivery_address,
        )

        session["last_user_booking"] = booking_result
        flash(f"AI booking completed: {booking_result['booking_id']}", "success")
        return redirect(url_for("user_dashboard"))

    @app.route("/user/restaurant/<restaurant_name>")
    @login_required
    def restaurant_details(restaurant_name):
        """Display detailed restaurant information with reviews, menu, and booking options."""
        # Load all restaurants
        restaurants, _ = _load_user_restaurant_catalog(app.config["DATASET_FOLDER"], Review)
        
        # Find the selected restaurant
        restaurant = None
        for r in restaurants:
            if r["name"].lower() == restaurant_name.lower():
                restaurant = r
                break
        
        if not restaurant:
            flash("Restaurant not found.", "warning")
            return redirect(url_for("user_dashboard"))
        
        # Load reviews for this restaurant
        from sqlalchemy import desc
        reviews = Review.query.filter_by(restaurant=restaurant_name).order_by(desc(Review.created_at)).limit(20).all()
        
        # Generate sample menu items based on cuisine
        menu_items = _generate_sample_menu(restaurant["cuisines"])
        
        return render_template(
            "restaurant_details.html",
            restaurant=restaurant,
            reviews=reviews,
            menu_items=menu_items,
        )


def _generate_sample_menu(cuisines_str):
    """Generate sample menu items based on cuisine type."""
    menu_samples = {
        "indian": [
            {"name": "Butter Chicken", "description": "Tender chicken in creamy tomato sauce", "price": 350},
            {"name": "Biryani", "description": "Fragrant rice with spiced meat", "price": 250},
            {"name": "Paneer Tikka", "description": "Grilled cottage cheese with spices", "price": 280},
            {"name": "Dal Makhani", "description": "Creamy lentil curry", "price": 220},
            {"name": "Naan", "description": "Traditional Indian bread", "price": 50},
        ],
        "chinese": [
            {"name": "Kung Pao Chicken", "description": "Stir-fried chicken with peanuts", "price": 320},
            {"name": "Hakka Noodles", "description": "Indo-Chinese noodles", "price": 180},
            {"name": "Gobi Manchurian", "description": "Fried cauliflower in tangy sauce", "price": 240},
            {"name": "Fried Rice", "description": "Egg or vegetable fried rice", "price": 160},
        ],
        "italian": [
            {"name": "Pasta Alfredo", "description": "Creamy pasta with parmesan", "price": 380},
            {"name": "Margherita Pizza", "description": "Fresh mozzarella and basil", "price": 320},
            {"name": "Risotto", "description": "Creamy Italian rice", "price": 340},
            {"name": "Carbonara", "description": "Pasta with bacon and cream", "price": 360},
        ],
        "mexican": [
            {"name": "Tacos", "description": "Soft tortillas with filling", "price": 280},
            {"name": "Burrito", "description": "Wrapped tortilla with rice and beans", "price": 300},
            {"name": "Quesadilla", "description": "Grilled tortilla with cheese", "price": 250},
        ],
        "thai": [
            {"name": "Pad Thai", "description": "Stir-fried rice noodles", "price": 280},
            {"name": "Green Curry", "description": "Spicy green curry with vegetables", "price": 320},
            {"name": "Tom Yum Soup", "description": "Spicy and tangy soup", "price": 200},
        ],
    }
    
    # Extract cuisine types from string
    cuisines = [c.strip().lower() for c in cuisines_str.split(",") if c.strip()]
    
    # Get menu items for matching cuisines
    menu = []
    for cuisine in cuisines:
        if cuisine in menu_samples:
            menu.extend(menu_samples[cuisine])
    
    # If no exact match, return a generic menu
    if not menu:
        menu = [
            {"name": "Signature Dish", "description": "Chef's special recommendation", "price": 350},
            {"name": "Appetizer Platter", "description": "Assorted starters", "price": 280},
            {"name": "Main Course", "description": "House specialty", "price": 400},
            {"name": "Dessert", "description": "Sweet treat", "price": 150},
        ]
    
    return menu[:8]  # Return max 8 items


# Example usage and testing
if __name__ == "__main__":
    # Initialize user manager
    manager = UserManager()
    
    # Create a test user
    success, msg, user = manager.create_user(
        email="john.doe@example.com",
        username="johndoe",
        password="SecurePass123",
        first_name="John",
        last_name="Doe",
        phone_number="+919876543210",
        preferences=UserPreferences(
            budget_min=500,
            budget_max=3000,
            preferred_cuisines=["indian", "italian"],
            dietary_restrictions=[DietaryRestriction.VEGETARIAN.value]
        )
    )
    
    if success:
        print(f"✓ User created: {user.username} ({user.user_id})")
        print(f"  Email: {user.email}")
        print(f"  Preferences: {user.preferences.to_dict()}")
    else:
        print(f"✗ Error: {msg}")
    
    # Authenticate user
    success, msg, authenticated_user = manager.authenticate_user(
        email="john.doe@example.com",
        password="SecurePass123"
    )
    
    if success:
        print(f"\n✓ Authentication successful for {authenticated_user.username}")
        print(f"  Last login: {authenticated_user.last_login}")
    else:
        print(f"\n✗ Authentication failed: {msg}")
    
    # Create session
    session_id = manager.create_session(user.user_id)
    print(f"\n✓ Session created: {session_id}")
    
    # Verify session
    verified_user_id = manager.verify_session(session_id)
    print(f"✓ Session verified for user: {verified_user_id}")
