"""Input validation and sanitization utilities."""
import re
import os
from typing import Optional, Tuple


def sanitize_string(text: str, max_length: int = 500) -> str:
    if not text:
        return ""
    
    text = str(text).strip()
    text = text[:max_length]
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text


def validate_restaurant_name(name: str) -> Tuple[bool, Optional[str]]:
    if not name or not name.strip():
        return False, "Restaurant name cannot be empty"
    
    name = name.strip()
    
    if len(name) < 2:
        return False, "Restaurant name must be at least 2 characters"
    
    if len(name) > 200:
        return False, "Restaurant name cannot exceed 200 characters"
    
    if not re.match(r'^[a-zA-Z0-9\s\'\-\.\,\&]+$', name):
        return False, "Restaurant name contains invalid characters"
    
    return True, None


def validate_filename(filename: str, allowed_extensions: set = None) -> Tuple[bool, Optional[str]]:
    if not filename:
        return False, "Filename cannot be empty"
    
    if allowed_extensions is None:
        allowed_extensions = {'csv', 'txt'}
    
    if '..' in filename or '/' in filename or '\\' in filename:
        return False, "Invalid filename: path traversal detected"
    
    if '.' not in filename:
        return False, "File must have an extension"
    
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in allowed_extensions:
        return False, f"File type '.{ext}' not allowed. Allowed types: {', '.join(allowed_extensions)}"
    
    if filename.startswith('.'):
        return False, "Hidden files not allowed"
    
    return True, None


def validate_file_size(file_path: str, max_size_mb: int = 16) -> Tuple[bool, Optional[str]]:
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    
    if size_mb > max_size_mb:
        return False, f"File size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
    
    return True, None


def sanitize_sql_input(text: str) -> str:
    if not text:
        return ""
    
    dangerous_patterns = [
        r';\s*DROP\s+TABLE',
        r';\s*DELETE\s+FROM',
        r';\s*UPDATE\s+',
        r';\s*INSERT\s+INTO',
        r'--',
        r'/\*',
        r'\*/',
        r'xp_',
        r'sp_',
    ]
    
    text = str(text)
    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()


def validate_rating(rating: any) -> Tuple[bool, Optional[float]]:
    if rating is None or rating == '':
        return True, None
    
    try:
        rating_float = float(rating)
        
        if rating_float < 0 or rating_float > 5:
            return False, None
        
        return True, round(rating_float, 2)
    except (ValueError, TypeError):
        return False, None


def validate_text_length(text: str, min_length: int = 10, max_length: int = 10000) -> Tuple[bool, Optional[str]]:
    if not text:
        return False, "Text cannot be empty"
    
    text = text.strip()
    length = len(text)
    
    if length < min_length:
        return False, f"Text must be at least {min_length} characters"
    
    if length > max_length:
        return False, f"Text cannot exceed {max_length} characters"
    
    return True, None


def safe_int_conversion(value: any, default: int = 0, min_val: int = None, max_val: int = None) -> int:
    try:
        result = int(value)
        
        if min_val is not None and result < min_val:
            return min_val
        if max_val is not None and result > max_val:
            return max_val
        
        return result
    except (ValueError, TypeError):
        return default
