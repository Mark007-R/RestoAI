"""Caching utilities for improved performance."""
from functools import wraps
from datetime import datetime, timedelta
import hashlib
import os


class SimpleCache:
    def __init__(self, cache_dir='cache', default_timeout=300):
        self._cache = {}
        self.cache_dir = cache_dir
        self.default_timeout = default_timeout
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
    
    def _make_key(self, key):
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        return hashlib.md5(str(key).encode()).hexdigest()
    
    def get(self, key):
        cache_key = self._make_key(key)
        if cache_key in self._cache:
            value, expires_at = self._cache[cache_key]
            if datetime.now() < expires_at:
                return value
            else:
                del self._cache[cache_key]
        return None
    
    def set(self, key, value, timeout=None):
        cache_key = self._make_key(key)
        timeout = timeout or self.default_timeout
        expires_at = datetime.now() + timedelta(seconds=timeout)
        self._cache[cache_key] = (value, expires_at)
    
    def delete(self, key):
        cache_key = self._make_key(key)
        if cache_key in self._cache:
            del self._cache[cache_key]
    
    def clear(self):
        self._cache.clear()
    
    def get_or_set(self, key, callback, timeout=None):
        value = self.get(key)
        if value is not None:
            return value
        value = callback()
        self.set(key, value, timeout)
        return value


def cached(timeout=300, key_func=None):
    cache = SimpleCache()
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            result = cache.get(cache_key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, timeout)
            return result
        wrapper.cache_clear = cache.clear
        return wrapper
    return decorator


def memoize(func):
    cache = {}
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    wrapper.cache_clear = cache.clear
    return wrapper