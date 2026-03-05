"""
Manager System Package
======================

Flask-based restaurant management and analytics system.

Main Components:
- app: Main Flask application
- manager: Manager routes and controllers
- analyzer: Review analysis engine
- scraper: Web scraping utilities
- rag_chat: RAG-based chat assistant
- config: Configuration settings

Quick Start:
    >>> from app import app
    >>> app.run(debug=True)

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "RestoAI Team"
__all__ = [
    "app",
    "db",
    "User",
    "Review",
]

# Note: Imports are handled in app.py to avoid circular dependencies
# This __init__.py primarily serves as package marker and documentation
