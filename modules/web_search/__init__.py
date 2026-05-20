"""
modules/web_search/__init__.py
Espone le classi pubbliche del modulo web_search.
"""

from modules.web_search.base_web_search import (
    SearchResponse,
    SearchResult,
    SearXNGClient,
)

__all__ = [
    "SearXNGClient",
    "SearchResult",
    "SearchResponse",
]
