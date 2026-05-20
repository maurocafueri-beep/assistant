"""
modules/memory/__init__.py
Punto di ingresso pubblico del modulo memory.

Uso:
    from modules.memory import MemoryManager, SaveResult, SearchResult
"""

from modules.memory.base_memory import (
    MemoryManager,
    SaveResult,
    SearchResult,
)

__all__ = [
    "MemoryManager",
    "SaveResult",
    "SearchResult",
]
