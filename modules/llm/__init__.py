"""
modules/llm
===========
Client Ollama per tutto il progetto.

Importare sempre da qui, mai da base.py direttamente:

    from modules.llm import OllamaClient, Message, Role, LLMResponse
    from core.context import ModelRole  # ModelRole vive in core, non qui
"""

from .base_llm import LLMResponse, Message, OllamaClient, Role

__all__ = [
    "OllamaClient",
    "LLMResponse",
    "Message",
    "Role",
]
