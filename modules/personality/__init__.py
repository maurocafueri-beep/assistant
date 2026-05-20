"""
modules/personality
===================
Gestione dei profili di personalità per l'assistente.

Importare sempre da qui:

    from modules.personality import PersonalityManager, PersonalityProfile
"""

from .base_personality import PersonalityManager, PersonalityProfile

__all__ = [
    "PersonalityManager",
    "PersonalityProfile",
]
