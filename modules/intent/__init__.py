"""
modules/intent
==============
Classificazione degli intenti del turno: instrada verso web e/o file.

    from modules.intent import Intent, IntentClassifier
"""

from modules.intent.base_intent import Intent, IntentClassifier

__all__ = [
    "Intent",
    "IntentClassifier",
]
