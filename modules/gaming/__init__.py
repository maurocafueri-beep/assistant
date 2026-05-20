"""
modules/gaming
==============
STATO: SOLO INTERFACCIA (nessuna implementazione concreta).

Definisce l'interfaccia astratta BaseGamingAgent (vedi base_gaming.py) ma
non esiste ancora una sottoclasse concreta, quindi il modulo non è
istanziabile né cablato nell'orchestratore. Roadmap.
"""

from .base_gaming import BaseGamingAgent, GameState

__all__ = ["BaseGamingAgent", "GameState"]
