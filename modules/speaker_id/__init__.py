"""
modules/speaker_id
==================
STATO: SOLO INTERFACCIA (nessuna implementazione concreta).

Definisce l'interfaccia astratta BaseSpeakerID (vedi base_speaker_id.py)
per identificazione/enrollment dei parlanti. Non esiste ancora una
sottoclasse concreta, quindi il modulo non è istanziabile né cablato
nell'orchestratore. Roadmap.
"""

from .base_speaker_id import BaseSpeakerID

__all__ = ["BaseSpeakerID"]
