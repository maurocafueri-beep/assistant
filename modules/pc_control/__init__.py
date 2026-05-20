"""
modules/pc_control
==================
STATO: SOLO INTERFACCIA (nessuna implementazione concreta).

Definisce l'interfaccia astratta BasePCControl (vedi base_pc_control.py)
per il controllo del PC (apri app/file, digita testo, screenshot,
clipboard). Non esiste ancora una sottoclasse concreta, quindi il modulo
non è istanziabile né cablato nell'orchestratore. Roadmap.

NOTA SICUREZZA: i flag in config.settings.PCControlSettings
(allow_delete, allow_sudo, safe_dirs) andranno rispettati
dall'implementazione concreta quando verrà scritta.
"""

from .base_pc_control import BasePCControl

__all__ = ["BasePCControl"]
