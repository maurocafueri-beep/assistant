"""
ui/
Interfaccia web locale per local-assistant.

Componenti:
    ui.bridge  → WSManager, UIBridge (sottoclasse di VoiceLoop)
    ui.server  → FastAPI app, WebSocket endpoint, REST API
    ui/static/ → frontend HTML/CSS/JS single-file

Uso rapido:
    venv-runtime/bin/python scripts/run_ui.py
"""

from ui.bridge import UIBridge, WSManager

__all__ = ["UIBridge", "WSManager"]
