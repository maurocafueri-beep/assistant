"""
ui/native_app.py
Applicazione nativa Qt Quick per local-assistant.

Nessuna webview, nessun server HTTP: QML (ui/qml/Main.qml) parla col
Backend (ui/native_backend.py) che gira l'UIBridge in un worker asyncio.

Non usare direttamente — usare scripts/run_native.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QGuiApplication, QIcon
from PyQt6.QtQml import QQmlApplicationEngine

from core.logger import logger
from ui.native_backend import Backend

_QML_DIR = Path(__file__).parent / "qml"
_ASSETS  = Path(__file__).parent / "assets"


def run(personality: Optional[str] = None, ptt_key: str = "space") -> int:
    app = QGuiApplication(sys.argv)
    app.setApplicationName("Local Assistant")
    app.setDesktopFileName("local-assistant")
    icon = _ASSETS / "icon.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))

    backend = Backend()

    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("backend", backend)
    engine.load(QUrl.fromLocalFile(str(_QML_DIR / "Main.qml")))
    if not engine.rootObjects():
        logger.error("ui.native | Main.qml non caricato — vedi errori QML sopra")
        return 1

    backend.start(personality=personality, ptt_key=ptt_key)
    app.aboutToQuit.connect(backend.shutdown)

    logger.info("ui.native | finestra Qt Quick pronta")
    return app.exec()
