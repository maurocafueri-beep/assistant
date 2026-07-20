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

from PyQt6.QtCore import QEvent, QObject, Qt, QUrl
from PyQt6.QtGui import QIcon
from PyQt6.QtQml import QQmlApplicationEngine
# QApplication (widgets) e non QGuiApplication: serve al QFileDialog di
# fallback del backend (pickFiles), che evita i portal inaffidabili.
from PyQt6.QtWidgets import QApplication

from core.logger import logger
from ui.native_backend import Backend

_QML_DIR = Path(__file__).parent / "qml"
_ASSETS  = Path(__file__).parent / "assets"


class HoldToTalkFilter(QObject):
    """Push-to-talk da tastiera: Alt premuto = registra, rilasciato = invia.

    Event filter a livello di applicazione (non QML): funziona qualunque
    item abbia il focus, campo di testo incluso. Wayland non permette
    hotkey globali senza portal, quindi vale con la finestra attiva —
    che è anche l'unico momento in cui ha senso dettare.
    """

    def __init__(self, backend: Backend) -> None:
        super().__init__()
        self._backend = backend
        self._held = False

    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:
        t = ev.type()
        if t == QEvent.Type.KeyPress and ev.key() == Qt.Key.Key_Alt:
            if not ev.isAutoRepeat() and not self._held:
                self._held = True
                self._backend.pttDown()
            return True
        if t == QEvent.Type.KeyRelease and ev.key() == Qt.Key.Key_Alt:
            if not ev.isAutoRepeat() and self._held:
                self._held = False
                self._backend.pttUp()
            return True
        # Alt+Tab e simili: la finestra perde il focus e il KeyRelease non
        # arriva mai — chiudiamo la registrazione per non lasciarla appesa.
        if t == QEvent.Type.ApplicationDeactivate and self._held:
            self._held = False
            self._backend.pttUp()
        return False


def run(personality: Optional[str] = None, ptt_key: str = "alt") -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Local Assistant")
    app.setDesktopFileName("local-assistant")
    icon = _ASSETS / "icon.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))

    backend = Backend()

    ptt_filter = HoldToTalkFilter(backend)
    app.installEventFilter(ptt_filter)

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
