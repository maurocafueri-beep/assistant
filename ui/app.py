"""
ui/app.py
Applicazione Qt nativa per local-assistant.

Avvolge la UI web (FastAPI + HTML) in una finestra nativa Ubuntu
tramite QWebEngineView. Il server FastAPI gira in un thread asyncio
separato; Qt occupa il main thread come richiesto da X11/Wayland.

Non usare direttamente — usare scripts/run_app.py.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

from PyQt6.QtCore    import QThread, pyqtSignal, QObject, QUrl, QTimer, QSize, Qt
from PyQt6.QtGui     import QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QSystemTrayIcon, QMenu,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore    import QWebEngineProfile, QWebEnginePage

from core.logger import logger

_ASSETS = Path(__file__).parent / "assets"
_APP_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# AsyncIO worker thread
# ---------------------------------------------------------------------------

class _AsyncWorker(QObject):
    """
    Gira il loop asyncio (UIBridge + uvicorn) in un QThread separato.
    Emette 'ready' quando il server è raggiungibile.
    """
    ready   = pyqtSignal()
    crashed = pyqtSignal(str)

    def __init__(
        self,
        personality: Optional[str],
        ptt_key: str,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._personality = personality
        self._ptt_key     = ptt_key

    def run(self) -> None:
        import asyncio

        from ui.bootstrap import serve

        async def _main() -> None:
            # L'intera sequenza di boot vive in ui.bootstrap.serve(),
            # condivisa con il launcher headless (scripts/run_ui.py). Qui
            # passiamo solo l'hook "server pronto": emettiamo il segnale Qt
            # che fa caricare la webview.
            await serve(
                personality     = self._personality,
                ptt_key         = self._ptt_key,
                on_server_ready = self.ready.emit,
            )

        try:
            asyncio.run(_main())
        except Exception as exc:
            logger.error("ui.app | async worker crashed: {}", exc)
            self.crashed.emit(str(exc))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class AssistantWindow(QMainWindow):
    """
    Finestra principale dell'applicazione.
    Contiene QWebEngineView che punta al server FastAPI locale.
    """

    def __init__(self) -> None:
        super().__init__()
        self._setup_window()
        self._setup_webview()
        self._setup_tray()
        self._setup_shortcuts()

    # -- Setup ----------------------------------------------------------------

    def _setup_window(self) -> None:
        self.setWindowTitle("local-assistant")

        # Icon
        icon_path = _ASSETS / "icon.png"
        if not icon_path.exists():
            icon_path = _ASSETS / "icon.svg"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Restore saved geometry, fallback to centered 1280x800
        self.resize(1280, 800)
        self._center_on_screen()

        # Dark background to avoid white flash during load
        self.setStyleSheet("QMainWindow { background: #080810; }")

    def _center_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen:
            sg = screen.availableGeometry()
            wg = self.frameGeometry()
            self.move(sg.center() - wg.center())

    def _setup_webview(self) -> None:
        # Dedicated off-the-record profile — no cookies/cache shared with system browser
        profile = QWebEngineProfile("local-assistant", self)
        profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.MemoryHttpCache)

        page = QWebEnginePage(profile, self)
        # Grant clipboard permissions so navigator.clipboard works
        page.featurePermissionRequested.connect(
            lambda origin, feature: page.setFeaturePermission(
                origin, feature,
                QWebEnginePage.PermissionPolicy.PermissionGrantedByUser
            )
        )

        # Intercept downloads → native "Save As" dialog so the user
        # chooses where to save the exported chat on their PC.
        profile.downloadRequested.connect(self._on_download)

        self._view = QWebEngineView(self)
        self._view.setPage(page)

        # Disable context menu (handled by the web UI)
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

        # Loading placeholder
        self._view.setHtml("""
        <!DOCTYPE html>
        <html>
        <head><style>
          body { margin:0; background:#080810; display:flex;
                 align-items:center; justify-content:center; height:100vh;
                 font-family: monospace; color: #3a3860; }
          .dot { animation: blink 1.2s ease-in-out infinite; }
          .dot:nth-child(2) { animation-delay:.2s }
          .dot:nth-child(3) { animation-delay:.4s }
          @keyframes blink { 0%,100%{opacity:.2} 50%{opacity:1} }
        </style></head>
        <body>
          <div>
            <span class="dot">◈</span>
            <span class="dot">◈</span>
            <span class="dot">◈</span>
          </div>
        </body>
        </html>
        """)

        central = QWidget(self)
        layout  = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        self.setCentralWidget(central)

    def _on_download(self, item) -> None:
        """Mostra una dialog nativa di salvataggio per i download."""
        from PyQt6.QtWidgets import QFileDialog
        from pathlib import Path as _P

        # Suggested filename from the download
        try:
            suggested = item.downloadFileName()
        except Exception:
            suggested = "chat-export.md"

        default_dir = str(_P.home() / "Documents")
        if not _P(default_dir).exists():
            default_dir = str(_P.home())

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Esporta chat",
            str(_P(default_dir) / suggested),
            "Markdown (*.md);;Testo (*.txt);;Tutti i file (*)",
        )

        if not path:
            item.cancel()
            return

        p = _P(path)
        item.setDownloadDirectory(str(p.parent))
        item.setDownloadFileName(p.name)
        item.accept()
        # Notify when finished
        try:
            item.isFinishedChanged.connect(
                lambda: logger.info("ui.app | chat esportata → {}", path)
            )
        except Exception:
            pass

    def _setup_tray(self) -> None:
        icon_path = _ASSETS / "icon.png"
        if not icon_path.exists():
            icon_path = _ASSETS / "icon.svg"

        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        icon = QIcon(str(icon_path)) if icon_path.exists() else QIcon()
        self._tray = QSystemTrayIcon(icon, self)
        self._tray.setToolTip("local-assistant")

        menu = QMenu()
        menu.addAction("Apri",   self.show_and_raise)
        menu.addAction("Nascondi", self.hide)
        menu.addSeparator()
        menu.addAction("Esci", QApplication.quit)
        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._tray_activated)
        self._tray.show()

    def _setup_shortcuts(self) -> None:
        # Ctrl+W / Cmd+W — hide to tray instead of close
        QShortcut(QKeySequence("Ctrl+W"), self).activated.connect(self.hide)
        # F5 — reload
        QShortcut(QKeySequence("F5"), self).activated.connect(
            lambda: self._view.reload()
        )
        # Ctrl+Q — quit
        QShortcut(QKeySequence("Ctrl+Q"), self).activated.connect(
            QApplication.quit
        )

    # -- Slots ----------------------------------------------------------------

    def load_app(self) -> None:
        """Chiamato quando il server FastAPI è pronto."""
        logger.info("ui.app | caricamento UI → {}", _APP_URL)
        self._view.load(QUrl(_APP_URL))

    def show_and_raise(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.isVisible():
                self.hide()
            else:
                self.show_and_raise()

    def server_crashed(self, error: str) -> None:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Errore", f"Il server è crashato:\n{error}")
        QApplication.quit()

    # -- Window events --------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Chiudi nella tray invece di terminare l'app."""
        if hasattr(self, "_tray") and QSystemTrayIcon.isSystemTrayAvailable():
            event.ignore()
            self.hide()
            self._tray.showMessage(
                "local-assistant",
                "L'app è ancora attiva nella tray.",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )
        else:
            event.accept()
            QApplication.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    personality: Optional[str] = None,
    ptt_key: str = "space",
) -> int:
    """
    Avvia l'applicazione Qt.
    Chiamare dal main thread.

    Returns:
        Exit code dell'applicazione.
    """
    # Required before QApplication on some Linux setups
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("local-assistant")
    app.setOrganizationName("local")
    app.setQuitOnLastWindowClosed(False)  # keep alive in tray

    # Set app icon
    icon_path = _ASSETS / "icon.png"
    if not icon_path.exists():
        icon_path = _ASSETS / "icon.svg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Create main window
    window = AssistantWindow()
    window.show()

    # Start async worker in a QThread
    thread = QThread()
    worker = _AsyncWorker(personality=personality, ptt_key=ptt_key)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.ready.connect(window.load_app)
    worker.crashed.connect(window.server_crashed)

    thread.start()

    exit_code = app.exec()
    thread.quit()
    thread.wait(3000)
    return exit_code
