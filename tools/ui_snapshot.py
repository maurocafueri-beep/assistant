"""
tools/ui_snapshot.py
Render offscreen della UI QML con backend stub e dati simulati → PNG.

Strumento di sviluppo: permette di iterare sul design senza avviare lo
stack (niente orchestratore/STT/TTS) e senza display.

Uso:
    QT_QPA_PLATFORM=offscreen venv-runtime/bin/python tools/ui_snapshot.py out chat
    # viste: chat | terminal | thinking | errors | empty
"""
import sys
from pathlib import Path

from PyQt6.QtCore import QUrl, QObject, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtQml import QQmlApplicationEngine
from PyQt6.QtQuick import QQuickWindow
from PyQt6 import sip

PROJECT = Path(__file__).parent.parent
OUT  = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ui_snap"
MODE = sys.argv[2] if len(sys.argv) > 2 else "chat"


class StubBackend(QObject):
    initReady          = pyqtSignal('QVariant')
    stateChanged       = pyqtSignal(str)
    chunkReceived      = pyqtSignal(str)
    userMessage        = pyqtSignal(str)
    statsChanged       = pyqtSignal('QVariant')
    sessionsChanged    = pyqtSignal('QVariant')
    sessionSwitched    = pyqtSignal('QVariant')
    modelChanged       = pyqtSignal(str)
    personalityChanged = pyqtSignal(str, str)
    voiceChanged       = pyqtSignal(str)
    ttsChanged         = pyqtSignal(bool)
    modeChanged        = pyqtSignal(str)
    modelsListed       = pyqtSignal('QVariant')
    backendError       = pyqtSignal(str)
    errorOccurred      = pyqtSignal('QVariant')
    uploadFinished     = pyqtSignal('QVariant')
    terminalEvent      = pyqtSignal('QVariant')

    @pyqtSlot(str, result=str)
    def uiSetting(self, key):
        import os
        if key == "ui_theme":
            return "dark" if os.environ.get("SNAP_DARK") == "1" else "light"
        return ""
    @pyqtSlot(str, str)
    def saveUiSetting(self, k, v): pass
    @pyqtSlot()
    def pickFiles(self): pass
    @pyqtSlot(str)
    def sendText(self, t): pass
    @pyqtSlot(str)
    def switchModel(self, n): pass
    @pyqtSlot(str)
    def switchPersonality(self, n): pass
    @pyqtSlot(str)
    def switchVoice(self, n): pass
    @pyqtSlot(str)
    def switchSession(self, s): pass
    @pyqtSlot(str)
    def deleteSession(self, s): pass
    @pyqtSlot(str)
    def uploadFile(self, u): pass
    @pyqtSlot(str)
    def setMode(self, m): self.modeChanged.emit(m)
    @pyqtSlot(str)
    def terminalPropose(self, t): pass
    @pyqtSlot(str)
    def terminalConfirm(self, p): pass
    @pyqtSlot(str)
    def terminalCancel(self, p): pass
    @pyqtSlot(str)
    def terminalSwitchModel(self, n): pass
    @pyqtSlot(bool)
    def setTtsEnabled(self, e): pass
    @pyqtSlot(str, str)
    def renameSession(self, a, b): pass
    @pyqtSlot()
    def requestModels(self): self.modelsListed.emit(["gemma-4-12B-heretic", "gemma4:12b"])
    @pyqtSlot()
    def requestTerminalState(self):
        self.terminalEvent.emit({"type": "terminal.state", "cwd": "/home/mauro/assistant",
                                 "current_model": "qwen3:14b", "models": ["qwen3:14b"],
                                 "pending": [], "history": []})
    @pyqtSlot()
    def cancelTurn(self): pass
    @pyqtSlot()
    def newSession(self): pass
    @pyqtSlot()
    def terminalReset(self): pass
    @pyqtSlot()
    def pttDown(self): pass
    @pyqtSlot()
    def pttUp(self): pass


def main() -> int:
    import time
    app = QGuiApplication(sys.argv)
    backend = StubBackend()
    engine = QQmlApplicationEngine()
    warnings: list = []
    engine.warnings.connect(lambda ws: warnings.extend(w.toString() for w in ws))
    engine.rootContext().setContextProperty("backend", backend)
    engine.load(QUrl.fromLocalFile(str(PROJECT / "ui" / "qml" / "Main.qml")))
    if not engine.rootObjects():
        print("ERRORE QML")
        return 1

    now = time.time()

    def populate() -> None:
        backend.initReady.emit({
            "state": "idle", "model": "gemma-4-12B-heretic", "personality": "dev",
            "voice": "mercoledì", "session": "s1", "tts_enabled": True,
            "sessions": [
                {"id": "s1", "name": "Ottimizzazione TTS", "last_active": now,
                 "preview": "Perfetto, procediamo con la prima."},
                {"id": "s2", "name": "Ricette veloci", "last_active": now - 90000,
                 "preview": "Grazie! Ottima carbonara."},
                {"id": "s3", "name": "Debug wake word", "last_active": now - 260000,
                 "preview": "Il trigger ora funziona bene."},
            ],
            "personalities": [{"name": "dev", "display_name": "Dev"}],
            "voices": ["mercoledì", "cloe"],
            "active_messages": [] if MODE == "empty" else [
                {"role": "user", "text": "Come posso rendere più veloce il TTS?"},
                {"role": "assistant", "text": "Tre leve principali:\n\n1. **Modello più piccolo** (0.6B): genera 3× più veloce\n2. **GPU dedicata**: isola la sintesi dal traffico LLM\n3. *Time-stretch* WSOLA per il parlato accelerato"},
                {"role": "user", "text": "Perfetto, procediamo con la prima."},
            ],
        })
        backend.statsChanged.emit({"latency": {"stt_ms": 240, "llm_ms": 1830,
                                               "tts_ms": 410, "total_ms": 2480}})
        if MODE == "terminal":
            backend.modeChanged.emit("terminal")
            backend.terminalEvent.emit({"type": "terminal.proposal",
                "proposal": {"proposal_id": "p1", "command": "ps aux --sort=-%mem | head -8",
                             "rationale": "Elenca i processi ordinati per memoria decrescente.",
                             "risk_level": "low", "needs_confirmation": True,
                             "cwd": "/home/mauro", "search_used": False, "sources": []}})
        elif MODE == "thinking":
            backend.userMessage.emit("E per la seconda leva?")
            backend.stateChanged.emit("thinking")
        elif MODE == "errors":
            backend.errorOccurred.emit({"source": "turn",
                                        "message": "LLM error: connessione a Ollama rifiutata"})

    def snap() -> None:
        w = sip.cast(engine.rootObjects()[0], QQuickWindow)
        img = w.grabWindow()
        img.save(f"{OUT}.png")
        print(f"salvato {OUT}.png | warnings: {len(warnings)}")
        for wr in warnings[:8]:
            print(" ", wr)
        app.quit()

    QTimer.singleShot(250, populate)
    QTimer.singleShot(1100, snap)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
