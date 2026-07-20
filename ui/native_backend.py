"""
ui/native_backend.py
Backend Qt in-process per la UI nativa Qt Quick (QML) di local-assistant.

Sostituisce l'intero stack web (FastAPI + uvicorn + WebSocket + HTML):
l'UIBridge esistente resta la fonte di verità per orchestratore, sessioni,
warmup e PTT, ma al posto del WSManager riceve un QtEmitter che inoltra i
payload (gli stessi dict del protocollo WS: init/state/chunk/…) come segnali
Qt verso QML. Niente porta HTTP, niente pagina, un solo processo.

Threading:
    - Main thread: QGuiApplication + QQmlApplicationEngine + Backend(QObject).
    - Worker: event loop asyncio con l'UIBridge (stesso pattern di ui/app.py).
    - asyncio → Qt: QtEmitter.broadcast() emette il segnale `event` (PyQt
      accoda automaticamente le emissioni cross-thread).
    - Qt → asyncio: gli slot del Backend usano run_coroutine_threadsafe
      sull'event loop del worker.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from core.logger import logger, setup_logging


# ---------------------------------------------------------------------------
# QtEmitter — rimpiazza WSManager: broadcast → segnale Qt
# ---------------------------------------------------------------------------

class QtEmitter(QObject):
    """
    Duck-type del WSManager per UIBridge: espone `broadcast(payload)` e
    `n_clients`. Ogni payload diventa un'emissione del segnale `event`,
    consumato dal Backend nel main thread.
    """
    event = pyqtSignal(dict)

    async def broadcast(self, payload: dict) -> None:
        self.event.emit(payload)

    async def connect(self, ws: Any) -> None:   # compat interfaccia
        pass

    def disconnect(self, ws: Any) -> None:      # compat interfaccia
        pass

    @property
    def n_clients(self) -> int:
        return 1


# ---------------------------------------------------------------------------
# Worker asyncio — UIBridge senza server HTTP
# ---------------------------------------------------------------------------

async def serve_native(
    emitter: QtEmitter,
    *,
    personality: Optional[str] = None,
    ptt_key: str = "alt",
    on_loop_ready=None,
) -> None:
    """
    Gemella di ui.bootstrap.serve ma senza uvicorn/FastAPI: stessa sequenza
    di ripristino (sessioni, impostazioni, cronologia LLM, RAG), stesso
    warmup e cleanup in background, poi il run() del bridge fino allo stop.

    on_loop_ready(ui_loop): hook invocato quando il bridge è costruito (prima
    di load), così il Backend può registrare i riferimenti per gli slot.
    """
    from core.uploads_cleanup import run_cleanup_at_boot
    from ui.bootstrap import (
        _restore_sessions_from_disk,
        _restore_ui_settings,
        _spawn_bg,
    )
    from ui.bridge import UIBridge
    from ui.server import _save_sessions_disk
    from ui.terminal_bridge import TerminalBridge

    setup_logging()

    ui_loop = UIBridge(ws_manager=emitter, ptt_key=ptt_key, personality=personality)
    # Il TerminalBridge serve a _restore_ui_settings (modalità/modelli
    # terminale); la UI nativa v1 non lo espone ancora, ma ripristinarlo
    # mantiene lo stato coerente con la vecchia UI.
    terminal_bridge = TerminalBridge(ws_manager=emitter)
    ui_loop.set_terminal_bridge(terminal_bridge)
    ui_loop.set_persist_callback(_save_sessions_disk)
    _restore_sessions_from_disk(ui_loop)

    if on_loop_ready is not None:
        try:
            on_loop_ready(ui_loop)
        except Exception as exc:
            logger.warning("ui.native | on_loop_ready: {}", exc)

    bg_tasks: set = set()
    try:
        async with ui_loop:
            await _restore_ui_settings(ui_loop, terminal_bridge)
            # Velocità voce salvata dalla sezione Sistema (slider): il client
            # TTS la legge alla costruzione, qui la ri-applichiamo a runtime.
            try:
                from ui.server import _load_ui_settings
                sp = (_load_ui_settings() or {}).get("tts_speed")
                if sp and ui_loop._tts is not None:
                    ui_loop._tts._speed = max(0.5, min(2.0, float(sp)))
                    logger.info("ui.native | velocità voce ripristinata → {}x", sp)
            except Exception as exc:
                logger.warning("ui.native | restore tts_speed: {}", exc)
            try:
                ui_loop.restore_histories_from_sessions()
            except Exception as exc:
                logger.warning("ui.native | restore histories: {}", exc)
            try:
                ui_loop.restore_rag_files_from_sessions()
            except Exception as exc:
                logger.warning("ui.native | restore rag_files: {}", exc)
            try:
                await ui_loop.broadcast_init()
            except Exception as exc:
                logger.warning("ui.native | broadcast_init: {}", exc)

            _spawn_bg(ui_loop.warmup(), label="warmup", registry=bg_tasks)
            _spawn_bg(run_cleanup_at_boot(), label="cleanup", registry=bg_tasks)

            await ui_loop.run()
    finally:
        try:
            await terminal_bridge.aclose()
        except Exception as exc:
            logger.debug("ui.native | aclose terminal_bridge: {}", exc)


# ---------------------------------------------------------------------------
# Upload — logica pura, riusata dallo slot e testabile in isolamento
# ---------------------------------------------------------------------------

def save_upload(path: str, session_id: str) -> dict:
    """
    Valida e copia un file locale in data/uploads/<session_id>/ con le stesse
    regole dell'endpoint web (estensione supportata, dimensione massima,
    filename sanificato). Ritorna {ok, path, name, size} o {ok, error}.
    """
    import shutil as _sh
    from datetime import datetime
    from pathlib import Path

    from config.settings import settings as _settings
    from ui.server import _ALLOWED_UPLOAD_EXTS, _safe_filename, _session_dir_for

    src = Path(path).expanduser()
    try:
        if not src.is_file():
            raise ValueError(f"file non trovato: {src}")
        ext = src.suffix.lower()
        if ext not in _ALLOWED_UPLOAD_EXTS:
            raise ValueError(
                f"estensione '{ext}' non supportata "
                f"({', '.join(sorted(_ALLOWED_UPLOAD_EXTS))})"
            )
        size = src.stat().st_size
        if size > _settings.file_analysis.max_file_bytes:
            raise ValueError(
                f"file troppo grande ({size} byte > "
                f"{_settings.file_analysis.max_file_bytes})"
            )
        target = (
            _session_dir_for(session_id)
            / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{_safe_filename(src.name)}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        _sh.copy2(src, target)
        return {"ok": True, "path": str(target), "name": src.name, "size": size}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Backend — l'oggetto esposto a QML
# ---------------------------------------------------------------------------

class Backend(QObject):
    """
    Facciata Qt per QML: traduce i payload del bridge in segnali granulari
    e gli input dell'utente in chiamate sull'event loop asyncio del worker.

    La mappa payload→segnale replica il protocollo WS della vecchia UI, così
    la semantica (init idempotente, chunk in streaming, session_switch con
    messaggi) resta identica e già collaudata.
    """

    # ── segnali verso QML ────────────────────────────────────────────
    initReady          = pyqtSignal('QVariant')   # payload init completo
    stateChanged       = pyqtSignal(str)          # idle/recording/thinking/…
    chunkReceived      = pyqtSignal(str)          # streaming risposta
    userMessage        = pyqtSignal(str)          # testo utente (eco da voce)
    statsChanged       = pyqtSignal('QVariant')   # stats + latency di turno
    sessionsChanged    = pyqtSignal('QVariant')   # lista sessioni
    sessionSwitched    = pyqtSignal('QVariant')   # {session, messages, name}
    modelChanged       = pyqtSignal(str)
    personalityChanged = pyqtSignal(str, str)     # name, display_name
    voiceChanged       = pyqtSignal(str)
    ttsChanged         = pyqtSignal(bool)
    modeChanged        = pyqtSignal(str)
    modelsListed       = pyqtSignal('QVariant')   # da requestModels()
    backendError       = pyqtSignal(str)
    errorOccurred      = pyqtSignal('QVariant')   # {source, message} → pannello errori
    uploadFinished     = pyqtSignal('QVariant')   # {ok, path, name} | {ok, error}
    terminalEvent      = pyqtSignal('QVariant')   # payload terminal.* integrale
    systemStatus       = pyqtSignal('QVariant')   # da requestSystemStatus()
    sttPartial         = pyqtSignal('QVariant')   # {text, final} — dettato live PTT

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._emitter = QtEmitter()
        self._emitter.event.connect(self._on_event)
        self._ui_loop: Any = None                     # UIBridge (worker thread)
        self._aio_loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    # ── ciclo di vita ────────────────────────────────────────────────

    def start(self, *, personality: Optional[str] = None, ptt_key: str = "alt") -> None:
        """Avvia il worker asyncio con l'UIBridge. Idempotente."""
        if self._thread is not None:
            return

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._aio_loop = loop

            def _ready(ui_loop: Any) -> None:
                self._ui_loop = ui_loop

            try:
                loop.run_until_complete(serve_native(
                    self._emitter,
                    personality=personality,
                    ptt_key=ptt_key,
                    on_loop_ready=_ready,
                ))
            except Exception as exc:
                logger.error("ui.native | worker terminato: {}", exc)
                self.backendError.emit(str(exc))
            finally:
                loop.close()

        self._thread = threading.Thread(target=_runner, name="ui-native-aio", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        """Ferma il bridge e attende la chiusura del worker (best-effort)."""
        if self._ui_loop is not None and self._aio_loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._ui_loop.stop(), self._aio_loop
                ).result(timeout=5)
            except Exception as exc:
                logger.debug("ui.native | shutdown stop(): {}", exc)
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    # ── payload bridge → segnali QML ─────────────────────────────────

    def _on_event(self, payload: dict) -> None:
        t = payload.get("type", "")
        if   t == "init":           self.initReady.emit(payload)
        elif t == "state":          self.stateChanged.emit(payload.get("value", "idle"))
        elif t == "chunk":          self.chunkReceived.emit(payload.get("text", ""))
        elif t == "user":           self.userMessage.emit(payload.get("text", ""))
        elif t == "stats":          self.statsChanged.emit(payload)
        elif t == "sessions":       self.sessionsChanged.emit(payload.get("sessions", []))
        elif t == "session_switch": self.sessionSwitched.emit(payload)
        elif t == "model":          self.modelChanged.emit(payload.get("name", ""))
        elif t == "personality":
            self.personalityChanged.emit(
                payload.get("name", ""), payload.get("display_name", ""),
            )
        elif t == "voice":          self.voiceChanged.emit(payload.get("name", ""))
        elif t == "tts":            self.ttsChanged.emit(bool(payload.get("enabled", True)))
        elif t == "mode":           self.modeChanged.emit(payload.get("mode", "chat"))
        elif t == "_models":        self.modelsListed.emit(payload.get("models", []))
        elif t == "_upload":        self.uploadFinished.emit(payload)
        elif t == "_system":        self.systemStatus.emit(payload)
        elif t == "stt_partial":    self.sttPartial.emit(payload)
        elif t == "error":          self.errorOccurred.emit(payload)
        elif t.startswith("terminal."):
            self.terminalEvent.emit(payload)
            # gli errori del terminale finiscono ANCHE nel pannello errori
            if t == "terminal.error":
                self.errorOccurred.emit(
                    {"source": "terminal", "message": payload.get("message", "")}
                )
        # tipi sconosciuti: ignorati (compat con estensioni future del bridge)

    # ── helper interni ───────────────────────────────────────────────

    def _call_async(self, coro) -> None:
        """Esegue una coroutine sull'event loop del worker (fire-and-forget)."""
        if self._aio_loop is None:
            logger.warning("ui.native | backend non ancora avviato")
            return
        fut = asyncio.run_coroutine_threadsafe(coro, self._aio_loop)

        def _done(f) -> None:
            exc = f.exception()
            if exc is not None:
                logger.warning("ui.native | chiamata async fallita: {}", exc)

        fut.add_done_callback(_done)

    def _call_on_loop(self, fn, *args, on_result=None) -> None:
        """
        Esegue un metodo SINCRONO del bridge dentro l'event loop del worker.

        Necessario anche per i metodi non-async: i loro `_emit` fanno
        `get_running_loop().create_task(...)` e, se invocati dal thread Qt
        (nessun loop attivo), scarterebbero l'evento in silenzio — la UI non
        riceverebbe mai la conferma (es. {"type":"model"} dopo switch_model).
        `on_result(ret)` opzionale, eseguito anch'esso nel worker.
        """
        if self._ui_loop is None:
            logger.warning("ui.native | bridge non ancora pronto")
            return

        async def _co() -> None:
            ret = fn(*args)
            if on_result is not None:
                on_result(ret)

        self._call_async(_co())

    def _save_setting(self, key: str, value: Any) -> None:
        from ui.server import _save_ui_settings
        _save_ui_settings({key: value})

    # ── slot chiamati da QML ─────────────────────────────────────────

    @pyqtSlot(str)
    def sendText(self, text: str) -> None:
        text = text.strip()
        if not text or self._ui_loop is None:
            return
        self._call_async(self._ui_loop.send_text(text))

    @pyqtSlot()
    def cancelTurn(self) -> None:
        self._call_on_loop(lambda: self._ui_loop.cancel_generation())

    @pyqtSlot(bool)
    def setTtsEnabled(self, enabled: bool) -> None:
        self._call_on_loop(
            lambda: self._ui_loop.set_tts_enabled(enabled),
            on_result=lambda _ok: self._save_setting("tts_enabled", enabled),
        )

    @pyqtSlot(str)
    def switchModel(self, name: str) -> None:
        self._call_on_loop(
            lambda: self._ui_loop.switch_model(name),
            on_result=lambda ok: self._save_setting("model", name) if ok else None,
        )

    @pyqtSlot(str)
    def switchPersonality(self, name: str) -> None:
        self._call_on_loop(
            lambda: self._ui_loop.switch_personality(name),
            on_result=lambda ok: self._save_setting("personality", name) if ok else None,
        )

    @pyqtSlot(str)
    def switchVoice(self, name: str) -> None:
        async def _do() -> None:
            ok = await self._ui_loop.switch_voice(name)
            if ok:
                self._save_setting("voice", name)
        if self._ui_loop is not None:
            self._call_async(_do())

    @pyqtSlot()
    def requestModels(self) -> None:
        async def _do() -> None:
            models = await self._ui_loop.list_models()
            self._emitter.event.emit({"type": "_models", "models": models})
        if self._ui_loop is not None:
            self._call_async(_do())

    @pyqtSlot()
    def requestInit(self) -> None:
        if self._ui_loop is not None:
            self._call_async(self._ui_loop.broadcast_init())

    # impostazioni UI (tema, preferenze del solo frontend) -----------------

    @pyqtSlot(str, result=str)
    def uiSetting(self, key: str) -> str:
        """Legge una preferenza da ui-settings.json (stringa, '' se assente).
        Lettura sincrona di un piccolo JSON locale: costo trascurabile."""
        from ui.server import _load_ui_settings
        try:
            return str(_load_ui_settings().get(key, "") or "")
        except Exception:
            return ""

    @pyqtSlot(str, str)
    def saveUiSetting(self, key: str, value: str) -> None:
        self._save_setting(key, value)

    @pyqtSlot(float)
    def setTtsSpeed(self, speed: float) -> None:
        """Velocità del parlato a runtime (time-stretch WSOLA server-side):
        applica subito al client TTS e persiste per i riavvii."""
        speed = max(0.5, min(2.0, float(speed)))

        def _apply() -> None:
            tts = getattr(self._ui_loop, "_tts", None)
            if tts is not None:
                tts._speed = speed

        self._call_on_loop(_apply,
                           on_result=lambda _r: self._save_setting("tts_speed", speed))

    # stato di sistema (sezione "Sistema" della UI) ------------------------

    @pyqtSlot()
    def requestSystemStatus(self) -> None:
        """
        Raccoglie lo stato di servizi, GPU e modelli ed emette il payload
        `_system` → segnale systemStatus. Best-effort su ogni voce: un
        servizio giù non blocca gli altri (probe con timeout corti).
        """
        if self._aio_loop is None:
            return

        async def _do() -> None:
            import asyncio as _aio

            import httpx

            from config.settings import settings as _s

            async def _gpus() -> list:
                def _run() -> list:
                    import subprocess
                    r = subprocess.run(
                        ["nvidia-smi",
                         "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, timeout=5,
                    )
                    out = []
                    for line in r.stdout.decode().strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 4:
                            out.append({"name": parts[0], "used": int(parts[1]),
                                        "total": int(parts[2]), "util": int(parts[3])})
                    return out
                try:
                    return await _aio.to_thread(_run)
                except Exception:
                    return []

            async def _ollama(client: "httpx.AsyncClient") -> dict:
                try:
                    v = await client.get(f"{_s.ollama.base_url}/api/version")
                    ps = await client.get(f"{_s.ollama.base_url}/api/ps")
                    models = [
                        {"name": m.get("name", "?"),
                         "vram_mb": round(m.get("size_vram", 0) / 1e6)}
                        for m in ps.json().get("models", [])
                    ]
                    return {"ok": True, "version": v.json().get("version", "?"),
                            "models": models}
                except Exception:
                    return {"ok": False, "version": "", "models": []}

            async def _tts(client: "httpx.AsyncClient") -> dict:
                try:
                    r = await client.get("http://127.0.0.1:8765/health")
                    j = r.json()
                    return {"ok": True, "model": j.get("model", ""),
                            "profile": j.get("profile", "")}
                except Exception:
                    return {"ok": False, "model": "", "profile": ""}

            async def _web(client: "httpx.AsyncClient") -> bool:
                try:
                    r = await client.get(_s.web_search.searxng_url)
                    return r.status_code < 500
                except Exception:
                    return False

            async with httpx.AsyncClient(timeout=2.5) as client:
                gpus, oll, tts, web = await _aio.gather(
                    _gpus(), _ollama(client), _tts(client), _web(client),
                )

            stats: dict = {}
            try:
                stats = self._ui_loop.stats.to_log_dict()
            except Exception:
                pass

            speed = 1.0
            try:
                speed = float(getattr(self._ui_loop._tts, "_speed", 1.0))
            except Exception:
                pass

            self._emitter.event.emit({
                "type": "_system",
                "gpus": gpus,
                "ollama": oll,
                "tts": tts,
                "web_ok": web,
                "wake": {"enabled": bool(getattr(_s.wake_word, "enabled", False)),
                         "model": getattr(_s.wake_word, "model", "")},
                "stats": stats,
                "tts_speed": speed,
            })

        self._call_async(_do())

    # sessioni ---------------------------------------------------------

    @pyqtSlot()
    def newSession(self) -> None:
        self._call_on_loop(lambda: self._ui_loop.new_session())

    @pyqtSlot(str)
    def switchSession(self, sid: str) -> None:
        self._call_on_loop(lambda: self._ui_loop.switch_session(sid))

    @pyqtSlot(str)
    def deleteSession(self, sid: str) -> None:
        self._call_on_loop(lambda: self._ui_loop.delete_session(sid))

    @pyqtSlot(str, str)
    def renameSession(self, sid: str, name: str) -> None:
        self._call_on_loop(lambda: self._ui_loop.rename_session(sid, name))

    # PTT (pulsante mic / barra spazio della finestra) ------------------

    @pyqtSlot()
    def pttDown(self) -> None:
        self._call_on_loop(lambda: self._ui_loop.ptt_down())

    @pyqtSlot()
    def pttUp(self) -> None:
        self._call_on_loop(lambda: self._ui_loop.ptt_up())

    # upload -------------------------------------------------------------

    @pyqtSlot()
    def pickFiles(self) -> None:
        """
        Apre il selettore file con il dialog Qt NON nativo. Su Hyprland il
        FileDialog QML passa dal portal FileChooser di xdg-desktop-portal,
        che a seconda del backend attivo può non rispondere (dialog che non
        appare mai): il widget Qt invece funziona sempre. Richiede che
        l'app sia una QApplication (vedi ui/native_app.py). Gira nel thread
        GUI (slot QML), quindi il dialog modale è legittimo qui.
        """
        try:
            from pathlib import Path

            from PyQt6.QtWidgets import QFileDialog
            files, _ = QFileDialog.getOpenFileNames(
                None, "Allega file", str(Path.home()),
                options=QFileDialog.Option.DontUseNativeDialog,
            )
        except Exception as exc:
            logger.warning("ui.native | pickFiles: {}", exc)
            self._emitter.event.emit({"type": "_upload", "ok": False, "error": str(exc)})
            return
        for f in files:
            self.uploadFile(f)

    @pyqtSlot(str)
    def uploadFile(self, file_url: str) -> None:
        """
        Copia un file locale in data/uploads/<sessione>/ (stesse validazioni
        dell'endpoint web: estensione supportata, dimensione, nome sano) ed
        emette uploadFinished. Il path risultante va incluso nel prossimo
        messaggio: il file_analysis lo riconosce dal testo, come nella
        vecchia UI. Accetta sia file:///… (drag&drop, FileDialog) sia path.
        """
        from PyQt6.QtCore import QUrl
        path = QUrl(file_url).toLocalFile() or file_url
        sid = self._ui_loop.session_id if self._ui_loop else "default"

        def _do() -> None:
            result = save_upload(path, sid)
            if not result.get("ok"):
                logger.warning("ui.native | upload '{}': {}", path, result.get("error"))
            result["type"] = "_upload"
            self._emitter.event.emit(result)

        # I/O su disco fuori dal thread Qt.
        import threading as _th
        _th.Thread(target=_do, daemon=True).start()

    # terminale agentico ---------------------------------------------------

    @pyqtSlot(str)
    def setMode(self, mode: str) -> None:
        if self._ui_loop is not None:
            self._call_async(self._ui_loop.set_mode(mode))

    @pyqtSlot(str)
    def terminalPropose(self, text: str) -> None:
        tb = getattr(self._ui_loop, "_terminal_bridge", None)
        if tb is not None and text.strip():
            self._call_async(tb.propose(text.strip()))

    @pyqtSlot(str)
    def terminalConfirm(self, proposal_id: str) -> None:
        tb = getattr(self._ui_loop, "_terminal_bridge", None)
        if tb is not None:
            self._call_async(tb.confirm(proposal_id))

    @pyqtSlot(str)
    def terminalCancel(self, proposal_id: str) -> None:
        tb = getattr(self._ui_loop, "_terminal_bridge", None)
        if tb is not None:
            self._call_async(tb.cancel(proposal_id))

    @pyqtSlot()
    def terminalReset(self) -> None:
        tb = getattr(self._ui_loop, "_terminal_bridge", None)
        if tb is not None:
            self._call_async(tb.reset())

    @pyqtSlot(str)
    def terminalSwitchModel(self, name: str) -> None:
        tb = getattr(self._ui_loop, "_terminal_bridge", None)
        if tb is not None:
            async def _do() -> None:
                ok = await tb.switch_model(name)
                if ok:
                    self._save_setting("terminal_model", name)
            self._call_async(_do())

    @pyqtSlot()
    def requestTerminalState(self) -> None:
        tb = getattr(self._ui_loop, "_terminal_bridge", None)
        if tb is None:
            return
        async def _do() -> None:
            models = await tb.list_visible_models()
            payload = dict(tb.state_payload())
            payload["type"] = "terminal.state"
            payload["models"] = models
            self._emitter.event.emit(payload)
        self._call_async(_do())
