"""
tests/test_native_backend.py
Backend Qt della UI nativa: traduzione payload bridge → segnali QML e
QtEmitter come sostituto del WSManager. Headless (QCoreApplication, niente
display); il worker asyncio NON viene avviato.
"""
import asyncio

import pytest

pytest.importorskip("PyQt6.QtCore")
from PyQt6.QtCore import QCoreApplication

from ui.native_backend import Backend, QtEmitter


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


@pytest.fixture()
def backend(qapp):
    return Backend()


def _capture(signal):
    received = []
    signal.connect(lambda *a: received.append(a))
    return received


class TestEventRouting:
    def test_init(self, backend):
        got = _capture(backend.initReady)
        payload = {"type": "init", "state": "idle", "model": "m"}
        backend._on_event(payload)
        assert got and got[0][0]["model"] == "m"

    def test_state(self, backend):
        got = _capture(backend.stateChanged)
        backend._on_event({"type": "state", "value": "thinking"})
        assert got == [("thinking",)]

    def test_chunk(self, backend):
        got = _capture(backend.chunkReceived)
        backend._on_event({"type": "chunk", "text": "ciao"})
        assert got == [("ciao",)]

    def test_user(self, backend):
        got = _capture(backend.userMessage)
        backend._on_event({"type": "user", "text": "domanda", "session": "s1"})
        assert got == [("domanda",)]

    def test_sessions(self, backend):
        got = _capture(backend.sessionsChanged)
        backend._on_event({"type": "sessions", "sessions": [{"id": "s1"}]})
        assert got and got[0][0] == [{"id": "s1"}]

    def test_session_switch(self, backend):
        got = _capture(backend.sessionSwitched)
        backend._on_event({"type": "session_switch", "session": "s2", "messages": []})
        assert got and got[0][0]["session"] == "s2"

    def test_model_personality_voice_tts(self, backend):
        m = _capture(backend.modelChanged)
        p = _capture(backend.personalityChanged)
        v = _capture(backend.voiceChanged)
        t = _capture(backend.ttsChanged)
        backend._on_event({"type": "model", "name": "gemma"})
        backend._on_event({"type": "personality", "name": "dev", "display_name": "Dev"})
        backend._on_event({"type": "voice", "name": "cloe"})
        backend._on_event({"type": "tts", "enabled": False})
        assert m == [("gemma",)]
        assert p == [("dev", "Dev")]
        assert v == [("cloe",)]
        assert t == [(False,)]

    def test_stats(self, backend):
        got = _capture(backend.statsChanged)
        backend._on_event({"type": "stats", "latency": {"llm_ms": 5}})
        assert got and got[0][0]["latency"]["llm_ms"] == 5

    def test_models_interni(self, backend):
        got = _capture(backend.modelsListed)
        backend._on_event({"type": "_models", "models": ["a", "b"]})
        assert got == [(["a", "b"],)]

    def test_tipo_sconosciuto_ignorato(self, backend):
        # Nessuna eccezione e nessun segnale: compat con estensioni future.
        got = _capture(backend.initReady)
        backend._on_event({"type": "qualcosa_di_nuovo", "x": 1})
        assert got == []


class TestQtEmitter:
    async def test_broadcast_emette_event(self, qapp):
        em = QtEmitter()
        got = _capture(em.event)
        await em.broadcast({"type": "state", "value": "idle"})
        assert got == [({"type": "state", "value": "idle"},)]

    def test_duck_type_wsmanager(self, qapp):
        # L'interfaccia attesa da UIBridge: broadcast, connect, disconnect,
        # n_clients. La presenza è garantita qui, la semantica dal bridge.
        em = QtEmitter()
        assert em.n_clients == 1
        assert asyncio.iscoroutinefunction(em.broadcast)
        assert asyncio.iscoroutinefunction(em.connect)


class TestSlotsSenzaWorker:
    """Gli slot chiamati prima dell'avvio del worker non devono esplodere."""

    def test_send_text_senza_loop(self, backend):
        backend.sendText("ciao")            # nessuna eccezione

    def test_call_on_loop_senza_bridge(self, backend):
        backend.cancelTurn()                # nessuna eccezione
        backend.switchModel("x")
        backend.pttDown(); backend.pttUp()

    def test_slot_terminale_senza_bridge(self, backend):
        backend.terminalPropose("ls")
        backend.terminalConfirm("p1")
        backend.terminalCancel("p1")
        backend.terminalReset()
        backend.requestTerminalState()


class TestErrorAndTerminalRouting:
    def test_error_va_al_pannello(self, backend):
        got = _capture(backend.errorOccurred)
        backend._on_event({"type": "error", "source": "turn", "message": "boom"})
        assert got and got[0][0]["message"] == "boom"

    def test_terminal_event_inoltrato(self, backend):
        got = _capture(backend.terminalEvent)
        backend._on_event({"type": "terminal.proposal", "proposal": {"proposal_id": "p1"}})
        assert got and got[0][0]["type"] == "terminal.proposal"

    def test_terminal_error_anche_nel_pannello(self, backend):
        term = _capture(backend.terminalEvent)
        errs = _capture(backend.errorOccurred)
        backend._on_event({"type": "terminal.error", "message": "comando fallito"})
        assert term and errs
        assert errs[0][0]["source"] == "terminal"

    def test_upload_routing(self, backend):
        got = _capture(backend.uploadFinished)
        backend._on_event({"type": "_upload", "ok": True, "path": "/x", "name": "x"})
        assert got and got[0][0]["ok"] is True


class TestSaveUpload:
    def test_upload_valido(self, tmp_path, monkeypatch):
        import ui.server as server_mod
        from ui.native_backend import save_upload
        monkeypatch.setattr(server_mod, "_session_dir_for",
                            lambda sid: tmp_path / "uploads" / sid)
        src = tmp_path / "doc.txt"
        src.write_text("contenuto")
        r = save_upload(str(src), "sess1")
        assert r["ok"] is True
        assert r["name"] == "doc.txt"
        assert "sess1" in r["path"]

    def test_estensione_non_supportata(self, tmp_path):
        from ui.native_backend import save_upload
        src = tmp_path / "malware.xyz"
        src.write_text("x")
        r = save_upload(str(src), "sess1")
        assert r["ok"] is False
        assert "estensione" in r["error"]

    def test_file_inesistente(self, tmp_path):
        from ui.native_backend import save_upload
        r = save_upload(str(tmp_path / "manca.pdf"), "sess1")
        assert r["ok"] is False
        assert "non trovato" in r["error"]

    def test_file_troppo_grande(self, tmp_path, monkeypatch):
        from config.settings import settings
        from ui.native_backend import save_upload
        monkeypatch.setattr(settings.file_analysis, "max_file_bytes", 4)
        src = tmp_path / "big.txt"
        src.write_text("ben più di quattro byte")
        r = save_upload(str(src), "sess1")
        assert r["ok"] is False
        assert "troppo grande" in r["error"]
