"""
tests/test_hold_to_talk.py
Push-to-talk da tastiera (Alt) e trascrizione live durante il PTT:
- HoldToTalkFilter: Alt premuto→pttDown, rilasciato→pttUp, auto-repeat
  ignorato, rilascio forzato quando la finestra perde il focus.
- UIBridge._stt_partial: emette stt_partial con il testo parziale.
Headless: nessun display, nessun worker asyncio.
"""
import asyncio

import pytest

pytest.importorskip("PyQt6.QtCore")
from PyQt6.QtCore import QCoreApplication, QEvent, Qt
from PyQt6.QtGui import QKeyEvent

from ui.native_app import HoldToTalkFilter


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


class _FakeBackend:
    def __init__(self):
        self.calls = []
    def pttDown(self): self.calls.append("down")
    def pttUp(self):   self.calls.append("up")


def _key(ev_type, key=Qt.Key.Key_Alt, autorep=False):
    return QKeyEvent(ev_type, key, Qt.KeyboardModifier.NoModifier, "", autorep, 1)


class TestHoldToTalkFilter:
    def test_press_release(self, qapp):
        be = _FakeBackend()
        f = HoldToTalkFilter(be)
        assert f.eventFilter(qapp, _key(QEvent.Type.KeyPress)) is True
        assert f.eventFilter(qapp, _key(QEvent.Type.KeyRelease)) is True
        assert be.calls == ["down", "up"]

    def test_autorepeat_ignorato(self, qapp):
        be = _FakeBackend()
        f = HoldToTalkFilter(be)
        f.eventFilter(qapp, _key(QEvent.Type.KeyPress))
        f.eventFilter(qapp, _key(QEvent.Type.KeyPress, autorep=True))
        f.eventFilter(qapp, _key(QEvent.Type.KeyRelease, autorep=True))
        f.eventFilter(qapp, _key(QEvent.Type.KeyRelease))
        assert be.calls == ["down", "up"]

    def test_release_senza_press_ignorata(self, qapp):
        be = _FakeBackend()
        f = HoldToTalkFilter(be)
        f.eventFilter(qapp, _key(QEvent.Type.KeyRelease))
        assert be.calls == []

    def test_altri_tasti_passano(self, qapp):
        be = _FakeBackend()
        f = HoldToTalkFilter(be)
        assert f.eventFilter(qapp, _key(QEvent.Type.KeyPress, key=Qt.Key.Key_A)) is False
        assert be.calls == []

    def test_deactivate_rilascia(self, qapp):
        be = _FakeBackend()
        f = HoldToTalkFilter(be)
        f.eventFilter(qapp, _key(QEvent.Type.KeyPress))
        f.eventFilter(qapp, QEvent(QEvent.Type.ApplicationDeactivate))
        assert be.calls == ["down", "up"]
        # il KeyRelease tardivo (arrivato dopo il deactivate) non raddoppia
        f.eventFilter(qapp, _key(QEvent.Type.KeyRelease))
        assert be.calls == ["down", "up"]


class _FakeSTTResult:
    def __init__(self, text):
        self.text = text
    def is_empty(self):
        return not self.text.strip()


class _FakeSTT:
    def __init__(self, text):
        self._text = text
        self.kwargs = None
    async def transcribe(self, audio, **kw):
        self.kwargs = kw
        return _FakeSTTResult(self._text)


class TestSttPartial:
    def _bridge(self, text):
        from ui.bridge import UIBridge
        b = UIBridge.__new__(UIBridge)
        b._stt = _FakeSTT(text)
        b.emitted = []
        b._emit = lambda p: b.emitted.append(p)
        return b

    async def test_emette_parziale(self):
        b = self._bridge("ciao mondo")
        await b._stt_partial(b"\x00\x00" * 1600)
        assert b.emitted == [{"type": "stt_partial", "text": "ciao mondo", "final": False}]
        # reattività prima della precisione: beam ridotto
        assert b._stt.kwargs.get("beam_size") == 1

    async def test_vuoto_non_emette(self):
        b = self._bridge("   ")
        await b._stt_partial(b"\x00\x00" * 1600)
        assert b.emitted == []

    async def test_errore_stt_non_solleva(self):
        b = self._bridge("x")
        async def boom(audio, **kw): raise RuntimeError("gpu occupata")
        b._stt.transcribe = boom
        await b._stt_partial(b"\x00\x00" * 1600)
        assert b.emitted == []
