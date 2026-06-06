"""
tests/test_intent.py
Classificatore di intenti (stadio 4): contratto a due livelli e parsing tollerante.
"""
from types import SimpleNamespace

import pytest

from modules.intent import Intent, IntentClassifier


class _FakeLLM:
    """Risponde con `content`, oppure solleva `exc` se impostata."""
    def __init__(self, content=None, exc=None):
        self._content = content
        self._exc = exc

    async def chat(self, messages, role, *, system=None, options=None, model=None):
        if self._exc is not None:
            raise self._exc
        return SimpleNamespace(content=self._content)


def _clf(content=None, exc=None):
    return IntentClassifier(_FakeLLM(content=content, exc=exc))


class TestClassify:
    async def test_singolo_intento(self):
        res = await _clf("WEB_SEARCH").classify("che tempo fa oggi?", has_file=False)
        assert res == {Intent.WEB_SEARCH}

    async def test_multi_intento(self):
        res = await _clf("WEB_SEARCH, FILE_GLOBAL").classify("q", has_file=True)
        assert res == {Intent.WEB_SEARCH, Intent.FILE_GLOBAL}

    async def test_tollera_la_prosa(self):
        res = await _clf("Direi WEB_SEARCH, perche' e' una notizia.").classify(
            "q", has_file=False)
        assert res == {Intent.WEB_SEARCH}

    async def test_none_esplicito_e_set_vuoto(self):
        res = await _clf("NONE").classify("ciao", has_file=False)
        assert res == set()              # riuscita, nessuno strumento

    async def test_garbage_e_none_fallback(self):
        res = await _clf("boh non saprei").classify("q", has_file=False)
        assert res is None               # fallita -> il chiamante usa il fallback

    async def test_eccezione_llm_e_none(self):
        res = await _clf(exc=RuntimeError("ollama giu'")).classify("q", has_file=True)
        assert res is None

    async def test_query_vuota_e_set_vuoto(self):
        res = await _clf("qualunque").classify("   ", has_file=True)
        assert res == set()


class TestFiltroFile:
    async def test_file_intent_scartato_senza_file(self):
        # FILE_GLOBAL senza documento -> riuscita ma vuota (NON fallback al web).
        res = await _clf("FILE_GLOBAL").classify("riassumi", has_file=False)
        assert res == set()

    async def test_web_tenuto_file_scartato_senza_file(self):
        res = await _clf("WEB_SEARCH, FILE_LOCAL").classify("q", has_file=False)
        assert res == {Intent.WEB_SEARCH}

    async def test_file_tenuto_con_file(self):
        res = await _clf("FILE_STRUCTURAL").classify("elenca i capitoli", has_file=True)
        assert res == {Intent.FILE_STRUCTURAL}
