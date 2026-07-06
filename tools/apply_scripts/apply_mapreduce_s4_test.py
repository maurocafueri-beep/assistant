#!/usr/bin/env python3
r"""
apply_mapreduce_s4_test.py — Test del classificatore di intenti (stadio 4).

Crea tests/test_intent.py: LLM finto pilotato, verifica singolo/multi intento,
tolleranza alla prosa, NONE -> set vuoto, garbage -> None, il filtro file-assente
(toglie FILE_* ma TIENE WEB_SEARCH), file tenuto quando c'e', eccezione -> None,
query vuota -> set vuoto. Niente Ollama.

Richiede apply_mapreduce_s4.py gia' applicato. CREA un file nuovo.

Uso:
    python3 apply_mapreduce_s4_test.py --check
    python3 apply_mapreduce_s4_test.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET = Path("tests/test_intent.py")

CONTENT = '''"""
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
'''


def main() -> int:
    ap = argparse.ArgumentParser(description="Test classificatore intenti (stadio 4).")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    path = Path(args.root) / TARGET
    if path.exists():
        print(f"\u2717 {TARGET} esiste gia': rifiuto di sovrascrivere.", file=sys.stderr)
        return 1
    base = Path(args.root) / "modules/intent/base_intent.py"
    if not base.is_file():
        print("\u2717 manca il classificatore: applica prima apply_mapreduce_s4.py.",
              file=sys.stderr)
        return 1

    print(f"  \u2713 {TARGET} pronto da creare.")
    if args.check:
        print("\n--check OK: nessuna scrittura.")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONTENT, encoding="utf-8")
    print(f"\n\u2713 Creato {TARGET}.")
    print("  Prossimi passi:")
    print("    SKIP_SLOW=1 make test       # +10 test verdi")
    print("    git add tests/test_intent.py")
    print("    git status")
    print('    git commit -m "test(intent): stadio 4 — contratto a due livelli e parsing"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
