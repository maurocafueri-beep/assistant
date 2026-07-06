#!/usr/bin/env python3
r"""
apply_mapreduce_s3_test.py — Test del precalcolo (stadio 3).

Crea tests/test_precompute.py: store su cartella temporanea (round-trip, load
mancante, has, drop) e funzione precompute con un motore finto (gira due domande,
impacchetta i risultati). Niente Ollama, niente disco reale.

Richiede apply_mapreduce_s3.py gia' applicato. CREA un file nuovo.

Uso:
    python3 apply_mapreduce_s3_test.py --check
    python3 apply_mapreduce_s3_test.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET = Path("tests/test_precompute.py")

CONTENT = '''"""
tests/test_precompute.py
Precalcolo (stadio 3): store JSON su disco + funzione precompute.
"""
from types import SimpleNamespace

import pytest

from modules.map_reduce import (
    CHAPTERS_QUESTION,
    Precomputed,
    PrecomputeStore,
    SUMMARY_QUESTION,
    precompute,
)


class _FakeEngine:
    """Motore finto: risponde in base alla domanda, registra le domande viste."""
    def __init__(self):
        self.questions = []

    async def run(self, *, question, blocks):
        self.questions.append(question)
        text = "RIASSUNTO" if question == SUMMARY_QUESTION else "CAPITOLI"
        return SimpleNamespace(content=text, n_blocks=len(blocks),
                               n_partials=1, n_llm_calls=1)


class TestPrecomputeStore:
    def test_round_trip(self, tmp_path):
        store = PrecomputeStore(cache_dir=tmp_path)
        pre = Precomputed(summary="S", chapters="C", computed_at=123.0, model="gemma4")
        store.save("fid1", pre)
        loaded = store.load("fid1")
        assert loaded == pre

    def test_load_missing_is_none(self, tmp_path):
        assert PrecomputeStore(cache_dir=tmp_path).load("inesistente") is None

    def test_has_and_drop(self, tmp_path):
        store = PrecomputeStore(cache_dir=tmp_path)
        assert store.has("fid") is False
        store.save("fid", Precomputed("s", "c", 1.0, "m"))
        assert store.has("fid") is True
        assert store.drop("fid") is True
        assert store.has("fid") is False
        assert store.drop("fid") is False     # gia' rimosso

    def test_unicode_preservato(self, tmp_path):
        store = PrecomputeStore(cache_dir=tmp_path)
        store.save("u", Precomputed("àèìòù — capitolo «uno»", "c", 1.0, "m"))
        assert store.load("u").summary == "àèìòù — capitolo «uno»"


class TestPrecompute:
    async def test_runs_both_questions(self):
        eng = _FakeEngine()
        pre = await precompute(eng, [{"text": "x"}], model="gemma4")
        assert pre.summary == "RIASSUNTO"
        assert pre.chapters == "CAPITOLI"
        assert pre.model == "gemma4"
        assert pre.computed_at > 0
        assert eng.questions == [SUMMARY_QUESTION, CHAPTERS_QUESTION]

    async def test_precompute_then_store(self, tmp_path):
        eng = _FakeEngine()
        pre = await precompute(eng, [{"text": "x"}], model="m")
        store = PrecomputeStore(cache_dir=tmp_path)
        store.save("fid", pre)
        assert store.load("fid").chapters == "CAPITOLI"
'''


def main() -> int:
    ap = argparse.ArgumentParser(description="Test precalcolo (stadio 3).")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    path = Path(args.root) / TARGET
    if path.exists():
        print(f"\u2717 {TARGET} esiste gia': rifiuto di sovrascrivere.", file=sys.stderr)
        return 1
    base = Path(args.root) / "modules/map_reduce/precompute.py"
    if not base.is_file():
        print("\u2717 manca precompute.py: applica prima apply_mapreduce_s3.py.", file=sys.stderr)
        return 1

    print(f"  \u2713 {TARGET} pronto da creare.")
    if args.check:
        print("\n--check OK: nessuna scrittura.")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONTENT, encoding="utf-8")
    print(f"\n\u2713 Creato {TARGET}.")
    print("  Prossimi passi:")
    print("    SKIP_SLOW=1 make test       # +6 test verdi")
    print("    git add tests/test_precompute.py")
    print("    git status")
    print('    git commit -m "test(map_reduce): stadio 3 — store JSON e precompute"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
