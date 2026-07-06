"""
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
    """Motore finto: risponde in base alla domanda, registra domande e modello."""
    def __init__(self):
        self.questions = []
        self.models = []

    async def run(self, *, question, blocks, model=None):
        self.questions.append(question)
        self.models.append(model)
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
        # il modello attivo arriva a OGNI chiamata del motore (vincolo VRAM)
        assert eng.models == ["gemma4", "gemma4"]

    async def test_precompute_then_store(self, tmp_path):
        eng = _FakeEngine()
        pre = await precompute(eng, [{"text": "x"}], model="m")
        store = PrecomputeStore(cache_dir=tmp_path)
        store.save("fid", pre)
        assert store.load("fid").chapters == "CAPITOLI"
