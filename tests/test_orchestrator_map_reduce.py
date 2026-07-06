"""
tests/test_orchestrator_map_reduce.py
Stadio 5b: _run_map_reduce on-demand e il gate conservativo di _run_file_rag.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import asyncio

import core.orchestrator as O
from core.context import AssistantContext, InputMode, ModelRole, OutputMode
from core.orchestrator import Orchestrator
from modules.intent import Intent
from modules.map_reduce import Precomputed, PrecomputeStore


def _ctx(**kw) -> AssistantContext:
    d = dict(
        user_text="riassumi il libro",
        session_id="s1",
        input_mode=InputMode.TEXT,
        output_mode=OutputMode.TEXT,
        model_role=ModelRole.CHAT,
        system_prompt="Sei un assistente.",
        personality_name="dev",
    )
    d.update(kw)
    return AssistantContext(**d)


def _chunk(i):
    return {"text": f"testo {i}", "page_start": i, "page_end": i, "index": i}


def _fake_result(content="SINTESI GLOBALE"):
    return SimpleNamespace(
        content=content,
        to_log_dict=lambda: {"n_blocks": 1, "n_partials": 1, "n_llm_calls": 2},
    )


def _orch_with_engine(chunks, result=None, store=None):
    orch = Orchestrator.__new__(Orchestrator)
    orch._model_overrides = {}
    orch._file_rag = MagicMock()
    orch._file_rag.get_ordered_chunks = AsyncMock(return_value=chunks)
    orch._map_reduce = MagicMock()
    orch._map_reduce.run = AsyncMock(return_value=result or _fake_result())
    orch._precompute_store = store   # None = cache 5c non disponibile
    return orch


_FILES = [{"file_id": "abc", "source": "libro.pdf"}]


class TestRunMapReduce:
    async def test_inietta_sintesi(self):
        orch = _orch_with_engine([_chunk(0), _chunk(1)])
        ctx = _ctx(metadata={"intents": {Intent.FILE_GLOBAL}, "rag_files": _FILES})
        before = ctx.system_prompt
        await orch._run_map_reduce(ctx)
        assert len(ctx.system_prompt) > len(before)
        assert "SINTESI GLOBALE" in ctx.system_prompt
        # la query reale e' passata come domanda
        assert orch._map_reduce.run.call_args.kwargs["question"] == "riassumi il libro"

    async def test_skip_senza_intento_mapreduce(self):
        orch = _orch_with_engine([_chunk(0)])
        ctx = _ctx(metadata={"intents": {Intent.FILE_LOCAL}, "rag_files": _FILES})
        await orch._run_map_reduce(ctx)
        orch._map_reduce.run.assert_not_called()

    async def test_skip_su_intents_none(self):
        orch = _orch_with_engine([_chunk(0)])
        ctx = _ctx(metadata={"intents": None, "rag_files": _FILES})
        await orch._run_map_reduce(ctx)
        orch._map_reduce.run.assert_not_called()

    async def test_skip_senza_file(self):
        orch = _orch_with_engine([_chunk(0)])
        ctx = _ctx(metadata={"intents": {Intent.FILE_GLOBAL}, "rag_files": []})
        await orch._run_map_reduce(ctx)
        orch._map_reduce.run.assert_not_called()

    async def test_posizionale_riduce_blocchi(self, monkeypatch):
        # Budget minuscolo -> ogni chunk un blocco: 6 blocchi -> primi 2 + ultimi 2.
        monkeypatch.setattr(O, "_MAPREDUCE_BLOCK_CHARS", 5)
        chunks = [_chunk(i) for i in range(6)]
        orch = _orch_with_engine(chunks)
        ctx = _ctx(metadata={"intents": {Intent.FILE_POSITIONAL}, "rag_files": _FILES})
        await orch._run_map_reduce(ctx)
        blocks = orch._map_reduce.run.call_args.kwargs["blocks"]
        assert len(blocks) == 4          # 2 primi + 2 ultimi


class TestFileRagGate:
    def _orch_file_rag(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._file_rag = MagicMock()
        orch._file_rag.search_file = AsyncMock(return_value=[])
        orch._personality = MagicMock()
        return orch

    async def test_salta_su_global_senza_locale(self):
        orch = self._orch_file_rag()
        ctx = _ctx(metadata={"intents": {Intent.FILE_GLOBAL}, "rag_files": _FILES})
        await orch._run_file_rag(ctx)
        orch._file_rag.search_file.assert_not_called()

    async def test_gira_su_locale(self):
        orch = self._orch_file_rag()
        ctx = _ctx(metadata={"intents": {Intent.FILE_LOCAL}, "rag_files": _FILES})
        await orch._run_file_rag(ctx)
        orch._file_rag.search_file.assert_called()

    async def test_gira_su_global_piu_locale(self):
        orch = self._orch_file_rag()
        ctx = _ctx(metadata={"intents": {Intent.FILE_GLOBAL, Intent.FILE_LOCAL},
                             "rag_files": _FILES})
        await orch._run_file_rag(ctx)
        orch._file_rag.search_file.assert_called()

    async def test_gira_su_none(self):
        orch = self._orch_file_rag()
        ctx = _ctx(metadata={"intents": None, "rag_files": _FILES})
        await orch._run_file_rag(ctx)
        orch._file_rag.search_file.assert_called()

    async def test_gira_su_vuoto(self):
        orch = self._orch_file_rag()
        ctx = _ctx(metadata={"intents": set(), "rag_files": _FILES})
        await orch._run_file_rag(ctx)
        orch._file_rag.search_file.assert_called()


def _pre(summary="RIASSUNTO PRECALCOLATO", chapters="CAPITOLI PRECALCOLATI"):
    return Precomputed(summary=summary, chapters=chapters,
                       computed_at=0.0, model="m")


def _store_with(pre):
    store = MagicMock()
    store.load = MagicMock(return_value=pre)
    return store


class TestPrecomputeFirst:
    """Stadio 5c: _run_map_reduce serve dal PrecomputeStore quando puo'."""

    async def test_global_da_cache_salta_motore(self):
        orch = _orch_with_engine([_chunk(0)], store=_store_with(_pre()))
        ctx = _ctx(metadata={"intents": {Intent.FILE_GLOBAL}, "rag_files": _FILES})
        await orch._run_map_reduce(ctx)
        orch._map_reduce.run.assert_not_called()
        assert "RIASSUNTO PRECALCOLATO" in ctx.system_prompt
        assert "CAPITOLI" not in ctx.system_prompt   # solo la parte richiesta

    async def test_strutturale_da_cache(self):
        orch = _orch_with_engine([_chunk(0)], store=_store_with(_pre()))
        ctx = _ctx(metadata={"intents": {Intent.FILE_STRUCTURAL}, "rag_files": _FILES})
        await orch._run_map_reduce(ctx)
        orch._map_reduce.run.assert_not_called()
        assert "CAPITOLI PRECALCOLATI" in ctx.system_prompt
        assert "RIASSUNTO" not in ctx.system_prompt

    async def test_posizionale_ignora_cache(self):
        store = _store_with(_pre())
        orch = _orch_with_engine([_chunk(0)], store=store)
        ctx = _ctx(metadata={"intents": {Intent.FILE_POSITIONAL}, "rag_files": _FILES})
        await orch._run_map_reduce(ctx)
        store.load.assert_not_called()
        orch._map_reduce.run.assert_called_once()

    async def test_cache_mancante_fallback_motore(self):
        orch = _orch_with_engine([_chunk(0)], store=_store_with(None))
        ctx = _ctx(metadata={"intents": {Intent.FILE_GLOBAL}, "rag_files": _FILES})
        await orch._run_map_reduce(ctx)
        orch._map_reduce.run.assert_called_once()

    async def test_cache_parziale_fallback_motore(self):
        # Due file, il secondo senza cache: on-demand per tutti.
        store = MagicMock()
        store.load = MagicMock(side_effect=[_pre(), None])
        orch = _orch_with_engine([_chunk(0)], store=store)
        files = [{"file_id": "a", "source": "a.pdf"},
                 {"file_id": "b", "source": "b.pdf"}]
        ctx = _ctx(metadata={"intents": {Intent.FILE_GLOBAL}, "rag_files": files})
        await orch._run_map_reduce(ctx)
        orch._map_reduce.run.assert_called_once()

    async def test_parte_richiesta_vuota_fallback_motore(self):
        # In cache ma col campo richiesto vuoto (es. summary=""): motore.
        orch = _orch_with_engine(
            [_chunk(0)], store=_store_with(_pre(summary="")),
        )
        ctx = _ctx(metadata={"intents": {Intent.FILE_GLOBAL}, "rag_files": _FILES})
        await orch._run_map_reduce(ctx)
        orch._map_reduce.run.assert_called_once()


class TestSpawnPrecompute:
    """_spawn_precompute: precalcolo in background all'indicizzazione."""

    def _orch(self, tmp_path, chunks=None):
        orch = _orch_with_engine(
            chunks or [_chunk(0)],
            store=PrecomputeStore(cache_dir=tmp_path),
        )
        return orch

    async def test_scrive_cache_in_background(self, tmp_path):
        orch = self._orch(tmp_path)
        ctx = _ctx()
        orch._bg_tasks = set()
        orch._spawn_precompute("abc", "libro.pdf", ctx)
        assert orch._bg_tasks
        await asyncio.gather(*orch._bg_tasks)
        assert orch._precompute_store.has("abc")
        pre = orch._precompute_store.load("abc")
        assert pre.summary == "SINTESI GLOBALE"
        # il motore ha girato per riassunto E capitoli, col modello del turno
        assert orch._map_reduce.run.call_count == 2
        assert orch._map_reduce.run.call_args.kwargs["model"]

    async def test_salta_se_gia_in_cache(self, tmp_path):
        orch = self._orch(tmp_path)
        orch._precompute_store.save("abc", _pre())
        orch._bg_tasks = set()
        orch._spawn_precompute("abc", "libro.pdf", _ctx())
        assert not orch._bg_tasks

    async def test_salta_senza_motore(self, tmp_path):
        orch = self._orch(tmp_path)
        orch._map_reduce = None
        orch._bg_tasks = set()
        orch._spawn_precompute("abc", "libro.pdf", _ctx())
        assert not orch._bg_tasks

    async def test_flag_disattivata_niente_task(self, tmp_path, monkeypatch):
        from config.settings import settings
        monkeypatch.setattr(
            settings.file_analysis, "precompute_on_index", False,
        )
        orch = self._orch(tmp_path)
        orch._bg_tasks = set()
        orch._spawn_precompute("abc", "libro.pdf", _ctx())
        assert not orch._bg_tasks

    async def test_errore_nel_precalcolo_non_solleva(self, tmp_path):
        orch = self._orch(tmp_path)
        orch._file_rag.get_ordered_chunks = AsyncMock(side_effect=RuntimeError("boom"))
        orch._bg_tasks = set()
        orch._spawn_precompute("abc", "libro.pdf", _ctx())
        await asyncio.gather(*orch._bg_tasks)   # non deve propagare
        assert not orch._precompute_store.has("abc")
