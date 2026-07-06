"""
tests/test_orchestrator_map_reduce.py
Stadio 5b: _run_map_reduce on-demand e il gate conservativo di _run_file_rag.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import core.orchestrator as O
from core.context import AssistantContext, InputMode, ModelRole, OutputMode
from core.orchestrator import Orchestrator
from modules.intent import Intent


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


def _orch_with_engine(chunks, result=None):
    orch = Orchestrator.__new__(Orchestrator)
    orch._model_overrides = {}
    orch._file_rag = MagicMock()
    orch._file_rag.get_ordered_chunks = AsyncMock(return_value=chunks)
    orch._map_reduce = MagicMock()
    orch._map_reduce.run = AsyncMock(return_value=result or _fake_result())
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
