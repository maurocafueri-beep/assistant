#!/usr/bin/env python3
r"""
apply_mapreduce_s5b_test.py — Test dello stadio 5b.

Crea tests/test_orchestrator_map_reduce.py (harness autosufficiente):
_run_map_reduce inietta la sintesi, salta senza intento map-reduce / con None /
senza file, e il posizionale riduce ai primi+ultimi N blocchi; piu' il gate
conservativo di _run_file_rag (salta su {FILE_GLOBAL}, gira su {FILE_LOCAL}/None).

Richiede gli stadi 5a+5b applicati. CREA un file nuovo.

Uso:
    python3 apply_mapreduce_s5b_test.py --check
    python3 apply_mapreduce_s5b_test.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET = Path("tests/test_orchestrator_map_reduce.py")

CONTENT = '''"""
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
'''


def main() -> int:
    ap = argparse.ArgumentParser(description="Test stadio 5b.")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    path = Path(args.root) / TARGET
    if path.exists():
        print(f"\u2717 {TARGET} esiste gia': rifiuto di sovrascrivere.", file=sys.stderr)
        return 1
    orch = Path(args.root) / "core/orchestrator.py"
    if orch.is_file() and "_run_map_reduce" not in orch.read_text(encoding="utf-8"):
        print("\u2717 manca _run_map_reduce: applica prima apply_mapreduce_s5b.py.",
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
    print("    SKIP_SLOW=1 make test       # +11 test verdi")
    print("    git add tests/test_orchestrator_map_reduce.py")
    print("    git status")
    print('    git commit -m "test(map_reduce): stadio 5b — _run_map_reduce e gate di _run_file_rag"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
