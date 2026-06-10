"""
tests/test_orchestrator_web_search.py
Test suite per l'integrazione web_search in core/orchestrator.py.

Copre gli helper puri (_should_search, _clean_query, _format_search_block)
e il metodo _run_web_search con mock completi — nessuna rete reale.

asyncio_mode = "auto" → @pytest.mark.asyncio non necessario.

Esecuzione:
    venv-runtime/bin/pytest tests/test_orchestrator_web_search.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.context import AssistantContext, InputMode, ModelRole, OutputMode
from modules.intent import Intent
from core.orchestrator import (
    Orchestrator,
    OrchestratorStatus,
    _clean_query,
    _format_search_block,
    _should_search,
)
from modules.web_search import SearchResponse, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ctx(**kwargs) -> AssistantContext:
    defaults = dict(
        user_text="Ciao, come stai?",
        session_id="sess-test",
        input_mode=InputMode.TEXT,
        output_mode=OutputMode.TEXT,
        model_role=ModelRole.CHAT,
        system_prompt="Sei un assistente utile.",
        personality_name="dev",
    )
    defaults.update(kwargs)
    return AssistantContext(**defaults)


def _sample_results(n: int = 2) -> list[SearchResult]:
    return [
        SearchResult(
            title=f"Risultato {i}",
            url=f"https://example.com/{i}",
            snippet=f"Snippet del risultato {i}.",
            engine="google",
        )
        for i in range(1, n + 1)
    ]


def _make_ws_mock(resp: SearchResponse) -> MagicMock:
    ws = MagicMock()
    ws.search = AsyncMock(return_value=resp)
    ws.aclose = AsyncMock()
    return ws


def _make_personality_mock(allows: bool = True) -> MagicMock:
    pm = MagicMock()
    active = MagicMock()
    active.allows_tool = MagicMock(return_value=allows)
    pm.active = active
    return pm


# ---------------------------------------------------------------------------
# _should_search
# ---------------------------------------------------------------------------

class TestShouldSearch:
    def test_empty_string(self):
        assert _should_search("") is False

    def test_no_trigger(self):
        assert _should_search("Spiegami come funziona asyncio") is False

    def test_chitchat(self):
        assert _should_search("Ciao, come stai oggi?") is False

    @pytest.mark.parametrize("text", [
        "Cerca online le novità su Python",
        "Fai una ricerca sui modelli LLM",
        "Cosa dicono del nuovo iPhone?",
        "ultime notizie intelligenza artificiale",
        "Search the web for asyncio tutorials",
        "Guarda su internet il meteo di domani",
    ])
    def test_triggers_detected(self, text):
        assert _should_search(text) is True

    def test_case_insensitive(self):
        assert _should_search("CERCA ONLINE qualcosa") is True


# ---------------------------------------------------------------------------
# _clean_query
# ---------------------------------------------------------------------------

class TestCleanQuery:
    def test_removes_trigger(self):
        q = _clean_query("Cerca online i modelli LLM open source")
        assert "cerca online" not in q.lower()
        assert "modelli LLM open source" in q

    def test_normalizes_whitespace(self):
        q = _clean_query("Cerca online   i   migliori   editor")
        assert "  " not in q

    def test_strips_punctuation(self):
        q = _clean_query("Cosa dicono del nuovo iPhone?")
        assert not q.endswith("?")
        assert "nuovo iPhone" in q

    def test_fallback_when_empty(self):
        # solo trigger, niente contenuto → ritorna testo originale strippato
        q = _clean_query("ultime notizie")
        assert q  # non vuoto


# ---------------------------------------------------------------------------
# _format_search_block
# ---------------------------------------------------------------------------

class TestFormatSearchBlock:
    def test_empty_returns_empty(self):
        assert _format_search_block([]) == ""

    def test_contains_titles_and_urls(self):
        block = _format_search_block(_sample_results(2))
        assert "Risultato 1" in block
        assert "https://example.com/1" in block
        assert "Risultato 2" in block

    def test_has_header_and_footer(self):
        block = _format_search_block(_sample_results(1))
        assert "RISULTATI RICERCA WEB" in block
        assert block.strip().endswith("---")


# ---------------------------------------------------------------------------
# OrchestratorStatus — campo web_search
# ---------------------------------------------------------------------------

class TestOrchestratorStatusWebSearch:
    def test_default_false(self):
        s = OrchestratorStatus()
        assert s.web_search_ok is False

    def test_in_log_dict(self):
        s = OrchestratorStatus(web_search_ok=True)
        assert s.to_log_dict()["web_search"] is True


# ---------------------------------------------------------------------------
# Orchestrator._run_web_search
# ---------------------------------------------------------------------------

class TestRunWebSearch:
    async def test_skip_when_client_none(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._web_search = None
        ctx = make_ctx(user_text="Cerca online qualcosa")
        await orch._run_web_search(ctx)
        assert ctx.tool_calls == []

    async def test_skip_when_no_trigger(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._web_search = _make_ws_mock(SearchResponse(query="x"))
        orch._personality = _make_personality_mock(allows=True)
        ctx = make_ctx(user_text="Spiegami i decoratori Python")
        await orch._run_web_search(ctx)
        orch._web_search.search.assert_not_called()
        assert ctx.tool_calls == []

    async def test_skip_when_tool_not_allowed(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._web_search = _make_ws_mock(SearchResponse(query="x"))
        orch._personality = _make_personality_mock(allows=False)
        ctx = make_ctx(user_text="Cerca online le ultime news")
        await orch._run_web_search(ctx)
        orch._web_search.search.assert_not_called()
        assert ctx.tool_calls == []

    async def test_injects_results_into_system_prompt(self):
        resp = SearchResponse(
            query="modelli llm",
            results=_sample_results(2),
            elapsed_ms=42.0,
            total_found=2,
        )
        orch = Orchestrator.__new__(Orchestrator)
        orch._web_search = _make_ws_mock(resp)
        orch._personality = _make_personality_mock(allows=True)

        ctx = make_ctx(user_text="Cerca online i modelli LLM open source")
        prompt_before = ctx.system_prompt
        await orch._run_web_search(ctx)

        assert len(ctx.system_prompt) > len(prompt_before)
        assert "RISULTATI RICERCA WEB" in ctx.system_prompt
        assert "https://example.com/1" in ctx.system_prompt

    async def test_registers_tool_call(self):
        resp = SearchResponse(
            query="x", results=_sample_results(3), total_found=3
        )
        orch = Orchestrator.__new__(Orchestrator)
        orch._web_search = _make_ws_mock(resp)
        orch._personality = _make_personality_mock(allows=True)

        ctx = make_ctx(user_text="Fai una ricerca sui database vettoriali")
        await orch._run_web_search(ctx)

        assert len(ctx.tool_calls) == 1
        assert ctx.tool_calls[0]["tool"] == "web_search"
        assert len(ctx.tool_results) == 1
        assert "web_search" in ctx.timings

    async def test_nonfatal_on_search_exception(self):
        orch = Orchestrator.__new__(Orchestrator)
        ws = MagicMock()
        ws.search = AsyncMock(side_effect=RuntimeError("boom"))
        orch._web_search = ws
        orch._personality = _make_personality_mock(allows=True)

        ctx = make_ctx(user_text="Cerca online qualcosa")
        # non deve sollevare eccezioni
        await orch._run_web_search(ctx)
        assert ctx.tool_calls == []
        assert ctx.error is None

    async def test_nonfatal_on_searxng_error_field(self):
        resp = SearchResponse(query="x", error="timeout")
        orch = Orchestrator.__new__(Orchestrator)
        orch._web_search = _make_ws_mock(resp)
        orch._personality = _make_personality_mock(allows=True)

        ctx = make_ctx(user_text="Cerca online news")
        await orch._run_web_search(ctx)
        assert ctx.tool_calls == []
        assert ctx.error is None

    async def test_empty_results_no_injection(self):
        resp = SearchResponse(query="x", results=[], total_found=0)
        orch = Orchestrator.__new__(Orchestrator)
        orch._web_search = _make_ws_mock(resp)
        orch._personality = _make_personality_mock(allows=True)

        ctx = make_ctx(user_text="Cerca online xyzqwerty123")
        prompt_before = ctx.system_prompt
        await orch._run_web_search(ctx)

        assert ctx.system_prompt == prompt_before
        assert ctx.tool_calls == []


class TestRunWebSearchIntents:
    """Gate del web pilotato dagli intenti, con fallback al keyword su None."""

    async def test_intento_web_attiva_senza_keyword(self):
        resp = SearchResponse(query="x", results=_sample_results(1), total_found=1)
        orch = Orchestrator.__new__(Orchestrator)
        orch._web_search = _make_ws_mock(resp)
        orch._personality = _make_personality_mock(allows=True)
        # Nessuna keyword nel testo, ma c'e' l'intento WEB_SEARCH.
        ctx = make_ctx(user_text="Spiegami i decoratori Python",
                       metadata={"intents": {Intent.WEB_SEARCH}})
        await orch._run_web_search(ctx)
        orch._web_search.search.assert_called_once()

    async def test_set_vuoto_non_cerca_nonostante_keyword(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._web_search = _make_ws_mock(SearchResponse(query="x"))
        orch._personality = _make_personality_mock(allows=True)
        # Keyword presente, ma il classificatore ha detto "niente strumenti".
        ctx = make_ctx(user_text="Cerca online le ultime news",
                       metadata={"intents": set()})
        await orch._run_web_search(ctx)
        orch._web_search.search.assert_not_called()

    async def test_none_fallback_keyword_attiva(self):
        resp = SearchResponse(query="x", results=_sample_results(1), total_found=1)
        orch = Orchestrator.__new__(Orchestrator)
        orch._web_search = _make_ws_mock(resp)
        orch._personality = _make_personality_mock(allows=True)
        ctx = make_ctx(user_text="Cerca online qualcosa",
                       metadata={"intents": None})
        await orch._run_web_search(ctx)
        orch._web_search.search.assert_called_once()

    async def test_none_fallback_keyword_inattiva(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._web_search = _make_ws_mock(SearchResponse(query="x"))
        orch._personality = _make_personality_mock(allows=True)
        ctx = make_ctx(user_text="Spiegami asyncio",
                       metadata={"intents": None})
        await orch._run_web_search(ctx)
        orch._web_search.search.assert_not_called()


class TestClassifyIntents:
    """_classify_intents popola ctx.metadata['intents'] e calcola has_file."""

    async def test_popola_metadata_has_file_false(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._session_rag_files = {}
        clf = MagicMock()
        clf.classify = AsyncMock(return_value={Intent.WEB_SEARCH})
        orch._intent_classifier = clf
        ctx = make_ctx(user_text="q", session_id="s1")
        await orch._classify_intents(ctx)
        assert ctx.metadata["intents"] == {Intent.WEB_SEARCH}
        assert clf.classify.call_args.kwargs["has_file"] is False

    async def test_has_file_true_con_rag_files(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._session_rag_files = {"s1": [{"file_id": "x"}]}
        clf = MagicMock()
        clf.classify = AsyncMock(return_value=set())
        orch._intent_classifier = clf
        ctx = make_ctx(user_text="q", session_id="s1")
        await orch._classify_intents(ctx)
        assert clf.classify.call_args.kwargs["has_file"] is True

    async def test_senza_classificatore_parcheggia_none(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._intent_classifier = None
        ctx = make_ctx(user_text="q", session_id="s1")
        await orch._classify_intents(ctx)
        assert ctx.metadata["intents"] is None

