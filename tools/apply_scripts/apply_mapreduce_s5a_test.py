#!/usr/bin/env python3
r"""
apply_mapreduce_s5a_test.py — Test dello stadio 5a.

Estende tests/test_orchestrator_web_search.py: gating del web via intento
(WEB_SEARCH attiva senza keyword; set vuoto non cerca nonostante la keyword;
None ricade sul keyword), e _classify_intents (popola metadata, has_file dai
rag_files, None senza classificatore). I test esistenti restano verdi: senza
intents in metadata, il gate ricade sul keyword come prima.

Due modifiche a tests/test_orchestrator_web_search.py. CRITICO: nessuna rete.

Uso:
    python3 apply_mapreduce_s5a_test.py --check
    python3 apply_mapreduce_s5a_test.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET = "tests/test_orchestrator_web_search.py"

NEW_TESTS = '''


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
'''

EDITS: list[tuple[str, str, str]] = [
    (
        "import Intent",
        r'''from core.context import AssistantContext, InputMode, ModelRole, OutputMode''',
        r'''from core.context import AssistantContext, InputMode, ModelRole, OutputMode
from modules.intent import Intent''',
    ),
    (
        "classi di test in coda",
        r'''        ctx = make_ctx(user_text="Cerca online xyzqwerty123")
        prompt_before = ctx.system_prompt
        await orch._run_web_search(ctx)

        assert ctx.system_prompt == prompt_before
        assert ctx.tool_calls == []''',
        r'''        ctx = make_ctx(user_text="Cerca online xyzqwerty123")
        prompt_before = ctx.system_prompt
        await orch._run_web_search(ctx)

        assert ctx.system_prompt == prompt_before
        assert ctx.tool_calls == []''' + NEW_TESTS,
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Test stadio 5a.")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    path = Path(args.root) / TARGET
    if not path.is_file():
        print(f"\u2717 file non trovato: {path}", file=sys.stderr)
        return 1
    txt = path.read_text(encoding="utf-8")

    errors: list[str] = []
    for label, old, _new in EDITS:
        n = txt.count(old)
        if n == 0:
            errors.append(f"  \u2717 [{label}] blocco NON trovato (0).")
        elif n > 1:
            errors.append(f"  \u2717 [{label}] blocco AMBIGUO ({n}).")
        else:
            print(f"  \u2713 [{label}] ancoraggio OK.")

    if errors:
        print("\nABORT.\n", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 1

    if args.check:
        print("\n--check OK: 2 blocchi combaciano. Nessuna scrittura.")
        return 0

    for _label, old, new in EDITS:
        txt = txt.replace(old, new, 1)
    path.write_text(txt, encoding="utf-8")
    print("  \u2713 scritto", TARGET)
    print("\n\u2713 Applicato.")
    print("  Prossimi passi:")
    print("    SKIP_SLOW=1 make test       # +7 test verdi")
    print("    git add tests/test_orchestrator_web_search.py")
    print("    git status")
    print('    git commit -m "test(intent): stadio 5a — gating web via intenti e _classify_intents"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
