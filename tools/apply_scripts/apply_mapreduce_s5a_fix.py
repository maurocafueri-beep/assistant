#!/usr/bin/env python3
r"""
apply_mapreduce_s5a_fix.py — Fix del bug has_file al primo turno.

_classify_intents leggeva has_file da self._session_rag_files, che si aggiorna
solo a fine turno: al PRIMO turno con un file appena caricato era ancora vuoto,
quindi has_file=False e la domanda globale cadeva sul RAG semantico invece che
sul map-reduce. Fix: leggere ctx.metadata["rag_files"], la stessa fonte di
_run_file_rag/_run_map_reduce, gia' popolata da _run_file_analysis nel turno.

Una modifica a core/orchestrator.py + due test allineati in
tests/test_orchestrator_web_search.py. Richiede lo stadio 5a applicato.

Uso:
    python3 apply_mapreduce_s5a_fix.py --check
    python3 apply_mapreduce_s5a_fix.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EDITS: list[tuple[str, str, str, str]] = [
    (
        "1) orchestrator: has_file da ctx.metadata",
        "core/orchestrator.py",
        r'''        has_file = bool(self._session_rag_files.get(ctx.session_id))''',
        r'''        # I file vivono in ctx.metadata["rag_files"] (stessa fonte di
        # _run_file_rag/_run_map_reduce), gia' popolata da _run_file_analysis in
        # questo turno: cosi' has_file e' corretto anche al PRIMO turno con un
        # file appena caricato (self._session_rag_files si aggiorna solo a fine
        # turno, sarebbe in ritardo).
        has_file = bool(ctx.metadata.get("rag_files"))''',
    ),
    (
        "2) test: has_file=False senza rag_files",
        "tests/test_orchestrator_web_search.py",
        r'''    async def test_popola_metadata_has_file_false(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._session_rag_files = {}
        clf = MagicMock()
        clf.classify = AsyncMock(return_value={Intent.WEB_SEARCH})
        orch._intent_classifier = clf
        ctx = make_ctx(user_text="q", session_id="s1")
        await orch._classify_intents(ctx)
        assert ctx.metadata["intents"] == {Intent.WEB_SEARCH}
        assert clf.classify.call_args.kwargs["has_file"] is False''',
        r'''    async def test_popola_metadata_has_file_false(self):
        orch = Orchestrator.__new__(Orchestrator)
        clf = MagicMock()
        clf.classify = AsyncMock(return_value={Intent.WEB_SEARCH})
        orch._intent_classifier = clf
        ctx = make_ctx(user_text="q", session_id="s1")   # niente rag_files
        await orch._classify_intents(ctx)
        assert ctx.metadata["intents"] == {Intent.WEB_SEARCH}
        assert clf.classify.call_args.kwargs["has_file"] is False''',
    ),
    (
        "3) test: has_file=True dai rag_files in metadata",
        "tests/test_orchestrator_web_search.py",
        r'''    async def test_has_file_true_con_rag_files(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._session_rag_files = {"s1": [{"file_id": "x"}]}
        clf = MagicMock()
        clf.classify = AsyncMock(return_value=set())
        orch._intent_classifier = clf
        ctx = make_ctx(user_text="q", session_id="s1")
        await orch._classify_intents(ctx)
        assert clf.classify.call_args.kwargs["has_file"] is True''',
        r'''    async def test_has_file_true_con_rag_files(self):
        orch = Orchestrator.__new__(Orchestrator)
        clf = MagicMock()
        clf.classify = AsyncMock(return_value=set())
        orch._intent_classifier = clf
        # I file vivono in ctx.metadata["rag_files"] (popolati da file_analysis).
        ctx = make_ctx(user_text="q", session_id="s1",
                       metadata={"rag_files": [{"file_id": "x"}]})
        await orch._classify_intents(ctx)
        assert clf.classify.call_args.kwargs["has_file"] is True''',
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Fix has_file al primo turno.")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root)
    errors: list[str] = []
    contents: dict[str, str] = {}
    for label, rel, old, _new in EDITS:
        path = root / rel
        if not path.is_file():
            errors.append(f"  \u2717 [{label}] file non trovato: {path}")
            continue
        if rel not in contents:
            contents[rel] = path.read_text(encoding="utf-8")
        n = contents[rel].count(old)
        if n == 0:
            errors.append(f"  \u2717 [{label}] blocco NON trovato in {rel} (0). 5a applicato?")
        elif n > 1:
            errors.append(f"  \u2717 [{label}] blocco AMBIGUO in {rel} ({n}).")
        else:
            print(f"  \u2713 [{label}] ancoraggio OK in {rel}.")

    if errors:
        print("\nABORT.\n", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 1

    if args.check:
        print("\n--check OK: 3 blocchi combaciano. Nessuna scrittura.")
        return 0

    patched = dict(contents)
    for _label, rel, old, new in EDITS:
        patched[rel] = patched[rel].replace(old, new, 1)
    for rel, txt in patched.items():
        (root / rel).write_text(txt, encoding="utf-8")
        print(f"  \u2713 scritto {rel}")

    print("\n\u2713 Applicato.")
    print("  Prossimi passi:")
    print("    SKIP_SLOW=1 make test")
    print("    git add core/orchestrator.py tests/test_orchestrator_web_search.py")
    print("    git status")
    print('    git commit -m "fix(intent): has_file da ctx.metadata, corretto al primo turno con file"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
