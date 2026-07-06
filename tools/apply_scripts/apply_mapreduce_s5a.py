#!/usr/bin/env python3
r"""
apply_mapreduce_s5a.py — Stadio 5a: routing del web via classificatore.

Cabla il classificatore di intenti in orchestrator.turn() ACCANTO a _should_search,
e fa decidere il web dall'intento WEB_SEARCH con fallback alle keyword quando la
classificazione fallisce (None). NON tocca il RAG file (resta com'e' oggi): i
cinque intenti file vengono calcolati e parcheggiati ma non ancora consumati
(quello e' lo stadio 5b).

Sei modifiche, tutte in core/orchestrator.py:
  1. import del classificatore
  2. costruzione in load()
  3. reset a None nel close
  4. helper _classify_intents
  5. chiamata in turn() dopo _run_file_analysis
  6. gate del web in _run_web_search (intento + fallback keyword)

ATTENZIONE: orchestrator.py e' il file piu' cambiato del progetto. Se un --check
non combacia, FERMATI e incolla il contesto reale di quel punto.

Uso:
    python3 apply_mapreduce_s5a.py --check
    python3 apply_mapreduce_s5a.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET = "core/orchestrator.py"

EDITS: list[tuple[str, str, str]] = [
    (
        "1) import classificatore",
        r'''from modules.file_rag import FileRAG''',
        r'''from modules.file_rag import FileRAG
from modules.intent import Intent, IntentClassifier''',
    ),
    (
        "2) costruzione in load()",
        r'''        # --- LLM (httpx, leggero) ---
        self._llm = OllamaClient()''',
        r'''        # --- LLM (httpx, leggero) ---
        self._llm = OllamaClient()

        # --- Classificatore di intenti (gira sul modello chat, gia' in VRAM) ---
        self._intent_classifier = IntentClassifier(self._llm)''',
    ),
    (
        "3) reset a None nel close",
        r'''        self._file_analyzer = None
        self._file_rag = None
        self._loaded = False
        logger.info("orchestrator | chiuso")''',
        r'''        self._file_analyzer = None
        self._file_rag = None
        self._intent_classifier = None
        self._loaded = False
        logger.info("orchestrator | chiuso")''',
    ),
    (
        "4) helper _classify_intents",
        r'''    async def _run_web_search(self, ctx: AssistantContext) -> None:
        """
        Se il testo utente contiene una keyword-trigger e il profilo attivo''',
        r'''    async def _classify_intents(self, ctx: AssistantContext) -> None:
        """
        Classifica la query del turno e parcheggia il risultato in
        ctx.metadata["intents"]: set[Intent] (riuscita, anche vuoto) oppure None
        (fallita -> i consumatori usano il fallback). Va chiamato DOPO
        _run_file_analysis, cosi' has_file vede anche il file appena caricato.
        Non fatale: senza classificatore parcheggia None.
        """
        if not self._intent_classifier:
            ctx.metadata["intents"] = None
            return
        has_file = bool(self._session_rag_files.get(ctx.session_id))
        ctx.metadata["intents"] = await self._intent_classifier.classify(
            ctx.user_text, has_file=has_file,
        )

    async def _run_web_search(self, ctx: AssistantContext) -> None:
        """
        Se il testo utente contiene una keyword-trigger e il profilo attivo''',
    ),
    (
        "5) chiamata in turn() dopo file_analysis",
        r'''        await self._run_file_analysis(ctx)

        # 4.45 File RAG: recupera passaggi rilevanti dai file grandi indicizzati''',
        r'''        await self._run_file_analysis(ctx)

        # 4.42 Classificazione intenti del turno: instrada il web (e, dagli stadi
        # successivi, i percorsi sui file). Popola ctx.metadata["intents"].
        await self._classify_intents(ctx)

        # 4.45 File RAG: recupera passaggi rilevanti dai file grandi indicizzati''',
    ),
    (
        "6) gate del web (intento + fallback keyword)",
        r'''        # Trigger: keyword nel testo utente
        if not _should_search(ctx.user_text):
            return''',
        r'''        # Trigger: intento WEB_SEARCH dal classificatore; se la classificazione
        # e' fallita (None), fallback all'euristica keyword.
        intents = ctx.metadata.get("intents")
        if intents is None:
            triggered = _should_search(ctx.user_text)
        else:
            triggered = Intent.WEB_SEARCH in intents
        if not triggered:
            return''',
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Stadio 5a: routing web via classificatore.")
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
        print("\nABORT: ancoraggi non combaciano (orchestrator.py diverge?).\n", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        print("\nFermati e incolla il contesto reale dei punti falliti.", file=sys.stderr)
        return 1

    if args.check:
        print("\n--check OK: tutti e 6 i blocchi combaciano. Nessuna scrittura.")
        return 0

    for _label, old, new in EDITS:
        txt = txt.replace(old, new, 1)
    path.write_text(txt, encoding="utf-8")
    print("  \u2713 scritto", TARGET)

    print("\n\u2713 Applicato.")
    print("  Prossimi passi:")
    print("    SKIP_SLOW=1 make test")
    print("    git add core/orchestrator.py")
    print("    git status")
    print('    git commit -m "feat(intent): stadio 5a — routing del web via classificatore, fallback keyword"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
