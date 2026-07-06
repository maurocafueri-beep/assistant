#!/usr/bin/env python3
r"""
apply_mapreduce_s5b.py — Stadio 5b: map-reduce on-demand cablato nel turno.

Cabla i tre intenti file (FILE_GLOBAL/STRUCTURAL/POSITIONAL) a un nuovo
_run_map_reduce: get_ordered_chunks -> iter_blocks -> engine.run -> inietta la
sintesi nel system prompt (poi il modello principale streamma). Solo on-demand
(il "guarda prima il precalcolato" e' il 5c). _run_file_rag ottiene un gate
CONSERVATIVO: si fa da parte solo se gli intenti instradano al map-reduce SENZA
intento locale; in tutti gli altri casi (locale, None, vuoto) gira come oggi.

Sette modifiche a core/orchestrator.py. Richiede lo stadio 5a applicato.

ATTENZIONE: file molto cambiato. Se un --check non combacia, FERMATI e incolla.

Uso:
    python3 apply_mapreduce_s5b.py --check
    python3 apply_mapreduce_s5b.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET = "core/orchestrator.py"

EDITS: list[tuple[str, str, str]] = [
    (
        "1) import del motore",
        r'''from modules.intent import Intent, IntentClassifier''',
        r'''from modules.intent import Intent, IntentClassifier
from modules.map_reduce import MapReduceEngine''',
    ),
    (
        "2) costruzione in load()",
        r'''        # --- Classificatore di intenti (gira sul modello chat, gia' in VRAM) ---
        self._intent_classifier = IntentClassifier(self._llm)''',
        r'''        # --- Classificatore di intenti (gira sul modello chat, gia' in VRAM) ---
        self._intent_classifier = IntentClassifier(self._llm)

        # --- Motore map-reduce (scansione globale dei documenti) ---
        self._map_reduce = MapReduceEngine(self._llm)''',
    ),
    (
        "3) reset a None nel close",
        r'''        self._intent_classifier = None
        self._loaded = False''',
        r'''        self._intent_classifier = None
        self._map_reduce = None
        self._loaded = False''',
    ),
    (
        "4) costanti + _format_map_reduce_block",
        r'''    lines.append(_FILE_RAG_FOOTER)
    return "\n".join(lines)''',
        r'''    lines.append(_FILE_RAG_FOOTER)
    return "\n".join(lines)


_MAPREDUCE_INTENTS = {
    Intent.FILE_GLOBAL, Intent.FILE_STRUCTURAL, Intent.FILE_POSITIONAL,
}
_MAPREDUCE_BLOCK_CHARS = 40000      # ~10K token per blocco sul context 16K
_POSITIONAL_BLOCKS = 2              # primi N + ultimi N per le domande posizionali

_MAP_REDUCE_HEADER = (
    "\n\n---\nSINTESI DAL DOCUMENTO COMPLETO (ottenuta scandendo l'intero file; "
    "basati su questa per rispondere e cita le pagine indicate):\n"
)


def _format_map_reduce_block(content: str) -> str:
    """Formatta la sintesi map-reduce come blocco di contesto da iniettare."""
    if not content or not content.strip():
        return ""
    return _MAP_REDUCE_HEADER + content.strip() + "\n"''',
    ),
    (
        "5) gate conservativo in _run_file_rag",
        r'''        if self._file_rag is None:
            return
        rag_files = ctx.metadata.get("rag_files") or []''',
        r'''        if self._file_rag is None:
            return
        # Si fa da parte solo se il classificatore ha instradato al map-reduce
        # (file globale/strutturale/posizionale) SENZA intento locale: in quel
        # caso risponde _run_map_reduce. Negli altri casi (locale, None, vuoto)
        # il RAG semantico gira come oggi — nessuna rete tolta.
        intents = ctx.metadata.get("intents")
        if isinstance(intents, set) and (intents & _MAPREDUCE_INTENTS) and (
            Intent.FILE_LOCAL not in intents
        ):
            return
        rag_files = ctx.metadata.get("rag_files") or []''',
    ),
    (
        "6) metodo _run_map_reduce",
        r'''        logger.info(
            "orchestrator._run_file_rag | chunk_iniettati={} files={}",
            len(all_chunks), len(rag_files),
        )''',
        r'''        logger.info(
            "orchestrator._run_file_rag | chunk_iniettati={} files={}",
            len(all_chunks), len(rag_files),
        )

    async def _run_map_reduce(self, ctx: AssistantContext) -> None:
        """
        Per gli intenti FILE_GLOBAL/STRUCTURAL/POSITIONAL: scandisce il documento
        a blocchi (on-demand) e inietta la sintesi nel system prompt; poi il
        modello principale streamma la risposta finale. Best-effort: non solleva.
        Su intents None (classificatore fallito) NON parte: copre il RAG semantico.
        """
        if self._file_rag is None or self._map_reduce is None:
            return
        intents = ctx.metadata.get("intents")
        if not isinstance(intents, set):
            return
        mr = intents & _MAPREDUCE_INTENTS
        if not mr:
            return
        rag_files = ctx.metadata.get("rag_files") or []
        if not rag_files:
            return
        query = ctx.user_text
        if not query.strip():
            return

        positional_only = mr == {Intent.FILE_POSITIONAL}
        t0 = time.monotonic()
        blocks: list[dict] = []
        for f in rag_files:
            try:
                chunks = await self._file_rag.get_ordered_chunks(f["file_id"])
            except Exception as exc:
                logger.warning(
                    "orchestrator._run_map_reduce | chunks '{}': {}", f.get("source"), exc
                )
                continue
            fb = list(FileRAG.iter_blocks(chunks, _MAPREDUCE_BLOCK_CHARS))
            if positional_only and len(fb) > 2 * _POSITIONAL_BLOCKS:
                fb = fb[:_POSITIONAL_BLOCKS] + fb[-_POSITIONAL_BLOCKS:]
            blocks.extend(fb)

        if not blocks:
            return
        try:
            result = await self._map_reduce.run(question=query, blocks=blocks)
        except Exception as exc:
            logger.warning("orchestrator._run_map_reduce | motore: {}", exc)
            return
        ctx.set_timing("map_reduce", (time.monotonic() - t0) * 1000)

        block = _format_map_reduce_block(result.content)
        if block:
            ctx.system_prompt = (ctx.system_prompt or "") + block
        logger.info("orchestrator._run_map_reduce | {}", result.to_log_dict())''',
    ),
    (
        "7) chiamata in turn() dopo file_rag",
        r'''        await self._run_file_rag(ctx)

        # 4.5 Web search (se trigger keyword + tool consentito dal profilo)''',
        r'''        await self._run_file_rag(ctx)

        # 4.47 Map-reduce on-demand: domande globali/strutturali/posizionali sul
        # documento (instradate dal classificatore). Inietta la sintesi.
        await self._run_map_reduce(ctx)

        # 4.5 Web search (se trigger keyword + tool consentito dal profilo)''',
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Stadio 5b: map-reduce on-demand.")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    path = Path(args.root) / TARGET
    if not path.is_file():
        print(f"\u2717 file non trovato: {path}", file=sys.stderr)
        return 1
    txt = path.read_text(encoding="utf-8")
    if "_run_map_reduce" in txt:
        print("\u2717 _run_map_reduce gia' presente: 5b gia' applicato?", file=sys.stderr)
        return 1

    errors: list[str] = []
    for label, old, _new in EDITS:
        n = txt.count(old)
        if n == 0:
            errors.append(f"  \u2717 [{label}] blocco NON trovato (0). 5a applicato?")
        elif n > 1:
            errors.append(f"  \u2717 [{label}] blocco AMBIGUO ({n}).")
        else:
            print(f"  \u2713 [{label}] ancoraggio OK.")

    if errors:
        print("\nABORT: ancoraggi non combaciano.\n", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        print("\nFermati e incolla il contesto reale dei punti falliti.", file=sys.stderr)
        return 1

    if args.check:
        print("\n--check OK: tutti e 7 i blocchi combaciano. Nessuna scrittura.")
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
    print('    git commit -m "feat(map_reduce): stadio 5b — map-reduce on-demand nel turno"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
