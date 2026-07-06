#!/usr/bin/env python3
r"""
apply_mapreduce_s4.py — Stadio 4: il classificatore di intenti.

Una chiamata LLM corta classifica la query in un insieme di intenti dal
vocabolario fisso (WEB_SEARCH, FILE_LOCAL, FILE_GLOBAL, FILE_STRUCTURAL,
FILE_POSITIONAL). Contratto a due livelli: set[Intent] (anche vuoto) = riuscita,
None = fallita -> il chiamante usa il fallback. Gli intenti FILE_* valgono solo
se c'e' un documento in sessione.

Crea il modulo modules/intent/ (base_intent.py + __init__.py). NON cabla nulla
nell'orchestrator e NON rimuove _should_search (lo swap e' lo stadio 5).

CREA file nuovi. Aborta se base_intent.py esiste gia'.

Uso:
    python3 apply_mapreduce_s4.py --check
    python3 apply_mapreduce_s4.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path("modules/intent")

INIT = '''"""
modules/intent
==============
Classificazione degli intenti del turno: instrada verso web e/o file.

    from modules.intent import Intent, IntentClassifier
"""

from modules.intent.base_intent import Intent, IntentClassifier

__all__ = [
    "Intent",
    "IntentClassifier",
]
'''

BASE = '''"""
modules/intent/base_intent.py
IntentClassifier — instrada il turno. Una chiamata LLM corta classifica la query
in un insieme di intenti dal vocabolario fisso (web + file). Sostituira'
l'euristica keyword del web (stadio 5), che resta come FALLBACK quando la
classificazione fallisce.

Contratto a due livelli:
  - set[Intent] (anche vuoto) = classificazione RIUSCITA (vuoto = nessuno strumento)
  - None                      = classificazione FALLITA -> il chiamante usa il fallback

Gli intenti FILE_* valgono solo se la sessione ha un documento: lo diciamo nel
prompt E filtriamo a posteriori (doppia difesa).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from core.context import ModelRole
from core.logger import logger
from modules.llm import Message, OllamaClient, Role


class Intent(str, Enum):
    WEB_SEARCH      = "WEB_SEARCH"
    FILE_LOCAL      = "FILE_LOCAL"
    FILE_GLOBAL     = "FILE_GLOBAL"
    FILE_STRUCTURAL = "FILE_STRUCTURAL"
    FILE_POSITIONAL = "FILE_POSITIONAL"


_FILE_INTENTS = {
    Intent.FILE_LOCAL,
    Intent.FILE_GLOBAL,
    Intent.FILE_STRUCTURAL,
    Intent.FILE_POSITIONAL,
}

_SYSTEM = (
    "Classifichi la richiesta dell'utente per decidere quali strumenti servono. "
    "Rispondi SOLO con le etichette pertinenti separate da virgola, senza "
    "spiegazioni.\\n"
    "Etichette:\\n"
    "WEB_SEARCH = serve cercare sul web (notizie, fatti aggiornati, info non note)\\n"
    "FILE_LOCAL = domanda puntuale su un punto preciso del documento\\n"
    "FILE_GLOBAL = riassunto o panoramica dell'intero documento\\n"
    "FILE_STRUCTURAL = struttura del documento (elenco capitoli, indice, sezioni)\\n"
    "FILE_POSITIONAL = cosa accade all'inizio o alla fine del documento\\n"
    "Se non serve alcuno strumento, rispondi: NONE\\n"
    "Usa le etichette FILE_* solo se e' presente un documento in sessione."
)


class IntentClassifier:
    """Classificatore di intenti via LLM (iniettato, testabile con un fake)."""

    def __init__(
        self,
        llm: OllamaClient,
        *,
        num_predict: int   = 24,
        temperature: float = 0.0,
    ) -> None:
        self._llm         = llm
        self._num_predict = num_predict
        self._temperature = temperature

    async def classify(self, query: str, *, has_file: bool) -> "Optional[set[Intent]]":
        if not query or not query.strip():
            return set()

        user = (
            f"Documento in sessione: {'si' if has_file else 'no'}\\n\\n"
            f"Richiesta: {query}"
        )
        try:
            resp = await self._llm.chat(
                [Message(role=Role.USER, content=user)],
                ModelRole.CHAT,
                system=_SYSTEM,
                options={
                    "think":       False,
                    "temperature": self._temperature,
                    "num_predict": self._num_predict,
                },
            )
            raw = resp.content or ""
        except Exception as exc:
            logger.warning("intent | classify fallita: {}", exc)
            return None

        up = raw.upper()
        recognized = {it for it in Intent if it.value in up}
        saw_none = "NONE" in up

        # Nulla di riconoscibile (ne' etichette ne' NONE) -> fallimento, fallback.
        if not recognized and not saw_none:
            logger.warning("intent | risposta non interpretabile: {!r}", raw[:80])
            return None

        # Classificazione riuscita: applica il filtro file-assente (doppia difesa).
        if not has_file:
            recognized = {it for it in recognized if it not in _FILE_INTENTS}

        logger.info(
            "intent | {} (has_file={})",
            sorted(i.value for i in recognized), has_file,
        )
        return recognized
'''


def main() -> int:
    ap = argparse.ArgumentParser(description="Stadio 4: classificatore di intenti.")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root)
    base = root / PKG / "base_intent.py"
    init = root / PKG / "__init__.py"

    if base.exists():
        print(f"\u2717 {PKG}/base_intent.py esiste gia': rifiuto di sovrascrivere.",
              file=sys.stderr)
        return 1

    print(f"  \u2713 {PKG}/ pronto da creare (base + __init__).")
    if args.check:
        print("\n--check OK: nessuna scrittura.")
        return 0

    (root / PKG).mkdir(parents=True, exist_ok=True)
    init.write_text(INIT, encoding="utf-8")
    base.write_text(BASE, encoding="utf-8")
    print(f"  \u2713 creato {PKG}/__init__.py")
    print(f"  \u2713 creato {PKG}/base_intent.py")

    print("\n\u2713 Modulo creato.")
    print("  Prossimi passi:")
    print("    SKIP_SLOW=1 make test")
    print("    git add modules/intent/")
    print("    git status")
    print('    git commit -m "feat(intent): stadio 4 — classificatore LLM degli intenti del turno"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
