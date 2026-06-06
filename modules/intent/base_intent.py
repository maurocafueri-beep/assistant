"""
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
    "spiegazioni.\n"
    "Etichette:\n"
    "WEB_SEARCH = serve cercare sul web (notizie, fatti aggiornati, info non note)\n"
    "FILE_LOCAL = domanda puntuale su un punto preciso del documento\n"
    "FILE_GLOBAL = riassunto o panoramica dell'intero documento\n"
    "FILE_STRUCTURAL = struttura del documento (elenco capitoli, indice, sezioni)\n"
    "FILE_POSITIONAL = cosa accade all'inizio o alla fine del documento\n"
    "Se non serve alcuno strumento, rispondi: NONE\n"
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
            f"Documento in sessione: {'si' if has_file else 'no'}\n\n"
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
