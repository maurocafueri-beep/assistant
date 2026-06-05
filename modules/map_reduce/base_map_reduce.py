"""
modules/map_reduce/base_map_reduce.py
MapReduceEngine — riassunti e risposte GLOBALI su un documento intero.

Perché esiste: il RAG semantico recupera i chunk SIMILI alla domanda, e fallisce
sulle domande che richiedono tutto il documento o la sua struttura ("riassumi il
libro", "elenca i capitoli", "cosa succede alla fine"). Il map-reduce scandisce
l'intero documento a blocchi: il map estrae da ogni blocco cio' che serve alla
domanda, il reduce fonde i parziali in una risposta finale. Il reduce e'
gerarchico (fonde a gruppi e ricorre), cosi' regge anche i libri grandi.

Disaccoppiato dallo stadio 1: lavora sui blocchi gia' pronti (output di
FileRAG.iter_blocks). Sara' l'orchestrator a concatenare
get_ordered_chunks -> iter_blocks -> run.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.context import ModelRole
from core.logger import logger
from modules.llm import Message, OllamaClient, Role

# ---------------------------------------------------------------------------
# Prompt (generici: lo stadio 4 li specializzera' per tipo di domanda)
# ---------------------------------------------------------------------------

_MAP_SYSTEM = (
    "Analizzi una porzione di un documento piu' ampio. Ricevi una DOMANDA e un "
    "BLOCCO di testo con il suo intervallo di pagine. Estrai SOLO le informazioni "
    "del blocco rilevanti alla domanda, in forma sintetica, citando il numero di "
    "pagina. Se il blocco non contiene NULLA di rilevante alla domanda, rispondi "
    "esattamente con la sola parola: NIENTE. Non inventare e non aggiungere nulla "
    "che non sia nel blocco."
)

_REDUCE_SYSTEM = (
    "Fondi estratti parziali ottenuti analizzando un intero documento per "
    "rispondere a una domanda. Ricevi la DOMANDA e una lista di ESTRATTI, ognuno "
    "con i suoi riferimenti di pagina. Produci UNA risposta finale, completa e "
    "coerente, conservando i riferimenti di pagina. Non aggiungere informazioni "
    "che non siano negli estratti."
)

_EMPTY_SENTINEL = "NIENTE"
_NOT_FOUND = (
    "Non ho trovato informazioni rilevanti nel documento per questa domanda."
)


@dataclass
class MapReduceResult:
    """Esito di una scansione map-reduce."""
    content:     str
    n_blocks:    int
    n_partials:  int
    n_llm_calls: int

    def to_log_dict(self) -> dict:
        return {
            "n_blocks":    self.n_blocks,
            "n_partials":  self.n_partials,
            "n_llm_calls": self.n_llm_calls,
            "content_len": len(self.content),
        }


class MapReduceEngine:
    """
    Motore map-reduce su un documento. L'LLM e' iniettato cosi' il motore e'
    testabile con un client finto, senza Ollama.
    """

    def __init__(
        self,
        llm: OllamaClient,
        *,
        map_num_predict:    int   = 512,
        reduce_num_predict: int   = 1024,
        reduce_block_chars: int   = 12000,
        temperature:        float = 0.2,
    ) -> None:
        self._llm                = llm
        self._map_num_predict    = map_num_predict
        self._reduce_num_predict = reduce_num_predict
        self._reduce_block_chars = reduce_block_chars
        self._temperature        = temperature
        self._calls              = 0

    async def run(self, *, question: str, blocks: list[dict]) -> MapReduceResult:
        """
        map (un blocco per volta) + reduce gerarchico.
        `blocks` e' l'output di FileRAG.iter_blocks.
        """
        self._calls = 0
        n_blocks = len(blocks)

        # -- MAP (sequenziale) -------------------------------------------------
        partials: list[str] = []
        for blk in blocks:
            piece = await self._map_block(question, blk)
            if piece is not None:
                partials.append(piece)
        logger.info(
            "map_reduce.map | blocchi={} parziali_rilevanti={}",
            n_blocks, len(partials),
        )

        # -- REDUCE (gerarchico) ----------------------------------------------
        if not partials:
            return MapReduceResult(_NOT_FOUND, n_blocks, 0, self._calls)

        content = await self._reduce(question, partials)
        result = MapReduceResult(content, n_blocks, len(partials), self._calls)
        logger.info("map_reduce.run | {}", result.to_log_dict())
        return result

    # -- map -------------------------------------------------------------------

    async def _map_block(self, question: str, block: dict) -> "str | None":
        ps = block.get("page_start", 0)
        pe = block.get("page_end", 0)
        user = (
            f"DOMANDA: {question}\n\n"
            f"BLOCCO (pagine {ps}-{pe}):\n{block.get('text', '')}"
        )
        out = (await self._chat(_MAP_SYSTEM, user, self._map_num_predict)).strip()
        if not out or out.upper().startswith(_EMPTY_SENTINEL):
            return None
        # Prefissa le pagine: il reduce le conserva anche se il map le omette.
        return f"[pagine {ps}-{pe}] {out}"

    # -- reduce (gerarchico) ---------------------------------------------------

    async def _reduce(self, question: str, partials: list[str]) -> str:
        groups = self._group(partials)
        if len(groups) == 1:
            return await self._reduce_one(question, groups[0])
        if len(groups) == len(partials):
            # Parziali tutti oltre-budget: impossibile raggruppare ancora.
            # Fondi tutto in un colpo (best effort) per garantire la terminazione.
            return await self._reduce_one(question, partials)
        intermediates = [await self._reduce_one(question, g) for g in groups]
        return await self._reduce(question, intermediates)

    def _group(self, partials: list[str]) -> "list[list[str]]":
        groups: list[list[str]] = []
        cur: list[str] = []
        size = 0
        for p in partials:
            if cur and size + len(p) > self._reduce_block_chars:
                groups.append(cur)
                cur, size = [], 0
            cur.append(p)
            size += len(p)
        if cur:
            groups.append(cur)
        return groups

    async def _reduce_one(self, question: str, partials: list[str]) -> str:
        joined = "\n\n---\n\n".join(partials)
        user = f"DOMANDA: {question}\n\nESTRATTI:\n{joined}"
        out = await self._chat(_REDUCE_SYSTEM, user, self._reduce_num_predict)
        return out.strip()

    # -- chiamata LLM ----------------------------------------------------------

    async def _chat(self, system: str, user: str, num_predict: int) -> str:
        self._calls += 1
        resp = await self._llm.chat(
            [Message(role=Role.USER, content=user)],
            ModelRole.CHAT,
            system=system,
            options={
                "think":       False,
                "temperature": self._temperature,
                "num_predict": num_predict,
            },
        )
        return resp.content or ""
