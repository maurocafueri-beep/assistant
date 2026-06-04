"""
modules/file_rag/base_file_rag.py
FileRAG — indicizzazione e ricerca semantica su file di grandi dimensioni.

Perché esiste: il file_analysis inietta inline solo i primi ~8000 caratteri di
un file (limite di contesto). Per un libro da centinaia di pagine questo
significa "vedere" solo le prime pagine. FileRAG indicizza l'INTERO documento
in chunk con embedding, così qualsiasi domanda mirata ("in che capitolo
compare X?", "cosa dice a proposito di Y?") recupera in 1-3 secondi i passaggi
rilevanti — indipendentemente da dove si trovano nel libro.

Distinzione dal riassunto map-reduce (Commit successivo): il RAG risponde a
domande PUNTUALI recuperando pochi chunk; il map-reduce produce un riassunto
GLOBALE leggendo tutto. Condividono il chunking, sono operazioni diverse.

Architettura: riusa MemoryManager con una collection ChromaDB dedicata per
file (`file_<hash>`), separata dalla memoria conversazionale. Zero codice
ChromaDB duplicato: chunking + ciclo di vita qui, persistenza in MemoryManager.

API:
    rag = FileRAG()
    await rag.load()
    file_id = await rag.index_file(path, text)      # idempotente per contenuto
    chunks  = await rag.search_file(file_id, query, top_k)
    rag.is_indexed(file_id)
    await rag.drop_file(file_id)
    await rag.aclose()
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from config.settings import settings
from core.context import MemoryChunk
from core.logger import logger
from modules.memory import MemoryManager


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

# Prefisso delle collection ChromaDB dedicate ai file. Il cleanup (Commit 6)
# riconosce le collection orfane da questo prefisso.
FILE_COLLECTION_PREFIX = "file_"

# Marker di pagina prodotto da _extract_pdf in file_analysis. Lo usiamo per
# chunkare rispettando i confini di pagina e tracciare il numero di pagina.
_PAGE_MARKER_RE = re.compile(r"^--- pagina (\d+) ---$", re.MULTILINE)

# Stima caratteri→token per l'italiano: ~4 char/token. Per il chunking RAG la
# precisione non conta, basta che i chunk siano "circa" della dimensione
# voluta — quindi niente tokenizer, nessuna dipendenza aggiuntiva.
_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FileChunk:
    """Un chunk di testo pronto per l'indicizzazione."""
    text: str
    page_start: int   # prima pagina coperta (1-based); 0 se ignota
    page_end:   int   # ultima pagina coperta
    index:      int   # posizione ordinale del chunk nel documento

    def to_metadata(self, file_id: str, source: str) -> dict[str, Any]:
        """Metadata ChromaDB per questo chunk (solo tipi scalari)."""
        return {
            "source":     source,
            "file_id":    file_id,
            "page_start": self.page_start,
            "page_end":   self.page_end,
            "chunk_index": self.index,
        }


@dataclass
class IndexResult:
    """Esito di index_file()."""
    file_id:        str
    chunks_indexed: int
    already_indexed: bool = False
    pages:          int = 0
    elapsed_ms:     float = 0.0

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "file_id":         self.file_id,
            "chunks":          self.chunks_indexed,
            "already_indexed": self.already_indexed,
            "pages":           self.pages,
            "elapsed_ms":      round(self.elapsed_ms, 1),
        }


# ---------------------------------------------------------------------------
# Funzioni pure — chunking & hashing (testabili senza ChromaDB)
# ---------------------------------------------------------------------------

def compute_file_id(text: str) -> str:
    """
    ID stabile di un file dal suo CONTENUTO estratto (non dal path): così lo
    stesso libro caricato due volte (path diversi, timestamp diversi) condivide
    la stessa collection e non viene re-indicizzato. SHA-256 troncato a 16 hex.
    """
    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return h[:16]


def collection_name_for(file_id: str) -> str:
    """Nome della collection ChromaDB per un file_id."""
    return f"{FILE_COLLECTION_PREFIX}{file_id}"


def _split_pages(text: str) -> list[tuple[int, str]]:
    """
    Divide il testo nei suoi blocchi-pagina usando i marker prodotti da
    _extract_pdf (`--- pagina N ---`). Ritorna [(numero_pagina, testo), ...].

    Se non ci sono marker (es. file non-PDF: docx, txt), ritorna un unico
    blocco con pagina 0 (= "pagina ignota"). Così il chunking funziona per
    qualunque tipo di file, non solo PDF.
    """
    matches = list(_PAGE_MARKER_RE.finditer(text))
    if not matches:
        stripped = text.strip()
        return [(0, stripped)] if stripped else []

    pages: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        page_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            pages.append((page_num, body))
    return pages


def chunk_text(
    text: str,
    *,
    target_tokens: int,
    overlap_tokens: int = 0,
) -> list[FileChunk]:
    """
    Spezza il testo in chunk di ~target_tokens, rispettando i confini di pagina.

    Strategia (decisione di design del Commit 7):
      - pagine corte ACCORPATE fino a raggiungere ~target_tokens;
      - pagine lunghe SPEZZATE in più chunk (per paragrafo dove possibile,
        altrimenti per taglio netto);
      - ogni chunk traccia page_start/page_end così l'LLM può citare la pagina.

    overlap_tokens: caratteri di coda della pagina precedente ripetuti in testa
    al chunk successivo quando si spezza una pagina lunga (continuità di
    contesto). 0 = nessun overlap.

    Stima token via caratteri (_CHARS_PER_TOKEN): per il RAG è sufficiente.
    """
    target_chars  = max(200, target_tokens * _CHARS_PER_TOKEN)
    overlap_chars = max(0, overlap_tokens * _CHARS_PER_TOKEN)

    pages = _split_pages(text)
    chunks: list[FileChunk] = []
    idx = 0

    # Buffer di accorpamento per pagine corte consecutive.
    buf_text: str = ""
    buf_start: int = 0
    buf_end: int = 0

    def flush_buffer() -> None:
        nonlocal buf_text, buf_start, buf_end, idx
        body = buf_text.strip()
        if body:
            chunks.append(FileChunk(
                text=body, page_start=buf_start, page_end=buf_end, index=idx,
            ))
            idx += 1
        buf_text = ""

    for page_num, body in pages:
        if len(body) > target_chars:
            # Pagina lunga: prima svuota il buffer accumulato, poi spezza
            # questa pagina in più chunk.
            flush_buffer()
            pieces = _split_long_page(body, target_chars, overlap_chars)
            for piece in pieces:
                chunks.append(FileChunk(
                    text=piece, page_start=page_num, page_end=page_num, index=idx,
                ))
                idx += 1
            continue

        # Pagina corta: prova ad accorparla nel buffer.
        candidate_len = len(buf_text) + len(body) + 2  # +2 per il separatore
        if buf_text and candidate_len > target_chars:
            flush_buffer()
        if not buf_text:
            buf_start = page_num
        buf_text = (buf_text + "\n\n" + body) if buf_text else body
        buf_end = page_num

    flush_buffer()
    return chunks


def _split_long_page(body: str, target_chars: int, overlap_chars: int) -> list[str]:
    """
    Spezza una singola pagina lunga in pezzi di ~target_chars.
    Prova a tagliare sui confini di paragrafo (doppio newline); se un
    paragrafo è da solo più lungo del target, lo taglia comunque netto.
    Applica overlap_chars di coda tra un pezzo e il successivo.
    """
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    cur = ""

    def emit(s: str) -> None:
        s = s.strip()
        if s:
            pieces.append(s)

    for para in paragraphs:
        if len(para) > target_chars:
            # Paragrafo gigante: svuota il corrente e taglia netto il paragrafo.
            if cur:
                emit(cur)
                cur = ""
            for i in range(0, len(para), target_chars):
                emit(para[i:i + target_chars])
            continue
        if cur and len(cur) + len(para) + 2 > target_chars:
            emit(cur)
            # Overlap: ripeti la coda del pezzo appena emesso.
            cur = (cur[-overlap_chars:] + "\n\n" + para) if overlap_chars else para
        else:
            cur = (cur + "\n\n" + para) if cur else para
    emit(cur)
    return pieces


# ---------------------------------------------------------------------------
# FileRAG
# ---------------------------------------------------------------------------

class FileRAG:
    """
    Indicizza file di grandi dimensioni in collection ChromaDB dedicate e ne
    permette la ricerca semantica. Riusa MemoryManager per la persistenza.

    Args:
        persist_dir:     Directory ChromaDB (default: settings.memory.chroma_persist_dir).
        chunk_tokens:    Dimensione target dei chunk in token
                         (default: settings.file_analysis.rag_chunk_target_tokens).
        overlap_tokens:  Overlap tra chunk di una pagina lunga (default 0).
        top_k:           Risultati default per search_file
                         (default: settings.file_analysis.rag_top_k).

    Note di ciclo di vita: una FileRAG mantiene una cache di MemoryManager per
    collection aperta. aclose() li chiude tutti.
    """

    def __init__(
        self,
        persist_dir:    Optional[str] = None,
        chunk_tokens:   int           = 0,
        overlap_tokens: int           = 0,
        top_k:          int           = 0,
    ) -> None:
        fa = settings.file_analysis
        self._persist_dir    = persist_dir or str(settings.memory.chroma_persist_dir)
        self._chunk_tokens   = chunk_tokens   or getattr(fa, "rag_chunk_target_tokens", 800)
        self._overlap_tokens = overlap_tokens or getattr(fa, "rag_overlap_tokens", 0)
        self._top_k          = top_k          or getattr(fa, "rag_top_k", 5)
        # Cache di MemoryManager per collection: file_id → MemoryManager
        self._managers: dict[str, MemoryManager] = {}
        self._loaded = False

    # -- context manager -------------------------------------------------------

    async def __aenter__(self) -> "FileRAG":
        await self.load()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def load(self) -> None:
        """Niente di pesante da inizializzare qui: i manager sono lazy per file."""
        self._loaded = True

    async def aclose(self) -> None:
        """Chiude tutti i MemoryManager aperti."""
        for mgr in self._managers.values():
            try:
                await mgr.aclose()
            except Exception as exc:
                logger.debug("file_rag | aclose manager: {}", exc)
        self._managers.clear()
        self._loaded = False

    # -- factory interna -------------------------------------------------------

    async def _manager_for(self, file_id: str) -> MemoryManager:
        """Ottiene (o crea e carica) il MemoryManager per la collection del file."""
        mgr = self._managers.get(file_id)
        if mgr is None:
            mgr = MemoryManager(
                persist_dir=self._persist_dir,
                collection_name=collection_name_for(file_id),
                top_k=self._top_k,
            )
            await mgr.load()
            self._managers[file_id] = mgr
        return mgr

    # -- API pubblica ----------------------------------------------------------

    async def index_file(
        self,
        text:   str,
        *,
        source: str,
        file_id: Optional[str] = None,
    ) -> IndexResult:
        """
        Indicizza il testo estratto di un file nella sua collection dedicata.

        Args:
            text:    Testo estratto (con eventuali marker `--- pagina N ---`).
            source:  Etichetta leggibile del file (es. nome file) per i metadata.
            file_id: ID esplicito; se None, calcolato dall'hash del contenuto.

        Idempotente: se la collection esiste già con dei chunk, NON re-indicizza
        (riapri lo stesso libro → riuso immediato). Best-effort sui singoli
        chunk: un embedding fallito viene loggato e saltato.

        Returns:
            IndexResult con file_id, numero chunk e flag already_indexed.
        """
        import time
        t0 = time.monotonic()
        fid = file_id or compute_file_id(text)

        mgr = await self._manager_for(fid)

        # Idempotenza: se ci sono già chunk, salta.
        existing = await mgr.count()
        if existing > 0:
            logger.info("file_rag.index | '{}' già indicizzato ({} chunk) — skip", fid, existing)
            return IndexResult(
                file_id=fid, chunks_indexed=existing, already_indexed=True,
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )

        chunks = chunk_text(
            text,
            target_tokens=self._chunk_tokens,
            overlap_tokens=self._overlap_tokens,
        )
        pages = max((c.page_end for c in chunks), default=0)

        indexed = 0
        for c in chunks:
            try:
                await mgr.save(
                    c.text,
                    metadata=c.to_metadata(fid, source),
                    chunk_id=f"{fid}_{c.index}",
                )
                indexed += 1
            except Exception as exc:
                logger.warning("file_rag.index | chunk {} fallito: {}", c.index, exc)

        result = IndexResult(
            file_id=fid, chunks_indexed=indexed, already_indexed=False,
            pages=pages, elapsed_ms=(time.monotonic() - t0) * 1000,
        )
        logger.info("file_rag.index | {}", result.to_log_dict())
        return result

    async def search_file(
        self,
        file_id: str,
        query:   str,
        top_k:   Optional[int] = None,
    ) -> list[MemoryChunk]:
        """
        Ricerca semantica nei chunk di un file indicizzato.
        Ritorna lista vuota se il file non è indicizzato o la query è vuota.
        """
        if not query.strip():
            return []
        mgr = await self._manager_for(file_id)
        if await mgr.count() == 0:
            return []
        return await mgr.search(query, top_k=top_k if top_k is not None else self._top_k)

    async def get_ordered_chunks(self, file_id: str) -> list[dict]:
        """
        Tutti i chunk del file in ordine di documento (per chunk_index), come
        lista di {"text", "page_start", "page_end", "index"}. Per le scansioni
        complete (map-reduce): NON è una ricerca per similarità.
        Lista vuota se il file non è indicizzato.
        """
        mgr   = await self._manager_for(file_id)
        items = await mgr.get_all()
        out: list[dict] = []
        for it in items:
            m = it.get("metadata") or {}
            out.append({
                "text":       it.get("text") or "",
                "page_start": m.get("page_start", 0),
                "page_end":   m.get("page_end", 0),
                "index":      m.get("chunk_index", 0),
            })
        out.sort(key=lambda c: c["index"])
        return out

    @staticmethod
    def iter_blocks(chunks: list[dict], block_chars: int) -> Iterator[dict]:
        """
        Raggruppa chunk ordinati in blocchi fino a ~block_chars caratteri. Ogni
        blocco: {"text", "page_start", "page_end", "n_chunks"}, con le pagine del
        primo e dell'ultimo chunk del blocco. Puro: la dimensione la decide il
        chiamante (lo stadio map la prenderà da settings).
        """
        buf: list[str] = []
        size = 0
        p_start = None
        p_end = None
        n = 0
        for c in chunks:
            t = c.get("text") or ""
            if buf and size + len(t) > block_chars:
                yield {"text": "\n\n".join(buf), "page_start": p_start,
                       "page_end": p_end, "n_chunks": n}
                buf, size, p_start, p_end, n = [], 0, None, None, 0
            buf.append(t)
            size += len(t)
            if p_start is None:
                p_start = c.get("page_start", 0)
            p_end = c.get("page_end", 0)
            n += 1
        if buf:
            yield {"text": "\n\n".join(buf), "page_start": p_start,
                   "page_end": p_end, "n_chunks": n}

    async def is_indexed(self, file_id: str) -> bool:
        """True se la collection del file esiste e contiene chunk."""
        mgr = await self._manager_for(file_id)
        try:
            return await mgr.count() > 0
        except Exception:
            return False

    async def drop_file(self, file_id: str) -> bool:
        """
        Rimuove completamente la collection di un file (usato dal cleanup
        quando il file scade da uploads/). Ritorna True se rimossa.
        """
        mgr = await self._manager_for(file_id)
        try:
            await mgr.clear()
            # Chiudi e dimentica il manager: la collection resta vuota su disco
            # (ChromaDB non cancella la collection con clear, ma senza chunk è
            # inerte; la rimozione fisica della collection avverrà nel Commit 7c
            # via delete_collection sul client).
            await mgr.aclose()
            self._managers.pop(file_id, None)
            logger.info("file_rag.drop | '{}' svuotato", file_id)
            return True
        except Exception as exc:
            logger.warning("file_rag.drop | '{}' fallito: {}", file_id, exc)
            return False
