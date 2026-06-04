"""
modules/memory/base_memory.py
MemoryManager — salvataggio e ricerca di chunk di testo con embeddings
via OllamaClient + persistenza su ChromaDB.

API pubblica:
    manager.save(text, metadata, chunk_id)    → SaveResult         (async)
    manager.search(query, top_k)              → list[MemoryChunk]  (async)
    manager.delete(chunk_id)                  → bool               (async)
    manager.clear()                           → int                (async)
    manager.count()                           → int                (async)
    manager.populate_context(ctx, query, top_k) → None             (async)

Uso rapido:
    async with MemoryManager() as mem:
        await mem.save("Python è un linguaggio.", {"source": "tutorial"})
        chunks = await mem.search("linguaggio di programmazione")

Uso standalone:
    mem = MemoryManager()
    await mem.load()
    await mem.save("...")
    await mem.aclose()
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from config.settings import settings
from core.context import AssistantContext, MemoryChunk
from core.logger import logger
from modules.llm import OllamaClient  # leggero da importare: solo httpx


# ---------------------------------------------------------------------------
# Dataclasses pubbliche
# ---------------------------------------------------------------------------

@dataclass
class SaveResult:
    """Esito di una chiamata save()."""
    chunk_id:   str
    text_len:   int
    elapsed_ms: float

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "chunk_id":   self.chunk_id,
            "text_len":   self.text_len,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }

    def __str__(self) -> str:
        return f"<SaveResult id='{self.chunk_id}' len={self.text_len} {self.elapsed_ms:.0f}ms>"


@dataclass
class SearchResult:
    """Esito di una chiamata search() — usata internamente per il log."""
    chunks:     list[MemoryChunk]
    query_len:  int
    elapsed_ms: float
    top_k:      int

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "n_results":  len(self.chunks),
            "query_len":  self.query_len,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "top_k":      self.top_k,
        }


# ---------------------------------------------------------------------------
# Helpers ChromaDB — eseguiti in executor (import pesante: chromadb)
# ---------------------------------------------------------------------------

def _get_or_create_collection(persist_dir: str, collection_name: str):
    """
    Inizializza ChromaDB PersistentClient e ottieni/crea la collection.
    Eseguire in executor: 'import chromadb' è lento (~200 ms).

    Returns:
        (chromadb.PersistentClient, chromadb.Collection)
    """
    import chromadb  # import pesante — solo in executor

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def _chroma_add(
    collection,
    chunk_id:  str,
    embedding: list[float],
    text:      str,
    metadata:  Optional[dict[str, Any]],
) -> None:
    collection.add(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata],
    )


def _chroma_query(
    collection,
    embedding: list[float],
    n_results: int,
    where: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


def _chroma_delete(collection, chunk_id: str) -> None:
    collection.delete(ids=[chunk_id])


def _chroma_clear(collection) -> int:
    """Rimuove tutti i documenti. Restituisce il numero di chunk eliminati."""
    result = collection.get(include=[])
    all_ids: list[str] = result.get("ids", [])
    if all_ids:
        collection.delete(ids=all_ids)
    return len(all_ids)


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

COLLECTION_NAME = "assistant_memory"


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Gestisce la memoria semantica dell'assistente.

    Salva chunk di testo con il loro embedding (generato da OllamaClient)
    in una collection ChromaDB persistente su disco.
    Ricerca per similarità coseno e popola AssistantContext.retrieved_memories.

    Args:
        persist_dir:      Directory di persistenza ChromaDB
                          (default: settings.memory.chroma_persist_dir).
        collection_name:  Nome della collection ChromaDB.
        top_k:            Numero di risultati default per search()
                          (default: settings.memory.rag_top_k).

    Esempio — context manager:
        async with MemoryManager() as mem:
            await mem.save("Testo da ricordare.", {"source": "conversazione"})
            chunks = await mem.search("Testo")

    Esempio — standalone:
        mem = MemoryManager()
        await mem.load()
        await mem.save("...")
        await mem.aclose()
    """

    def __init__(
        self,
        persist_dir:     Optional[str] = None,
        collection_name: str           = COLLECTION_NAME,
        top_k:           int           = 0,
    ) -> None:
        self._persist_dir     = persist_dir or str(settings.memory.chroma_persist_dir)
        self._collection_name = collection_name
        self._top_k           = top_k or settings.memory.rag_top_k
        self._collection      = None
        self._chroma_client   = None
        self._llm:  Optional[OllamaClient] = None
        self._loaded: bool = False

    # -- context manager -------------------------------------------------------

    async def __aenter__(self) -> "MemoryManager":
        await self.load()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Chiude le risorse (OllamaClient). ChromaDB non richiede flush esplicito."""
        if self._llm is not None:
            await self._llm.aclose()
            self._llm = None
        self._loaded = False
        logger.debug("memory | chiuso")

    # -- inizializzazione ------------------------------------------------------

    async def load(self) -> None:
        """
        Inizializza ChromaDB (in executor) e OllamaClient.
        Chiamato automaticamente dal context manager.
        """
        loop = asyncio.get_running_loop()

        # ChromaDB: import pesante → executor
        self._chroma_client, self._collection = await loop.run_in_executor(
            None,
            lambda: _get_or_create_collection(self._persist_dir, self._collection_name),
        )

        self._llm = OllamaClient()
        self._loaded = True

        n = await loop.run_in_executor(None, lambda: self._collection.count())
        logger.info(
            "memory | inizializzato | dir={} collection={} chunks={}",
            self._persist_dir,
            self._collection_name,
            n,
        )

    # -- API pubblica ----------------------------------------------------------

    async def save(
        self,
        text:     str,
        metadata: Optional[dict[str, Any]] = None,
        chunk_id: Optional[str]            = None,
    ) -> SaveResult:
        """
        Salva un chunk di testo con embedding in ChromaDB.

        Args:
            text:      Testo da salvare e indicizzare.
            metadata:  Dizionario di metadati (source, timestamp, …).
                       Valori None vengono rimossi (ChromaDB non li accetta).
            chunk_id:  ID univoco; auto-generato (UUID4) se non fornito.

        Returns:
            SaveResult con chunk_id, lunghezza testo ed elapsed_ms.
        """
        self._require_loaded()
        t0 = time.monotonic()

        cid  = chunk_id or str(uuid.uuid4())
        # ChromaDB rifiuta None nei singoli valori dei metadati e, dalla 1.x,
        # rifiuta anche i dict vuoti: in quel caso passa None (accettato).
        meta = {k: v for k, v in (metadata or {}).items() if v is not None}
        meta = meta or None

        # Embedding via OllamaClient (nomic-embed-text)
        vectors = await self._llm.embed(text)
        embedding: list[float] = vectors[0]

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: _chroma_add(self._collection, cid, embedding, text, meta),
        )

        elapsed = (time.monotonic() - t0) * 1000
        result  = SaveResult(chunk_id=cid, text_len=len(text), elapsed_ms=elapsed)
        logger.debug("memory.save | {}", result.to_log_dict())
        return result

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> list[MemoryChunk]:
        """
        Ricerca semantica per similarità coseno.

        Args:
            query:  Testo di query.
            top_k:  Numero massimo di risultati
                    (default: settings.memory.rag_top_k).

        Returns:
            Lista di MemoryChunk ordinati per rilevanza decrescente.
            Lista vuota se la collection è vuota.
        """
        self._require_loaded()
        t0 = time.monotonic()
        k  = top_k if top_k is not None else self._top_k

        loop  = asyncio.get_running_loop()
        count = await loop.run_in_executor(None, lambda: self._collection.count())
        if count == 0:
            logger.debug("memory.search | collection vuota — skip")
            return []

        # Embedding della query
        vectors   = await self._llm.embed(query)
        embedding = vectors[0]

        # Memoria per-sessione: se session_id è dato, recupera solo i suoi chunk.
        where    = {"session_id": session_id} if session_id else None
        actual_k = min(k, count)
        raw      = await loop.run_in_executor(
            None,
            lambda: _chroma_query(self._collection, embedding, actual_k, where=where),
        )

        docs:      list[str]           = (raw.get("documents") or [[]])[0]
        metas:     list[dict]          = (raw.get("metadatas") or [[]])[0]
        distances: list[float]         = (raw.get("distances") or [[]])[0]

        chunks: list[MemoryChunk] = []
        for doc, meta, dist in zip(docs, metas, distances):
            # Distanza coseno ∈ [0, 2] → rilevanza ∈ [0, 1]
            # dist=0 → identici → relevance=1.0
            relevance = round(max(0.0, 1.0 - dist), 4)
            source    = str((meta or {}).get("source", "memory"))
            chunks.append(MemoryChunk(
                content=doc,
                source=source,
                relevance_score=relevance,
                metadata=dict(meta or {}),
            ))

        elapsed = (time.monotonic() - t0) * 1000
        sr = SearchResult(
            chunks=chunks, query_len=len(query),
            elapsed_ms=elapsed, top_k=k,
        )
        logger.debug("memory.search | {}", sr.to_log_dict())
        return chunks

    async def delete(self, chunk_id: str) -> bool:
        """
        Rimuove un chunk per ID.

        Returns:
            True se l'operazione è riuscita, False in caso di errore.
        """
        self._require_loaded()
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: _chroma_delete(self._collection, chunk_id),
            )
            logger.debug("memory.delete | id={}", chunk_id)
            return True
        except Exception as exc:
            logger.warning("memory.delete | errore id={}: {}", chunk_id, exc)
            return False

    async def clear(self) -> int:
        """
        Rimuove tutti i chunk dalla collection.

        Returns:
            Numero di chunk rimossi.
        """
        self._require_loaded()
        loop = asyncio.get_running_loop()
        n = await loop.run_in_executor(
            None,
            lambda: _chroma_clear(self._collection),
        )
        logger.info("memory.clear | rimossi {} chunk", n)
        return n

    async def count(self) -> int:
        """Numero di chunk presenti nella collection."""
        self._require_loaded()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._collection.count())

    async def get_all(self) -> list[dict]:
        """
        Tutti i chunk della collection (testo + metadata), senza ordine
        garantito e senza similarità: per le scansioni complete (es. map-reduce),
        non per il retrieval mirato.
        """
        self._require_loaded()
        loop = asyncio.get_running_loop()

        def _fetch() -> list[dict]:
            res   = self._collection.get(include=["documents", "metadatas"])
            docs  = res.get("documents") or []
            metas = res.get("metadatas") or []
            return [{"text": d, "metadata": m or {}} for d, m in zip(docs, metas)]

        return await loop.run_in_executor(None, _fetch)

    async def populate_context(
        self,
        ctx:   AssistantContext,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """
        Popola AssistantContext.retrieved_memories con i chunk più rilevanti.
        Imposta anche ctx.timings["memory"].

        Args:
            ctx:    Contesto da popolare.
            query:  Testo di ricerca (default: ctx.user_text).
            top_k:  Numero massimo di chunk (default: settings.memory.rag_top_k).
        """
        self._require_loaded()
        q = query or ctx.user_text
        if not q.strip():
            logger.debug("memory.populate | query vuota — skip (turn={})", ctx.turn_id)
            return

        t0     = time.monotonic()
        # Recupera solo dalla sessione corrente: ciò che è stato detto in una
        # chat non deve riaffiorare in un'altra.
        chunks = await self.search(q, top_k=top_k, session_id=ctx.session_id)
        ctx.retrieved_memories = chunks
        ctx.set_timing("memory", (time.monotonic() - t0) * 1000)

        logger.debug(
            "memory.populate | turn={} chunks={}",
            ctx.turn_id,
            len(chunks),
        )

    # -- interno ---------------------------------------------------------------

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "MemoryManager non inizializzato — "
                "usa 'async with MemoryManager()' oppure chiama 'await mem.load()'"
            )

    # -- rappresentazione ------------------------------------------------------

    def __repr__(self) -> str:
        if self._loaded:
            return (
                f"<MemoryManager collection='{self._collection_name}' "
                f"dir='{self._persist_dir}'>"
            )
        return "<MemoryManager [non caricato]>"
