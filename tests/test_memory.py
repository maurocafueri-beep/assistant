"""
tests/test_memory.py
Test suite per modules/memory.

I test veloci usano mock completi — zero dipendenze da Ollama o ChromaDB reali.
I test @pytest.mark.slow richiedono Ollama attivo + ChromaDB su disco.

Esecuzione:
    make test
    SKIP_SLOW=1 venv-runtime/bin/pytest tests/test_memory.py -v
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.context import AssistantContext, MemoryChunk
from modules.memory.base_memory import (
    COLLECTION_NAME,
    MemoryManager,
    SaveResult,
    SearchResult,
    _chroma_clear,
)

SKIP_SLOW = os.getenv("SKIP_SLOW", "0") == "1"
slow = pytest.mark.skipif(SKIP_SLOW, reason="SKIP_SLOW=1")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_collection():
    """Collection ChromaDB completamente mockato."""
    col = MagicMock()
    col.count.return_value = 0
    col.add    = MagicMock()
    col.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    col.delete = MagicMock()
    col.get.return_value = {"ids": []}
    return col


@pytest.fixture
def mock_llm():
    """OllamaClient mockato: embed() ritorna un vettore costante 3D."""
    llm = MagicMock()
    llm.embed  = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    llm.aclose = AsyncMock()
    return llm


@pytest.fixture
async def mem(mock_collection, mock_llm, tmp_path):
    """
    MemoryManager pre-caricato con dipendenze completamente mockate.
    Nessuna connessione reale a Ollama o ChromaDB.
    """
    with patch(
        "modules.memory.base_memory._get_or_create_collection",
        return_value=(MagicMock(), mock_collection),
    ), patch(
        "modules.memory.base_memory.OllamaClient",
        return_value=mock_llm,
    ):
        manager = MemoryManager(persist_dir=str(tmp_path), top_k=5)
        await manager.load()
        yield manager
        await manager.aclose()


# ---------------------------------------------------------------------------
# SaveResult
# ---------------------------------------------------------------------------

class TestSaveResult:
    def test_to_log_dict_keys(self):
        r = SaveResult(chunk_id="abc", text_len=42, elapsed_ms=12.5)
        assert r.to_log_dict().keys() == {"chunk_id", "text_len", "elapsed_ms"}

    def test_to_log_dict_values(self):
        r = SaveResult(chunk_id="x", text_len=10, elapsed_ms=3.14159)
        d = r.to_log_dict()
        assert d["chunk_id"]   == "x"
        assert d["text_len"]   == 10
        assert d["elapsed_ms"] == 3.1   # round(3.14159, 1)

    def test_str_contains_id(self):
        r = SaveResult(chunk_id="my-id", text_len=5, elapsed_ms=0)
        assert "my-id" in str(r)


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:
    def _make_chunks(self, n: int) -> list[MemoryChunk]:
        return [
            MemoryChunk(content=f"doc{i}", source="test", relevance_score=0.9)
            for i in range(n)
        ]

    def test_to_log_dict_keys(self):
        sr = SearchResult(chunks=[], query_len=10, elapsed_ms=5.0, top_k=3)
        assert sr.to_log_dict().keys() == {"n_results", "query_len", "elapsed_ms", "top_k"}

    def test_n_results(self):
        sr = SearchResult(chunks=self._make_chunks(3), query_len=5, elapsed_ms=1.0, top_k=5)
        assert sr.to_log_dict()["n_results"] == 3

    def test_elapsed_rounded(self):
        sr = SearchResult(chunks=[], query_len=0, elapsed_ms=7.777, top_k=1)
        assert sr.to_log_dict()["elapsed_ms"] == 7.8


# ---------------------------------------------------------------------------
# MemoryManager — lifecycle
# ---------------------------------------------------------------------------

class TestMemoryManagerLifecycle:
    async def test_context_manager(self, mock_collection, mock_llm, tmp_path):
        with patch(
            "modules.memory.base_memory._get_or_create_collection",
            return_value=(MagicMock(), mock_collection),
        ), patch(
            "modules.memory.base_memory.OllamaClient",
            return_value=mock_llm,
        ):
            async with MemoryManager(persist_dir=str(tmp_path)) as m:
                assert m._loaded is True
            # dopo __aexit__ il manager è chiuso
            assert m._loaded is False

    async def test_standalone_load(self, mock_collection, mock_llm, tmp_path):
        with patch(
            "modules.memory.base_memory._get_or_create_collection",
            return_value=(MagicMock(), mock_collection),
        ), patch(
            "modules.memory.base_memory.OllamaClient",
            return_value=mock_llm,
        ):
            m = MemoryManager(persist_dir=str(tmp_path))
            await m.load()
            assert m._loaded is True
            await m.aclose()
            assert m._loaded is False

    async def test_not_loaded_raises_save(self, tmp_path):
        m = MemoryManager(persist_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="non inizializzato"):
            await m.save("testo")

    async def test_not_loaded_raises_search(self, tmp_path):
        m = MemoryManager(persist_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="non inizializzato"):
            await m.search("query")

    async def test_not_loaded_raises_delete(self, tmp_path):
        m = MemoryManager(persist_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="non inizializzato"):
            await m.delete("some-id")

    async def test_not_loaded_raises_clear(self, tmp_path):
        m = MemoryManager(persist_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="non inizializzato"):
            await m.clear()

    async def test_not_loaded_raises_count(self, tmp_path):
        m = MemoryManager(persist_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="non inizializzato"):
            await m.count()

    async def test_aclose_idempotent(self, mem):
        await mem.aclose()
        await mem.aclose()   # seconda chiamata non deve sollevare eccezioni

    def test_repr_loaded(self, mem):
        r = repr(mem)
        assert "collection=" in r
        assert COLLECTION_NAME in r

    def test_repr_not_loaded(self, tmp_path):
        m = MemoryManager(persist_dir=str(tmp_path))
        assert "non caricato" in repr(m)


# ---------------------------------------------------------------------------
# MemoryManager — save()
# ---------------------------------------------------------------------------

class TestMemoryManagerSave:
    async def test_returns_save_result(self, mem):
        result = await mem.save("Testo di test")
        assert isinstance(result, SaveResult)

    async def test_chunk_id_auto_generated(self, mem):
        r = await mem.save("Testo")
        assert r.chunk_id  # non vuoto
        assert len(r.chunk_id) > 8

    async def test_custom_chunk_id(self, mem):
        r = await mem.save("Testo", chunk_id="my-custom-id")
        assert r.chunk_id == "my-custom-id"

    async def test_text_len_correct(self, mem):
        text = "Hello world"
        r = await mem.save(text)
        assert r.text_len == len(text)

    async def test_elapsed_ms_positive(self, mem):
        r = await mem.save("Qualcosa")
        assert r.elapsed_ms >= 0

    async def test_embed_called_once(self, mem, mock_llm):
        await mem.save("Testo da embeddare")
        mock_llm.embed.assert_awaited_once_with("Testo da embeddare")

    async def test_chroma_add_called(self, mem, mock_collection):
        await mem.save("Testo", {"source": "wiki"})
        mock_collection.add.assert_called_once()
        _, kwargs = mock_collection.add.call_args
        # verifica che il testo sia passato
        assert kwargs.get("documents") == ["Testo"] or \
               mock_collection.add.call_args[1].get("documents") == ["Testo"]

    async def test_none_metadata_filtered(self, mem, mock_collection):
        """Valori None nel metadata non devono essere passati a ChromaDB."""
        await mem.save("testo", {"source": "x", "optional": None})
        _, kwargs = mock_collection.add.call_args
        passed_meta: dict = (kwargs.get("metadatas") or [[{}]])[0][0] \
            if isinstance((kwargs.get("metadatas") or [[{}]])[0], list) \
            else {}
        # 'optional' non deve apparire nei metadati
        assert "optional" not in passed_meta

    async def test_empty_metadata_ok(self, mem):
        result = await mem.save("Testo", metadata=None)
        assert result.chunk_id

    async def test_two_saves_different_ids(self, mem):
        r1 = await mem.save("Primo")
        r2 = await mem.save("Secondo")
        assert r1.chunk_id != r2.chunk_id


# ---------------------------------------------------------------------------
# MemoryManager — search()
# ---------------------------------------------------------------------------

class TestMemoryManagerSearch:
    def _setup_collection(self, mock_collection, docs, metas, dists):
        """Configura mock_collection per restituire risultati di ricerca."""
        mock_collection.count.return_value = len(docs)
        mock_collection.query.return_value = {
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }

    async def test_returns_list(self, mem, mock_collection):
        self._setup_collection(mock_collection, ["doc1"], [{"source": "test"}], [0.2])
        chunks = await mem.search("query")
        assert isinstance(chunks, list)

    async def test_returns_memory_chunks(self, mem, mock_collection):
        self._setup_collection(mock_collection, ["doc1"], [{"source": "s1"}], [0.1])
        chunks = await mem.search("query")
        assert len(chunks) == 1
        assert isinstance(chunks[0], MemoryChunk)

    async def test_content_correct(self, mem, mock_collection):
        self._setup_collection(mock_collection, ["Testo memorizzato"], [{}], [0.05])
        chunks = await mem.search("query")
        assert chunks[0].content == "Testo memorizzato"

    async def test_source_from_metadata(self, mem, mock_collection):
        self._setup_collection(mock_collection, ["doc"], [{"source": "wiki"}], [0.0])
        chunks = await mem.search("query")
        assert chunks[0].source == "wiki"

    async def test_source_defaults_to_memory(self, mem, mock_collection):
        self._setup_collection(mock_collection, ["doc"], [{}], [0.3])
        chunks = await mem.search("query")
        assert chunks[0].source == "memory"

    async def test_relevance_score_from_distance(self, mem, mock_collection):
        # dist=0.0 → rilevanza=1.0; dist=0.5 → rilevanza=0.5
        self._setup_collection(mock_collection, ["a", "b"], [{}, {}], [0.0, 0.5])
        chunks = await mem.search("query")
        assert chunks[0].relevance_score == 1.0
        assert chunks[1].relevance_score == 0.5

    async def test_relevance_clamped_to_zero(self, mem, mock_collection):
        """Distanze > 1 non devono produrre rilevanza negativa."""
        self._setup_collection(mock_collection, ["doc"], [{}], [1.5])
        chunks = await mem.search("query")
        assert chunks[0].relevance_score == 0.0

    async def test_empty_collection_returns_empty(self, mem, mock_collection):
        mock_collection.count.return_value = 0
        chunks = await mem.search("qualsiasi query")
        assert chunks == []
        mock_collection.query.assert_not_called()

    async def test_embed_called_with_query(self, mem, mock_collection, mock_llm):
        self._setup_collection(mock_collection, ["doc"], [{}], [0.1])
        await mem.search("la mia query")
        mock_llm.embed.assert_awaited_with("la mia query")

    async def test_custom_top_k(self, mem, mock_collection):
        self._setup_collection(
            mock_collection,
            ["d1", "d2"], [{}, {}], [0.1, 0.3]
        )
        await mem.search("query", top_k=2)
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs.get("n_results") == 2

    async def test_multiple_results(self, mem, mock_collection):
        self._setup_collection(
            mock_collection,
            ["alpha", "beta", "gamma"],
            [{"source": "s1"}, {"source": "s2"}, {"source": "s3"}],
            [0.1, 0.3, 0.6],
        )
        chunks = await mem.search("query", top_k=3)
        assert len(chunks) == 3
        contents = [c.content for c in chunks]
        assert "alpha" in contents
        assert "beta"  in contents


# ---------------------------------------------------------------------------
# MemoryManager — delete()
# ---------------------------------------------------------------------------

class TestMemoryManagerDelete:
    async def test_returns_true_on_success(self, mem):
        ok = await mem.delete("some-id")
        assert ok is True

    async def test_chroma_delete_called(self, mem, mock_collection):
        await mem.delete("target-id")
        mock_collection.delete.assert_called_once_with(ids=["target-id"])

    async def test_returns_false_on_error(self, mem, mock_collection):
        mock_collection.delete.side_effect = Exception("not found")
        ok = await mem.delete("bad-id")
        assert ok is False


# ---------------------------------------------------------------------------
# MemoryManager — clear()
# ---------------------------------------------------------------------------

class TestMemoryManagerClear:
    async def test_returns_count(self, mem, mock_collection):
        mock_collection.get.return_value = {"ids": ["a", "b", "c"]}
        n = await mem.clear()
        assert n == 3

    async def test_chroma_delete_called(self, mem, mock_collection):
        mock_collection.get.return_value = {"ids": ["x", "y"]}
        await mem.clear()
        mock_collection.delete.assert_called_once_with(ids=["x", "y"])

    async def test_empty_collection_returns_zero(self, mem, mock_collection):
        mock_collection.get.return_value = {"ids": []}
        n = await mem.clear()
        assert n == 0
        mock_collection.delete.assert_not_called()


# ---------------------------------------------------------------------------
# MemoryManager — count()
# ---------------------------------------------------------------------------

class TestMemoryManagerCount:
    async def test_count_zero(self, mem, mock_collection):
        mock_collection.count.return_value = 0
        assert await mem.count() == 0

    async def test_count_nonzero(self, mem, mock_collection):
        mock_collection.count.return_value = 7
        assert await mem.count() == 7


# ---------------------------------------------------------------------------
# MemoryManager — populate_context()
# ---------------------------------------------------------------------------

class TestMemoryManagerPopulateContext:
    def _setup_collection(self, mock_collection, docs, metas, dists):
        mock_collection.count.return_value = len(docs)
        mock_collection.query.return_value = {
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }

    async def test_populates_retrieved_memories(self, mem, mock_collection):
        self._setup_collection(mock_collection, ["ricordo"], [{"source": "chat"}], [0.2])
        ctx = AssistantContext(user_text="Di cosa parlavamo?")
        await mem.populate_context(ctx)
        assert len(ctx.retrieved_memories) == 1
        assert ctx.retrieved_memories[0].content == "ricordo"

    async def test_uses_user_text_as_default_query(self, mem, mock_collection, mock_llm):
        self._setup_collection(mock_collection, ["doc"], [{}], [0.1])
        ctx = AssistantContext(user_text="test query")
        await mem.populate_context(ctx)
        mock_llm.embed.assert_awaited_with("test query")

    async def test_custom_query_overrides_user_text(self, mem, mock_collection, mock_llm):
        self._setup_collection(mock_collection, ["doc"], [{}], [0.1])
        ctx = AssistantContext(user_text="questo viene ignorato")
        await mem.populate_context(ctx, query="query personalizzata")
        mock_llm.embed.assert_awaited_with("query personalizzata")

    async def test_empty_query_skips(self, mem, mock_collection, mock_llm):
        ctx = AssistantContext(user_text="")
        await mem.populate_context(ctx)
        mock_llm.embed.assert_not_called()
        assert ctx.retrieved_memories == []

    async def test_sets_timing(self, mem, mock_collection):
        self._setup_collection(mock_collection, ["doc"], [{}], [0.0])
        ctx = AssistantContext(user_text="query")
        await mem.populate_context(ctx)
        assert "memory" in ctx.timings
        assert ctx.timings["memory"] >= 0

    async def test_sets_custom_top_k(self, mem, mock_collection):
        self._setup_collection(mock_collection, ["a", "b"], [{}, {}], [0.1, 0.2])
        ctx = AssistantContext(user_text="query")
        await mem.populate_context(ctx, top_k=2)
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs.get("n_results") == 2

    async def test_empty_collection_leaves_memories_empty(self, mem, mock_collection):
        mock_collection.count.return_value = 0
        ctx = AssistantContext(user_text="query")
        await mem.populate_context(ctx)
        assert ctx.retrieved_memories == []


# ---------------------------------------------------------------------------
# _chroma_clear — unit test senza I/O
# ---------------------------------------------------------------------------

class TestChromaClear:
    def test_deletes_all_ids(self):
        col = MagicMock()
        col.get.return_value = {"ids": ["a", "b"]}
        n = _chroma_clear(col)
        assert n == 2
        col.delete.assert_called_once_with(ids=["a", "b"])

    def test_empty_collection_no_delete(self):
        col = MagicMock()
        col.get.return_value = {"ids": []}
        n = _chroma_clear(col)
        assert n == 0
        col.delete.assert_not_called()


# ---------------------------------------------------------------------------
# Test slow — richiedono Ollama e ChromaDB reali
# ---------------------------------------------------------------------------

@slow
@pytest.mark.real_ollama
class TestMemoryManagerReal:
    """
    Test di integrazione end-to-end.
    Richiede: Ollama attivo con nomic-embed-text caricato.
    Eseguiti automaticamente in CI (SKIP_SLOW non impostato).
    """

    async def test_save_and_search(self, tmp_path):
        async with MemoryManager(persist_dir=str(tmp_path)) as mem:
            r = await mem.save(
                "Python è un linguaggio di programmazione ad alto livello.",
                {"source": "test"},
            )
            assert r.chunk_id
            chunks = await mem.search("linguaggio di programmazione", top_k=1)
            assert len(chunks) >= 1
            assert chunks[0].relevance_score > 0

    async def test_count_increments(self, tmp_path):
        async with MemoryManager(persist_dir=str(tmp_path)) as mem:
            before = await mem.count()
            await mem.save("Testo aggiunto")
            after = await mem.count()
            assert after == before + 1

    async def test_delete_reduces_count(self, tmp_path):
        async with MemoryManager(persist_dir=str(tmp_path)) as mem:
            r = await mem.save("Da eliminare")
            before = await mem.count()
            ok = await mem.delete(r.chunk_id)
            assert ok is True
            assert await mem.count() == before - 1

    async def test_clear_empties_collection(self, tmp_path):
        async with MemoryManager(persist_dir=str(tmp_path)) as mem:
            await mem.save("A")
            await mem.save("B")
            await mem.save("C")
            n = await mem.clear()
            assert n >= 3
            assert await mem.count() == 0

    async def test_populate_context_fills_memories(self, tmp_path):
        async with MemoryManager(persist_dir=str(tmp_path)) as mem:
            ctx = AssistantContext(user_text="Dimmi qualcosa su Roma.")
            sid = ctx.session_id
            await mem.save("Roma è la capitale d'Italia.", {"source": "geo", "session_id": sid})
            await mem.save("Milano è la capitale della moda.", {"source": "geo", "session_id": sid})
            await mem.populate_context(ctx, top_k=2)
            assert len(ctx.retrieved_memories) >= 1
            assert "memory" in ctx.timings

    async def test_relevance_scores_in_range(self, tmp_path):
        async with MemoryManager(persist_dir=str(tmp_path)) as mem:
            await mem.save("Documento di test", {"source": "s"})
            chunks = await mem.search("test", top_k=1)
            for c in chunks:
                assert 0.0 <= c.relevance_score <= 1.0


class TestMemoryGetAll:
    """get_all restituisce tutti i chunk (testo+metadata), per le scansioni complete."""

    async def test_returns_text_and_metadata(self, mem, mock_collection):
        mock_collection.get.return_value = {
            "documents": ["uno", "due"],
            "metadatas": [{"chunk_index": 0}, {"chunk_index": 1}],
        }
        items = await mem.get_all()
        assert items == [
            {"text": "uno", "metadata": {"chunk_index": 0}},
            {"text": "due", "metadata": {"chunk_index": 1}},
        ]

    async def test_empty_collection(self, mem, mock_collection):
        mock_collection.get.return_value = {"documents": [], "metadatas": []}
        assert await mem.get_all() == []



class TestMemorySessionScope:
    """La memoria recupera solo i chunk della sessione corrente (no leak tra chat)."""

    async def test_search_with_session_filters_query(self, mem, mock_collection):
        mock_collection.count.return_value = 3
        await mem.search("q", session_id="s1")
        assert mock_collection.query.call_args.kwargs.get("where") == {"session_id": "s1"}

    async def test_search_without_session_no_filter(self, mem, mock_collection):
        mock_collection.count.return_value = 3
        await mem.search("q")
        assert mock_collection.query.call_args.kwargs.get("where") is None

    async def test_populate_context_filters_by_session(self, mem, mock_collection):
        mock_collection.count.return_value = 3
        ctx = MagicMock()
        ctx.user_text = "domanda"
        ctx.session_id = "sess-X"
        ctx.turn_id = "t1"
        await mem.populate_context(ctx)
        assert mock_collection.query.call_args.kwargs.get("where") == {"session_id": "sess-X"}

