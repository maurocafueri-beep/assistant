"""
tests/test_file_rag.py
Test per modules/file_rag — chunking puro + FileRAG con MemoryManager fake.

Le funzioni di chunking/hashing sono pure: testate direttamente. FileRAG è
testato con un fake di MemoryManager (in-memory) così verifichiamo idempotenza,
metadata e ciclo di vita senza dipendere da ChromaDB/Ollama.
"""

from __future__ import annotations

import pytest

from modules.file_rag.base_file_rag import (
    FILE_COLLECTION_PREFIX,
    FileChunk,
    FileRAG,
    chunk_text,
    collection_name_for,
    compute_file_id,
    _split_pages,
)


# ---------------------------------------------------------------------------
# Funzioni pure: hashing & naming
# ---------------------------------------------------------------------------

class TestFileId:
    def test_stable_for_same_content(self):
        a = compute_file_id("contenuto del libro")
        b = compute_file_id("contenuto del libro")
        assert a == b

    def test_differs_for_different_content(self):
        assert compute_file_id("libro uno") != compute_file_id("libro due")

    def test_length_16(self):
        assert len(compute_file_id("x")) == 16

    def test_collection_name(self):
        fid = compute_file_id("x")
        assert collection_name_for(fid) == f"{FILE_COLLECTION_PREFIX}{fid}"


# ---------------------------------------------------------------------------
# _split_pages
# ---------------------------------------------------------------------------

class TestSplitPages:
    def test_pdf_markers(self):
        text = "--- pagina 1 ---\nPrima pagina.\n\n--- pagina 2 ---\nSeconda pagina."
        pages = _split_pages(text)
        assert pages == [(1, "Prima pagina."), (2, "Seconda pagina.")]

    def test_no_markers_single_block(self):
        text = "Un documento senza marker di pagina, tipo un txt."
        pages = _split_pages(text)
        assert len(pages) == 1
        assert pages[0][0] == 0  # pagina ignota
        assert "documento senza marker" in pages[0][1]

    def test_empty(self):
        assert _split_pages("   ") == []

    def test_skips_empty_pages(self):
        text = "--- pagina 1 ---\nTesto.\n\n--- pagina 2 ---\n\n--- pagina 3 ---\nAltro."
        pages = _split_pages(text)
        # pagina 2 vuota viene saltata
        assert [p[0] for p in pages] == [1, 3]


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_short_pages_accorpate(self):
        # 3 pagine corte → dovrebbero stare in 1 chunk con target ampio
        text = (
            "--- pagina 1 ---\nAlfa.\n\n"
            "--- pagina 2 ---\nBeta.\n\n"
            "--- pagina 3 ---\nGamma."
        )
        chunks = chunk_text(text, target_tokens=800)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.page_start == 1 and c.page_end == 3
        assert "Alfa" in c.text and "Gamma" in c.text

    def test_long_page_split(self):
        # 1 pagina molto lunga → più chunk, stessa pagina
        body = "Frase. " * 2000  # ~14000 char
        text = f"--- pagina 5 ---\n{body}"
        chunks = chunk_text(text, target_tokens=200)  # ~800 char target
        assert len(chunks) > 1
        assert all(c.page_start == 5 and c.page_end == 5 for c in chunks)

    def test_chunk_indices_sequential(self):
        text = "--- pagina 1 ---\n" + ("x " * 3000)
        chunks = chunk_text(text, target_tokens=100)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_accorpamento_rispetta_target(self):
        # 10 pagine da ~400 char con target ~800 char (200 tok) → ~5 chunk
        pages = "".join(
            f"--- pagina {i} ---\n{'a' * 400}\n\n" for i in range(1, 11)
        )
        chunks = chunk_text(pages, target_tokens=200)
        # Ogni chunk non deve superare di molto il target (tolleranza: 1 pagina)
        assert all(len(c.text) <= 200 * 4 + 400 for c in chunks)
        assert len(chunks) >= 4

    def test_non_pdf_text(self):
        # Testo senza marker → 1 blocco pagina 0, eventualmente spezzato
        text = "parola " * 50
        chunks = chunk_text(text, target_tokens=800)
        assert len(chunks) == 1
        assert chunks[0].page_start == 0

    def test_empty_text(self):
        assert chunk_text("", target_tokens=800) == []

    def test_metadata_scalar_only(self):
        c = FileChunk(text="x", page_start=1, page_end=2, index=0)
        meta = c.to_metadata("abc123", "libro.pdf")
        assert meta["file_id"] == "abc123"
        assert meta["source"] == "libro.pdf"
        assert meta["page_start"] == 1 and meta["page_end"] == 2
        # tutti scalari (ChromaDB non accetta liste/dict nei metadata)
        assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())


# ---------------------------------------------------------------------------
# FileRAG con MemoryManager fake
# ---------------------------------------------------------------------------

class _FakeManager:
    """MemoryManager fake in-memory per testare FileRAG senza ChromaDB."""
    def __init__(self, *a, **k):
        self.collection_name = k.get("collection_name", "")
        self._store: list[tuple[str, str, dict]] = []
        self.loaded = False

    async def load(self):
        self.loaded = True

    async def aclose(self):
        self.loaded = False

    async def count(self):
        return len(self._store)

    async def save(self, text, metadata=None, chunk_id=None):
        self._store.append((chunk_id, text, metadata or {}))
        class _R:  # SaveResult-like
            pass
        r = _R(); r.chunk_id = chunk_id; r.text_len = len(text); r.elapsed_ms = 0.1
        return r

    async def search(self, query, top_k=None):
        from core.context import MemoryChunk
        out = []
        for cid, text, meta in self._store[: (top_k or 5)]:
            out.append(MemoryChunk(
                content=text, source=meta.get("source", "file"),
                relevance_score=0.9, metadata=meta,
            ))
        return out

    async def clear(self):
        n = len(self._store); self._store.clear(); return n


@pytest.fixture
def rag(monkeypatch):
    """FileRAG che usa _FakeManager invece del vero MemoryManager."""
    import modules.file_rag.base_file_rag as mod
    monkeypatch.setattr(mod, "MemoryManager", _FakeManager)
    r = FileRAG(persist_dir="/tmp/fake", chunk_tokens=200, top_k=3)
    return r


class TestFileRAG:
    async def test_index_then_search(self, rag):
        await rag.load()
        text = "--- pagina 1 ---\n" + ("contenuto " * 500)
        res = await rag.index_file(text, source="libro.pdf")
        assert res.chunks_indexed > 0
        assert not res.already_indexed
        chunks = await rag.search_file(res.file_id, "contenuto")
        assert len(chunks) > 0
        assert chunks[0].metadata["file_id"] == res.file_id

    async def test_idempotent(self, rag):
        await rag.load()
        text = "--- pagina 1 ---\n" + ("testo " * 500)
        first = await rag.index_file(text, source="x.pdf")
        second = await rag.index_file(text, source="x.pdf")
        assert second.already_indexed
        assert second.chunks_indexed == first.chunks_indexed

    async def test_search_empty_query(self, rag):
        await rag.load()
        res = await rag.index_file("--- pagina 1 ---\nabc", source="x.pdf")
        assert await rag.search_file(res.file_id, "   ") == []

    async def test_is_indexed(self, rag):
        await rag.load()
        res = await rag.index_file("--- pagina 1 ---\n" + "z " * 300, source="x.pdf")
        assert await rag.is_indexed(res.file_id) is True
        assert await rag.is_indexed("inesistente0000000") is False

    async def test_drop(self, rag):
        await rag.load()
        res = await rag.index_file("--- pagina 1 ---\n" + "z " * 300, source="x.pdf")
        assert await rag.drop_file(res.file_id) is True

    async def test_same_content_same_id(self, rag):
        await rag.load()
        a = await rag.index_file("--- pagina 1 ---\nidentico", source="a.pdf")
        # nuovo FileRAG, stesso contenuto → stesso file_id
        assert a.file_id == compute_file_id("--- pagina 1 ---\nidentico")
