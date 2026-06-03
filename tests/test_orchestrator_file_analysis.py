"""
tests/test_orchestrator_file_analysis.py
Test suite per l'integrazione file_analysis in core/orchestrator.py.

Copre gli helper puri (_extract_file_paths, _format_file_analysis_block,
FILE_PATH_RE) e il metodo _run_file_analysis con mock completi —
nessuna analisi reale di file (FileAnalyzer è mockato).

asyncio_mode = "auto" → @pytest.mark.asyncio non necessario.

Esecuzione:
    venv-runtime/bin/pytest tests/test_orchestrator_file_analysis.py -v
"""

from __future__ import annotations

from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.context import AssistantContext, InputMode, MemoryChunk, ModelRole, OutputMode
from core.orchestrator import (
    Orchestrator,
    OrchestratorStatus,
    _extract_file_paths,
    _format_file_analysis_block,
    _format_file_rag_block,
)
from modules.file_analysis import AnalysisResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ctx(**kwargs) -> AssistantContext:
    defaults = dict(
        user_text="Ciao, come stai?",
        session_id="sess-test",
        input_mode=InputMode.TEXT,
        output_mode=OutputMode.TEXT,
        model_role=ModelRole.CHAT,
        system_prompt="Sei un assistente utile.",
        personality_name="dev",
    )
    defaults.update(kwargs)
    return AssistantContext(**defaults)


def _make_result(
    path: str = "/tmp/x.pdf",
    file_type: str = "pdf",
    content: str = "Contenuto di esempio.",
    error: str | None = None,
    truncated: bool = False,
    metadata: dict | None = None,
    full_content: str | None = None,
) -> AnalysisResult:
    """Costruisce un AnalysisResult per i mock."""
    return AnalysisResult(
        path=path,
        file_type=file_type,
        content=content if error is None else "",
        char_count=len(content) if error is None else 0,
        original_char_count=len(content) if error is None else 0,
        truncated=truncated,
        elapsed_ms=12.3,
        extractor=f"{file_type}_mock",
        error=error,
        metadata=metadata if metadata is not None else {},
        full_content=full_content,
    )


def _make_fa_mock(results: list[AnalysisResult] | None = None) -> MagicMock:
    """
    Mock di FileAnalyzer: analyze() restituisce risultati nell'ordine
    in cui viene chiamato.
    """
    fa = MagicMock()
    if results is None:
        fa.analyze = AsyncMock(return_value=_make_result())
    else:
        fa.analyze = AsyncMock(side_effect=list(results))
    fa.aclose = AsyncMock()
    return fa


def _make_personality_mock(allows: bool = True) -> MagicMock:
    pm = MagicMock()
    active = MagicMock()
    active.allows_tool = MagicMock(return_value=allows)
    pm.active = active
    return pm


# ===========================================================================
# _extract_file_paths
# ===========================================================================

class TestExtractFilePaths:
    def test_empty_string(self):
        assert _extract_file_paths("") == []

    def test_no_path_in_text(self):
        assert _extract_file_paths("Ciao, come stai oggi?") == []

    def test_absolute_path(self):
        paths = _extract_file_paths("Apri /home/mauro/Scaricati/contratto.pdf grazie")
        assert paths == ["/home/mauro/Scaricati/contratto.pdf"]

    def test_home_relative_path(self):
        paths = _extract_file_paths("riassumi ~/Scaricati/note.md per favore")
        assert paths == ["~/Scaricati/note.md"]

    def test_dot_relative_path(self):
        paths = _extract_file_paths("controlla ./report.csv adesso")
        assert paths == ["./report.csv"]

    def test_parent_relative_path(self):
        paths = _extract_file_paths("guarda ../docs/manual.pdf")
        assert paths == ["../docs/manual.pdf"]

    def test_multiple_paths(self):
        paths = _extract_file_paths(
            "Confronta ~/a.pdf con ~/Scaricati/b.pdf"
        )
        assert paths == ["~/a.pdf", "~/Scaricati/b.pdf"]

    def test_dedup_preserves_order(self):
        paths = _extract_file_paths(
            "Analizza ~/x.txt poi commenta ~/x.txt e ~/y.txt"
        )
        assert paths == ["~/x.txt", "~/y.txt"]

    def test_max_paths_cap(self):
        paths = _extract_file_paths(
            "Tre file: ~/a.pdf ~/b.pdf ~/c.pdf ~/d.pdf ~/e.pdf",
            max_paths=3,
        )
        assert len(paths) == 3
        assert paths == ["~/a.pdf", "~/b.pdf", "~/c.pdf"]

    def test_unsupported_extension_ignored(self):
        # .exe non è una delle estensioni supportate → non catturato
        assert _extract_file_paths("apri ~/foo.exe") == []

    def test_naked_filename_not_captured(self):
        # contratto.pdf senza dir → ambiguo, non catturato
        assert _extract_file_paths("ho visto contratto.pdf ieri") == []

    def test_case_insensitive_extension(self):
        paths = _extract_file_paths("vedi ~/X.PDF")
        assert paths == ["~/X.PDF"]

    def test_strips_trailing_punctuation(self):
        # Path seguito da virgola/punto/?/!: deve toglierli
        assert _extract_file_paths("apri ~/file.pdf.") == ["~/file.pdf"]
        assert _extract_file_paths("apri ~/file.pdf,") == ["~/file.pdf"]
        assert _extract_file_paths("apri ~/file.pdf?") == ["~/file.pdf"]

    def test_path_in_backticks(self):
        # I backtick vanno strippati
        paths = _extract_file_paths("apri `~/Scaricati/x.pdf` per me")
        assert paths == ["~/Scaricati/x.pdf"]

    def test_path_in_double_quotes(self):
        paths = _extract_file_paths('apri "~/foo.pdf"')
        assert paths == ["~/foo.pdf"]

    def test_audio_extensions(self):
        for ext in ("wav", "mp3", "ogg", "flac", "m4a", "opus"):
            paths = _extract_file_paths(f"trascrivi ~/Musica/x.{ext}")
            assert paths == [f"~/Musica/x.{ext}"], f"fallito per {ext}"

    def test_all_doc_extensions(self):
        for ext in ("pdf", "docx", "txt", "md", "log", "rst",
                    "yaml", "yml", "html", "htm", "json", "csv", "tsv", "xml"):
            paths = _extract_file_paths(f"vedi /tmp/x.{ext}")
            assert paths == [f"/tmp/x.{ext}"], f"fallito per {ext}"


# ===========================================================================
# _format_file_analysis_block
# ===========================================================================

class TestFormatFileAnalysisBlock:
    def test_empty_returns_empty(self):
        assert _format_file_analysis_block([]) == ""

    def test_skips_error_results(self):
        results = [_make_result(error="[not_found] missing")]
        assert _format_file_analysis_block(results) == ""

    def test_skips_empty_content(self):
        results = [_make_result(content="   \n\t")]
        assert _format_file_analysis_block(results) == ""

    def test_single_file_block(self):
        results = [_make_result(path="/tmp/a.pdf", content="Testo del PDF.")]
        block = _format_file_analysis_block(results)
        assert "CONTENUTO FILE ANALIZZATI" in block
        assert "/tmp/a.pdf" in block
        assert "Testo del PDF." in block
        assert "tipo: pdf" in block

    def test_multiple_files(self):
        results = [
            _make_result(path="/tmp/a.pdf", content="A"),
            _make_result(path="/tmp/b.md",  content="B", file_type="text"),
        ]
        block = _format_file_analysis_block(results)
        assert "/tmp/a.pdf" in block and "/tmp/b.md" in block
        assert block.count("tipo:") == 2

    def test_truncated_marked_in_header(self):
        results = [_make_result(
            path="/tmp/big.pdf",
            content="abc",
            truncated=True,
        )]
        # patch dei char_count perché _make_result li allinea a len(content)
        results[0].char_count          = 8000
        results[0].original_char_count = 50000
        block = _format_file_analysis_block(results)
        assert "troncato" in block
        assert "8000" in block and "50000" in block

    def test_rag_preview_truncates_when_full_content(self):
        # File grande (full_content valorizzato) + preview attivo: inline ridotto
        # a un estratto introduttivo, il dettaglio lo porta il RAG.
        big = "x" * 8000
        results = [_make_result(
            path="/tmp/libro.pdf", content=big,
            full_content="--- pagina 1 ---\n" + big,
        )]
        block = _format_file_analysis_block(results, rag_preview_chars=2000)
        assert "estratto introduttivo" in block
        assert "PASSAGGI RILEVANTI" in block        # il marcatore rimanda al RAG
        assert len(block) < 4000                    # molto piu' corto degli 8000 char

    def test_rag_preview_ignored_for_small_files(self):
        # File piccolo (full_content None): nessun troncamento anche con preview.
        results = [_make_result(
            path="/tmp/nota.txt", content="contenuto intero", file_type="text",
        )]
        block = _format_file_analysis_block(results, rag_preview_chars=2000)
        assert "contenuto intero" in block
        assert "estratto introduttivo" not in block


# ===========================================================================
# OrchestratorStatus — campo file_analysis_ok
# ===========================================================================

class TestOrchestratorStatusFileAnalysis:
    def test_default_false(self):
        s = OrchestratorStatus()
        assert s.file_analysis_ok is False

    def test_in_log_dict(self):
        s = OrchestratorStatus(file_analysis_ok=True)
        d = s.to_log_dict()
        assert d["file_analysis"] is True

    def test_log_dict_has_all_keys(self):
        s = OrchestratorStatus()
        d = s.to_log_dict()
        assert "file_analysis" in d
        # gli altri campi storici devono essere ancora presenti
        for k in ("llm", "memory", "personality", "stt", "tts",
                  "web_search", "sessions"):
            assert k in d


# ===========================================================================
# Orchestrator._run_file_analysis
# ===========================================================================

class TestRunFileAnalysisSkip:
    async def test_skip_when_analyzer_none(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._file_analyzer = None
        orch._file_rag = None
        ctx = make_ctx(user_text="riassumi ~/Scaricati/x.pdf")
        await orch._run_file_analysis(ctx)
        assert ctx.tool_calls == []

    async def test_skip_when_no_path_in_text(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._file_analyzer = _make_fa_mock()
        orch._file_rag = None
        orch._personality   = _make_personality_mock(allows=True)
        ctx = make_ctx(user_text="Spiegami i decoratori Python")
        await orch._run_file_analysis(ctx)
        orch._file_analyzer.analyze.assert_not_called()
        assert ctx.tool_calls == []

    async def test_skip_when_tool_not_allowed(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._file_analyzer = _make_fa_mock()
        orch._file_rag = None
        orch._personality   = _make_personality_mock(allows=False)
        ctx = make_ctx(user_text="riassumi ~/foo.pdf")
        await orch._run_file_analysis(ctx)
        orch._file_analyzer.analyze.assert_not_called()
        assert ctx.tool_calls == []


class TestRunFileAnalysisHappy:
    async def test_injects_content_into_system_prompt(self):
        result = _make_result(
            path="/home/mauro/Scaricati/contratto.pdf",
            content="Articolo 1. Le parti convengono...",
        )
        orch = Orchestrator.__new__(Orchestrator)
        orch._file_analyzer = _make_fa_mock([result])
        orch._file_rag = None
        orch._personality   = _make_personality_mock(allows=True)

        ctx = make_ctx(user_text="Riassumi /home/mauro/Scaricati/contratto.pdf")
        before = ctx.system_prompt
        await orch._run_file_analysis(ctx)

        assert len(ctx.system_prompt) > len(before)
        assert "CONTENUTO FILE ANALIZZATI" in ctx.system_prompt
        assert "Articolo 1" in ctx.system_prompt
        assert "/home/mauro/Scaricati/contratto.pdf" in ctx.system_prompt

    async def test_registers_tool_call(self):
        result = _make_result(path="/tmp/a.pdf", content="testo")
        orch = Orchestrator.__new__(Orchestrator)
        orch._file_analyzer = _make_fa_mock([result])
        orch._file_rag = None
        orch._personality   = _make_personality_mock(allows=True)

        ctx = make_ctx(user_text="apri /tmp/a.pdf")
        await orch._run_file_analysis(ctx)

        assert len(ctx.tool_calls) == 1
        tc = ctx.tool_calls[0]
        assert tc["tool"] == "file_analysis"
        assert "/tmp/a.pdf" in tc["args"]["paths"]
        assert "file_analysis" in ctx.timings

    async def test_calls_analyze_for_each_path(self):
        # Mock con due risultati, uno per chiamata
        results = [
            _make_result(path="/tmp/a.pdf", content="A"),
            _make_result(path="/tmp/b.md",  content="B", file_type="text"),
        ]
        orch = Orchestrator.__new__(Orchestrator)
        orch._file_analyzer = _make_fa_mock(results)
        orch._file_rag = None
        orch._personality   = _make_personality_mock(allows=True)

        ctx = make_ctx(user_text="confronta /tmp/a.pdf e /tmp/b.md")
        await orch._run_file_analysis(ctx)

        assert orch._file_analyzer.analyze.await_count == 2

    async def test_max_paths_cap_three(self):
        """Se l'utente menziona 5 path, ne vengono analizzati al massimo 3."""
        results = [_make_result(path=f"/tmp/{x}.pdf") for x in "abc"]
        orch = Orchestrator.__new__(Orchestrator)
        orch._file_analyzer = _make_fa_mock(results)
        orch._file_rag = None
        orch._personality   = _make_personality_mock(allows=True)

        ctx = make_ctx(user_text="vedi /tmp/a.pdf /tmp/b.pdf /tmp/c.pdf /tmp/d.pdf /tmp/e.pdf")
        await orch._run_file_analysis(ctx)

        # max_files_per_turn default = 3
        assert orch._file_analyzer.analyze.await_count == 3


class TestRunFileAnalysisErrors:
    async def test_error_result_not_injected_but_logged(self):
        bad = _make_result(error="[not_found] /tmp/x.pdf inesistente")
        orch = Orchestrator.__new__(Orchestrator)
        orch._file_analyzer = _make_fa_mock([bad])
        orch._file_rag = None
        orch._personality   = _make_personality_mock(allows=True)

        ctx = make_ctx(user_text="apri /tmp/x.pdf")
        before = ctx.system_prompt
        await orch._run_file_analysis(ctx)

        # nessun contenuto iniettato (tutto in errore)
        assert ctx.system_prompt == before
        # tool_call comunque registrato (per diagnostica)
        assert len(ctx.tool_calls) == 1

    async def test_mixed_results_only_good_injected(self):
        good = _make_result(path="/tmp/good.pdf", content="OK content")
        bad  = _make_result(path="/tmp/bad.pdf", error="[not_found] missing")
        orch = Orchestrator.__new__(Orchestrator)
        orch._file_analyzer = _make_fa_mock([good, bad])
        orch._file_rag = None
        orch._personality   = _make_personality_mock(allows=True)

        ctx = make_ctx(user_text="vedi /tmp/good.pdf e /tmp/bad.pdf")
        await orch._run_file_analysis(ctx)

        assert "OK content" in ctx.system_prompt
        assert "/tmp/good.pdf" in ctx.system_prompt
        # il bad path è citato solo nel tool_result, non nel system_prompt
        assert "/tmp/bad.pdf" not in ctx.system_prompt

    async def test_analyzer_exception_non_fatal(self):
        """
        Se analyze() solleva (asyncio.gather con return_exceptions=True),
        il file viene saltato senza bloccare gli altri.
        """
        good = _make_result(path="/tmp/a.pdf", content="testo")
        fa = MagicMock()
        # Prima chiamata solleva, seconda OK
        fa.analyze = AsyncMock(side_effect=[RuntimeError("boom"), good])
        fa.aclose  = AsyncMock()

        orch = Orchestrator.__new__(Orchestrator)
        orch._file_analyzer = fa
        orch._file_rag = None
        orch._personality   = _make_personality_mock(allows=True)

        ctx = make_ctx(user_text="apri /tmp/a.pdf e /tmp/b.pdf")
        await orch._run_file_analysis(ctx)

        assert ctx.error is None
        # uno solo dei due è arrivato a destinazione
        assert "/tmp/a.pdf" in ctx.system_prompt or "/tmp/b.pdf" in ctx.system_prompt


class TestRunFileAnalysisFairShare:
    async def test_fair_share_applied_when_over_total_cap(self, monkeypatch):
        """
        Tre file da 6000 char ciascuno (18000 totali) > max_total_chars
        di default (12000) → ogni file viene troncato a 4000 char.
        """
        long_text = "x" * 6000
        results = [
            _make_result(path=f"/tmp/{c}.txt", content=long_text, file_type="text")
            for c in "abc"
        ]
        orch = Orchestrator.__new__(Orchestrator)
        orch._file_analyzer = _make_fa_mock(results)
        orch._file_rag = None
        orch._personality   = _make_personality_mock(allows=True)

        ctx = make_ctx(user_text="vedi /tmp/a.txt /tmp/b.txt /tmp/c.txt")
        await orch._run_file_analysis(ctx)

        # Ogni AnalysisResult deve essere stato troncato a ~4000 char
        # (max_total_chars=12000 / 3 file = 4000)
        for r in results:
            # Il content è stato modificato in place, contiene la nota globale
            assert "troncato dal cap globale" in r.content
            # char_count rispecchia il troncamento
            assert r.char_count <= 4000 + 200  # 200 di margine per la nota


# ===========================================================================
# Smoke test integrazione: l'init di Orchestrator espone il nuovo parametro
# ===========================================================================

class TestOrchestratorInitFileAnalysis:
    def test_enable_file_analysis_param_default(self):
        o = Orchestrator(enable_tts=False, enable_stt=False)
        assert o._enable_file_analysis is True
        assert o._file_analyzer is None  # ancora non caricato

    def test_enable_file_analysis_param_false(self):
        o = Orchestrator(
            enable_tts=False, enable_stt=False, enable_file_analysis=False,
        )
        assert o._enable_file_analysis is False


# ===========================================================================
# FileAnalysisSettings (dal config)
# ===========================================================================

class TestFileAnalysisSettings:
    def test_settings_fields(self):
        from config.settings import settings
        cfg = settings.file_analysis
        assert cfg.max_chars_per_file > 0
        assert cfg.max_files_per_turn > 0
        assert cfg.max_total_chars > 0
        assert cfg.max_file_bytes > 0

    def test_safe_dirs_default_none(self):
        from config.settings import FileAnalysisSettings
        cfg = FileAnalysisSettings()
        assert cfg.safe_dirs is None

    def test_env_override_max_chars(self, monkeypatch):
        monkeypatch.setenv("FILE_ANALYSIS_MAX_CHARS_PER_FILE", "5000")
        from config.settings import FileAnalysisSettings
        cfg = FileAnalysisSettings()
        assert cfg.max_chars_per_file == 5000

    def test_field_validator_accepts_csv_safe_dirs(self):
        """Override programmatico con CSV: il validator converte in lista."""
        from config.settings import FileAnalysisSettings
        cfg = FileAnalysisSettings(safe_dirs="/home, /tmp")
        assert cfg.safe_dirs == ["/home", "/tmp"]
# ===========================================================================
# APPENDI QUESTO IN CODA A tests/test_orchestrator_file_analysis.py
# (commit page_count — _format_file_analysis_block mostra page_count nell'header)
# ===========================================================================

# Richiede l'helper _make_result gia' presente nel file e l'import di
# _format_file_analysis_block. Se _make_result NON accetta il kwarg `metadata`,
# aggiungilo alla sua firma con default None e passalo ad AnalysisResult come
# `metadata=metadata or {}` (vedi commento sotto).
#
# Se _make_result e' definito senza `metadata`, sostituisci la sua firma:
#     def _make_result(..., metadata: dict | None = None) -> AnalysisResult:
# e nel corpo:
#     metadata = metadata if metadata is not None else {},
# dentro la costruzione di AnalysisResult.


class TestPageCountInHeader:
    """
    Il page_count, quando presente nei metadata del risultato, deve comparire
    in chiaro nell'header del blocco file. Cosi' l'LLM lo legge come fatto
    diretto invece di dedurlo dal testo troncato.
    """

    def test_page_count_shown(self):
        r = _make_result(
            path="/tmp/libro.pdf",
            content="Lorem ipsum...",
            metadata={"page_count": 287, "pages_with_text": 287},
        )
        block = _format_file_analysis_block([r])
        assert "287 pagine" in block

    def test_singular(self):
        r = _make_result(
            path="/tmp/foglio.pdf",
            content="abc",
            metadata={"page_count": 1, "pages_with_text": 1},
        )
        block = _format_file_analysis_block([r])
        assert "1 pagina" in block
        assert "1 pagine" not in block

    def test_absent_for_non_pdf(self):
        r = _make_result(path="/tmp/note.md", content="testo", file_type="text")
        block = _format_file_analysis_block([r])
        assert "pagina" not in block and "pagine" not in block

    def test_coexists_with_truncation(self):
        r = _make_result(
            path="/tmp/libro.pdf",
            content="x",
            truncated=True,
            metadata={"page_count": 287, "pages_with_text": 287},
        )
        r.char_count = 8000
        r.original_char_count = 547000
        block = _format_file_analysis_block([r])
        assert "287 pagine" in block
        assert "testo troncato" in block
        assert "8000" in block and "547000" in block

class _FakeFileRAG:
    """FileRAG fake per testare l'innesto senza ChromaDB."""
    def __init__(self):
        self.indexed: dict[str, str] = {}     # file_id -> source
        self.search_returns: list = []         # MemoryChunk da restituire

    async def load(self):
        pass

    async def aclose(self):
        pass

    async def index_file(self, text, *, source, file_id=None):
        from modules.file_rag.base_file_rag import compute_file_id, IndexResult
        fid = file_id or compute_file_id(text)
        already = fid in self.indexed
        self.indexed[fid] = source
        return IndexResult(file_id=fid, chunks_indexed=3, already_indexed=already)

    async def search_file(self, file_id, query, top_k=None):
        return list(self.search_returns)


def _make_orch_with_rag():
    """Orchestrator minimale con file_rag fake, senza load() completo."""
    orch = Orchestrator.__new__(Orchestrator)
    orch._file_rag = _FakeFileRAG()
    return orch


class TestFormatFileRagBlock:
    def test_empty(self):
        assert _format_file_rag_block([]) == ""

    def test_shows_source_and_page(self):
        chunks = [MemoryChunk(
            content="testo del passaggio",
            source="libro.pdf",
            relevance_score=0.9,
            metadata={"source": "libro.pdf", "page_start": 142, "page_end": 142},
        )]
        block = _format_file_rag_block(chunks)
        assert "libro.pdf" in block
        assert "pagina 142" in block
        assert "testo del passaggio" in block

    def test_page_range(self):
        chunks = [MemoryChunk(
            content="x", source="l.pdf", relevance_score=0.8,
            metadata={"source": "l.pdf", "page_start": 10, "page_end": 12},
        )]
        block = _format_file_rag_block(chunks)
        assert "pagine 10-12" in block


class TestIndexLargeFiles:
    async def test_indexes_only_large(self):
        orch = _make_orch_with_rag()
        ctx = AssistantContext(user_text="domanda", session_id="s1")
        results = [
            _make_result(path="/up/small.pdf", content="piccolo"),  # no full_content
            _make_result(path="/up/libro.pdf", content="estratto",
                         full_content="--- pagina 1 ---\n" + "x" * 50000),
        ]
        await orch._index_large_files(ctx, results)
        # solo il file grande è stato indicizzato
        assert len(orch._file_rag.indexed) == 1
        assert ctx.metadata["rag_files"][0]["source"] == "libro.pdf"

    async def test_no_large_files_noop(self):
        orch = _make_orch_with_rag()
        ctx = AssistantContext(user_text="q", session_id="s1")
        results = [_make_result(path="/up/x.pdf", content="piccolo")]
        await orch._index_large_files(ctx, results)
        assert orch._file_rag.indexed == {}
        assert "rag_files" not in ctx.metadata or ctx.metadata["rag_files"] == []

    async def test_dedup_file_id(self):
        orch = _make_orch_with_rag()
        ctx = AssistantContext(user_text="q", session_id="s1")
        big = "--- pagina 1 ---\n" + "y" * 50000
        results = [_make_result(path="/up/a.pdf", content="e", full_content=big)]
        await orch._index_large_files(ctx, results)
        await orch._index_large_files(ctx, results)  # stesso contenuto
        # un solo file_id registrato, niente duplicati
        assert len(ctx.metadata["rag_files"]) == 1

    async def test_rag_disabled_noop(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._file_rag = None  # RAG non disponibile
        ctx = AssistantContext(user_text="q", session_id="s1")
        results = [_make_result(path="/up/a.pdf", content="e",
                                full_content="--- pagina 1 ---\n" + "z" * 50000)]
        await orch._index_large_files(ctx, results)  # non deve sollevare
        assert "rag_files" not in ctx.metadata or not ctx.metadata.get("rag_files")


class TestRunFileRag:
    async def test_no_rag_files_noop(self):
        orch = _make_orch_with_rag()
        ctx = AssistantContext(user_text="q", session_id="s1")
        await orch._run_file_rag(ctx)
        assert "PASSAGGI RILEVANTI" not in (ctx.system_prompt or "")

    async def test_injects_retrieved_chunks(self):
        orch = _make_orch_with_rag()
        orch._file_rag.search_returns = [MemoryChunk(
            content="passaggio rilevante", source="libro.pdf",
            relevance_score=0.95,
            metadata={"source": "libro.pdf", "page_start": 50, "page_end": 50},
        )]
        ctx = AssistantContext(user_text="parlami del capitolo", session_id="s1")
        ctx.metadata["rag_files"] = [{"file_id": "abc1234567890000", "source": "libro.pdf"}]
        await orch._run_file_rag(ctx)
        assert "passaggio rilevante" in ctx.system_prompt
        assert "pagina 50" in ctx.system_prompt

    async def test_empty_query_noop(self):
        orch = _make_orch_with_rag()
        ctx = AssistantContext(user_text="   ", session_id="s1")
        ctx.metadata["rag_files"] = [{"file_id": "x", "source": "l.pdf"}]
        await orch._run_file_rag(ctx)
        assert not (ctx.system_prompt or "")


class TestSessionRagPersistence:
    """Commit 7c: i rag_files della sessione sopravvivono tra i turni.

    Verifica in isolamento le due cuciture estratte da turn():
      - _sync_session_rag  (sync-in:   store di sessione → ctx)
      - _persist_session_rag (write-back: ctx → store di sessione)
    Stesso stile di TestIndexLargeFiles / TestRunFileRag: nessun turn() intero.
    """

    def _orch(self):
        # _make_orch_with_rag costruisce via __new__, quindi lo store di sessione
        # (popolato in __init__) va impostato a mano, come si fa per _file_rag.
        orch = _make_orch_with_rag()
        orch._session_rag_files = defaultdict(list)
        return orch

    async def test_persist_then_sync_round_trip(self):
        """Turno 1 indicizza un file grande e lo persiste; turno 2 (ctx nuovo,
        stessa sessione, NESSUN path nel testo) lo ritrova via sync-in."""
        orch = self._orch()
        big = "--- pagina 1 ---\n" + "x" * 50000

        # Turno 1: file citato → indicizzato → write-back
        ctx1 = AssistantContext(user_text="analizza /up/libro.pdf", session_id="s1")
        orch._sync_session_rag(ctx1)  # store vuoto: parte da []
        await orch._index_large_files(
            ctx1,
            [_make_result(path="/up/libro.pdf", content="estratto", full_content=big)],
        )
        orch._persist_session_rag(ctx1)
        assert orch._session_rag_files["s1"], "il file doveva restare nello store di sessione"
        assert orch._session_rag_files["s1"][0]["source"] == "libro.pdf"

        # Turno 2: ctx NUOVO, stessa sessione, nessun path → sync-in ripopola
        ctx2 = AssistantContext(user_text="cosa succede ai mezzelfi?", session_id="s1")
        orch._sync_session_rag(ctx2)
        assert ctx2.metadata["rag_files"] == orch._session_rag_files["s1"]

        # e il retrieval trova i chunk del file caricato al turno precedente
        orch._file_rag.search_returns = [MemoryChunk(
            content="lo sterminio dei mezzelfi", source="libro.pdf",
            relevance_score=0.95,
            metadata={"source": "libro.pdf", "page_start": 157, "page_end": 157},
        )]
        await orch._run_file_rag(ctx2)
        assert "lo sterminio dei mezzelfi" in ctx2.system_prompt
        assert "pagina 157" in ctx2.system_prompt

    async def test_sync_is_defensive_copy(self):
        """Il sync-in copia la lista: un append nel ctx non muta lo store."""
        orch = self._orch()
        orch._session_rag_files["s1"] = [{"file_id": "a", "source": "x.pdf"}]
        ctx = AssistantContext(user_text="q", session_id="s1")
        orch._sync_session_rag(ctx)
        ctx.metadata["rag_files"].append({"file_id": "b", "source": "y.pdf"})
        assert len(orch._session_rag_files["s1"]) == 1, "lo store non deve essere mutato"

    async def test_sessions_are_isolated(self):
        """Sessioni diverse non condividono i rag_files."""
        orch = self._orch()
        ctx_a = AssistantContext(user_text="q", session_id="A")
        await orch._index_large_files(
            ctx_a,
            [_make_result(path="/up/a.pdf", content="e",
                          full_content="--- pagina 1 ---\n" + "a" * 50000)],
        )
        orch._persist_session_rag(ctx_a)
        ctx_b = AssistantContext(user_text="q", session_id="B")
        orch._sync_session_rag(ctx_b)
        assert ctx_b.metadata["rag_files"] == [], "la sessione B non deve vedere i file di A"

    async def test_empty_turn_does_not_clobber(self):
        """Un turno senza file non azzera i rag_files gia' memorizzati."""
        orch = self._orch()
        orch._session_rag_files["s1"] = [{"file_id": "a", "source": "x.pdf"}]
        ctx = AssistantContext(user_text="domanda senza file", session_id="s1")
        orch._sync_session_rag(ctx)        # ctx.metadata["rag_files"] = [a]
        orch._persist_session_rag(ctx)     # write-back di [a], non []
        assert orch._session_rag_files["s1"] == [{"file_id": "a", "source": "x.pdf"}]
