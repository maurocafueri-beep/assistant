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

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.context import AssistantContext, InputMode, ModelRole, OutputMode
from core.orchestrator import (
    Orchestrator,
    OrchestratorStatus,
    _extract_file_paths,
    _format_file_analysis_block,
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
        ctx = make_ctx(user_text="riassumi ~/Scaricati/x.pdf")
        await orch._run_file_analysis(ctx)
        assert ctx.tool_calls == []

    async def test_skip_when_no_path_in_text(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._file_analyzer = _make_fa_mock()
        orch._personality   = _make_personality_mock(allows=True)
        ctx = make_ctx(user_text="Spiegami i decoratori Python")
        await orch._run_file_analysis(ctx)
        orch._file_analyzer.analyze.assert_not_called()
        assert ctx.tool_calls == []

    async def test_skip_when_tool_not_allowed(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._file_analyzer = _make_fa_mock()
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
        orch._personality   = _make_personality_mock(allows=True)

        ctx = make_ctx(user_text="confronta /tmp/a.pdf e /tmp/b.md")
        await orch._run_file_analysis(ctx)

        assert orch._file_analyzer.analyze.await_count == 2

    async def test_max_paths_cap_three(self):
        """Se l'utente menziona 5 path, ne vengono analizzati al massimo 3."""
        results = [_make_result(path=f"/tmp/{x}.pdf") for x in "abc"]
        orch = Orchestrator.__new__(Orchestrator)
        orch._file_analyzer = _make_fa_mock(results)
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
