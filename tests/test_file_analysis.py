"""
tests/test_file_analysis.py
Test suite per modules/file_analysis.

asyncio_mode = "auto" configurato in pyproject.toml → le funzioni async sono
raccolte automaticamente, @pytest.mark.asyncio non è necessario.

Strategia:
    - Estrattori sincroni testati su file temporanei in tmp_path.
    - PDF: pypdf scrive PDF di pagine vuote (sufficiente per coprire il
      wrapper; estrazione di testo reale ricade su un mock).
    - Audio: WhisperSTT mockato (no Whisper, no torch, no rete).
    - Safety: safe_dirs=[] negli helper diretti, safe_dirs=[tmp_path] nei
      test integrati per non interagire col filesystem reale.

Esecuzione:
    make test
    venv-runtime/bin/pytest tests/test_file_analysis.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.file_analysis import (
    AnalysisResult,
    FileAnalysisError,
    FileAnalyzer,
)
from modules.file_analysis.base_file_analysis import (
    _AUDIO_EXTS,
    _CSV_EXTS,
    _DOCX_EXTS,
    _HTML_EXTS,
    _JSON_EXTS,
    _PDF_EXTS,
    _TEXT_EXTS,
    _XML_EXTS,
    _classify_path,
    _extract_csv,
    _extract_docx,
    _extract_html,
    _extract_json,
    _extract_pdf,
    _extract_text,
    _extract_xml,
    _is_safe_path,
    _resolve_path,
    _truncate,
)


# ===========================================================================
# Helper puri
# ===========================================================================

class TestResolvePath:
    def test_expanduser(self):
        p = _resolve_path("~/foo.txt")
        assert str(p).startswith(str(Path.home()))
        assert p.name == "foo.txt"

    def test_absolute_path(self):
        p = _resolve_path("/tmp/x.txt")
        assert str(p) == "/tmp/x.txt"

    def test_relative_resolved_to_absolute(self):
        p = _resolve_path("./pippo.txt")
        assert p.is_absolute()

    def test_double_dot_normalised(self):
        p = _resolve_path("/tmp/foo/../bar.txt")
        assert str(p) == "/tmp/bar.txt"


class TestIsSafePath:
    def test_empty_safe_dirs_permits_everything(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("ok")
        assert _is_safe_path(f, []) is True

    def test_inside_safe_dir(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("ok")
        assert _is_safe_path(f, [str(tmp_path)]) is True

    def test_outside_safe_dir(self, tmp_path):
        # /etc è fuori da tmp_path per definizione
        assert _is_safe_path(Path("/etc/passwd"), [str(tmp_path)]) is False

    def test_multiple_safe_dirs_any_match(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("ok")
        assert _is_safe_path(f, ["/nonesistente", str(tmp_path)]) is True

    def test_expanduser_in_safe_dirs(self):
        home = Path.home()
        # un file ipotetico in home — non serve esista, _is_safe_path
        # non lo verifica
        assert _is_safe_path(home / "foo.txt", ["~"]) is True


class TestClassifyPath:
    @pytest.mark.parametrize("ext,expected", [
        (".pdf",  "pdf"),
        (".PDF",  "pdf"),         # case insensitive
        (".docx", "docx"),
        (".txt",  "text"),
        (".md",   "text"),
        (".rst",  "text"),
        (".yaml", "text"),
        (".yml",  "text"),
        (".log",  "text"),
        (".html", "html"),
        (".htm",  "html"),
        (".json", "json"),
        (".csv",  "csv"),
        (".tsv",  "csv"),
        (".xml",  "xml"),
        (".wav",  "audio"),
        (".mp3",  "audio"),
        (".ogg",  "audio"),
        (".flac", "audio"),
        (".m4a",  "audio"),
        (".opus", "audio"),
        (".xyz",  "unknown"),
        (".exe",  "unknown"),
        ("",      "unknown"),
    ])
    def test_extensions(self, ext, expected):
        assert _classify_path(Path(f"file{ext}")) == expected


class TestTruncate:
    def test_short_content_unchanged(self):
        text, orig, was = _truncate("ciao", 100)
        assert text == "ciao"
        assert orig == 4
        assert was is False

    def test_exactly_at_limit_unchanged(self):
        s = "x" * 100
        text, orig, was = _truncate(s, 100)
        assert text == s
        assert was is False

    def test_long_content_truncated(self):
        s = "a" * 500
        text, orig, was = _truncate(s, 100)
        assert was is True
        assert orig == 500
        assert text.startswith("a" * 100)
        assert "troncato" in text.lower()
        assert "100" in text and "500" in text

    def test_empty_content(self):
        text, orig, was = _truncate("", 100)
        assert text == ""
        assert orig == 0
        assert was is False


# ===========================================================================
# FileAnalysisError
# ===========================================================================

class TestFileAnalysisError:
    def test_basic(self):
        err = FileAnalysisError("not_found", "File mancante")
        assert err.code == "not_found"
        assert err.message == "File mancante"
        assert err.path is None

    def test_with_path(self):
        err = FileAnalysisError("too_large", "Troppo grande", path="/tmp/x.pdf")
        assert err.path == "/tmp/x.pdf"

    def test_str_without_path(self):
        err = FileAnalysisError("unsupported", "ext .xyz")
        s = str(err)
        assert "unsupported" in s
        assert "ext .xyz" in s

    def test_str_with_path(self):
        err = FileAnalysisError("not_found", "missing", path="/x/y.pdf")
        s = str(err)
        assert "/x/y.pdf" in s

    def test_is_exception(self):
        with pytest.raises(FileAnalysisError):
            raise FileAnalysisError("x", "y")


# ===========================================================================
# AnalysisResult
# ===========================================================================

class TestAnalysisResult:
    def _make(self, **kw) -> AnalysisResult:
        defaults = dict(
            path="/tmp/x.txt",
            file_type="text",
            content="ciao",
            char_count=4,
            original_char_count=4,
            truncated=False,
            elapsed_ms=1.2,
            extractor="text_raw",
        )
        defaults.update(kw)
        return AnalysisResult(**defaults)

    def test_is_empty_true(self):
        assert self._make(content="", char_count=0).is_empty() is True

    def test_is_empty_only_whitespace(self):
        assert self._make(content="   \n\t", char_count=5).is_empty() is True

    def test_is_empty_false(self):
        assert self._make(content="ciao").is_empty() is False

    def test_to_log_dict_keys(self):
        d = self._make().to_log_dict()
        assert set(d.keys()) == {
            "path", "file_type", "char_count", "orig_char_count",
            "truncated", "elapsed_ms", "extractor", "error",
        }

    def test_to_log_dict_truncates_path(self):
        long_path = "/" + "x" * 200 + "/file.txt"
        d = self._make(path=long_path).to_log_dict()
        assert len(d["path"]) <= 80

    def test_metadata_field(self):
        r = self._make(metadata={"file_size_bytes": 1024})
        assert r.metadata["file_size_bytes"] == 1024


# ===========================================================================
# Estrattori — TXT/MD/YAML
# ===========================================================================

class TestExtractText:
    def test_simple_txt(self, tmp_path):
        f = tmp_path / "ciao.txt"
        f.write_text("Riga uno\nRiga due", encoding="utf-8")
        assert _extract_text(f) == "Riga uno\nRiga due"

    def test_md_with_unicode(self, tmp_path):
        f = tmp_path / "note.md"
        f.write_text("# Mercoledì\nÈ già notte 🌙", encoding="utf-8")
        out = _extract_text(f)
        assert "Mercoledì" in out
        assert "🌙" in out

    def test_invalid_utf8_recovered(self, tmp_path):
        f = tmp_path / "bad.txt"
        # mix di UTF-8 valido + byte non decodificabili
        f.write_bytes(b"Ciao \xff\xfe mondo")
        out = _extract_text(f)
        assert "Ciao" in out and "mondo" in out
        # i byte non validi vengono sostituiti con U+FFFD, non solleva

    def test_empty_file(self, tmp_path):
        f = tmp_path / "vuoto.txt"
        f.write_text("", encoding="utf-8")
        assert _extract_text(f) == ""


# ===========================================================================
# Estrattori — JSON
# ===========================================================================

class TestExtractJson:
    def test_pretty_prints_valid_json(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"nome":"Mauro","eta":42}', encoding="utf-8")
        out = _extract_json(f)
        # pretty-print → ha indentazione
        assert "\n" in out
        assert "Mauro" in out
        assert "42" in out

    def test_unicode_preserved(self, tmp_path):
        f = tmp_path / "u.json"
        f.write_text('{"giorno":"mercoledì"}', encoding="utf-8")
        out = _extract_json(f)
        assert "mercoledì" in out          # ensure_ascii=False

    def test_malformed_returns_raw(self, tmp_path):
        f = tmp_path / "broken.json"
        f.write_text('{"nome": "Mauro"', encoding="utf-8")
        out = _extract_json(f)
        # fallback al testo grezzo, NON solleva
        assert "Mauro" in out

    def test_nested_structure(self, tmp_path):
        f = tmp_path / "nested.json"
        data = {"a": [1, 2, {"b": "c"}]}
        f.write_text(json.dumps(data), encoding="utf-8")
        out = _extract_json(f)
        assert "a" in out and "b" in out and "c" in out


# ===========================================================================
# Estrattori — CSV/TSV
# ===========================================================================

class TestExtractCsv:
    def test_comma_separated(self, tmp_path):
        f = tmp_path / "people.csv"
        f.write_text("nome,eta\nMauro,42\nGwen,30\n", encoding="utf-8")
        out = _extract_csv(f)
        assert "nome" in out and "Mauro" in out and "Gwen" in out
        # separatore visivo " | " applicato
        assert " | " in out

    def test_tab_separated(self, tmp_path):
        f = tmp_path / "people.tsv"
        f.write_text("nome\teta\nMauro\t42\n", encoding="utf-8")
        out = _extract_csv(f)
        assert "Mauro" in out

    def test_semicolon_separated(self, tmp_path):
        f = tmp_path / "people.csv"
        f.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")
        out = _extract_csv(f)
        # sniffer dovrebbe riconoscere ";", e dopo il formatter usiamo " | "
        assert "1" in out and "6" in out

    def test_preview_truncates_many_rows(self, tmp_path):
        f = tmp_path / "big.csv"
        # 1 header + 100 righe dati = 101 righe totali
        rows = ["col\n"] + [f"r{i}\n" for i in range(100)]
        f.write_text("".join(rows), encoding="utf-8")
        out = _extract_csv(f)
        assert "anteprima" in out.lower()
        assert "101" in out                # totale segnalato in coda

    def test_empty_file(self, tmp_path):
        f = tmp_path / "vuoto.csv"
        f.write_text("", encoding="utf-8")
        assert _extract_csv(f) == ""


# ===========================================================================
# Estrattori — XML
# ===========================================================================

class TestExtractXml:
    def test_returns_raw(self, tmp_path):
        f = tmp_path / "data.xml"
        f.write_text("<root><a>1</a></root>", encoding="utf-8")
        out = _extract_xml(f)
        assert "<root>" in out and "<a>1</a>" in out


# ===========================================================================
# Estrattori — HTML
# ===========================================================================

class TestExtractHtml:
    def test_strips_script_and_style(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_text(dedent("""\
            <html>
              <head>
                <style>body{color:red}</style>
                <script>alert(1)</script>
              </head>
              <body>
                <h1>Titolo</h1>
                <p>Paragrafo visibile.</p>
              </body>
            </html>
        """), encoding="utf-8")
        out = _extract_html(f)
        assert "Titolo" in out
        assert "Paragrafo visibile." in out
        assert "alert" not in out
        assert "color:red" not in out

    def test_normalizes_whitespace(self, tmp_path):
        f = tmp_path / "p.html"
        f.write_text("<p>uno</p>\n\n\n<p>due</p>\n\n", encoding="utf-8")
        out = _extract_html(f)
        # niente blocchi di righe vuote consecutive
        assert "\n\n\n" not in out
        assert "uno" in out and "due" in out

    def test_minimal_html(self, tmp_path):
        f = tmp_path / "x.html"
        f.write_text("<p>solo testo</p>", encoding="utf-8")
        assert "solo testo" in _extract_html(f)


# ===========================================================================
# Estrattori — DOCX
# ===========================================================================

class TestExtractDocx:
    def test_paragraphs(self, tmp_path):
        from docx import Document
        doc = Document()
        doc.add_paragraph("Primo paragrafo.")
        doc.add_paragraph("Secondo paragrafo con accenti: caffè.")
        f = tmp_path / "doc.docx"
        doc.save(str(f))

        out = _extract_docx(f)
        assert "Primo paragrafo." in out
        assert "caffè" in out

    def test_table(self, tmp_path):
        from docx import Document
        doc = Document()
        doc.add_paragraph("Intro")
        table = doc.add_table(rows=2, cols=2)
        table.rows[0].cells[0].text = "Nome"
        table.rows[0].cells[1].text = "Età"
        table.rows[1].cells[0].text = "Mauro"
        table.rows[1].cells[1].text = "42"
        f = tmp_path / "tbl.docx"
        doc.save(str(f))

        out = _extract_docx(f)
        assert "Intro" in out
        assert "Nome" in out and "Mauro" in out and "42" in out
        # le celle della stessa riga sono tab-separated
        assert "Nome\tEtà" in out
        assert "Mauro\t42" in out

    def test_empty_doc(self, tmp_path):
        from docx import Document
        doc = Document()
        f = tmp_path / "vuoto.docx"
        doc.save(str(f))
        assert _extract_docx(f) == ""


# ===========================================================================
# Estrattori — PDF
# ===========================================================================

class TestExtractPdf:
    """
    Test reali: pypdf può creare PDF di pagine vuote facilmente, e
    _extract_pdf non deve sollevare. Non possiamo scrivere testo PDF da
    zero senza reportlab, quindi l'estrazione su testo vero è coperta
    dall'integration test del progetto e dallo smoke.
    """

    def test_empty_pages_no_crash(self, tmp_path):
        from pypdf import PdfWriter

        w = PdfWriter()
        w.add_blank_page(width=72, height=72)
        w.add_blank_page(width=72, height=72)
        f = tmp_path / "blank.pdf"
        with open(f, "wb") as fh:
            w.write(fh)

        out = _extract_pdf(f)
        # Pagine vuote → stringa vuota, non eccezione
        assert isinstance(out, str)

    def test_malformed_pdf_raises(self, tmp_path):
        # un file non-PDF con estensione .pdf solleva — il wrapper
        # in analyze() catturerà come extraction_failed
        f = tmp_path / "fake.pdf"
        f.write_bytes(b"non sono un pdf")
        with pytest.raises(Exception):
            _extract_pdf(f)


# ===========================================================================
# FileAnalyzer — API pubblica
# ===========================================================================

class TestFileAnalyzerSupport:
    def test_supported_extensions_completeness(self):
        fa = FileAnalyzer(safe_dirs=[])
        all_exts = (
            _TEXT_EXTS | _PDF_EXTS | _DOCX_EXTS | _HTML_EXTS
            | _JSON_EXTS | _CSV_EXTS | _XML_EXTS | _AUDIO_EXTS
        )
        assert fa.supported_extensions() == set(all_exts)

    def test_is_supported_known(self):
        fa = FileAnalyzer(safe_dirs=[])
        assert fa.is_supported("foo.pdf")  is True
        assert fa.is_supported("foo.DOCX") is True
        assert fa.is_supported("foo.mp3")  is True

    def test_is_supported_unknown(self):
        fa = FileAnalyzer(safe_dirs=[])
        assert fa.is_supported("foo.xyz")  is False
        assert fa.is_supported("foo")      is False


class TestFileAnalyzerInit:
    def test_default_chars_threshold(self):
        fa = FileAnalyzer(safe_dirs=[])
        assert fa._max_chars == 8000

    def test_custom_chars_threshold(self):
        fa = FileAnalyzer(max_chars_inline=2000, safe_dirs=[])
        assert fa._max_chars == 2000

    def test_safe_dirs_from_setting_when_none(self):
        # Quando safe_dirs è None → eredita da settings.pc_control.safe_dirs
        from config.settings import settings
        fa = FileAnalyzer(safe_dirs=None)
        assert fa._safe_dirs == list(settings.pc_control.safe_dirs)

    def test_explicit_empty_safe_dirs(self):
        fa = FileAnalyzer(safe_dirs=[])
        assert fa._safe_dirs == []

    def test_stt_none_by_default(self):
        fa = FileAnalyzer(safe_dirs=[])
        assert fa._stt is None


class TestFileAnalyzerContextManager:
    async def test_aenter_aexit(self):
        async with FileAnalyzer(safe_dirs=[]) as fa:
            assert isinstance(fa, FileAnalyzer)

    async def test_aclose_noop(self):
        fa = FileAnalyzer(safe_dirs=[])
        # non deve sollevare e può essere chiamata più volte
        await fa.aclose()
        await fa.aclose()


# ===========================================================================
# FileAnalyzer.analyze — happy path
# ===========================================================================

class TestAnalyzeText:
    async def test_simple_txt(self, tmp_path):
        f = tmp_path / "ciao.txt"
        f.write_text("Riga uno\nRiga due", encoding="utf-8")

        async with FileAnalyzer(safe_dirs=[str(tmp_path)]) as fa:
            r = await fa.analyze(f)

        assert r.error is None
        assert r.file_type == "text"
        assert r.extractor == "text_raw"
        assert "Riga uno" in r.content
        assert r.truncated is False
        assert r.original_char_count == len("Riga uno\nRiga due")
        assert r.char_count == r.original_char_count
        assert r.metadata["file_size_bytes"] > 0
        assert r.elapsed_ms >= 0

    async def test_md_path_as_str(self, tmp_path):
        f = tmp_path / "note.md"
        f.write_text("# Titolo\nCorpo.", encoding="utf-8")

        async with FileAnalyzer(safe_dirs=[str(tmp_path)]) as fa:
            r = await fa.analyze(str(f))   # accetta anche str

        assert r.error is None
        assert "Titolo" in r.content


class TestAnalyzeJson:
    async def test_pretty_printed(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"k":1}', encoding="utf-8")

        async with FileAnalyzer(safe_dirs=[str(tmp_path)]) as fa:
            r = await fa.analyze(f)

        assert r.error is None
        assert r.file_type == "json"
        assert r.extractor == "json_stdlib"
        assert "\n" in r.content                # indentato


class TestAnalyzeCsv:
    async def test_csv_pretty(self, tmp_path):
        f = tmp_path / "x.csv"
        f.write_text("a,b\n1,2\n", encoding="utf-8")

        async with FileAnalyzer(safe_dirs=[str(tmp_path)]) as fa:
            r = await fa.analyze(f)

        assert r.error is None
        assert r.file_type == "csv"
        assert r.extractor == "csv_stdlib"
        assert " | " in r.content


class TestAnalyzeHtml:
    async def test_html_get_text(self, tmp_path):
        f = tmp_path / "p.html"
        f.write_text("<p>Salve</p>", encoding="utf-8")

        async with FileAnalyzer(safe_dirs=[str(tmp_path)]) as fa:
            r = await fa.analyze(f)

        assert r.error is None
        assert r.file_type == "html"
        assert "Salve" in r.content


class TestAnalyzeDocx:
    async def test_docx_paragraphs(self, tmp_path):
        from docx import Document
        doc = Document()
        doc.add_paragraph("Una riga.")
        f = tmp_path / "x.docx"
        doc.save(str(f))

        async with FileAnalyzer(safe_dirs=[str(tmp_path)]) as fa:
            r = await fa.analyze(f)

        assert r.error is None
        assert r.file_type == "docx"
        assert r.extractor == "docx_python-docx"
        assert "Una riga." in r.content


class TestAnalyzePdf:
    async def test_pdf_blank_no_crash(self, tmp_path):
        from pypdf import PdfWriter
        w = PdfWriter()
        w.add_blank_page(width=72, height=72)
        f = tmp_path / "blank.pdf"
        with open(f, "wb") as fh:
            w.write(fh)

        async with FileAnalyzer(safe_dirs=[str(tmp_path)]) as fa:
            r = await fa.analyze(f)

        assert r.error is None
        assert r.file_type == "pdf"
        assert r.extractor == "pdf_pypdf"


# ===========================================================================
# FileAnalyzer.analyze — error paths
# ===========================================================================

class TestAnalyzeErrors:
    async def test_file_not_found(self, tmp_path):
        async with FileAnalyzer(safe_dirs=[str(tmp_path)]) as fa:
            r = await fa.analyze(tmp_path / "non_esiste.txt")

        assert r.error is not None
        assert "not_found" in r.error
        assert r.content == ""
        assert r.char_count == 0

    async def test_directory_not_a_file(self, tmp_path):
        async with FileAnalyzer(safe_dirs=[str(tmp_path)]) as fa:
            r = await fa.analyze(tmp_path)

        assert r.error is not None
        assert "not_found" in r.error

    async def test_out_of_safe_dirs(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("ok", encoding="utf-8")

        # safe_dirs punta altrove → richiesta rifiutata
        async with FileAnalyzer(safe_dirs=["/usr"]) as fa:
            r = await fa.analyze(f)

        assert r.error is not None
        assert "out_of_safe_dirs" in r.error

    async def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "x.xyz"
        f.write_text("ok", encoding="utf-8")

        async with FileAnalyzer(safe_dirs=[str(tmp_path)]) as fa:
            r = await fa.analyze(f)

        assert r.error is not None
        assert "unsupported" in r.error
        assert "xyz" in r.error.lower() or ".xyz" in r.error

    async def test_too_large(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("ciao", encoding="utf-8")

        async with FileAnalyzer(safe_dirs=[str(tmp_path)], max_file_bytes=2) as fa:
            r = await fa.analyze(f)

        assert r.error is not None
        assert "too_large" in r.error
        assert r.file_type == "text"

    async def test_extraction_exception_caught(self, tmp_path):
        """Un'eccezione interna dell'estrattore → error 'extraction_failed', no re-raise."""
        f = tmp_path / "fake.pdf"
        f.write_bytes(b"definitivamente non un PDF")

        async with FileAnalyzer(safe_dirs=[str(tmp_path)]) as fa:
            r = await fa.analyze(f)

        assert r.error is not None
        assert "extraction_failed" in r.error
        assert r.file_type == "pdf"


# ===========================================================================
# FileAnalyzer.analyze — truncation
# ===========================================================================

class TestAnalyzeTruncation:
    async def test_short_file_not_truncated(self, tmp_path):
        f = tmp_path / "small.txt"
        f.write_text("ciao", encoding="utf-8")

        async with FileAnalyzer(safe_dirs=[str(tmp_path)], max_chars_inline=100) as fa:
            r = await fa.analyze(f)

        assert r.truncated is False
        assert r.char_count == 4

    async def test_long_file_truncated(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 1000, encoding="utf-8")

        async with FileAnalyzer(safe_dirs=[str(tmp_path)], max_chars_inline=100) as fa:
            r = await fa.analyze(f)

        assert r.truncated is True
        assert r.original_char_count == 1000
        # contenuto = 100 char + notice → maggiore di 100 ma di poco
        assert len(r.content) > 100
        assert "troncato" in r.content.lower()


# ===========================================================================
# FileAnalyzer.analyze — audio (mock STT)
# ===========================================================================

def _make_mock_stt(text: str = "Trascrizione di prova.") -> MagicMock:
    """Mock di WhisperSTT con transcribe_file che ritorna un STTResult-like."""
    stt = MagicMock()
    fake_result = MagicMock()
    fake_result.text = text
    stt.transcribe_file = AsyncMock(return_value=fake_result)
    return stt


class TestAnalyzeAudio:
    async def test_audio_without_stt_fails(self, tmp_path):
        f = tmp_path / "x.mp3"
        f.write_bytes(b"\x00" * 100)   # contenuto non significativo

        async with FileAnalyzer(safe_dirs=[str(tmp_path)], stt=None) as fa:
            r = await fa.analyze(f)

        assert r.error is not None
        assert "stt_unavailable" in r.error
        assert r.file_type == "audio"
        assert r.content == ""

    async def test_audio_with_stt_calls_transcribe_file(self, tmp_path):
        f = tmp_path / "voice.wav"
        f.write_bytes(b"\x00" * 100)

        stt = _make_mock_stt("Ciao Claude, oggi piove.")
        async with FileAnalyzer(safe_dirs=[str(tmp_path)], stt=stt) as fa:
            r = await fa.analyze(f)

        assert r.error is None
        assert r.file_type == "audio"
        assert r.extractor == "audio_stt"
        assert r.content == "Ciao Claude, oggi piove."
        stt.transcribe_file.assert_awaited_once()

    async def test_audio_stt_exception_maps_to_extraction_failed(self, tmp_path):
        f = tmp_path / "voice.mp3"
        f.write_bytes(b"\x00" * 100)

        stt = MagicMock()
        stt.transcribe_file = AsyncMock(side_effect=RuntimeError("whisper down"))

        async with FileAnalyzer(safe_dirs=[str(tmp_path)], stt=stt) as fa:
            r = await fa.analyze(f)

        assert r.error is not None
        assert "extraction_failed" in r.error
        assert "whisper down" in r.error

    async def test_audio_empty_transcription_is_ok(self, tmp_path):
        f = tmp_path / "silence.wav"
        f.write_bytes(b"\x00" * 100)

        stt = _make_mock_stt("")
        async with FileAnalyzer(safe_dirs=[str(tmp_path)], stt=stt) as fa:
            r = await fa.analyze(f)

        # trascrizione vuota → contenuto vuoto, NON un errore
        assert r.error is None
        assert r.content == ""
        assert r.is_empty()

    async def test_audio_all_extensions(self, tmp_path):
        """Tutte le estensioni audio devono essere accettate dal dispatch."""
        stt = _make_mock_stt("ok")
        async with FileAnalyzer(safe_dirs=[str(tmp_path)], stt=stt) as fa:
            for ext in (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".opus"):
                f = tmp_path / f"x{ext}"
                f.write_bytes(b"\x00" * 50)
                r = await fa.analyze(f)
                assert r.error is None, f"errore inatteso per {ext}: {r.error}"
                assert r.file_type == "audio"
