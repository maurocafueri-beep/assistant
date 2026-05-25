"""
tests/test_ui_uploads.py
Test suite per l'endpoint POST /api/uploads in ui/server.py.

Strategia:
- TestClient di FastAPI (sync) — niente WebSocket, niente Bridge avviato.
- WSManager mockato per non aprire connessioni reali.
- File temporanei costruiti al volo con tmp_path.
- settings.data_dir viene puntato a tmp_path/data via monkeypatch così
  i file caricati restano isolati dal vero filesystem dell'utente.

Esecuzione:
    venv-runtime/bin/pytest tests/test_ui_uploads.py -v
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from ui.server import (
    _ALLOWED_UPLOAD_EXTS,
    _safe_filename,
    _session_dir_for,
    create_app,
)


# ===========================================================================
# Helpers — fixture e mock
# ===========================================================================

@pytest.fixture
def mock_ws_manager():
    """WSManager mockato (broadcast no-op)."""
    ws = MagicMock()
    ws.broadcast = AsyncMock()
    ws.n_clients = 0
    return ws


@pytest.fixture
def client_with_loop(mock_ws_manager, monkeypatch, tmp_path):
    """
    TestClient con un loop mockato (session_id='test-sess') e settings
    puntati a tmp_path/data.
    """
    # Reindirizza data_dir
    from config.settings import settings as _s
    monkeypatch.setattr(_s, "data_dir", tmp_path / "data")
    # Aggiorna anche il modulo ui.server che ha cachato _UPLOADS_DIR all'import
    from ui import server as _srv
    monkeypatch.setattr(_srv, "_UPLOADS_DIR", tmp_path / "data" / "uploads")

    app, state = create_app(mock_ws_manager)
    loop = MagicMock()
    loop.session_id = "test-sess"
    state["loop"] = loop
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_without_loop(mock_ws_manager, monkeypatch, tmp_path):
    """TestClient con loop=None (caso: upload prima che il bridge sia pronto)."""
    from config.settings import settings as _s
    monkeypatch.setattr(_s, "data_dir", tmp_path / "data")
    from ui import server as _srv
    monkeypatch.setattr(_srv, "_UPLOADS_DIR", tmp_path / "data" / "uploads")

    app, state = create_app(mock_ws_manager)
    state["loop"] = None
    with TestClient(app) as c:
        yield c


# ===========================================================================
# _safe_filename
# ===========================================================================

class TestSafeFilename:
    def test_simple_name_preserved(self):
        assert _safe_filename("contratto.pdf") == "contratto.pdf"

    def test_spaces_to_underscore(self):
        assert _safe_filename("Contratto firmato.PDF") == "Contratto_firmato.PDF"

    def test_unicode_accents_stripped(self):
        assert _safe_filename("résumé_2024.docx") == "resume_2024.docx"

    def test_path_traversal_blocked(self):
        # "../etc/passwd" → solo basename → "passwd"
        assert _safe_filename("../etc/passwd") == "passwd"

    def test_double_slash(self):
        assert _safe_filename("/home/mauro/x.pdf") == "x.pdf"

    def test_special_chars_replaced(self):
        assert _safe_filename("file <weird> &^%$.txt") == "file_weird_.txt"

    def test_empty_name(self):
        assert _safe_filename("") == "file"

    def test_none_name(self):
        assert _safe_filename(None) == "file"

    def test_long_name_truncated(self):
        name = "a" * 200 + ".pdf"
        result = _safe_filename(name)
        assert len(result) <= 80

    def test_only_special_chars(self):
        # tutto viene normalizzato in '_' che poi viene strippato
        assert _safe_filename("...") == "file"

    def test_leading_dot_stripped(self):
        assert _safe_filename(".hidden") == "hidden"


# ===========================================================================
# _session_dir_for
# ===========================================================================

class TestSessionDir:
    def test_creates_directory(self, tmp_path, monkeypatch):
        from ui import server as _srv
        monkeypatch.setattr(_srv, "_UPLOADS_DIR", tmp_path / "uploads")
        d = _session_dir_for("sess-abc")
        assert d.exists()
        assert d.is_dir()
        assert d.parent == tmp_path / "uploads"

    def test_none_session_id_uses_default(self, tmp_path, monkeypatch):
        from ui import server as _srv
        monkeypatch.setattr(_srv, "_UPLOADS_DIR", tmp_path / "uploads")
        d = _session_dir_for(None)
        assert d.name == "default"

    def test_sanitizes_unsafe_session_id(self, tmp_path, monkeypatch):
        """Anche se il session_id avesse caratteri esotici, va sanificato."""
        from ui import server as _srv
        monkeypatch.setattr(_srv, "_UPLOADS_DIR", tmp_path / "uploads")
        d = _session_dir_for("../escape")
        # Non deve uscire dalla cartella uploads
        assert str(d).startswith(str(tmp_path / "uploads"))


# ===========================================================================
# _ALLOWED_UPLOAD_EXTS — sanity check
# ===========================================================================

class TestAllowedExtensions:
    def test_contains_pdf_and_docx(self):
        assert ".pdf" in _ALLOWED_UPLOAD_EXTS
        assert ".docx" in _ALLOWED_UPLOAD_EXTS

    def test_contains_audio_extensions(self):
        for ext in (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".opus"):
            assert ext in _ALLOWED_UPLOAD_EXTS, f"manca {ext}"

    def test_does_not_contain_exe(self):
        assert ".exe" not in _ALLOWED_UPLOAD_EXTS
        assert ".sh" not in _ALLOWED_UPLOAD_EXTS


# ===========================================================================
# POST /api/uploads — happy path
# ===========================================================================

class TestUploadsHappyPath:
    def test_simple_txt_upload(self, client_with_loop, tmp_path):
        files = {"file": ("note.txt", io.BytesIO(b"Hello world"), "text/plain")}
        r = client_with_loop.post("/api/uploads", files=files)
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["name"] == "note.txt"
        assert body["size"] == 11
        assert body["ext"]  == ".txt"
        # path deve esistere su filesystem
        target = Path(body["path"])
        assert target.exists()
        assert target.read_bytes() == b"Hello world"

    def test_path_under_session_dir(self, client_with_loop, tmp_path):
        files = {"file": ("x.md", io.BytesIO(b"# title"), "text/markdown")}
        r = client_with_loop.post("/api/uploads", files=files)
        body = r.json()
        # /tmp.../data/uploads/test-sess/<ts>_x.md
        assert "/uploads/test-sess/" in body["path"]
        assert body["path"].endswith("_x.md")

    def test_timestamp_prefix(self, client_with_loop):
        files = {"file": ("x.txt", io.BytesIO(b"a"), "text/plain")}
        r = client_with_loop.post("/api/uploads", files=files)
        name = Path(r.json()["path"]).name
        # Es: 20260525093000_x.txt
        assert "_" in name
        prefix = name.split("_", 1)[0]
        assert len(prefix) == 14 and prefix.isdigit()

    def test_unicode_filename_sanitized(self, client_with_loop):
        files = {"file": ("résumé 2024.docx", io.BytesIO(b"fake docx"), "application/octet-stream")}
        r = client_with_loop.post("/api/uploads", files=files)
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "resume_2024.docx"

    def test_default_session_when_no_loop(self, client_without_loop):
        files = {"file": ("x.txt", io.BytesIO(b"a"), "text/plain")}
        r = client_without_loop.post("/api/uploads", files=files)
        assert r.status_code == 200
        assert "/uploads/default/" in r.json()["path"]


# ===========================================================================
# POST /api/uploads — error paths
# ===========================================================================

class TestUploadsErrors:
    def test_unsupported_extension(self, client_with_loop):
        files = {"file": ("malware.exe", io.BytesIO(b"MZ..."), "application/octet-stream")}
        r = client_with_loop.post("/api/uploads", files=files)
        assert r.status_code == 400
        body = r.json()
        assert "detail" in body
        assert ".exe" in body["detail"]

    def test_no_extension(self, client_with_loop):
        files = {"file": ("noext", io.BytesIO(b"x"), "application/octet-stream")}
        r = client_with_loop.post("/api/uploads", files=files)
        assert r.status_code == 400

    def test_too_large(self, client_with_loop, monkeypatch):
        """File oltre max_file_bytes deve dare 413."""
        from config.settings import settings as _s
        monkeypatch.setattr(_s.file_analysis, "max_file_bytes", 10)
        files = {"file": ("big.txt", io.BytesIO(b"x" * 100), "text/plain")}
        r = client_with_loop.post("/api/uploads", files=files)
        assert r.status_code == 413
        body = r.json()
        assert "troppo grande" in body["detail"].lower()

    def test_at_limit_ok(self, client_with_loop, monkeypatch):
        """File esattamente a max_file_bytes deve passare."""
        from config.settings import settings as _s
        monkeypatch.setattr(_s.file_analysis, "max_file_bytes", 10)
        files = {"file": ("ok.txt", io.BytesIO(b"x" * 10), "text/plain")}
        r = client_with_loop.post("/api/uploads", files=files)
        assert r.status_code == 200

    def test_uppercase_extension_accepted(self, client_with_loop):
        """Pdf.PDF deve essere accettato (estensione case-insensitive)."""
        files = {"file": ("Doc.PDF", io.BytesIO(b"%PDF-1.4"), "application/pdf")}
        r = client_with_loop.post("/api/uploads", files=files)
        assert r.status_code == 200
        assert r.json()["ext"] == ".pdf"


# ===========================================================================
# POST /api/uploads — supporti diversi tipi
# ===========================================================================

class TestUploadsTypes:
    @pytest.mark.parametrize("ext,content", [
        (".pdf",  b"%PDF-1.4"),
        (".docx", b"PK\x03\x04 fake zip"),
        (".txt",  b"plain text"),
        (".md",   b"# md"),
        (".html", b"<html></html>"),
        (".json", b'{"k":1}'),
        (".csv",  b"a,b\n1,2"),
        (".xml",  b"<root/>"),
        (".wav",  b"RIFF.... fake"),
        (".mp3",  b"ID3 fake"),
        (".ogg",  b"OggS fake"),
        (".flac", b"fLaC fake"),
        (".m4a",  b"....ftypM4A fake"),
        (".opus", b"OpusHead fake"),
    ])
    def test_all_supported_extensions(self, client_with_loop, ext, content):
        files = {"file": (f"sample{ext}", io.BytesIO(content), "application/octet-stream")}
        r = client_with_loop.post("/api/uploads", files=files)
        assert r.status_code == 200, f"fallito per {ext}: {r.text}"
        body = r.json()
        assert body["ok"] is True
        assert body["ext"] == ext
        assert Path(body["path"]).exists()
