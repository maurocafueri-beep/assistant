"""
tests/test_uploads_cleanup.py
Test per core/uploads_cleanup — pulizia di data/uploads/ al boot.

Strategia: usiamo tmp_path per costruire alberi di upload realistici e
iniettiamo `now` (timestamp di riferimento) per simulare l'età dei file
senza dover manipolare davvero il mtime su scala di giorni. Per testare i
file "vecchi" impostiamo il loro mtime nel passato con os.utime.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from core.uploads_cleanup import (
    DEFAULT_SUBDIR,
    CleanupReport,
    cleanup_uploads,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DAY = 86_400


def _touch(path: Path, *, age_days: float, content: bytes = b"x") -> Path:
    """Crea un file con mtime impostato a `age_days` giorni fa."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    past = time.time() - age_days * _DAY
    os.utime(path, (past, past))
    return path


@pytest.fixture
def uploads(tmp_path: Path) -> Path:
    d = tmp_path / "uploads"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Casi base
# ---------------------------------------------------------------------------

class TestCleanupUploads:

    def test_missing_dir_is_skipped(self, tmp_path):
        report = cleanup_uploads(
            tmp_path / "non_esiste",
            retention_session=7, retention_default=3,
        )
        assert report.skipped_reason is not None
        assert report.files_removed == 0

    def test_empty_dir_no_op(self, uploads):
        report = cleanup_uploads(uploads, retention_session=7, retention_default=3)
        assert report.files_removed == 0
        assert report.dirs_removed == 0

    def test_recent_session_file_kept(self, uploads):
        _touch(uploads / "sess_abc" / "20260529_a.pdf", age_days=2)
        report = cleanup_uploads(uploads, retention_session=7, retention_default=3)
        assert report.files_removed == 0
        assert (uploads / "sess_abc" / "20260529_a.pdf").exists()

    def test_old_session_file_removed(self, uploads):
        _touch(uploads / "sess_abc" / "old.pdf", age_days=10)
        report = cleanup_uploads(uploads, retention_session=7, retention_default=3)
        assert report.files_removed == 1
        assert not (uploads / "sess_abc" / "old.pdf").exists()

    def test_emptied_session_dir_removed(self, uploads):
        _touch(uploads / "sess_abc" / "old.pdf", age_days=10)
        report = cleanup_uploads(uploads, retention_session=7, retention_default=3)
        # Il file scaduto è rimosso E la subdir di sessione, ora vuota, sparisce
        assert report.files_removed == 1
        assert report.dirs_removed == 1
        assert not (uploads / "sess_abc").exists()

    def test_session_dir_with_recent_file_kept(self, uploads):
        _touch(uploads / "sess_abc" / "old.pdf", age_days=10)
        _touch(uploads / "sess_abc" / "new.pdf", age_days=1)
        report = cleanup_uploads(uploads, retention_session=7, retention_default=3)
        # Solo il vecchio rimosso; la dir resta perché contiene ancora il recente
        assert report.files_removed == 1
        assert report.dirs_removed == 0
        assert (uploads / "sess_abc").exists()
        assert (uploads / "sess_abc" / "new.pdf").exists()


# ---------------------------------------------------------------------------
# default/ — retention diversa, dir mai rimossa
# ---------------------------------------------------------------------------

class TestDefaultSubdir:

    def test_default_uses_its_own_retention(self, uploads):
        # File a 5 giorni: oltre la retention default (3) ma sotto la session (7)
        _touch(uploads / DEFAULT_SUBDIR / "old.pdf", age_days=5)
        report = cleanup_uploads(uploads, retention_session=7, retention_default=3)
        assert report.files_removed == 1
        assert not (uploads / DEFAULT_SUBDIR / "old.pdf").exists()

    def test_default_dir_never_removed_even_if_empty(self, uploads):
        _touch(uploads / DEFAULT_SUBDIR / "old.pdf", age_days=5)
        report = cleanup_uploads(uploads, retention_session=7, retention_default=3)
        # Il file sparisce ma la directory default/ resta
        assert report.files_removed == 1
        assert report.dirs_removed == 0
        assert (uploads / DEFAULT_SUBDIR).is_dir()

    def test_default_recent_file_kept(self, uploads):
        _touch(uploads / DEFAULT_SUBDIR / "new.pdf", age_days=1)
        report = cleanup_uploads(uploads, retention_session=7, retention_default=3)
        assert report.files_removed == 0
        assert (uploads / DEFAULT_SUBDIR / "new.pdf").exists()

    def test_session_retention_does_not_apply_to_default(self, uploads):
        # Stesso file a 5 giorni in una sessione: con retention session=7 resta.
        _touch(uploads / "sess_x" / "f.pdf", age_days=5)
        # In default invece a 5 giorni con default=3 sparirebbe; qui verifichiamo
        # che il ramo sessione applichi 7, non 3.
        report = cleanup_uploads(uploads, retention_session=7, retention_default=3)
        assert report.files_removed == 0
        assert (uploads / "sess_x" / "f.pdf").exists()


# ---------------------------------------------------------------------------
# Retention infinita (0)
# ---------------------------------------------------------------------------

class TestInfiniteRetention:

    def test_zero_session_retention_disables(self, uploads):
        _touch(uploads / "sess_abc" / "ancient.pdf", age_days=999)
        report = cleanup_uploads(uploads, retention_session=0, retention_default=3)
        assert report.files_removed == 0
        assert (uploads / "sess_abc" / "ancient.pdf").exists()

    def test_zero_default_retention_disables(self, uploads):
        _touch(uploads / DEFAULT_SUBDIR / "ancient.pdf", age_days=999)
        report = cleanup_uploads(uploads, retention_session=7, retention_default=0)
        assert report.files_removed == 0
        assert (uploads / DEFAULT_SUBDIR / "ancient.pdf").exists()


# ---------------------------------------------------------------------------
# Robustezza
# ---------------------------------------------------------------------------

class TestRobustness:

    def test_bytes_freed_accounted(self, uploads):
        _touch(uploads / "sess_a" / "big.bin", age_days=10, content=b"y" * 1024)
        report = cleanup_uploads(uploads, retention_session=7, retention_default=3)
        assert report.bytes_freed == 1024

    def test_symlink_not_followed(self, uploads, tmp_path):
        # Un file esterno vecchio, e un symlink dentro uploads che vi punta.
        external = _touch(tmp_path / "external_old.pdf", age_days=999)
        link = uploads / "sess_a" / "link.pdf"
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(external)
        except OSError:
            pytest.skip("symlink non supportati su questo filesystem")
        report = cleanup_uploads(uploads, retention_session=7, retention_default=3)
        # Il symlink non viene seguito né rimosso come file scaduto, e il
        # target esterno resta intatto
        assert external.exists()
        assert report.files_removed == 0

    def test_mixed_tree(self, uploads):
        # Albero realistico: 2 sessioni + default, mix di età
        _touch(uploads / "sess_1" / "old1.pdf", age_days=10)   # rimosso
        _touch(uploads / "sess_1" / "new1.pdf", age_days=1)    # tenuto
        _touch(uploads / "sess_2" / "old2.pdf", age_days=20)   # rimosso → dir via
        _touch(uploads / DEFAULT_SUBDIR / "d_old.pdf", age_days=4)  # rimosso (def=3)
        _touch(uploads / DEFAULT_SUBDIR / "d_new.pdf", age_days=1)  # tenuto
        report = cleanup_uploads(uploads, retention_session=7, retention_default=3)
        assert report.files_removed == 3
        assert report.dirs_removed == 1   # solo sess_2 si è svuotata
        assert (uploads / "sess_1" / "new1.pdf").exists()
        assert not (uploads / "sess_2").exists()
        assert (uploads / DEFAULT_SUBDIR).is_dir()
        assert (uploads / DEFAULT_SUBDIR / "d_new.pdf").exists()

    def test_does_not_raise_on_report_type(self, uploads):
        report = cleanup_uploads(uploads, retention_session=7, retention_default=3)
        assert isinstance(report, CleanupReport)
        # Il report è serializzabile per il log
        d = report.to_log_dict()
        assert "files_removed" in d and "mb_freed" in d
