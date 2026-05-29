"""
core/uploads_cleanup.py
Pulizia automatica di data/uploads/ all'avvio dell'app.

Perché esiste: gli upload non sono transitori (l'utente può ispezionarli sul
filesystem dopo averli caricati), ma senza una pulizia periodica la cartella
cresce senza limiti. Il cleanup gira al boot, in background, best-effort.

Struttura attesa di uploads/ (creata da ui/server.py):
    data/uploads/<session_id>/<timestamp>_<filename>     ← legati a conversazione
    data/uploads/default/<timestamp>_<filename>          ← upload senza sessione

Due retention diverse perché i due tipi hanno valore diverso nel tempo: gli
upload di sessione accompagnano una conversazione (più longevi), quelli in
default/ sono one-shot (si possono buttare prima).

API:
    cleanup_uploads(uploads_dir, retention_session, retention_default) → CleanupReport

La funzione è pura rispetto alle settings (riceve i giorni come argomenti):
così è testabile senza monkeypatch e riusabile da contesti diversi (boot,
ma in futuro anche dal cleanup delle collection RAG orfane — Commit 7).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.logger import logger


# Nome della sottocartella per gli upload senza session_id (vedi ui/server.py
# _session_dir_for). Trattata diversamente: il suo CONTENUTO viene pulito con
# la retention "default", ma la directory in sé non viene mai rimossa.
DEFAULT_SUBDIR = "default"

_SECONDS_PER_DAY = 86_400


@dataclass
class CleanupReport:
    """Esito del cleanup, per logging e test."""
    files_removed:   int       = 0
    dirs_removed:    int       = 0
    bytes_freed:     int       = 0
    errors:          int       = 0
    skipped_reason:  str | None = None  # valorizzato se il cleanup non è partito
    removed_paths:   list[str] = field(default_factory=list)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "files_removed": self.files_removed,
            "dirs_removed":  self.dirs_removed,
            "mb_freed":      round(self.bytes_freed / (1024 * 1024), 2),
            "errors":        self.errors,
            "skipped":       self.skipped_reason,
        }


def _is_older_than(path: Path, max_age_days: int, *, now: float) -> bool:
    """
    True se il file è più vecchio di max_age_days, in base al mtime.

    Usa mtime (non ctime/atime): è il tempo di ultima modifica del contenuto,
    il più vicino a "quando l'utente l'ha caricato". Se il file sparisce tra
    lo scandir e questa chiamata (race con un'altra parte del sistema),
    propaga l'eccezione al chiamante che la conta come errore non fatale.
    """
    age_seconds = now - path.stat().st_mtime
    return age_seconds > (max_age_days * _SECONDS_PER_DAY)


def _clean_directory(
    directory: Path,
    retention_days: int,
    *,
    now: float,
    report: CleanupReport,
    remove_dir_if_empty: bool,
) -> None:
    """
    Pulisce ricorsivamente i file più vecchi di retention_days in `directory`.

    - retention_days == 0 → no-op (retention infinita per questo ramo).
    - Rimuove i file scaduti; salta i symlink (non li seguiamo, sicurezza).
    - Se remove_dir_if_empty e la directory resta vuota dopo la pulizia, la
      rimuove (caso: subdir di sessione ormai esaurita). La default/ passa
      con remove_dir_if_empty=False perché la sua dir non va mai eliminata.
    - Ogni errore su un singolo file è isolato: incrementa report.errors e
      prosegue, non aborta l'intero cleanup.
    """
    if retention_days <= 0:
        return
    if not directory.is_dir():
        return

    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        logger.warning("uploads_cleanup | impossibile leggere {}: {}", directory, exc)
        report.errors += 1
        return

    for entry in entries:
        try:
            # Sicurezza: non seguiamo symlink (potrebbero puntare fuori da uploads).
            if entry.is_symlink():
                continue
            if entry.is_dir():
                # Ricorsione: una subdir dentro una subdir di sessione.
                _clean_directory(
                    entry, retention_days, now=now, report=report,
                    remove_dir_if_empty=True,
                )
                continue
            if entry.is_file() and _is_older_than(entry, retention_days, now=now):
                size = entry.stat().st_size
                entry.unlink()
                report.files_removed += 1
                report.bytes_freed   += size
                report.removed_paths.append(str(entry))
        except FileNotFoundError:
            # Race: sparito nel frattempo. Non è un errore reale.
            continue
        except OSError as exc:
            logger.warning("uploads_cleanup | errore su {}: {}", entry, exc)
            report.errors += 1

    # Rimuovi la directory se richiesto e ora è vuota.
    if remove_dir_if_empty:
        try:
            if not any(directory.iterdir()):
                directory.rmdir()
                report.dirs_removed += 1
                report.removed_paths.append(str(directory) + "/")
        except OSError as exc:
            logger.warning("uploads_cleanup | impossibile rimuovere dir {}: {}", directory, exc)
            report.errors += 1


def cleanup_uploads(
    uploads_dir: Path,
    *,
    retention_session: int,
    retention_default: int,
    now: float | None = None,
) -> CleanupReport:
    """
    Pulisce data/uploads/ secondo le due retention.

    Args:
        uploads_dir:       Path di data/uploads/.
        retention_session: Giorni di retention per le subdir <session_id>/.
                           0 = infinita (no cleanup di quel ramo).
        retention_default: Giorni di retention per default/. 0 = infinita.
        now:               Timestamp di riferimento (default: time.time()).
                           Iniettabile per i test.

    Returns:
        CleanupReport con conteggi. Non solleva mai: ogni errore è isolato e
        contato. Se uploads_dir non esiste, ritorna un report con
        skipped_reason valorizzato (caso normale al primissimo avvio).
    """
    now = now if now is not None else time.time()
    report = CleanupReport()

    if not uploads_dir.exists():
        report.skipped_reason = "uploads_dir inesistente"
        return report
    if not uploads_dir.is_dir():
        report.skipped_reason = "uploads_dir non è una directory"
        return report

    try:
        subdirs = [d for d in uploads_dir.iterdir() if d.is_dir() and not d.is_symlink()]
    except OSError as exc:
        logger.warning("uploads_cleanup | impossibile leggere {}: {}", uploads_dir, exc)
        report.skipped_reason = f"errore lettura: {exc}"
        return report

    for sub in subdirs:
        if sub.name == DEFAULT_SUBDIR:
            # default/: pulisci il contenuto con la retention dedicata, ma NON
            # rimuovere la directory stessa (è ricreata di continuo da
            # _session_dir_for, eliminarla è solo lavoro sprecato).
            _clean_directory(
                sub, retention_default, now=now, report=report,
                remove_dir_if_empty=False,
            )
        else:
            # Subdir di sessione: retention più lunga, e se si svuota la
            # togliamo (la sessione è conclusa, niente più upload lì).
            _clean_directory(
                sub, retention_session, now=now, report=report,
                remove_dir_if_empty=True,
            )

    return report


async def run_cleanup_at_boot() -> CleanupReport:
    """
    Wrapper async per il boot: legge le settings, esegue il cleanup in un
    executor (I/O su disco, non bloccare l'event loop) e logga l'esito.

    Pensato per essere lanciato fire-and-forget da ui/app.py accanto al
    warmup. Gated su settings.file_analysis.uploads_cleanup_enabled.
    Best-effort: cattura qualsiasi eccezione e la logga senza propagarla.
    """
    import asyncio

    from config.settings import settings

    fa = settings.file_analysis
    if not getattr(fa, "uploads_cleanup_enabled", True):
        logger.debug("uploads_cleanup | disabilitato da settings")
        return CleanupReport(skipped_reason="disabilitato")

    uploads_dir = settings.data_dir / "uploads"
    try:
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(
            None,
            lambda: cleanup_uploads(
                uploads_dir,
                retention_session=fa.uploads_retention_days_session,
                retention_default=fa.uploads_retention_days_default,
            ),
        )
        if report.skipped_reason:
            logger.debug("uploads_cleanup | skip: {}", report.skipped_reason)
        else:
            logger.info("uploads_cleanup | {}", report.to_log_dict())
        return report
    except Exception as exc:
        logger.warning("uploads_cleanup | fallito (ignorato): {}", exc)
        return CleanupReport(errors=1, skipped_reason=str(exc))
