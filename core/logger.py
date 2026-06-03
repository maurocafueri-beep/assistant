"""
core/logger.py
Setup del logging centralizzato con loguru.
Importa `logger` da qui in tutti i moduli.
"""
import sys
from pathlib import Path
from loguru import logger as _logger

def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    if log_dir is None:
        # Path assoluto da settings (PROJECT_ROOT/logs): i log finiscono sempre
        # nella stessa cartella anche quando l'app parte dall'icona della dock
        # (working directory diversa da ~/assistant). Import lazy: evita cicli.
        from config.settings import settings
        log_dir = settings.log_dir
    _logger.remove()
    fmt = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "{message}"
    )
    _logger.add(sys.stderr, format=fmt, level=level, colorize=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    _logger.add(
        log_dir / "assistant.log",
        format=fmt, level=level,
        rotation="10 MB", retention="14 days", compression="zip",
    )
    _logger.add(
        log_dir / "errors.log",
        format=fmt, level="ERROR",
        rotation="5 MB", retention="30 days", compression="zip",
    )

logger = _logger
