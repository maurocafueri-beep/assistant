"""
modules/map_reduce/precompute.py
Precalcolo (ibrido): all'indicizzazione di un file giriamo il motore una volta
per le domande anticipabili (riassunto, elenco capitoli) e ne salviamo i
risultati su disco, cosi' quelle domande diventano istantanee invece di costare
una scansione ogni volta. Le domande globali arbitrarie restano on-demand.

Store: un JSON per file in data/map_reduce_cache/<file_id>.json. Nessun vincolo
di tipo o dimensione (testo lungo e strutturato), ciclo di vita agganciato al
file (drop quando scade da uploads/ — il wiring e' nello stadio 5).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from config.settings import settings
from core.logger import logger
from modules.map_reduce.base_map_reduce import MapReduceEngine

# Domande fisse del precalcolo (lo stadio 4 potra' affinarle per tipo).
SUMMARY_QUESTION = "Riassumi l'intero documento in modo completo e ordinato."
CHAPTERS_QUESTION = (
    "Elenca i capitoli o le sezioni principali del documento, con il titolo e la "
    "pagina in cui iniziano, nell'ordine in cui compaiono."
)


@dataclass
class Precomputed:
    """Risultati precalcolati per un file."""
    summary:     str
    chapters:    str
    computed_at: float
    model:       str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Precomputed":
        return cls(
            summary     = d.get("summary", ""),
            chapters    = d.get("chapters", ""),
            computed_at = d.get("computed_at", 0.0),
            model       = d.get("model", ""),
        )


class PrecomputeStore:
    """Cache su disco del precalcolo: un JSON per file."""

    def __init__(self, cache_dir: "Path | None" = None) -> None:
        self._dir = Path(cache_dir) if cache_dir else (settings.data_dir / "map_reduce_cache")

    def _path(self, file_id: str) -> Path:
        return self._dir / f"{file_id}.json"

    def has(self, file_id: str) -> bool:
        return self._path(file_id).is_file()

    def save(self, file_id: str, pre: "Precomputed") -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path(file_id).write_text(
            json.dumps(pre.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("map_reduce.precompute | salvato '{}'", file_id)

    def load(self, file_id: str) -> "Precomputed | None":
        p = self._path(file_id)
        if not p.is_file():
            return None
        try:
            return Precomputed.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("map_reduce.precompute | load '{}' fallito: {}", file_id, exc)
            return None

    def drop(self, file_id: str) -> bool:
        try:
            self._path(file_id).unlink()
            logger.info("map_reduce.precompute | rimosso '{}'", file_id)
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:
            logger.warning("map_reduce.precompute | drop '{}' fallito: {}", file_id, exc)
            return False


async def precompute(
    engine: MapReduceEngine,
    blocks: list[dict],
    *,
    model: str,
) -> "Precomputed":
    """
    Gira il motore per le domande anticipabili (riassunto + capitoli) e
    impacchetta i risultati. NON salva: il chiamante decide se e dove persistere.
    """
    summary  = (await engine.run(question=SUMMARY_QUESTION,  blocks=blocks)).content
    chapters = (await engine.run(question=CHAPTERS_QUESTION, blocks=blocks)).content
    return Precomputed(
        summary=summary, chapters=chapters, computed_at=time.time(), model=model,
    )
