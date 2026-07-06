#!/usr/bin/env python3
r"""
apply_mapreduce_s3.py — Map-reduce, stadio 3: il precalcolo (ibrido).

All'indicizzazione giriamo il motore una volta per le domande anticipabili
(riassunto + elenco capitoli) e ne salviamo i risultati su disco, cosi' diventano
istantanee invece di costare una scansione ogni volta. Le domande globali
arbitrarie restano on-demand.

Crea modules/map_reduce/precompute.py (store JSON + funzione precompute) e
aggiorna modules/map_reduce/__init__.py. NON cabla nulla nell'orchestrator
(quando far partire il precalcolo e come leggerlo e' lo stadio 5).

CREA un file e modifica __init__. Aborta se precompute.py esiste gia'.

Uso:
    python3 apply_mapreduce_s3.py --check
    python3 apply_mapreduce_s3.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PRECOMPUTE = Path("modules/map_reduce/precompute.py")
INIT = Path("modules/map_reduce/__init__.py")

PRECOMPUTE_SRC = '''"""
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
'''

INIT_OLD = '''from modules.map_reduce.base_map_reduce import MapReduceEngine, MapReduceResult

__all__ = [
    "MapReduceEngine",
    "MapReduceResult",
]
'''

INIT_NEW = '''from modules.map_reduce.base_map_reduce import MapReduceEngine, MapReduceResult
from modules.map_reduce.precompute import (
    CHAPTERS_QUESTION,
    Precomputed,
    PrecomputeStore,
    SUMMARY_QUESTION,
    precompute,
)

__all__ = [
    "MapReduceEngine",
    "MapReduceResult",
    "Precomputed",
    "PrecomputeStore",
    "precompute",
    "SUMMARY_QUESTION",
    "CHAPTERS_QUESTION",
]
'''


def main() -> int:
    ap = argparse.ArgumentParser(description="Map-reduce stadio 3: precalcolo.")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root)
    pre  = root / PRECOMPUTE
    init = root / INIT

    if pre.exists():
        print(f"\u2717 {PRECOMPUTE} esiste gia': rifiuto di sovrascrivere.", file=sys.stderr)
        return 1
    if not init.is_file():
        print(f"\u2717 manca {INIT}: applica prima lo stadio 2.", file=sys.stderr)
        return 1

    init_txt = init.read_text(encoding="utf-8")
    n = init_txt.count(INIT_OLD)
    if n == 0:
        print(f"\u2717 __init__ non combacia (atteso l'export dello stadio 2). 0 occorrenze.",
              file=sys.stderr)
        return 1
    if n > 1:
        print(f"\u2717 __init__ ambiguo ({n} occorrenze).", file=sys.stderr)
        return 1

    print(f"  \u2713 {PRECOMPUTE} pronto da creare.")
    print(f"  \u2713 ancoraggio __init__ OK.")
    if args.check:
        print("\n--check OK: nessuna scrittura.")
        return 0

    pre.write_text(PRECOMPUTE_SRC, encoding="utf-8")
    init.write_text(init_txt.replace(INIT_OLD, INIT_NEW, 1), encoding="utf-8")
    print(f"  \u2713 creato {PRECOMPUTE}")
    print(f"  \u2713 aggiornato {INIT}")

    print("\n\u2713 Applicato.")
    print("  Prossimi passi:")
    print("    SKIP_SLOW=1 make test")
    print("    git add modules/map_reduce/")
    print("    git status")
    print('    git commit -m "feat(map_reduce): stadio 3 — precalcolo riassunto/capitoli su disco"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
