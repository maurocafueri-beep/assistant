"""
modules/map_reduce
==================
Riassunti e risposte GLOBALI su un documento intero (map-reduce).

    from modules.map_reduce import MapReduceEngine, MapReduceResult
"""

from modules.map_reduce.base_map_reduce import MapReduceEngine, MapReduceResult
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
