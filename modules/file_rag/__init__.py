"""modules/file_rag — indicizzazione e ricerca semantica su file grandi."""
from .base_file_rag import (
    FileRAG,
    FileChunk,
    IndexResult,
    compute_file_id,
    collection_name_for,
    chunk_text,
    FILE_COLLECTION_PREFIX,
)

__all__ = [
    "FileRAG",
    "FileChunk",
    "IndexResult",
    "compute_file_id",
    "collection_name_for",
    "chunk_text",
    "FILE_COLLECTION_PREFIX",
]
