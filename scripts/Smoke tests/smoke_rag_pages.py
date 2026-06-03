#!/usr/bin/env python3
"""
smoke_rag_pages.py — Diagnostica del retrieval RAG su un file indicizzato.

Interroga direttamente la collection di un file (default: Nihal, file_id dedotto
dal nome collection file_98c70f533645fdbe) con alcune query e stampa i chunk
recuperati CON le loro pagine e la rilevanza. Serve a capire se il retrieval
tira fuori il chunk giusto (es. pag. 157 per lo sterminio dei mezzelfi) o no.

Bypassa UI/orchestrator: testa solo FileRAG.search_file su Ollama+ChromaDB reali.

Uso (dal venv-runtime, dalla radice del progetto):
    venv-runtime/bin/python scripts/smoke_rag_pages.py
    venv-runtime/bin/python scripts/smoke_rag_pages.py 98c70f533645fdbe
"""
import asyncio
import sys

from modules.file_rag.base_file_rag import FileRAG

FILE_ID = sys.argv[1] if len(sys.argv) > 1 else "98c70f533645fdbe"

QUERIES = [
    "a che pagina avviene lo sterminio dei mezzelfi",
    "sterminio dei mezzelfi popolo cancellato",
    "chi è Sennar mago allievo di Soana",
]


async def main() -> None:
    async with FileRAG() as frag:
        if not await frag.is_indexed(FILE_ID):
            print(f"⚠  file_id '{FILE_ID}' NON indicizzato (collection vuota o assente).")
            return
        for q in QUERIES:
            print(f"\n### query: {q!r}")
            chunks = await frag.search_file(FILE_ID, q, top_k=10)
            if not chunks:
                print("  (nessun chunk recuperato)")
                continue
            for i, c in enumerate(chunks, 1):
                m = c.metadata or {}
                ps, pe = m.get("page_start"), m.get("page_end")
                snippet = " ".join(c.content.split())[:90]
                print(f"  {i:>2}. pag {ps}-{pe} | rel {c.relevance_score:.3f} | {snippet!r}")


if __name__ == "__main__":
    asyncio.run(main())
