"""
scripts/smoke_memory.py
Test interattivo del modulo modules/memory.
Richiede: Ollama attivo con nomic-embed-text, ChromaDB installato.

Uso:
    venv-runtime/bin/python scripts/smoke_memory.py
    venv-runtime/bin/python scripts/smoke_memory.py --persist-dir /tmp/test_memory
    venv-runtime/bin/python scripts/smoke_memory.py --top-k 3

Lo script usa una directory temporanea isolata di default (non altera
i dati di produzione in data/embeddings/).
"""

import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.context import AssistantContext
from modules.memory import MemoryManager

# ---------------------------------------------------------------------------
# Corpus di test
# ---------------------------------------------------------------------------

CORPUS = [
    ("Python è un linguaggio di programmazione ad alto livello.", {"source": "wiki", "topic": "tech"}),
    ("Roma è la capitale d'Italia ed è una città ricca di storia.", {"source": "wiki", "topic": "geo"}),
    ("Il caffè è una delle bevande più consumate al mondo.", {"source": "wiki", "topic": "food"}),
    ("Machine learning è una branca dell'intelligenza artificiale.", {"source": "wiki", "topic": "tech"}),
    ("La pizza napoletana è un piatto tipico della cucina italiana.", {"source": "wiki", "topic": "food"}),
    ("Il Colosseo è un anfiteatro romano situato nel centro di Roma.", {"source": "wiki", "topic": "geo"}),
    ("FastAPI è un framework Python per costruire API REST ad alte prestazioni.", {"source": "wiki", "topic": "tech"}),
    ("Il parmigiano reggiano è un formaggio DOP prodotto in Emilia-Romagna.", {"source": "wiki", "topic": "food"}),
]

QUERIES = [
    ("programmazione e sviluppo software", "tech"),
    ("cucina e gastronomia italiana", "food"),
    ("storia e monumenti di Roma", "geo"),
    ("intelligenza artificiale e algoritmi", "tech"),
]


def _sep(char: str = "─", n: int = 58) -> None:
    print(char * n)


async def main(persist_dir: str | None, top_k: int) -> None:
    # Usa directory temporanea se non specificata
    tmp_ctx = tempfile.TemporaryDirectory() if persist_dir is None else None
    effective_dir = persist_dir or tmp_ctx.name

    print(f"{'='*58}")
    print("  smoke_memory.py — MemoryManager integration test")
    print(f"{'='*58}")
    print(f"  persist_dir : {effective_dir}")
    print(f"  top_k       : {top_k}")
    print()

    try:
        async with MemoryManager(persist_dir=effective_dir, top_k=top_k) as mem:

            # ------------------------------------------------------------------
            # Test 0 — stato iniziale
            # ------------------------------------------------------------------
            initial_count = await mem.count()
            print(f"[0] Stato iniziale: {initial_count} chunk nella collection")
            print()

            # ------------------------------------------------------------------
            # Test 1 — salvataggio corpus
            # ------------------------------------------------------------------
            print("[1] Salvataggio corpus ({} documenti)...".format(len(CORPUS)))
            t0 = time.time()
            saved_ids: list[str] = []
            for text, meta in CORPUS:
                result = await mem.save(text, meta)
                saved_ids.append(result.chunk_id)
                print(f"    ✓ [{result.elapsed_ms:5.0f}ms] id={result.chunk_id[:8]}… "
                      f"topic={meta.get('topic')} len={result.text_len}")
            elapsed_save = (time.time() - t0) * 1000
            count_after  = await mem.count()
            print(f"  Totale: {count_after} chunk | {elapsed_save:.0f}ms\n")

            # ------------------------------------------------------------------
            # Test 2 — ricerca semantica
            # ------------------------------------------------------------------
            print(f"[2] Ricerca semantica (top_k={top_k}):")
            for query, expected_topic in QUERIES:
                t0     = time.time()
                chunks = await mem.search(query, top_k=top_k)
                elapsed = (time.time() - t0) * 1000
                print(f"\n  Query: \"{query}\"  ({elapsed:.0f}ms)")
                for i, c in enumerate(chunks, 1):
                    topic = c.metadata.get("topic", "?")
                    hit   = "✓" if topic == expected_topic else "·"
                    print(f"    [{hit}] #{i} score={c.relevance_score:.4f} "
                          f"topic={topic:6s}  {c.content[:55]}…")
            print()

            # ------------------------------------------------------------------
            # Test 3 — populate_context
            # ------------------------------------------------------------------
            print("[3] populate_context() su AssistantContext:")
            ctx = AssistantContext(user_text="Dimmi qualcosa sulla programmazione in Python.")
            await mem.populate_context(ctx)
            print(f"    ctx.retrieved_memories: {len(ctx.retrieved_memories)} chunk")
            print(f"    ctx.timings['memory']:  {ctx.timings.get('memory', 0):.1f}ms")
            for i, c in enumerate(ctx.retrieved_memories, 1):
                print(f"      #{i} score={c.relevance_score:.4f}  {c.content[:60]}…")
            assert ctx.retrieved_memories, "ERRORE: retrieved_memories è vuoto!"
            print()

            # ------------------------------------------------------------------
            # Test 4 — delete
            # ------------------------------------------------------------------
            target_id = saved_ids[0]
            before = await mem.count()
            ok     = await mem.delete(target_id)
            after  = await mem.count()
            symbol = "✓" if (ok and after == before - 1) else "✗"
            print(f"[4] delete(id={target_id[:8]}…): {symbol}  "
                  f"count {before} → {after}")
            print()

            # ------------------------------------------------------------------
            # Test 5 — clear
            # ------------------------------------------------------------------
            before = await mem.count()
            n      = await mem.clear()
            after  = await mem.count()
            symbol = "✓" if after == 0 else "✗"
            print(f"[5] clear(): {symbol}  rimossi {n} chunk  count {before} → {after}")
            print()

            # ------------------------------------------------------------------
            # Test 6 — search su collection vuota
            # ------------------------------------------------------------------
            empty_chunks = await mem.search("qualsiasi query")
            symbol = "✓" if empty_chunks == [] else "✗"
            print(f"[6] search() su collection vuota: {symbol}  "
                  f"ritornati {len(empty_chunks)} chunk")
            print()

            # ------------------------------------------------------------------
            # Riepilogo
            # ------------------------------------------------------------------
            _sep("─")
            print("✓ Smoke test completato — MemoryManager operativo")
            print(f"  Collection: {mem._collection_name}")
            print(f"  Directory:  {effective_dir}")

    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    persist_dir: str | None = None
    top_k = 3

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--persist-dir="):
            persist_dir = arg.split("=", 1)[1]
        elif arg == "--persist-dir" and i + 1 < len(args):
            persist_dir = args[i + 1]; i += 1
        elif arg.startswith("--top-k="):
            top_k = int(arg.split("=", 1)[1])
        elif arg == "--top-k" and i + 1 < len(args):
            top_k = int(args[i + 1]); i += 1
        i += 1

    try:
        asyncio.run(main(persist_dir=persist_dir, top_k=top_k))
    except KeyboardInterrupt:
        print("\nInterrotto.")
    except Exception as exc:
        print(f"\n✗ ERRORE: {exc}")
        sys.exit(1)
