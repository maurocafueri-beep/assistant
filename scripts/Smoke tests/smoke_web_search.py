"""
scripts/smoke_web_search.py
Smoke test manuale per modules/web_search.

Richiede SearXNG attivo su settings.web_search.searxng_url.
Avviarlo con:
    cd ~/assistant && docker compose -f docker/docker-compose.yml up -d searxng

Esecuzione:
    cd ~/assistant
    venv-runtime/bin/python scripts/smoke_web_search.py
"""

import asyncio
import time

from modules.web_search import SearXNGClient


async def main() -> None:
    from config.settings import settings

    print(f"\n{'='*60}")
    print(f"  Smoke test — SearXNGClient")
    print(f"  URL: {settings.web_search.searxng_url}")
    print(f"  max_results: {settings.web_search.max_results}")
    print(f"  timeout: {settings.web_search.scrape_timeout}s")
    print(f"{'='*60}\n")

    async with SearXNGClient() as ws:

        # Test 0 — health check
        print("[0] Health check …")
        available = await ws.is_available()
        status = "✓ ONLINE" if available else "✗ OFFLINE"
        print(f"    SearXNG: {status}")
        if not available:
            print("    ⚠  SearXNG non raggiungibile — avvia il container e riprova.")
            print(f"       docker compose -f docker/docker-compose.yml up -d searxng")
            return

        # Test 1 — ricerca base in italiano
        print("\n[1] Ricerca base (italiano) …")
        start = time.time()
        resp = await ws.search("assistente vocale Python open source")
        elapsed = time.time() - start
        print(f"    Query:    '{resp.query}'")
        print(f"    Risultati: {len(resp.results)} / {resp.total_found} totali")
        print(f"    Tempo:    {elapsed:.2f}s ({resp.elapsed_ms:.0f} ms interno)")
        if resp.error:
            print(f"    Errore:   {resp.error}")
        for i, r in enumerate(resp.results[:3], 1):
            print(f"    [{i}] {r.title[:70]}")
            print(f"         {r.url[:90]}")
            if r.snippet:
                print(f"         ↳ {r.snippet[:100]}…")

        # Test 2 — ricerca tecnica in inglese
        print("\n[2] Ricerca tecnica (inglese) …")
        start = time.time()
        resp2 = await ws.search("asyncio python event loop tutorial", language="en-US")
        elapsed2 = time.time() - start
        print(f"    Query:    '{resp2.query}'")
        print(f"    Risultati: {len(resp2.results)} / {resp2.total_found} totali")
        print(f"    Tempo:    {elapsed2:.2f}s")
        if resp2.error:
            print(f"    Errore:   {resp2.error}")
        else:
            print(f"    Primo risultato: {resp2.results[0].title[:70] if resp2.results else '—'}")

        # Test 3 — max_results override
        print("\n[3] Limite risultati personalizzato (max=2) …")
        resp3 = await ws.search("open source LLM models", max_results=2)
        print(f"    Ottenuti: {len(resp3.results)} risultati (richiesti: 2)")
        assert len(resp3.results) <= 2, f"Attesi <= 2, ottenuti {len(resp3.results)}"
        print("    ✓ limite rispettato")

        # Test 4 — query vuota (deve tornare vuota senza crash)
        print("\n[4] Query vuota (gestione errori non fatali) …")
        resp4 = await ws.search("   ")
        assert resp4.is_empty(), "Query vuota deve restituire risposta vuota"
        assert resp4.error is None
        print("    ✓ gestita senza crash")

        # Test 5 — to_log_dict
        print("\n[5] to_log_dict …")
        if not resp.is_empty():
            d = resp.to_log_dict()
            print(f"    SearchResponse: {d}")
            d2 = resp.results[0].to_log_dict()
            print(f"    SearchResult:   {d2}")
            print("    ✓ formato log corretto")

    print(f"\n{'='*60}")
    print("  ✓ Smoke test completato con successo")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
