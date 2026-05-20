"""
modules/web_search/base_web_search.py
SearXNGClient — client asincrono per il motore di ricerca SearXNG locale.

Endpoints utilizzati:
    GET /search?q=<query>&format=json   → lista risultati JSON

API pubblica:
    client.search(query, **kw)          → SearchResponse   (async)
    client.is_available()               → bool             (async)

Uso tipico:
    async with SearXNGClient() as ws:
        resp = await ws.search("ultime notizie AI")
        for r in resp.results:
            print(r.title, r.url)

Errori non fatali:
    - SearXNG non raggiungibile → restituisce SearchResponse vuoto + log warning
    - Risposta malformata       → restituisce SearchResponse vuoto + log warning
    - Timeout                   → restituisce SearchResponse vuoto + log warning
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from config.settings import settings
from core.logger import logger


# ---------------------------------------------------------------------------
# Dataclasses pubbliche
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Un singolo risultato di ricerca restituito da SearXNG."""
    title:   str
    url:     str
    snippet: str
    engine:  str = ""
    score:   float = 0.0

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "title":   self.title[:80],
            "url":     self.url[:120],
            "snippet": self.snippet[:100],
            "engine":  self.engine,
        }


@dataclass
class SearchResponse:
    """Risposta completa da una query SearXNG."""
    query:         str
    results:       list[SearchResult] = field(default_factory=list)
    elapsed_ms:    float = 0.0
    total_found:   int   = 0
    error:         Optional[str] = None

    def is_empty(self) -> bool:
        return len(self.results) == 0

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "query":       self.query[:80],
            "n_results":   len(self.results),
            "total_found": self.total_found,
            "elapsed_ms":  round(self.elapsed_ms, 1),
            "error":       self.error,
        }


# ---------------------------------------------------------------------------
# Helpers interni
# ---------------------------------------------------------------------------

def _parse_results(data: dict, max_results: int) -> tuple[list[SearchResult], int]:
    """
    Estrae i SearchResult dal JSON SearXNG.
    Restituisce (risultati troncati, numero totale trovati).
    """
    raw        = data.get("results", [])
    total      = data.get("number_of_results") or len(raw)
    results: list[SearchResult] = []

    for item in raw[:max_results]:
        title   = str(item.get("title",   "") or "").strip()
        url     = str(item.get("url",     "") or "").strip()
        snippet = str(item.get("content", "") or "").strip()
        engine  = str(item.get("engine",  "") or "").strip()
        score   = float(item.get("score", 0.0) or 0.0)

        if not url:          # risultato senza URL → salta
            continue

        results.append(SearchResult(
            title=title,
            url=url,
            snippet=snippet,
            engine=engine,
            score=score,
        ))

    return results, int(total)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class SearXNGClient:
    """
    Client async per SearXNG locale. Usa un singolo httpx.AsyncClient per sessione.
    Preferire come async context manager.

    Args:
        base_url:    URL base di SearXNG  (default da settings.web_search.searxng_url).
        max_results: N. max risultati     (default da settings.web_search.max_results).
        timeout:     Timeout HTTP in sec  (default da settings.web_search.scrape_timeout).

    Esempio:
        async with SearXNGClient() as ws:
            resp = await ws.search("python asyncio")
            for r in resp.results:
                print(r.title, r.url)
    """

    def __init__(
        self,
        base_url:    Optional[str] = None,
        max_results: Optional[int] = None,
        timeout:     Optional[int] = None,
    ) -> None:
        self._base_url    = (base_url    or settings.web_search.searxng_url).rstrip("/")
        self._max_results = max_results  or settings.web_search.max_results
        self._timeout     = timeout      or settings.web_search.scrape_timeout

        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(float(self._timeout)),
            headers={"Accept": "application/json"},
        )

    # -- context manager -------------------------------------------------------

    async def __aenter__(self) -> "SearXNGClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # -- utility ---------------------------------------------------------------

    async def is_available(self) -> bool:
        """Verifica che SearXNG sia raggiungibile."""
        try:
            r = await self._http.get("/", timeout=5)
            return r.status_code < 500
        except Exception:
            return False

    # -- ricerca ---------------------------------------------------------------

    async def search(
        self,
        query:       str,
        *,
        max_results: Optional[int]       = None,
        engines:     Optional[list[str]] = None,
        language:    Optional[str]       = None,
        pageno:      int                 = 1,
    ) -> SearchResponse:
        """
        Esegue una ricerca su SearXNG e restituisce i risultati.

        Args:
            query:       Testo della query.
            max_results: N. max risultati da restituire (override default).
            engines:     Lista motori SearXNG (es. ["google","bing"]).
                         None = usa quelli configurati in SearXNG.
            language:    Codice lingua (es. "it-IT", "en-US"). None = auto.
            pageno:      Numero di pagina (default 1).

        Returns:
            SearchResponse con i risultati, sempre non-None.
            In caso di errore: SearchResponse vuoto con campo error valorizzato.
        """
        query = query.strip()
        if not query:
            logger.warning("web_search.search | query vuota — salto")
            return SearchResponse(query=query)

        n = max_results or self._max_results

        params: dict[str, Any] = {
            "q":      query,
            "format": "json",
            "pageno": pageno,
        }
        if engines:
            params["engines"] = ",".join(engines)
        if language:
            params["language"] = language

        logger.debug(
            "web_search.search | query='{}' max={} engines={}",
            query[:80], n, engines,
        )

        t0 = time.perf_counter()
        try:
            r = await self._http.get(f"/search?{urlencode(params)}")
            r.raise_for_status()
            data = r.json()
        except httpx.TimeoutException as exc:
            logger.warning(
                "web_search.search | timeout ({} s) per query='{}': {}",
                self._timeout, query[:60], exc,
            )
            return SearchResponse(query=query, error=f"timeout: {exc}")
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "web_search.search | HTTP {} per query='{}': {}",
                exc.response.status_code, query[:60], exc,
            )
            return SearchResponse(query=query, error=f"HTTP {exc.response.status_code}")
        except Exception as exc:
            logger.warning(
                "web_search.search | errore per query='{}': {}",
                query[:60], exc,
            )
            return SearchResponse(query=query, error=str(exc))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        try:
            results, total = _parse_results(data, n)
        except Exception as exc:
            logger.warning(
                "web_search.search | parsing fallito per query='{}': {}",
                query[:60], exc,
            )
            return SearchResponse(query=query, elapsed_ms=elapsed_ms, error=f"parse: {exc}")

        resp = SearchResponse(
            query=query,
            results=results,
            elapsed_ms=elapsed_ms,
            total_found=total,
        )
        logger.info(
            "web_search.search | {} | {}",
            query[:60],
            resp.to_log_dict(),
        )
        return resp
