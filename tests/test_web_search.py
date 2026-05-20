"""
tests/test_web_search.py
Test suite per modules/web_search.

asyncio_mode = "auto" configurato in pyproject.toml → le funzioni async
sono raccolte automaticamente, @pytest.mark.asyncio non è necessario.

Esecuzione:
    make test
    venv-runtime/bin/pytest tests/test_web_search.py -v

Test reali (richiedono SearXNG attivo):
    SKIP_SLOW=0 venv-runtime/bin/pytest tests/test_web_search.py -v -m slow
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.web_search import SearchResponse, SearchResult, SearXNGClient
from modules.web_search.base_web_search import _parse_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_search_resp(
    results: list[dict] | None = None,
    number_of_results: int = 0,
    status_code: int = 200,
) -> MagicMock:
    """Costruisce un mock di httpx.Response per /search."""
    if results is None:
        results = [
            {
                "title":   "Python asyncio — documentazione ufficiale",
                "url":     "https://docs.python.org/3/library/asyncio.html",
                "content": "asyncio è una libreria per scrivere codice concorrente con async/await.",
                "engine":  "google",
                "score":   0.9,
            },
            {
                "title":   "Real Python — asyncio tutorial",
                "url":     "https://realpython.com/async-io-python/",
                "content": "Tutorial completo su asyncio in Python.",
                "engine":  "bing",
                "score":   0.75,
            },
        ]
        number_of_results = number_of_results or len(results)

    m = MagicMock()
    m.status_code = status_code
    m.raise_for_status = MagicMock()
    m.json.return_value = {
        "results":           results,
        "number_of_results": number_of_results,
        "query":             "test query",
    }
    return m


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_fields(self):
        r = SearchResult(
            title="Titolo",
            url="https://example.com",
            snippet="Uno snippet di testo.",
        )
        assert r.title   == "Titolo"
        assert r.url     == "https://example.com"
        assert r.snippet == "Uno snippet di testo."
        assert r.engine  == ""
        assert r.score   == 0.0

    def test_to_log_dict_keys(self):
        r = SearchResult(title="T", url="https://x.com", snippet="S", engine="google")
        d = r.to_log_dict()
        assert set(d.keys()) == {"title", "url", "snippet", "engine"}

    def test_to_log_dict_truncation(self):
        long_title   = "A" * 200
        long_url     = "https://x.com/" + "b" * 200
        long_snippet = "C" * 200
        r = SearchResult(title=long_title, url=long_url, snippet=long_snippet)
        d = r.to_log_dict()
        assert len(d["title"])   <= 80
        assert len(d["url"])     <= 120
        assert len(d["snippet"]) <= 100


# ---------------------------------------------------------------------------
# SearchResponse
# ---------------------------------------------------------------------------

class TestSearchResponse:
    def test_is_empty_true(self):
        resp = SearchResponse(query="test")
        assert resp.is_empty() is True

    def test_is_empty_false(self):
        resp = SearchResponse(
            query="test",
            results=[SearchResult("T", "https://x.com", "S")],
        )
        assert resp.is_empty() is False

    def test_to_log_dict_keys(self):
        resp = SearchResponse(query="test", elapsed_ms=42.5, total_found=10)
        d = resp.to_log_dict()
        assert "query"       in d
        assert "n_results"   in d
        assert "total_found" in d
        assert "elapsed_ms"  in d
        assert "error"       in d

    def test_to_log_dict_values(self):
        r    = SearchResult("T", "https://x.com", "S")
        resp = SearchResponse(query="python asyncio", results=[r], elapsed_ms=33.1, total_found=5)
        d    = resp.to_log_dict()
        assert d["n_results"]   == 1
        assert d["total_found"] == 5
        assert d["elapsed_ms"]  == 33.1
        assert d["error"]       is None

    def test_error_field(self):
        resp = SearchResponse(query="test", error="timeout")
        assert resp.error == "timeout"
        assert resp.to_log_dict()["error"] == "timeout"


# ---------------------------------------------------------------------------
# _parse_results (helper interno)
# ---------------------------------------------------------------------------

class TestParseResults:
    def test_basic_parsing(self):
        data = {
            "results": [
                {"title": "T1", "url": "https://a.com", "content": "S1", "engine": "google"},
                {"title": "T2", "url": "https://b.com", "content": "S2", "engine": "bing"},
            ],
            "number_of_results": 42,
        }
        results, total = _parse_results(data, max_results=10)
        assert len(results) == 2
        assert total        == 42
        assert results[0].title  == "T1"
        assert results[0].engine == "google"
        assert results[1].url    == "https://b.com"

    def test_max_results_truncation(self):
        data = {
            "results": [{"title": f"T{i}", "url": f"https://x{i}.com", "content": ""} for i in range(10)],
            "number_of_results": 100,
        }
        results, total = _parse_results(data, max_results=3)
        assert len(results) == 3
        assert total        == 100

    def test_missing_url_skipped(self):
        data = {
            "results": [
                {"title": "Con URL",  "url": "https://good.com", "content": "ok"},
                {"title": "Senza URL","url": "",                 "content": "no"},
                {"title": "None URL", "url": None,               "content": "no"},
            ],
            "number_of_results": 3,
        }
        results, _ = _parse_results(data, max_results=10)
        assert len(results) == 1
        assert results[0].url == "https://good.com"

    def test_none_fields_handled(self):
        data = {
            "results": [
                {"title": None, "url": "https://x.com", "content": None, "engine": None, "score": None},
            ],
            "number_of_results": 1,
        }
        results, _ = _parse_results(data, max_results=10)
        assert len(results) == 1
        assert results[0].title   == ""
        assert results[0].snippet == ""
        assert results[0].engine  == ""
        assert results[0].score   == 0.0

    def test_empty_results(self):
        data = {"results": [], "number_of_results": 0}
        results, total = _parse_results(data, max_results=10)
        assert results == []
        assert total   == 0

    def test_missing_number_of_results(self):
        data = {
            "results": [{"title": "T", "url": "https://x.com", "content": "S"}],
        }
        results, total = _parse_results(data, max_results=10)
        assert len(results) == 1
        assert total        == 1   # fallback = len(raw)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class TestWebSearchSettings:
    def test_settings_fields(self):
        from config.settings import settings
        assert settings.web_search.searxng_url
        assert settings.web_search.max_results   > 0
        assert settings.web_search.scrape_timeout > 0

    def test_env_override_url(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://myhost:9999")
        from config.settings import WebSearchSettings
        s = WebSearchSettings()
        assert s.searxng_url == "http://myhost:9999"

    def test_env_override_max_results(self, monkeypatch):
        monkeypatch.setenv("WEB_SEARCH_MAX_RESULTS", "15")
        from config.settings import WebSearchSettings
        s = WebSearchSettings()
        assert s.max_results == 15


# ---------------------------------------------------------------------------
# SearXNGClient — mock httpx
# ---------------------------------------------------------------------------

class TestSearXNGClientInit:
    def test_default_params(self):
        from config.settings import settings
        client = SearXNGClient.__new__(SearXNGClient)
        # Verifica che __init__ legga da settings
        client2 = SearXNGClient()
        assert client2._max_results == settings.web_search.max_results
        assert client2._timeout     == settings.web_search.scrape_timeout

    def test_custom_params(self):
        client = SearXNGClient(
            base_url="http://custom:9090",
            max_results=3,
            timeout=5,
        )
        assert client._base_url    == "http://custom:9090"
        assert client._max_results == 3
        assert client._timeout     == 5

    def test_trailing_slash_stripped(self):
        client = SearXNGClient(base_url="http://localhost:8080/")
        assert not client._base_url.endswith("/")


class TestSearXNGClientSearch:
    async def test_returns_search_response(self):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
                   return_value=_mock_search_resp()):
            async with SearXNGClient() as ws:
                resp = await ws.search("python asyncio")

        assert isinstance(resp, SearchResponse)
        assert not resp.is_empty()
        assert resp.error is None

    async def test_result_fields_populated(self):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
                   return_value=_mock_search_resp()):
            async with SearXNGClient() as ws:
                resp = await ws.search("python asyncio")

        r = resp.results[0]
        assert r.title
        assert r.url.startswith("https://")
        assert r.snippet

    async def test_max_results_respected(self):
        # Costruiamo una risposta mock con 5 risultati
        raw = [
            {"title": f"T{i}", "url": f"https://x{i}.com", "content": f"S{i}"}
            for i in range(5)
        ]
        mock_resp = _mock_search_resp(results=raw, number_of_results=5)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            async with SearXNGClient(max_results=2) as ws:
                resp = await ws.search("test")

        assert len(resp.results) <= 2

    async def test_empty_query_returns_empty(self):
        async with SearXNGClient() as ws:
            resp = await ws.search("   ")

        assert resp.is_empty()
        assert resp.error is None

    async def test_timeout_returns_empty_nonfatal(self):
        import httpx as _httpx
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
                   side_effect=_httpx.TimeoutException("timeout")):
            async with SearXNGClient() as ws:
                resp = await ws.search("test query")

        assert resp.is_empty()
        assert resp.error is not None
        assert "timeout" in resp.error

    async def test_http_error_returns_empty_nonfatal(self):
        import httpx as _httpx
        mock = MagicMock()
        mock.status_code = 500
        err = _httpx.HTTPStatusError("500", request=MagicMock(), response=mock)
        mock.raise_for_status.side_effect = err

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock):
            async with SearXNGClient() as ws:
                resp = await ws.search("test query")

        assert resp.is_empty()
        assert resp.error is not None

    async def test_network_error_returns_empty_nonfatal(self):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
                   side_effect=ConnectionError("connection refused")):
            async with SearXNGClient() as ws:
                resp = await ws.search("test query")

        assert resp.is_empty()
        assert resp.error is not None

    async def test_elapsed_ms_populated(self):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
                   return_value=_mock_search_resp()):
            async with SearXNGClient() as ws:
                resp = await ws.search("test")

        assert resp.elapsed_ms >= 0

    async def test_total_found_populated(self):
        mock_resp = _mock_search_resp(number_of_results=999)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            async with SearXNGClient() as ws:
                resp = await ws.search("test")

        assert resp.total_found == 999

    async def test_engines_param_forwarded(self):
        captured: dict = {}

        async def fake_get(url, **kw):
            captured["url"] = url
            return _mock_search_resp()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=fake_get):
            async with SearXNGClient() as ws:
                await ws.search("test", engines=["google", "bing"])

        assert "engines=google%2Cbing" in captured["url"] or "engines=google" in captured["url"]

    async def test_language_param_forwarded(self):
        captured: dict = {}

        async def fake_get(url, **kw):
            captured["url"] = url
            return _mock_search_resp()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=fake_get):
            async with SearXNGClient() as ws:
                await ws.search("test", language="it-IT")

        assert "language=it-IT" in captured["url"]


class TestSearXNGClientAvailability:
    async def test_is_available_true(self):
        m = MagicMock()
        m.status_code = 200
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=m):
            async with SearXNGClient() as ws:
                assert await ws.is_available() is True

    async def test_is_available_true_on_redirect(self):
        """Anche 302/301 conta come disponibile (status < 500)."""
        m = MagicMock()
        m.status_code = 302
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=m):
            async with SearXNGClient() as ws:
                assert await ws.is_available() is True

    async def test_is_available_false_on_exception(self):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
                   side_effect=Exception("connection refused")):
            async with SearXNGClient() as ws:
                assert await ws.is_available() is False


class TestSearXNGClientContextManager:
    async def test_aclose_called_on_exit(self):
        closed = []

        async def fake_aclose():
            closed.append(True)

        with patch("httpx.AsyncClient.aclose", new_callable=AsyncMock, side_effect=fake_aclose):
            async with SearXNGClient():
                pass

        assert len(closed) == 1

    async def test_explicit_aclose(self):
        ws = SearXNGClient()
        await ws.aclose()   # non deve sollevare eccezioni


# ---------------------------------------------------------------------------
# Test reali (SearXNG live) — solo con SKIP_SLOW=0
# ---------------------------------------------------------------------------

@pytest.mark.slow
async def test_real_search_basic():
    """Richiede SearXNG attivo su settings.web_search.searxng_url."""
    skip = os.getenv("SKIP_SLOW", "1")
    if skip != "0":
        pytest.skip("test lento skippato (SKIP_SLOW != 0)")

    async with SearXNGClient() as ws:
        assert await ws.is_available(), "SearXNG non raggiungibile"
        resp = await ws.search("Python programming language")

    assert not resp.is_empty(), "Nessun risultato dalla ricerca reale"
    assert resp.elapsed_ms > 0
    for r in resp.results:
        assert r.url.startswith("http")


@pytest.mark.slow
async def test_real_search_italian():
    """Verifica la ricerca con query in italiano."""
    skip = os.getenv("SKIP_SLOW", "1")
    if skip != "0":
        pytest.skip("test lento skippato (SKIP_SLOW != 0)")

    async with SearXNGClient() as ws:
        resp = await ws.search("notizie tecnologia oggi", language="it-IT")

    assert not resp.is_empty()
    assert all(r.url for r in resp.results)


@pytest.mark.slow
async def test_real_search_max_results_limit():
    """Verifica che max_results venga rispettato sulla risposta reale."""
    skip = os.getenv("SKIP_SLOW", "1")
    if skip != "0":
        pytest.skip("test lento skippato (SKIP_SLOW != 0)")

    max_n = 3
    async with SearXNGClient(max_results=max_n) as ws:
        resp = await ws.search("open source AI models")

    assert len(resp.results) <= max_n
