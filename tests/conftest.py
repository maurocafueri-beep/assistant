"""tests/conftest.py — fixture condivise e auto-skip dei test su servizi reali"""
import struct
import sys
from functools import lru_cache
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Auto-skip dei test marcati real_* quando il servizio non risponde.
#
# Il marker @slow (SKIP_SLOW=1) resta lo skip manuale; questi probe evitano
# invece i timeout lunghi (240s l'uno sul TTS) quando un servizio è
# semplicemente spento: il test viene saltato subito con il motivo visibile
# in `pytest -rs`. Probe veloci (2s) e cachati per l'intera sessione.
# ---------------------------------------------------------------------------

_TTS_HEALTH_URL = "http://127.0.0.1:8765/health"   # porta default Qwen3TTS


@lru_cache(maxsize=2)
def _ollama_missing(require_chat: bool = False) -> "str | None":
    """None se Ollama risponde e ha i modelli richiesti, altrimenti il motivo.

    Base: server + modello embed (basta per i test memoria/RAG).
    require_chat=True richiede anche il modello chat (turni LLM reali).
    """
    import httpx
    from config.settings import settings
    try:
        r = httpx.get(f"{settings.ollama.base_url}/api/tags", timeout=2.0)
        r.raise_for_status()
    except Exception as exc:
        return f"Ollama non raggiungibile: {exc.__class__.__name__}"
    names = {m.get("name", "") for m in r.json().get("models", [])}

    def _present(model: str) -> bool:
        return model in names or model.split(":")[0] in {n.split(":")[0] for n in names}

    required = [settings.ollama.embed_model]
    if require_chat:
        required.append(settings.ollama.chat_model)
    for model in required:
        if not _present(model):
            return f"modello '{model}' non installato in Ollama"
    return None


@lru_cache(maxsize=1)
def _tts_missing() -> "str | None":
    """None se il server TTS risponde, altrimenti il motivo."""
    import httpx
    try:
        r = httpx.get(_TTS_HEALTH_URL, timeout=2.0)
        r.raise_for_status()
        return None
    except Exception as exc:
        return f"server TTS non attivo su {_TTS_HEALTH_URL}: {exc.__class__.__name__}"


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("real_ollama"):
            reason = _ollama_missing()
            if reason:
                item.add_marker(pytest.mark.skip(reason=reason))
        if item.get_closest_marker("real_ollama_chat"):
            reason = _ollama_missing(require_chat=True)
            if reason:
                item.add_marker(pytest.mark.skip(reason=reason))
        if item.get_closest_marker("real_tts"):
            reason = _tts_missing()
            if reason:
                item.add_marker(pytest.mark.skip(reason=reason))

@pytest.fixture(scope="session")
def sample_audio() -> bytes:
    num_samples = 16000
    raw = struct.pack(f"<{num_samples}h", *([0] * num_samples))
    return raw

@pytest.fixture(scope="session")
def sample_text() -> str:
    return "Ciao, come posso aiutarti oggi?"
