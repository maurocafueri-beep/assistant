"""
tests/test_llm.py
Test suite per modules/llm.

asyncio_mode = "auto" configurato in pyproject.toml → le funzioni async
sono raccolte automaticamente, @pytest.mark.asyncio non è necessario.

Esecuzione:
    make test
    venv-runtime/bin/pytest tests/test_llm.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.context import ModelRole
from modules.llm import LLMResponse, Message, OllamaClient, Role


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_user_msg() -> list[Message]:
    return [Message(Role.USER, "Ciao!")]


@pytest.fixture
def conversation() -> list[Message]:
    return [
        Message(Role.SYSTEM, "Sei un assistente utile."),
        Message(Role.USER, "Quanto fa 2+2?"),
        Message(Role.ASSISTANT, "4."),
        Message(Role.USER, "E 4+4?"),
    ]


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

class TestMessage:
    def test_to_dict_no_images(self):
        msg = Message(Role.USER, "ciao")
        d   = msg.to_dict()
        assert d == {"role": "user", "content": "ciao"}
        assert "images" not in d

    def test_to_dict_with_images(self):
        msg = Message(Role.USER, "descrivimi questo", images=["base64abc"])
        assert msg.to_dict()["images"] == ["base64abc"]

    def test_role_values(self):
        assert Role.SYSTEM.value    == "system"
        assert Role.USER.value      == "user"
        assert Role.ASSISTANT.value == "assistant"


# ---------------------------------------------------------------------------
# ModelRole — vive in core.context, qui verifichiamo solo che sia importabile
# e coerente con i valori attesi da _model_for_role
# ---------------------------------------------------------------------------

class TestModelRole:
    def test_values(self):
        assert ModelRole.CHAT.value   == "chat"
        assert ModelRole.CODE.value   == "code"
        assert ModelRole.VISION.value == "vision"


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------

class TestLLMResponse:
    def test_tokens_property(self):
        r = LLMResponse(
            "ok", "qwen3:14b-q8_0", ModelRole.CHAT,
            prompt_tokens=10, completion_tokens=20, total_tokens=30,
        )
        assert r.tokens == {"prompt": 10, "completion": 20, "total": 30}

    def test_defaults(self):
        r = LLMResponse("ok", "model", ModelRole.CHAT)
        assert r.done is True
        assert r.total_tokens == 0


# ---------------------------------------------------------------------------
# Settings — verifica che i model name vengano dai settings pydantic
# ---------------------------------------------------------------------------

class TestSettings:
    def test_model_names_not_empty(self):
        from config.settings import settings
        assert settings.ollama.chat_model
        assert settings.ollama.code_model
        assert settings.ollama.vision_model
        assert settings.ollama.embed_model

    def test_model_for_role_resolution(self):
        from config.settings import settings
        from modules.llm.base_llm import _model_for_role
        assert _model_for_role(ModelRole.CHAT)   == settings.ollama.chat_model
        assert _model_for_role(ModelRole.CODE)   == settings.ollama.code_model
        assert _model_for_role(ModelRole.VISION) == settings.ollama.vision_model

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_CHAT_MODEL", "llama3:8b")
        # Reistanziamo OllamaSettings per raccogliere il nuovo env
        from config.settings import OllamaSettings
        s = OllamaSettings()
        assert s.chat_model == "llama3:8b"


# ---------------------------------------------------------------------------
# Helpers — _build_payload
# ---------------------------------------------------------------------------

class TestBuildPayload:
    def test_system_injected_when_absent(self, single_user_msg):
        client = OllamaClient.__new__(OllamaClient)  # salta __init__
        payload = client._build_payload(
            "model", single_user_msg, None, "Sei un assistente.", stream=False
        )
        msgs = payload["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Sei un assistente."
        assert msgs[1]["role"] == "user"

    def test_system_not_duplicated(self, conversation):
        client = OllamaClient.__new__(OllamaClient)
        payload = client._build_payload(
            "model", conversation, None, "System duplicato", stream=False
        )
        system_msgs = [m for m in payload["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1

    def test_stream_flag(self, single_user_msg):
        client = OllamaClient.__new__(OllamaClient)
        assert client._build_payload("m", single_user_msg, None, None, True)["stream"] is True
        assert client._build_payload("m", single_user_msg, None, None, False)["stream"] is False

    def test_options_merged(self, single_user_msg):
        client = OllamaClient.__new__(OllamaClient)
        payload = client._build_payload(
            "m", single_user_msg, {"temperature": 0.1}, None, False
        )
        assert payload["options"]["temperature"] == 0.1


# ---------------------------------------------------------------------------
# OllamaClient.chat — mock httpx
# ---------------------------------------------------------------------------

def _mock_chat_resp(content: str = "risposta") -> MagicMock:
    m = MagicMock()
    m.status_code = 200
    m.raise_for_status = MagicMock()
    m.json.return_value = {
        "message":           {"role": "assistant", "content": content},
        "model":             "qwen3:14b-q8_0",
        "done":              True,
        "prompt_eval_count": 5,
        "eval_count":        10,
    }
    return m


class TestOllamaClientChat:
    async def test_returns_llm_response(self, single_user_msg):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
                   return_value=_mock_chat_resp("Ciao!")):
            async with OllamaClient() as llm:
                resp = await llm.chat(single_user_msg, ModelRole.CHAT)

        assert isinstance(resp, LLMResponse)
        assert resp.content == "Ciao!"
        assert resp.role == ModelRole.CHAT
        assert resp.done is True

    async def test_token_counts(self, single_user_msg):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
                   return_value=_mock_chat_resp()):
            async with OllamaClient() as llm:
                resp = await llm.chat(single_user_msg)

        assert resp.prompt_tokens     == 5
        assert resp.completion_tokens == 10
        assert resp.total_tokens      == 15

    async def test_model_override(self, single_user_msg):
        captured: dict = {}

        async def fake_post(url, *, json=None, **kw):
            captured.update(json or {})
            return _mock_chat_resp()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=fake_post):
            async with OllamaClient() as llm:
                await llm.chat(single_user_msg, model="llama3:8b")

        assert captured["model"] == "llama3:8b"


# ---------------------------------------------------------------------------
# OllamaClient.stream
# ---------------------------------------------------------------------------

class TestOllamaClientStream:
    async def test_yields_chunks(self, single_user_msg):
        lines = [
            '{"message":{"role":"assistant","content":"Cia"},"done":false}',
            '{"message":{"role":"assistant","content":"o!"},"done":true}',
        ]

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__  = AsyncMock(return_value=False)
        ctx.raise_for_status = MagicMock()

        async def fake_aiter_lines():
            for line in lines:
                yield line

        ctx.aiter_lines = fake_aiter_lines

        with patch("httpx.AsyncClient.stream", return_value=ctx):
            async with OllamaClient() as llm:
                chunks = [c async for c in llm.stream(single_user_msg)]

        assert chunks == ["Cia", "o!"]


# ---------------------------------------------------------------------------
# OllamaClient.embed
# ---------------------------------------------------------------------------

class TestOllamaClientEmbed:
    async def test_single_text(self):
        m = MagicMock()
        m.status_code = 200
        m.raise_for_status = MagicMock()
        m.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=m):
            async with OllamaClient() as llm:
                vecs = await llm.embed("testo")

        assert len(vecs) == 1
        assert vecs[0] == [0.1, 0.2, 0.3]

    async def test_multiple_texts(self):
        m = MagicMock()
        m.status_code = 200
        m.raise_for_status = MagicMock()
        m.json.return_value = {"embeddings": [[0.1], [0.2], [0.3]]}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=m):
            async with OllamaClient() as llm:
                vecs = await llm.embed(["a", "b", "c"])

        assert len(vecs) == 3

    async def test_fallback_old_ollama(self):
        """Se /api/embed ritorna 404, usa /api/embeddings per compatibilità."""
        not_found = MagicMock()
        not_found.status_code = 404

        ok = MagicMock()
        ok.raise_for_status = MagicMock()
        ok.json.return_value = {"embedding": [0.9, 0.8]}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
                   side_effect=[not_found, ok]):
            async with OllamaClient() as llm:
                vecs = await llm.embed("testo")

        assert vecs == [[0.9, 0.8]]


# ---------------------------------------------------------------------------
# OllamaClient.is_available / list_models
# ---------------------------------------------------------------------------

class TestOllamaClientUtility:
    async def test_is_available_true(self):
        m = MagicMock()
        m.status_code = 200
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=m):
            async with OllamaClient() as llm:
                assert await llm.is_available() is True

    async def test_is_available_false_on_exception(self):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
                   side_effect=Exception("connection refused")):
            async with OllamaClient() as llm:
                assert await llm.is_available() is False

    async def test_list_models(self):
        m = MagicMock()
        m.raise_for_status = MagicMock()
        m.json.return_value = {
            "models": [{"name": "qwen3:14b-q8_0"}, {"name": "nomic-embed-text"}]
        }
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=m):
            async with OllamaClient() as llm:
                models = await llm.list_models()

        assert "qwen3:14b-q8_0"   in models
        assert "nomic-embed-text" in models
