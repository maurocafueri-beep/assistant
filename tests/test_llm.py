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
from modules.llm.base_llm import _StripThink, _strip_think_tags


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

    def test_num_ctx_default_from_settings(self, single_user_msg):
        """
        Senza num_ctx esplicito nelle options, _build_payload usa il default
        dai settings (16384). Senza questo, Ollama userebbe il suo 4096
        interno — limite troppo basso per file analysis e chat lunghe.
        """
        client = OllamaClient.__new__(OllamaClient)
        payload = client._build_payload(
            "m", single_user_msg, None, None, False
        )
        assert payload["options"]["num_ctx"] == 16384

    def test_num_ctx_caller_wins(self, single_user_msg):
        """
        Se il chiamante mette num_ctx nelle options, quello prevale sul
        default dei settings: serve al map-reduce del riassunto per usare
        context più grandi su singoli chunk senza ricommittare nulla.
        """
        client = OllamaClient.__new__(OllamaClient)
        payload = client._build_payload(
            "m", single_user_msg, {"num_ctx": 32768}, None, False
        )
        assert payload["options"]["num_ctx"] == 32768

    def test_num_ctx_env_override_settings(self, single_user_msg, monkeypatch):
        """
        L'env OLLAMA_NUM_CTX dev'essere onorata: l'utente alza il context
        via env, _build_payload lo propaga senza richiedere altro.
        """
        from config.settings import settings
        monkeypatch.setattr(settings.ollama, "num_ctx", 24576)
        client = OllamaClient.__new__(OllamaClient)
        payload = client._build_payload(
            "m", single_user_msg, None, None, False
        )
        assert payload["options"]["num_ctx"] == 24576


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


# ---------------------------------------------------------------------------
# Filtro <think>...</think> sui chunk LLM
# ---------------------------------------------------------------------------

class TestStripThink:
    """
    Copre i casi che il filtro deve gestire:
      - chunk pulito (no-op),
      - tag interi in un singolo chunk,
      - tag spezzato su più chunk (sia apertura che chiusura),
      - variante <thinking>...</thinking>,
      - blocchi multipli,
      - stream tronco a metà thinking,
      - prefisso sospetto seguito da falso allarme.
    """

    @staticmethod
    def _feed(chunks):
        s = _StripThink(); out = ""
        for c in chunks:
            out += s.feed(c)
        out += s.flush()
        return out

    def test_no_tags_is_noop(self):
        assert self._feed(["ciao", " mondo"]) == "ciao mondo"

    def test_whole_block_single_chunk(self):
        assert self._feed(["A<think>X</think>B"]) == "AB"

    def test_open_split_across_chunks(self):
        assert self._feed(["A<", "think>X</think>B"]) == "AB"

    def test_close_split_across_chunks(self):
        assert self._feed(["A<think>X</thi", "nk>B"]) == "AB"

    def test_thinking_variant(self):
        assert self._feed(["A<thinking>X</thinking>B"]) == "AB"

    def test_multiple_blocks(self):
        assert self._feed(
            ["A<think>x</think>B<thinking>y</thinking>C"]
        ) == "ABC"

    def test_unterminated_think_truncated(self):
        # Stream interrotto mentre eravamo dentro <think>: nessun residuo emesso.
        assert self._feed(["A<think>x"]) == "A"

    def test_char_by_char(self):
        assert self._feed(list("A<think>X</think>B")) == "AB"

    def test_false_positive_partial_then_unrelated(self):
        # Un prefisso che sembra l'inizio di un tag, ma poi prosegue diverso,
        # deve essere emesso integro (non perso).
        assert self._feed(["A<th", "ello"]) == "A<thello"

    def test_less_than_not_a_tag(self):
        assert self._feed(["a < b"]) == "a < b"

    def test_single_shot_helper(self):
        big = "A<think>ragionamento</think>B<thinking>altro</thinking>C"
        assert _strip_think_tags(big) == "ABC"
        assert _strip_think_tags("no tags here") == "no tags here"
