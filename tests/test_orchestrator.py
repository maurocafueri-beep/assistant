"""
tests/test_orchestrator.py
Test suite per core/orchestrator.py.

I test veloci usano mock completi — nessuna dipendenza da Ollama, ChromaDB,
Whisper o il server TTS.
I test @pytest.mark.slow richiedono tutto lo stack attivo.

Esecuzione:
    make test
    SKIP_SLOW=1 venv-runtime/bin/pytest tests/test_orchestrator.py -v
"""

from __future__ import annotations

import os
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from core.context import (
    AssistantContext,
    InputMode,
    MemoryChunk,
    ModelRole,
    OutputMode,
)
from core.orchestrator import (
    Orchestrator,
    OrchestratorStatus,
    _build_messages,
    _format_memory_block,
    _trim_history,
)

SKIP_SLOW = os.getenv("SKIP_SLOW", "0") == "1"
slow = pytest.mark.skipif(SKIP_SLOW, reason="SKIP_SLOW=1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ctx(**kwargs) -> AssistantContext:
    defaults = dict(
        user_text="Ciao, come stai?",
        session_id="sess-test",
        input_mode=InputMode.TEXT,
        output_mode=OutputMode.TEXT,
        model_role=ModelRole.CHAT,
        system_prompt="Sei un assistente utile.",
    )
    defaults.update(kwargs)
    return AssistantContext(**defaults)


async def _fake_stream(messages, role, **kw) -> AsyncGenerator[str, None]:
    """Simula uno streaming LLM che emette 3 chunk."""
    for chunk in ["Sto ", "bene, ", "grazie!"]:
        yield chunk


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.stream = _fake_stream
    llm.chat   = AsyncMock(return_value=MagicMock(content="Risposta completa."))
    llm.aclose = AsyncMock()
    return llm


@pytest.fixture
def mock_memory():
    mem = MagicMock()
    mem.load            = AsyncMock()
    mem.aclose          = AsyncMock()
    mem.save            = AsyncMock(return_value=MagicMock(chunk_id="abc", text_len=10, elapsed_ms=5.0))
    mem.search          = AsyncMock(return_value=[])
    mem.populate_context = AsyncMock()
    return mem


@pytest.fixture
def mock_personality():
    pm = MagicMock()
    pm.load         = AsyncMock()
    pm._loaded      = True
    pm.active       = MagicMock(name="default", system_prompt="Sei un assistente.")
    pm.switch       = MagicMock()
    pm.apply_to_context = MagicMock()
    return pm


@pytest.fixture
async def orch(mock_llm, mock_memory, mock_personality):
    """
    Orchestrator pre-inizializzato con tutti i moduli mockati.
    STT e TTS disabilitati per i test veloci.
    """
    with (
        patch("core.orchestrator.OllamaClient",      return_value=mock_llm),
        patch("core.orchestrator.MemoryManager",     return_value=mock_memory),
        patch("core.orchestrator.PersonalityManager", return_value=mock_personality),
    ):
        o = Orchestrator(enable_tts=False, enable_stt=False)
        await o.load()
        yield o
        await o.aclose()


# ---------------------------------------------------------------------------
# Test helpers puri
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_base(self):
        ctx = make_ctx(system_prompt="System.", user_text="Hello")
        msgs = _build_messages(ctx)
        roles = [m.role.value for m in msgs]
        assert roles[0] == "system"
        assert roles[-1] == "user"
        assert msgs[-1].content == "Hello"

    def test_memory_injected_in_system(self):
        ctx = make_ctx(
            system_prompt="System.",
            retrieved_memories=[
                MemoryChunk("Chunk A", "conv", 0.9, {})
            ],
        )
        msgs = _build_messages(ctx)
        assert "Chunk A" in msgs[0].content

    def test_history_included(self):
        ctx = make_ctx(
            conversation_history=[
                {"role": "user",      "content": "Prima domanda"},
                {"role": "assistant", "content": "Prima risposta"},
            ],
            user_text="Seconda domanda",
        )
        msgs = _build_messages(ctx)
        contents = [m.content for m in msgs]
        assert "Prima domanda" in contents
        assert "Prima risposta" in contents
        assert "Seconda domanda" in contents

    def test_no_system_no_text(self):
        ctx = make_ctx(system_prompt="", user_text="")
        msgs = _build_messages(ctx)
        # Nessun messaggio system, nessun user
        assert all(m.role.value != "system" for m in msgs)


class TestFormatMemoryBlock:
    def test_empty(self):
        assert _format_memory_block([]) == ""

    def test_single_chunk(self):
        chunk = MemoryChunk("Contenuto test", "wiki", 0.85, {})
        result = _format_memory_block([chunk])
        assert "Contenuto test" in result
        assert "0.85" in result
        assert "wiki" in result

    def test_multiple_chunks(self):
        chunks = [
            MemoryChunk(f"Chunk {i}", "src", 0.9 - i * 0.1, {})
            for i in range(3)
        ]
        result = _format_memory_block(chunks)
        for i in range(3):
            assert f"Chunk {i}" in result


class TestTrimHistory:
    def test_no_trim_needed(self):
        history = [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        assert _trim_history(history, 10) == history

    def test_trim_keeps_recent(self):
        history = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        trimmed = _trim_history(history, 4)
        assert len(trimmed) == 4
        assert trimmed[-1]["content"] == "msg9"

    def test_zero_means_no_limit(self):
        history = [{"role": "user", "content": f"msg{i}"} for i in range(100)]
        assert _trim_history(history, 0) == history


# ---------------------------------------------------------------------------
# Test Orchestrator — inizializzazione
# ---------------------------------------------------------------------------

class TestOrchestratorInit:
    async def test_repr_loaded(self, orch):
        assert "Orchestrator" in repr(orch)

    async def test_repr_not_loaded(self):
        o = Orchestrator(enable_tts=False, enable_stt=False)
        assert "non caricato" in repr(o)

    async def test_status(self, orch):
        s = orch.status
        assert isinstance(s, OrchestratorStatus)
        assert s.llm_ok
        assert s.personality_ok

    async def test_require_loaded_raises(self):
        o = Orchestrator(enable_tts=False, enable_stt=False)
        ctx = make_ctx()
        with pytest.raises(RuntimeError, match="non inizializzato"):
            async for _ in o.turn(ctx):
                pass

    async def test_double_aclose_safe(self, orch):
        await orch.aclose()
        await orch.aclose()  # non deve sollevare


# ---------------------------------------------------------------------------
# Test turn() streaming
# ---------------------------------------------------------------------------

class TestTurnStreaming:
    async def test_turn_yields_chunks(self, orch):
        ctx = make_ctx()
        chunks = []
        async for chunk in orch.turn(ctx):
            chunks.append(chunk)
        assert chunks == ["Sto ", "bene, ", "grazie!"]

    async def test_assistant_text_set_after_turn(self, orch):
        ctx = make_ctx()
        async for _ in orch.turn(ctx):
            pass
        assert ctx.assistant_text == "Sto bene, grazie!"

    async def test_timing_llm_set(self, orch):
        ctx = make_ctx()
        async for _ in orch.turn(ctx):
            pass
        assert "llm" in ctx.timings
        assert ctx.timings["llm"] >= 0

    async def test_memory_populated(self, orch, mock_memory):
        ctx = make_ctx()
        async for _ in orch.turn(ctx):
            pass
        mock_memory.populate_context.assert_called_once()

    async def test_personality_applied(self, orch, mock_personality):
        ctx = make_ctx()
        async for _ in orch.turn(ctx):
            pass
        mock_personality.apply_to_context.assert_called_once_with(ctx)

    async def test_empty_user_text_skipped(self, orch):
        ctx = make_ctx(user_text="   ")
        chunks = []
        async for chunk in orch.turn(ctx):
            chunks.append(chunk)
        assert chunks == []

    async def test_turn_saves_to_memory(self, orch, mock_memory):
        ctx = make_ctx()
        async for _ in orch.turn(ctx):
            pass
        # Deve aver salvato almeno il testo utente
        assert mock_memory.save.call_count >= 1

    async def test_llm_error_sets_ctx_error(self, mock_memory, mock_personality):
        """Se LLM fallisce, ctx.error viene settato e il turn si interrompe."""
        bad_llm = MagicMock()

        async def _fail_stream(*a, **kw):
            raise RuntimeError("Ollama down")
            yield  # rende la funzione un generator

        bad_llm.stream = _fail_stream
        bad_llm.aclose = AsyncMock()

        with (
            patch("core.orchestrator.OllamaClient",      return_value=bad_llm),
            patch("core.orchestrator.MemoryManager",     return_value=mock_memory),
            patch("core.orchestrator.PersonalityManager", return_value=mock_personality),
        ):
            o = Orchestrator(enable_tts=False, enable_stt=False)
            await o.load()
            ctx = make_ctx()
            async for _ in o.turn(ctx):
                pass
            assert ctx.error is not None
            assert "LLM" in ctx.error
            await o.aclose()

    async def test_memory_error_non_fatal(self, mock_llm, mock_personality):
        """Se la memoria fallisce, il turn continua senza RAG."""
        bad_memory = MagicMock()
        bad_memory.load             = AsyncMock()
        bad_memory.aclose           = AsyncMock()
        bad_memory.save             = AsyncMock()
        bad_memory.populate_context = AsyncMock(side_effect=RuntimeError("chroma down"))

        with (
            patch("core.orchestrator.OllamaClient",      return_value=mock_llm),
            patch("core.orchestrator.MemoryManager",     return_value=bad_memory),
            patch("core.orchestrator.PersonalityManager", return_value=mock_personality),
        ):
            o = Orchestrator(enable_tts=False, enable_stt=False)
            await o.load()
            ctx = make_ctx()
            chunks = []
            async for chunk in o.turn(ctx):
                chunks.append(chunk)
            # Il turn deve completarsi normalmente
            assert chunks == ["Sto ", "bene, ", "grazie!"]
            assert ctx.error is None
            await o.aclose()


# ---------------------------------------------------------------------------
# Test turn_sync()
# ---------------------------------------------------------------------------

class TestTurnSync:
    async def test_returns_populated_ctx(self, orch):
        ctx = make_ctx()
        result = await orch.turn_sync(ctx)
        assert result is ctx
        assert result.assistant_text == "Sto bene, grazie!"

    async def test_ctx_error_propagated(self, mock_memory, mock_personality):
        bad_llm = MagicMock()

        async def _fail(*a, **kw):
            raise RuntimeError("down")
            yield

        bad_llm.stream = _fail
        bad_llm.aclose = AsyncMock()

        with (
            patch("core.orchestrator.OllamaClient",      return_value=bad_llm),
            patch("core.orchestrator.MemoryManager",     return_value=mock_memory),
            patch("core.orchestrator.PersonalityManager", return_value=mock_personality),
        ):
            o = Orchestrator(enable_tts=False, enable_stt=False)
            await o.load()
            ctx = make_ctx()
            result = await o.turn_sync(ctx)
            assert result.error is not None
            await o.aclose()


# ---------------------------------------------------------------------------
# Test conversation history
# ---------------------------------------------------------------------------

class TestConversationHistory:
    async def test_history_grows_across_turns(self, orch):
        for i in range(3):
            ctx = make_ctx(
                user_text=f"Domanda {i}",
                session_id="sess-history",
            )
            async for _ in orch.turn(ctx):
                pass

        history = orch._session_histories["sess-history"]
        # 3 turni × 2 messaggi (user + assistant) = 6
        assert len(history) == 6

    async def test_history_trimmed_to_window(self, mock_llm, mock_memory, mock_personality):
        """Con context_window=4 la history non supera 4 messaggi."""
        with (
            patch("core.orchestrator.OllamaClient",      return_value=mock_llm),
            patch("core.orchestrator.MemoryManager",     return_value=mock_memory),
            patch("core.orchestrator.PersonalityManager", return_value=mock_personality),
        ):
            o = Orchestrator(enable_tts=False, enable_stt=False, context_window=4)
            await o.load()

            for i in range(5):
                ctx = make_ctx(user_text=f"msg {i}", session_id="sess-trim")
                async for _ in o.turn(ctx):
                    pass

            history = o._session_histories["sess-trim"]
            assert len(history) <= 4
            await o.aclose()

    async def test_sessions_are_isolated(self, orch):
        for sid in ("sessA", "sessB"):
            ctx = make_ctx(user_text="Ciao", session_id=sid)
            async for _ in orch.turn(ctx):
                pass

        assert "sessA" in orch._session_histories
        assert "sessB" in orch._session_histories
        # Le due sessioni non si contaminano
        histA = orch._session_histories["sessA"]
        histB = orch._session_histories["sessB"]
        assert histA is not histB

    async def test_clear_session(self, orch):
        ctx = make_ctx(session_id="sess-clear")
        async for _ in orch.turn(ctx):
            pass
        orch.clear_session("sess-clear")
        assert "sess-clear" not in orch._session_histories


# ---------------------------------------------------------------------------
# Test add_memory()
# ---------------------------------------------------------------------------

class TestAddMemory:
    async def test_add_memory_ok(self, orch, mock_memory):
        result = await orch.add_memory("Testo da ricordare.", {"source": "test"})
        assert result is not None
        mock_memory.save.assert_called()

    async def test_add_memory_no_memory_module(self, mock_llm, mock_personality):
        """Se la memoria non è disponibile, ritorna None senza eccezioni."""
        with (
            patch("core.orchestrator.OllamaClient",      return_value=mock_llm),
            patch("core.orchestrator.MemoryManager",     side_effect=RuntimeError("no chroma")),
            patch("core.orchestrator.PersonalityManager", return_value=mock_personality),
        ):
            o = Orchestrator(enable_tts=False, enable_stt=False)
            await o.load()
            result = await o.add_memory("test", {})
            assert result is None
            await o.aclose()


# ---------------------------------------------------------------------------
# Test switch_personality()
# ---------------------------------------------------------------------------

class TestSwitchPersonality:
    async def test_switch_calls_personality(self, orch, mock_personality):
        orch.switch_personality("dev")
        mock_personality.switch.assert_called_once_with("dev")

    async def test_switch_before_load_raises(self):
        o = Orchestrator(enable_tts=False, enable_stt=False)
        with pytest.raises(RuntimeError):
            o.switch_personality("dev")


# ---------------------------------------------------------------------------
# Test STT path
# ---------------------------------------------------------------------------

class TestSTTPath:
    async def test_voice_input_without_audio_sets_error(self, orch):
        ctx = make_ctx(input_mode=InputMode.VOICE, raw_audio=None)
        async for _ in orch.turn(ctx):
            pass
        assert ctx.error is not None
        assert "raw_audio" in ctx.error

    async def test_voice_input_without_stt_module_sets_error(self, orch):
        # orch ha enable_stt=False → self._stt è None
        ctx = make_ctx(
            input_mode=InputMode.VOICE,
            raw_audio=b"\x00" * 1000,
        )
        async for _ in orch.turn(ctx):
            pass
        assert ctx.error is not None

    async def test_stt_error_sets_ctx_error(self, mock_llm, mock_memory, mock_personality):
        bad_stt = MagicMock()
        # WhisperSTT non ha load() — il cleanup avviene tramite __aexit__
        bad_stt.__aexit__  = AsyncMock(return_value=None)
        bad_stt.transcribe = AsyncMock(side_effect=RuntimeError("whisper down"))

        with (
            patch("core.orchestrator.OllamaClient",      return_value=mock_llm),
            patch("core.orchestrator.MemoryManager",     return_value=mock_memory),
            patch("core.orchestrator.PersonalityManager", return_value=mock_personality),
        ):
            o = Orchestrator(enable_tts=False, enable_stt=False)
            await o.load()
            o._stt = bad_stt  # inietta STT difettoso manualmente

            ctx = make_ctx(
                input_mode=InputMode.VOICE,
                raw_audio=b"\x00" * 1000,
            )
            async for _ in o.turn(ctx):
                pass

            assert ctx.error is not None
            assert "STT" in ctx.error
            await o.aclose()


# ---------------------------------------------------------------------------
# Test TTS — errore non fatale
# ---------------------------------------------------------------------------

class TestTTSPath:
    async def test_tts_error_non_fatal(self, mock_llm, mock_memory, mock_personality):
        """TTS fallisce → il turn si completa in testo senza errore."""
        bad_tts = MagicMock()
        # Qwen3TTS non ha load() né aclose() — il cleanup avviene tramite __aexit__
        bad_tts.__aexit__  = AsyncMock(return_value=None)
        bad_tts.synthesize = AsyncMock(side_effect=RuntimeError("tts down"))

        with (
            patch("core.orchestrator.OllamaClient",      return_value=mock_llm),
            patch("core.orchestrator.MemoryManager",     return_value=mock_memory),
            patch("core.orchestrator.PersonalityManager", return_value=mock_personality),
        ):
            o = Orchestrator(enable_tts=False, enable_stt=False)
            await o.load()
            o._tts = bad_tts  # inietta TTS difettoso

            ctx = make_ctx(output_mode=OutputMode.VOICE)
            chunks = []
            async for chunk in o.turn(ctx):
                chunks.append(chunk)

            assert chunks                # testo ricevuto
            assert ctx.error is None     # nessun errore fatale
            assert ctx.audio_chunks == []  # nessun audio prodotto
            await o.aclose()

    async def test_tts_audio_set_on_success(self, mock_llm, mock_memory, mock_personality):
        good_tts = MagicMock()
        # Qwen3TTS non ha load() né aclose() — il cleanup avviene tramite __aexit__
        good_tts.__aexit__  = AsyncMock(return_value=None)
        good_tts.synthesize = AsyncMock(return_value=MagicMock(audio_bytes=b"RIFF...."))

        with (
            patch("core.orchestrator.OllamaClient",      return_value=mock_llm),
            patch("core.orchestrator.MemoryManager",     return_value=mock_memory),
            patch("core.orchestrator.PersonalityManager", return_value=mock_personality),
        ):
            o = Orchestrator(enable_tts=False, enable_stt=False)
            await o.load()
            o._tts = good_tts

            ctx = make_ctx(output_mode=OutputMode.VOICE)
            async for _ in o.turn(ctx):
                pass

            assert ctx.audio_chunks == [b"RIFF...."]
            await o.aclose()


# ---------------------------------------------------------------------------
# Test OrchestratorStatus
# ---------------------------------------------------------------------------

class TestOrchestratorStatus:
    def test_to_log_dict(self):
        s = OrchestratorStatus(
            llm_ok=True, memory_ok=True, personality_ok=True,
            stt_ok=False, tts_ok=False, active_sessions=2,
        )
        d = s.to_log_dict()
        assert d["llm"] is True
        assert d["sessions"] == 2


# ---------------------------------------------------------------------------
# Slow tests (richiedono stack reale)
# ---------------------------------------------------------------------------

@slow
class TestOrchestratorIntegration:
    """Richiedono Ollama attivo, ChromaDB su disco, Whisper, server TTS."""

    async def test_text_turn_real(self):
        async with Orchestrator(enable_tts=False, enable_stt=False) as orch:
            ctx = make_ctx(user_text="Chi sei?")
            result = await orch.turn_sync(ctx)
            assert result.assistant_text
            assert not result.error

    async def test_memory_persists_across_turns(self):
        async with Orchestrator(enable_tts=False, enable_stt=False) as orch:
            ctx1 = make_ctx(user_text="Mi chiamo Mauro.", session_id="integ")
            await orch.turn_sync(ctx1)

            ctx2 = make_ctx(user_text="Come mi chiamo?", session_id="integ")
            result = await orch.turn_sync(ctx2)
            assert result.assistant_text
