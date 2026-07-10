"""
tests/test_voice_loop_mute.py
Regressione del deadlock "mute durante la generazione": interrupt_tts()
svuota la coda audio e in una race può ingoiare anche il sentinel None che
chiude il player — il turno restava appeso per sempre in stato speaking.
Il fix (producer_done + get a timeout nel player) garantisce la chiusura
del turno anche se il sentinel viene rubato.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.context import AssistantContext, InputMode, ModelRole, OutputMode
from core.voice_loop import VoiceLoop, VoiceLoopStats


def _ctx() -> AssistantContext:
    return AssistantContext(
        user_text="domanda", session_id="s1",
        input_mode=InputMode.TEXT, output_mode=OutputMode.TEXT,
        model_role=ModelRole.CHAT, personality_name="dev",
    )


def _voice_loop(sentences: list[str]) -> VoiceLoop:
    vl = VoiceLoop.__new__(VoiceLoop)
    vl._state = "idle"
    vl._active_audio_queue = None
    vl._tts_enabled = True
    vl._tts_speaking_start = 0.0
    vl._tts_last_play_end = 0.0
    vl._last_llm_ms = vl._last_stt_ms = vl._last_tts_ms = 0.0
    vl._stats = VoiceLoopStats()
    vl._cancel_event = asyncio.Event()
    vl._is_speaking = False
    vl._echo_block_until = 0.0
    vl._audio_tasks = set()

    tts = MagicMock()
    tts.synthesize = AsyncMock(
        return_value=SimpleNamespace(audio_bytes=b"AUDIO", duration_s=0.1)
    )
    async def _play(_audio, **_kw):
        await asyncio.sleep(0.01)
    tts.play = AsyncMock(side_effect=_play)
    tts.stop_playback = AsyncMock()
    vl._tts = tts

    orch = MagicMock()
    async def _turn(_ctx):
        for s in sentences:
            yield s
            await asyncio.sleep(0.005)
    orch.turn = _turn
    vl._orch = orch
    return vl


class TestMuteDuranteGenerazione:
    async def test_turno_si_chiude_anche_se_il_sentinel_viene_rubato(self):
        """
        Un 'ladro' svuota di continuo la coda audio (come fa interrupt_tts
        al mute), rubando anche il sentinel None. Il turno deve chiudersi
        comunque entro il timeout. Ripetuto più volte per colpire la race.
        """
        for _ in range(5):
            vl = _voice_loop([
                "Prima frase abbastanza lunga per il TTS. ",
                "Seconda frase ugualmente lunga e valida. ",
            ])

            stealing = True
            async def _thief():
                while stealing:
                    q = vl._active_audio_queue
                    if q is not None:
                        try:
                            while True:
                                q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await asyncio.sleep(0)

            thief = asyncio.create_task(_thief())
            try:
                # Senza il fix questo await pende per sempre → TimeoutError.
                await asyncio.wait_for(vl._stream_and_speak(_ctx()), timeout=5.0)
            finally:
                stealing = False
                thief.cancel()
                try:
                    await thief
                except asyncio.CancelledError:
                    pass
            assert vl._active_audio_queue is None   # turno chiuso pulito

    async def test_mute_a_meta_turno_non_blocca(self):
        """Mute reale a metà turno: set _tts_enabled False + drain, il turno
        completa e lo stato non resta 'speaking'."""
        vl = _voice_loop([
            "Prima frase abbastanza lunga per il TTS. ",
            "Seconda frase ugualmente lunga e valida. ",
        ])

        async def _mute_soon():
            await asyncio.sleep(0.02)
            vl._tts_enabled = False
            q = vl._active_audio_queue
            if q is not None:
                try:
                    while True:
                        q.get_nowait()
                except asyncio.QueueEmpty:
                    pass

        muter = asyncio.create_task(_mute_soon())
        # Senza il fix il turno non completerebbe (deadlock sul player):
        # il ripristino dello stato a fine turno spetta a _process_turn.
        await asyncio.wait_for(vl._stream_and_speak(_ctx()), timeout=5.0)
        await muter
        assert vl._active_audio_queue is None   # turno chiuso pulito
