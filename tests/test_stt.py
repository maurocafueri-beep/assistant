"""
tests/test_stt.py
Test suite per modules/stt.

I test puri (helper audio, dataclass) girano senza modelli.
I test @pytest.mark.slow richiedono faster-whisper + Silero VAD caricati:
saltali in CI con SKIP_SLOW=1.

Esecuzione:
    make test
    SKIP_SLOW=1 venv-runtime/bin/pytest tests/test_stt.py -v
"""

from __future__ import annotations

import os
import struct

import numpy as np
import pytest

from modules.stt.base_stt import (
    _bytes_to_float32,
    _float32_to_bytes,
    _duration_s,
    STTResult,
    STTSegment,
    SAMPLE_RATE,
    WhisperSTT,
)

SKIP_SLOW = os.getenv("SKIP_SLOW", "0") == "1"
slow = pytest.mark.skipif(SKIP_SLOW, reason="SKIP_SLOW=1")


class TestAudioHelpers:
    def test_bytes_to_float32_zero(self):
        raw = struct.pack("<4h", 0, 0, 0, 0)
        arr = _bytes_to_float32(raw)
        assert arr.shape == (4,)
        assert np.allclose(arr, 0.0)

    def test_bytes_to_float32_max(self):
        raw = struct.pack("<1h", 32767)
        arr = _bytes_to_float32(raw)
        assert pytest.approx(arr[0], abs=1e-4) == 1.0

    def test_roundtrip(self):
        rng = np.random.default_rng(42)
        original = rng.uniform(-0.9, 0.9, 1600).astype(np.float32)
        roundtrip = _bytes_to_float32(_float32_to_bytes(original))
        assert np.allclose(original, roundtrip, atol=1.0 / 32768)

    def test_duration_s(self, sample_audio):
        d = _duration_s(sample_audio)
        assert pytest.approx(d, abs=0.01) == 1.0

    def test_duration_empty(self):
        assert _duration_s(b"") == 0.0


class TestSTTResult:
    def _make_result(self, text: str) -> STTResult:
        return STTResult(
            text=text,
            language="it",
            segments=[STTSegment(0.0, 1.0, text, "it")],
            duration_s=1.0,
            inference_ms=100.0,
        )

    def test_is_empty_blank(self):
        assert self._make_result("   ").is_empty()

    def test_is_empty_nonempty(self):
        assert not self._make_result("Ciao").is_empty()

    def test_to_log_dict_keys(self):
        d = self._make_result("Test").to_log_dict()
        assert set(d.keys()) == {
            "text_len", "language", "segments",
            "duration_s", "inference_ms", "vad_kept_s",
        }


@slow
@pytest.mark.asyncio
class TestWhisperSTTLoad:
    async def test_context_manager(self):
        async with WhisperSTT() as stt:
            assert stt._whisper is not None
            assert stt._vad is not None

    async def test_exit_clears_models(self):
        stt = WhisperSTT()
        async with stt:
            pass
        assert stt._whisper is None


@slow
@pytest.mark.asyncio
class TestTranscribeSilence:
    async def test_transcribe_silence_no_crash(self, sample_audio):
        async with WhisperSTT() as stt:
            result = await stt.transcribe(sample_audio)
        assert isinstance(result, STTResult)

    async def test_transcribe_with_vad_silence(self, sample_audio):
        async with WhisperSTT() as stt:
            result = await stt.transcribe_with_vad(sample_audio)
        assert result.is_empty()
        assert result.vad_kept_s < 0.2

    async def test_inference_ms_positive(self, sample_audio):
        async with WhisperSTT() as stt:
            result = await stt.transcribe(sample_audio)
        assert result.inference_ms >= 0


@slow
@pytest.mark.asyncio
class TestTranscribeTone:
    @pytest.fixture
    def tone_audio(self) -> bytes:
        t = np.linspace(0, 2.0, 2 * SAMPLE_RATE, endpoint=False)
        arr = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        return arr.tobytes()

    async def test_vad_keeps_tone(self, tone_audio):
        async with WhisperSTT() as stt:
            result = await stt.transcribe_with_vad(tone_audio)
        assert result.duration_s == pytest.approx(2.0, abs=0.05)

    async def test_result_has_language(self, tone_audio):
        async with WhisperSTT() as stt:
            result = await stt.transcribe_with_vad(tone_audio)
        assert len(result.language) >= 2
