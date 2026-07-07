"""
tests/test_tts.py
Test suite per modules/tts (Qwen3TTS con voice cloning).

I test puri (helper audio, dataclass, sentence splitting) girano senza server.
I test che usano Qwen3TTS mockano il server HTTP — zero dipendenze esterne.
I test @pytest.mark.slow richiedono il server TTS reale attivo:
saltali con SKIP_SLOW=1.

Esecuzione:
    make test
    SKIP_SLOW=1 venv-runtime/bin/pytest tests/test_tts.py -v
"""

from __future__ import annotations

import io
import os
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from modules.tts.base_tts import (
    OUTPUT_SAMPLE_RATE,
    Qwen3TTS,
    TTSChunk,
    TTSResult,
    _duration_from_wav,
    _resolve_language,
    _split_sentences,
    _wav_bytes_to_float32,
)

SKIP_SLOW = os.getenv("SKIP_SLOW", "0") == "1"
slow = pytest.mark.skipif(SKIP_SLOW, reason="SKIP_SLOW=1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_s: float = 1.0, sr: int = OUTPUT_SAMPLE_RATE) -> bytes:
    """Genera WAV bytes PCM int16 silenziosi di durata specificata."""
    n_samples = int(sr * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.zeros(n_samples, dtype=np.int16).tobytes())
    return buf.getvalue()


def _make_mock_http(wav_bytes: bytes | None = None) -> MagicMock:
    """Crea un AsyncClient mockato che restituisce WAV bytes."""
    audio = wav_bytes or _make_wav_bytes(1.0)
    resp  = MagicMock()
    resp.status_code = 200
    resp.content     = audio
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "status":   "ok",
        "profile":  "mercoledì",
        "profiles": ["mercoledì", "squib"],
        "model":    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    }

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__  = AsyncMock(return_value=False)
    mock_client.post       = AsyncMock(return_value=resp)
    mock_client.get        = AsyncMock(return_value=resp)
    mock_client.aclose     = AsyncMock()
    return mock_client


@pytest.fixture
def mock_tts() -> Qwen3TTS:
    """Qwen3TTS con server mockato — non avvia alcun processo."""
    tts = Qwen3TTS.__new__(Qwen3TTS)
    tts._profile           = "mercoledì"
    tts._language          = "it"
    tts._cuda_device_index = 1
    tts._port              = 8765
    tts._server_timeout_s  = 240.0
    tts._base_url          = "http://127.0.0.1:8765"
    tts._process           = None
    tts._http              = _make_mock_http()
    tts._owns_server       = False
    return tts


# ---------------------------------------------------------------------------
# Helper audio
# ---------------------------------------------------------------------------

class TestAudioHelpers:
    def test_wav_bytes_to_float32_silent(self):
        wav = _make_wav_bytes(0.1)
        arr = _wav_bytes_to_float32(wav)
        assert arr.dtype == np.float32
        assert np.allclose(arr, 0.0)

    def test_wav_bytes_to_float32_shape(self):
        wav = _make_wav_bytes(1.0)
        arr = _wav_bytes_to_float32(wav)
        assert len(arr) == OUTPUT_SAMPLE_RATE

    def test_duration_from_wav(self):
        wav = _make_wav_bytes(2.0)
        assert _duration_from_wav(wav) == pytest.approx(2.0, abs=0.01)

    def test_duration_from_wav_empty(self):
        assert _duration_from_wav(b"") == 0.0

    def test_duration_from_wav_invalid(self):
        assert _duration_from_wav(b"\x00\x01\x02") == 0.0


# ---------------------------------------------------------------------------
# Split sentences
# ---------------------------------------------------------------------------

class TestSplitSentences:
    def test_single_sentence(self):
        assert _split_sentences("Ciao mondo.") == ["Ciao mondo."]

    def test_multiple_sentences(self):
        s = _split_sentences("Prima frase. Seconda frase. Terza frase.")
        assert len(s) >= 2

    def test_empty_string(self):
        assert _split_sentences("") == [""]

    def test_short_fragments_merged(self):
        s = _split_sentences("Sì. Certo che posso aiutarti con questo problema.")
        assert all(len(sent.split()) >= 2 for sent in s)

    def test_exclamation_split(self):
        s = _split_sentences("Ottimo! Procedo subito con la sintesi.")
        assert len(s) >= 1

    def test_no_trailing_empty(self):
        s = _split_sentences("Frase uno. Frase due.")
        assert all(sent.strip() for sent in s)


# ---------------------------------------------------------------------------
# Resolve language
# ---------------------------------------------------------------------------

class TestResolveLanguage:
    @pytest.mark.parametrize("code,expected", [
        ("it",    "italian"),
        ("en",    "english"),
        ("en-us", "english"),
        ("fr",    "french"),
        ("de",    "german"),
        ("es",    "spanish"),
        ("IT",    "italian"),   # case-insensitive
        ("EN",    "english"),
        ("xx",    "italian"),   # fallback → italiano
    ])
    def test_mapping(self, code, expected):
        assert _resolve_language(code) == expected


# ---------------------------------------------------------------------------
# TTSChunk / TTSResult
# ---------------------------------------------------------------------------

class TestTTSChunk:
    def test_fields(self):
        chunk = TTSChunk(
            text="Ciao", audio_bytes=b"\x00\x01",
            sample_rate=24000, duration_s=0.5,
            inference_ms=50.0, index=0, is_last=True,
        )
        assert chunk.text        == "Ciao"
        assert chunk.is_last     is True
        assert chunk.sample_rate == 24000

    def test_defaults(self):
        chunk = TTSChunk("ok", b"", 24000, 0.0, 0.0)
        assert chunk.index   == 0
        assert chunk.is_last is False


class TestTTSResult:
    def _make(self, text: str = "Testo", audio: bytes | None = None) -> TTSResult:
        return TTSResult(
            text=text,
            audio_bytes=audio if audio is not None else _make_wav_bytes(1.0),
            sample_rate=OUTPUT_SAMPLE_RATE,
            duration_s=1.0,
            inference_ms=120.0,
            voice="mercoledì",
            language="italian",
        )

    def test_is_empty_false(self):
        assert not self._make().is_empty()

    def test_is_empty_true(self):
        assert self._make(audio=b"").is_empty()

    def test_to_log_dict_keys(self):
        assert self._make().to_log_dict().keys() == {
            "text_len", "duration_s", "inference_ms", "voice", "language", "chunks"
        }

    def test_to_log_dict_values(self):
        d = self._make("Ciao").to_log_dict()
        assert d["text_len"] == 4
        assert d["voice"]    == "mercoledì"
        assert d["language"] == "italian"


# ---------------------------------------------------------------------------
# Qwen3TTS — init e properties
# ---------------------------------------------------------------------------

class TestQwen3TTSInit:
    def test_properties(self, mock_tts):
        assert mock_tts.profile     == "mercoledì"
        assert mock_tts.language    == "it"
        assert mock_tts.sample_rate == OUTPUT_SAMPLE_RATE

    async def test_not_initialized_raises_synthesize(self):
        tts = Qwen3TTS.__new__(Qwen3TTS)
        tts._http = None
        with pytest.raises(RuntimeError, match="non inizializzato"):
            await tts.synthesize("ciao")

    async def test_not_initialized_raises_stream(self):
        tts = Qwen3TTS.__new__(Qwen3TTS)
        tts._http = None
        async def dummy(c): pass
        with pytest.raises(RuntimeError, match="non inizializzato"):
            await tts.stream_sentences("x", dummy)

    def test_constructor_defaults_from_settings(self):
        from config.settings import settings
        with patch.object(Qwen3TTS, "_ensure_server", AsyncMock()), \
             patch("httpx.AsyncClient"):
            tts = Qwen3TTS()
        assert tts._profile           == settings.tts.voice
        assert tts._cuda_device_index == settings.tts.cuda_device_index

    def test_constructor_overrides(self):
        with patch.object(Qwen3TTS, "_ensure_server", AsyncMock()), \
             patch("httpx.AsyncClient"):
            tts = Qwen3TTS(profile="squib", language="en", cuda_device_index=0)
        assert tts._profile           == "squib"
        assert tts._language          == "en"
        assert tts._cuda_device_index == 0


# ---------------------------------------------------------------------------
# Qwen3TTS.synthesize
# ---------------------------------------------------------------------------

class TestQwen3TTSSynthesize:
    async def test_returns_tts_result(self, mock_tts, sample_text):
        result = await mock_tts.synthesize(sample_text)
        assert isinstance(result, TTSResult)

    async def test_non_empty_audio(self, mock_tts, sample_text):
        result = await mock_tts.synthesize(sample_text)
        assert not result.is_empty()

    async def test_empty_text_returns_empty_result(self, mock_tts):
        result = await mock_tts.synthesize("   ")
        assert result.is_empty()
        mock_tts._http.post.assert_not_called()

    async def test_sample_rate_correct(self, mock_tts, sample_text):
        result = await mock_tts.synthesize(sample_text)
        assert result.sample_rate == OUTPUT_SAMPLE_RATE

    async def test_voice_in_result(self, mock_tts, sample_text):
        result = await mock_tts.synthesize(sample_text)
        assert result.voice == "mercoledì"

    async def test_language_resolved(self, mock_tts, sample_text):
        result = await mock_tts.synthesize(sample_text, language="en")
        call_kwargs = mock_tts._http.post.call_args
        assert "english" in str(call_kwargs)

    async def test_post_called_once(self, mock_tts, sample_text):
        await mock_tts.synthesize(sample_text)
        mock_tts._http.post.assert_called_once()

    async def test_inference_ms_positive(self, mock_tts, sample_text):
        result = await mock_tts.synthesize(sample_text)
        assert result.inference_ms >= 0.0

    async def test_duration_from_wav(self, mock_tts):
        wav_2s = _make_wav_bytes(2.0)
        mock_tts._http = _make_mock_http(wav_2s)
        result = await mock_tts.synthesize("Due secondi.")
        assert result.duration_s == pytest.approx(2.0, abs=0.05)


# ---------------------------------------------------------------------------
# Qwen3TTS.stream_sentences (callback)
# ---------------------------------------------------------------------------

class TestQwen3TTSStreamSentences:
    async def test_callback_called_for_each_sentence(self, mock_tts):
        text   = "Prima frase. Seconda frase. Terza frase lunga abbastanza."
        chunks: list[TTSChunk] = []

        async def collect(c: TTSChunk) -> None:
            chunks.append(c)

        await mock_tts.stream_sentences(text, collect)
        assert len(chunks) >= 1
        assert all(isinstance(c, TTSChunk) for c in chunks)

    async def test_last_chunk_flagged(self, mock_tts, sample_text):
        chunks: list[TTSChunk] = []
        async def collect(c): chunks.append(c)
        await mock_tts.stream_sentences(sample_text, collect)
        assert chunks[-1].is_last is True

    async def test_index_sequential(self, mock_tts):
        text   = "Frase uno. Frase due lunga. Frase tre ancora lunga."
        chunks: list[TTSChunk] = []
        async def collect(c): chunks.append(c)
        await mock_tts.stream_sentences(text, collect)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    async def test_not_initialized_raises(self):
        tts = Qwen3TTS.__new__(Qwen3TTS)
        tts._http = None
        with pytest.raises(RuntimeError):
            await tts.stream_sentences("x", lambda c: None)

    async def test_post_called_per_sentence(self, mock_tts):
        text = "Prima frase lunga. Seconda frase lunga. Terza frase lunga."
        async def noop(c): pass
        await mock_tts.stream_sentences(text, noop)
        call_count = mock_tts._http.post.call_count
        assert call_count >= 2


# ---------------------------------------------------------------------------
# Qwen3TTS.stream_sentences_gen (async generator)
# ---------------------------------------------------------------------------

class TestQwen3TTSStreamGen:
    async def test_yields_chunks(self, mock_tts):
        text   = "Prima frase. Seconda frase lunga abbastanza."
        chunks = [c async for c in mock_tts.stream_sentences_gen(text)]
        assert len(chunks) >= 1
        assert all(isinstance(c, TTSChunk) for c in chunks)

    async def test_last_chunk_is_last(self, mock_tts, sample_text):
        chunks = [c async for c in mock_tts.stream_sentences_gen(sample_text)]
        assert chunks[-1].is_last is True

    async def test_matches_callback_count(self, mock_tts):
        text = "Frase uno. Frase due lunga. Frase tre ancora lunga."
        cb_chunks: list[TTSChunk] = []
        async def collect(c): cb_chunks.append(c)
        await mock_tts.stream_sentences(text, collect)

        mock_tts._http.post.reset_mock()
        gen_chunks = [c async for c in mock_tts.stream_sentences_gen(text)]
        assert len(gen_chunks) == len(cb_chunks)

    async def test_not_initialized_raises(self):
        tts = Qwen3TTS.__new__(Qwen3TTS)
        tts._http = None
        with pytest.raises(RuntimeError):
            async for _ in tts.stream_sentences_gen("x"):
                pass


# ---------------------------------------------------------------------------
# Qwen3TTS.synthesize_to_file
# ---------------------------------------------------------------------------

class TestQwen3TTSSynthesizeToFile:
    async def test_creates_wav_file(self, mock_tts, sample_text, tmp_path):
        out  = tmp_path / "output.wav"
        path = await mock_tts.synthesize_to_file(sample_text, out)
        assert path == out
        assert out.exists()

    async def test_wav_is_valid(self, mock_tts, sample_text, tmp_path):
        out = tmp_path / "test.wav"
        await mock_tts.synthesize_to_file(sample_text, out)
        with wave.open(str(out), "rb") as wf:
            assert wf.getnchannels()  == 1
            assert wf.getsampwidth()  == 2
            assert wf.getframerate()  == OUTPUT_SAMPLE_RATE
            assert wf.getnframes()    >  0

    async def test_creates_parent_dirs(self, mock_tts, sample_text, tmp_path):
        out = tmp_path / "subdir" / "nested" / "audio.wav"
        await mock_tts.synthesize_to_file(sample_text, out)
        assert out.exists()

    async def test_returns_path_object(self, mock_tts, sample_text, tmp_path):
        out      = tmp_path / "out.wav"
        returned = await mock_tts.synthesize_to_file(sample_text, str(out))
        assert isinstance(returned, Path)


# ---------------------------------------------------------------------------
# Qwen3TTS.play / say
# ---------------------------------------------------------------------------

class TestQwen3TTSPlay:
    # play() avvia la riproduzione (non bloccante) via _start_playback e poi
    # attende in modo asincrono finché lo stream sounddevice è attivo. Nei test
    # mockiamo l'avvio e forziamo lo stream a "non attivo" così l'attesa termina
    # subito senza toccare un device audio reale.
    async def test_play_calls_sounddevice(self, mock_tts):
        with patch("modules.tts.base_tts._start_playback") as mock_play, \
             patch("sounddevice.get_stream") as mock_get_stream:
            mock_get_stream.return_value.active = False
            await mock_tts.play(_make_wav_bytes(0.5))
        mock_play.assert_called_once()

    async def test_play_empty_audio_no_call(self, mock_tts):
        with patch("modules.tts.base_tts._start_playback") as mock_play:
            await mock_tts.play(b"")
        mock_play.assert_not_called()

    async def test_play_handles_closed_stream(self, mock_tts):
        # Regressione mute: durante il mute, interrupt_tts → stop_playback chiama
        # sd.stop() che CHIUDE lo stream; il successivo accesso a .active solleva.
        # play() deve uscire pulito (riproduzione finita), NON propagare l'errore.
        with patch("modules.tts.base_tts._start_playback"), \
             patch("sounddevice.get_stream", side_effect=RuntimeError("closed")):
            await mock_tts.play(_make_wav_bytes(0.5))  # non deve sollevare

    async def test_say_returns_result(self, mock_tts, sample_text):
        with patch("modules.tts.base_tts._start_playback"), \
             patch("sounddevice.get_stream") as mock_get_stream:
            mock_get_stream.return_value.active = False
            result = await mock_tts.say(sample_text)
        assert isinstance(result, TTSResult)

    async def test_say_calls_both_synth_and_play(self, mock_tts, sample_text):
        with patch("modules.tts.base_tts._start_playback") as mock_play, \
             patch("sounddevice.get_stream") as mock_get_stream:
            mock_get_stream.return_value.active = False
            await mock_tts.say(sample_text)
        mock_tts._http.post.assert_called_once()
        mock_play.assert_called_once()


# ---------------------------------------------------------------------------
# Qwen3TTS.server_info / available_profiles
# ---------------------------------------------------------------------------

class TestQwen3TTSServerInfo:
    async def test_server_info_keys(self, mock_tts):
        info = await mock_tts.server_info()
        assert "status"   in info
        assert "profile"  in info
        assert "profiles" in info

    async def test_available_profiles(self, mock_tts):
        profiles = await mock_tts.available_profiles()
        assert isinstance(profiles, list)
        assert len(profiles) >= 1

    async def test_not_initialized_raises(self):
        tts = Qwen3TTS.__new__(Qwen3TTS)
        tts._http = None
        with pytest.raises(RuntimeError):
            await tts.server_info()


# ---------------------------------------------------------------------------
# Test slow — richiedono server reale
# ---------------------------------------------------------------------------

@slow
@pytest.mark.real_tts
class TestQwen3TTSRealServer:
    async def test_context_manager(self):
        async with Qwen3TTS() as tts:
            info = await tts.server_info()
            assert info["status"] == "ok"

    async def test_synthesize_non_empty(self, sample_text):
        async with Qwen3TTS() as tts:
            result = await tts.synthesize(sample_text)
        assert not result.is_empty()
        assert result.duration_s > 0.0

    async def test_synthesize_italian(self):
        async with Qwen3TTS() as tts:
            result = await tts.synthesize(
                "Buongiorno, piacere di conoscerti.", language="it"
            )
        assert result.language == "italian"
        assert not result.is_empty()

    async def test_stream_sentences_real(self, sample_text):
        chunks: list[TTSChunk] = []
        async with Qwen3TTS() as tts:
            async def collect(c: TTSChunk) -> None:
                chunks.append(c)
            await tts.stream_sentences(sample_text, collect)
        assert len(chunks) >= 1
        assert chunks[-1].is_last is True

    async def test_switch_profile(self):
        async with Qwen3TTS(profile="mercoledì") as tts:
            profiles = await tts.available_profiles()
            if len(profiles) > 1:
                alt = next(p for p in profiles if p != "mercoledì")
                import httpx
                async with httpx.AsyncClient() as c:
                    r = await c.post(f"http://127.0.0.1:8765/switch/{alt}")
                assert r.json()["profile"] == alt

    async def test_synthesize_to_file_real(self, sample_text, tmp_path):
        out = tmp_path / "real_output.wav"
        async with Qwen3TTS() as tts:
            path = await tts.synthesize_to_file(sample_text, out)
        assert path.exists()
        assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Warmup (prima inferenza usa-e-getta)
# ---------------------------------------------------------------------------

class TestWarmup:
    async def test_warmup_calls_synthesize(self, mock_tts):
        ok = await mock_tts.warmup()
        assert ok is True
        # warmup() passa per synthesize() → POST /synthesize
        mock_tts._http.post.assert_awaited()

    async def test_warmup_not_initialized_is_noop(self):
        tts = Qwen3TTS.__new__(Qwen3TTS)
        tts._http = None
        ok = await tts.warmup()
        assert ok is False

    async def test_warmup_failure_is_soft(self, mock_tts):
        mock_tts._http.post = AsyncMock(side_effect=RuntimeError("server giù"))
        ok = await mock_tts.warmup()  # non deve sollevare
        assert ok is False
