"""
modules/stt/base_stt.py
WhisperSTT — Speech-to-Text con faster-whisper + Silero VAD.

Flussi supportati:
    1. transcribe(audio)              → STTResult   (blocco singolo, no VAD)
    2. transcribe_with_vad(audio)     → STTResult   (filtra silenzio con VAD)
    3. stream_mic(callback)           → None        (acquisisce dal mic in loop,
                                                     chiama callback su ogni frase)

Formato audio atteso: PCM int16 LE, mono, 16 000 Hz (SAMPLE_RATE).

Uso rapido:
    async with WhisperSTT() as stt:
        result = await stt.transcribe_with_vad(raw_pcm_bytes)
        print(result.text)
"""

from __future__ import annotations

import asyncio
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

import numpy as np

from config.settings import settings
from core.logger import logger

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16_000          # Hz — fisso per Whisper e Silero
CHANNELS:    int = 1               # mono
DTYPE:       str = "int16"         # formato raw bytes in entrata
_INT16_MAX: float = 32_768.0


# ---------------------------------------------------------------------------
# Dataclasses pubblici
# ---------------------------------------------------------------------------

@dataclass
class STTSegment:
    """Un segmento trascritto da Whisper."""
    start:       float          # secondi dall'inizio
    end:         float
    text:        str
    language:    str
    confidence:  float = 1.0    # avg log-prob normalizzato in [0, 1]
    no_speech:   float = 0.0    # no_speech_prob da Whisper


@dataclass
class STTResult:
    """Risultato completo della trascrizione."""
    text:          str
    language:      str
    segments:      list[STTSegment] = field(default_factory=list)
    duration_s:    float = 0.0
    inference_ms:  float = 0.0
    vad_kept_s:    float = 0.0    # secondi di audio sopravvissuti al VAD
    word_count:    int   = 0

    def is_empty(self) -> bool:
        return not self.text.strip()

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "text_len":     len(self.text),
            "language":     self.language,
            "segments":     len(self.segments),
            "duration_s":   round(self.duration_s, 2),
            "inference_ms": round(self.inference_ms, 1),
            "vad_kept_s":   round(self.vad_kept_s, 2),
        }


# ---------------------------------------------------------------------------
# Helper audio
# ---------------------------------------------------------------------------

def _bytes_to_float32(raw: bytes) -> np.ndarray:
    """PCM int16 bytes → numpy float32 normalizzato in [-1, 1]."""
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return samples / _INT16_MAX


def _float32_to_bytes(arr: np.ndarray) -> bytes:
    """numpy float32 → PCM int16 bytes."""
    clipped = np.clip(arr * _INT16_MAX, -_INT16_MAX, _INT16_MAX - 1)
    return clipped.astype(np.int16).tobytes()


def _duration_s(raw: bytes) -> float:
    n_samples = len(raw) // 2  # int16 = 2 byte
    return n_samples / SAMPLE_RATE


# ---------------------------------------------------------------------------
# Client principale
# ---------------------------------------------------------------------------

class WhisperSTT:
    """
    Client STT asincrono basato su faster-whisper + Silero VAD.

    Il modello Whisper viene caricato in un thread pool all'init (pesante),
    il VAD è leggero e viene ricaricato ad ogni istanza.

    Args:
        model_name:    Override del modello (default da settings.stt.model).
        device:        "cpu" | "cuda" | "auto".
        compute_type:  "int8" | "float16" | "float32".
        language:      Codice lingua ISO-639-1 ("it", "en", …). None = auto-detect.
    """

    def __init__(
        self,
        model_name:   Optional[str] = None,
        device:       Optional[str] = None,
        compute_type: Optional[str] = None,
        language:     Optional[str] = None,
    ) -> None:
        self._model_name   = model_name   or settings.stt.model
        self._device       = device       or settings.stt.device
        self._compute_type = compute_type or settings.stt.compute_type
        self._language     = language     or settings.stt.language

        self._whisper: Any   = None   # faster_whisper.WhisperModel
        self._vad:     Any   = None   # silero VAD pipeline
        self._vad_utils: Any = None   # get_speech_timestamps, collect_chunks

        self._loop: asyncio.AbstractEventLoop | None = None

    # -- context manager -------------------------------------------------------

    async def __aenter__(self) -> "WhisperSTT":
        await self._load_models()
        return self

    async def __aexit__(self, *_: Any) -> None:
        # faster-whisper non ha cleanup esplicito; azzeriamo i riferimenti
        self._whisper = None
        self._vad     = None

    # -- caricamento modelli ---------------------------------------------------

    async def _load_models(self) -> None:
        """Carica Whisper e VAD in un executor per non bloccare il loop."""
        self._loop = asyncio.get_running_loop()
        logger.info(
            "stt | caricamento Whisper '{}' [device={}, compute={}]",
            self._model_name, self._device, self._compute_type,
        )
        t0 = time.perf_counter()
        await self._loop.run_in_executor(None, self._load_sync)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("stt | modelli pronti ({:.0f} ms)", elapsed)

    def _load_sync(self) -> None:
        """Eseguito in thread — importa e carica i modelli pesanti."""
        from faster_whisper import WhisperModel
        import torch

        # Whisper
        self._whisper = WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
        )

        # Silero VAD — usa torch.hub, prima volta scarica ~2 MB
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._vad       = model
        self._vad_utils = utils  # (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)

    # -- trascrizione base (senza VAD) -----------------------------------------

    async def transcribe(
        self,
        audio: bytes,
        *,
        initial_prompt: Optional[str] = None,
        beam_size:      int = 5,
        temperature:    float = 0.0,
    ) -> STTResult:
        """
        Trascrive un blocco PCM int16 senza pre-filtraggio VAD.
        Utile quando la sorgente è già pulita (es. file audio) o il VAD
        è già stato applicato a monte.
        """
        if self._whisper is None:
            raise RuntimeError("WhisperSTT non inizializzato — usa 'async with WhisperSTT()'")

        loop      = asyncio.get_running_loop()
        audio_f32 = _bytes_to_float32(audio)
        duration  = _duration_s(audio)

        t0 = time.perf_counter()
        segments_raw, info = await loop.run_in_executor(
            None,
            lambda: self._transcribe_sync(audio_f32, initial_prompt, beam_size, temperature),
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        segments  = self._parse_segments(segments_raw, info.language)
        full_text = " ".join(s.text.strip() for s in segments if s.no_speech < 0.6)

        result = STTResult(
            text=full_text.strip(),
            language=info.language,
            segments=segments,
            duration_s=duration,
            inference_ms=inference_ms,
            vad_kept_s=duration,
            word_count=len(full_text.split()),
        )
        logger.debug("stt.transcribe | {}", result.to_log_dict())
        return result

    def _transcribe_sync(
        self,
        audio_f32:      np.ndarray,
        initial_prompt: Optional[str],
        beam_size:      int,
        temperature:    float,
    ):
        """Eseguito in thread."""
        return self._whisper.transcribe(
            audio_f32,
            language=self._language or None,
            initial_prompt=initial_prompt,
            beam_size=beam_size,
            temperature=temperature,
            vad_filter=False,   # gestiamo il VAD esternamente
            word_timestamps=False,
        )

    # -- trascrizione con VAD --------------------------------------------------

    async def transcribe_with_vad(
        self,
        audio: bytes,
        *,
        vad_threshold:      Optional[float] = None,
        silence_duration_s: Optional[float] = None,
        initial_prompt:     Optional[str]   = None,
        beam_size:          int = 5,
        temperature:        float = 0.0,
    ) -> STTResult:
        """
        Pipeline completa: VAD → Whisper.

        1. Silero VAD individua i segmenti vocali.
        2. I chunk vengono riconcatenati ed inviati a Whisper.
        3. Se nessun segmento vocale → STTResult vuoto immediato.

        Args:
            vad_threshold:      Soglia VAD [0, 1] (default da settings).
            silence_duration_s: Silenzio minimo per chiudere un segmento.
        """
        if self._whisper is None or self._vad is None:
            raise RuntimeError("WhisperSTT non inizializzato — usa 'async with WhisperSTT()'")

        threshold = vad_threshold      or settings.stt.vad_threshold
        silence_s = silence_duration_s or settings.stt.vad_silence_duration

        loop      = asyncio.get_running_loop()
        audio_f32 = _bytes_to_float32(audio)
        duration  = _duration_s(audio)

        # Fase 1 — VAD in executor
        filtered_f32 = await loop.run_in_executor(
            None,
            lambda: self._apply_vad(audio_f32, threshold, silence_s),
        )

        vad_kept_s = len(filtered_f32) / SAMPLE_RATE
        logger.debug(
            "stt.vad | {:.1f}s → {:.1f}s mantenuti ({:.0f}%)",
            duration,
            vad_kept_s,
            (vad_kept_s / duration * 100) if duration > 0 else 0,
        )

        if vad_kept_s < 0.2:
            logger.debug("stt | nessun parlato rilevato dal VAD")
            return STTResult(
                text="", language=self._language or "it",
                duration_s=duration, vad_kept_s=vad_kept_s,
            )

        # Fase 2 — Whisper sull'audio filtrato
        t0 = time.perf_counter()
        segments_raw, info = await loop.run_in_executor(
            None,
            lambda: self._transcribe_sync(filtered_f32, initial_prompt, beam_size, temperature),
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        segments  = self._parse_segments(segments_raw, info.language)
        full_text = " ".join(s.text.strip() for s in segments if s.no_speech < 0.6)

        result = STTResult(
            text=full_text.strip(),
            language=info.language,
            segments=segments,
            duration_s=duration,
            inference_ms=inference_ms,
            vad_kept_s=vad_kept_s,
            word_count=len(full_text.split()),
        )
        logger.info("stt | '{}' ({:.0f} ms)", result.text[:80], inference_ms)
        return result

    def _apply_vad(
        self,
        audio_f32: np.ndarray,
        threshold: float,
        silence_s: float,
    ) -> np.ndarray:
        """
        Applica Silero VAD e restituisce solo i campioni con parlato.
        Eseguito in thread.
        """
        import torch

        get_speech_timestamps, _, _, _, collect_chunks = self._vad_utils
        tensor = torch.from_numpy(audio_f32)

        speech_timestamps = get_speech_timestamps(
            tensor,
            self._vad,
            threshold=threshold,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=int(silence_s * 1000),
            return_seconds=False,
        )

        if not speech_timestamps:
            return np.array([], dtype=np.float32)

        chunks = collect_chunks(speech_timestamps, tensor)
        return chunks.numpy()

    # -- helper parsing segmenti -----------------------------------------------

    @staticmethod
    def _parse_segments(segments_raw, language: str) -> list[STTSegment]:
        """Converte i segmenti faster-whisper in STTSegment."""
        result = []
        for seg in segments_raw:
            avg_logprob    = getattr(seg, "avg_logprob",    0.0) or 0.0
            no_speech_prob = getattr(seg, "no_speech_prob", 0.0) or 0.0
            # normalizza avg_logprob (tipicamente in [-1, 0]) → [0, 1]
            confidence = float(np.clip(np.exp(avg_logprob), 0.0, 1.0))
            result.append(STTSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                language=language,
                confidence=confidence,
                no_speech=no_speech_prob,
            ))
        return result

    # -- mic streaming ---------------------------------------------------------

    async def stream_mic(
        self,
        callback: Callable[[STTResult], Coroutine[Any, Any, None]],
        *,
        chunk_ms:      int   = 30,    # dimensione chunk per il VAD iterator
        max_silence_s: float = 1.5,   # silenzio per chiudere la frase
        min_speech_s:  float = 0.3,   # durata minima per mandare a Whisper
        device_index:  Optional[int]          = None,
        stop_event:    Optional[asyncio.Event] = None,
    ) -> None:
        """
        Acquisisce dal microfono in loop, rileva frasi con VAD iterator,
        trascrive e chiama `callback(result)` per ogni frase completa.

        Il loop termina quando `stop_event` viene settato.

        Args:
            callback:       Coroutine chiamata con ogni STTResult non vuoto.
            chunk_ms:       Durata chunk audio per l'iteratore VAD (ms).
            max_silence_s:  Silenzio dopo cui la frase viene inviata a Whisper.
            min_speech_s:   Durata minima del parlato per evitare false attivazioni.
            device_index:   Indice dispositivo sounddevice (None = default).
            stop_event:     asyncio.Event per arrestare il loop dall'esterno.
        """
        import sounddevice as sd
        import torch

        if self._vad is None:
            raise RuntimeError("WhisperSTT non inizializzato")

        _, _, _, VADIterator, _ = self._vad_utils
        stop_event = stop_event or asyncio.Event()
        chunk_size = int(SAMPLE_RATE * chunk_ms / 1000)
        silence_ms = int(max_silence_s * 1000)

        vad_iter = VADIterator(
            self._vad,
            threshold=settings.stt.vad_threshold,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=silence_ms,
        )

        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        def _mic_callback(indata, frames, time_info, status):
            if status:
                logger.warning("stt.mic | status={}", status)
            audio_queue.put_nowait(indata.tobytes())

        speech_buffer: list[np.ndarray] = []

        logger.info("stt.mic | avvio acquisizione (chunk={}ms, silenzio={}ms)", chunk_ms, silence_ms)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=chunk_size,
            device=device_index,
            callback=_mic_callback,
        ):
            while not stop_event.is_set():
                try:
                    raw_chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                f32_chunk = _bytes_to_float32(raw_chunk)
                tensor    = torch.from_numpy(f32_chunk)

                speech_dict = vad_iter(tensor, return_seconds=False)

                if speech_dict:
                    if "start" in speech_dict:
                        speech_buffer.clear()
                    speech_buffer.append(f32_chunk)

                    if "end" in speech_dict:
                        duration = len(speech_buffer) * chunk_ms / 1000
                        if duration >= min_speech_s:
                            combined = np.concatenate(speech_buffer)
                            raw      = _float32_to_bytes(combined)

                            async def _do_transcribe():
                                result = await self.transcribe(raw, beam_size=3, temperature=0.0)
                                if not result.is_empty():
                                    await callback(result)

                            asyncio.ensure_future(_do_transcribe())
                        speech_buffer.clear()
                        vad_iter.reset_states()
                else:
                    if speech_buffer:
                        speech_buffer.append(f32_chunk)

        logger.info("stt.mic | acquisizione terminata")

    # -- convenienza: trascrivi file -------------------------------------------

    async def transcribe_file(self, path: Path | str) -> STTResult:
        """
        Trascrive un file audio qualsiasi.
        Converte internamente in PCM float32 tramite soundfile.
        """
        import soundfile as sf

        path = Path(path)
        loop = asyncio.get_running_loop()

        def _read():
            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)          # stereo → mono
            if sr != SAMPLE_RATE:
                # resample naïve — per produzione usare librosa o soxr
                import scipy.signal as sps
                samples = int(len(data) * SAMPLE_RATE / sr)
                data    = sps.resample(data, samples)
            return data

        audio_f32 = await loop.run_in_executor(None, _read)
        return await self.transcribe_with_vad(_float32_to_bytes(audio_f32))
