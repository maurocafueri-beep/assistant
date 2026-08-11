"""
modules/tts/base_tts.py
Qwen3TTS — client asincrono per il server TTS locale (Qwen3-TTS con voice cloning).

Il server gira in un venv separato (venv-tts) con transformers 4.57.3.
Questo modulo gestisce il ciclo di vita del server e comunica via HTTP.

Flussi supportati:
    1. synthesize(text)                   → TTSResult              (blocco singolo)
    2. synthesize_to_file(text, path)     → Path                   (salva WAV su disco)
    3. stream_sentences(text, callback)   → None                   (frase per frase,
                                                                     callback su ogni TTSChunk)
    4. stream_sentences_gen(text)         → AsyncGenerator[TTSChunk]
    5. play(audio_bytes)                  → None                   (riproduce via sounddevice)
    6. say(text)                          → TTSResult              (sintetizza + riproduci)

Formato audio prodotto: WAV PCM int16 LE, mono, 24 000 Hz.

Uso rapido:
    async with Qwen3TTS() as tts:
        result = await tts.synthesize("Ciao, come posso aiutarti?")
        await tts.play(result.audio_bytes)

Uso streaming (bassa latenza, si aggancia all'output LLM):
    async with Qwen3TTS() as tts:
        async def on_chunk(chunk: TTSChunk) -> None:
            await tts.play(chunk.audio_bytes)
        await tts.stream_sentences("Prima frase. Seconda frase.", callback=on_chunk)

Il server viene avviato automaticamente al primo utilizzo e fermato all'uscita
dal context manager. I modelli vengono mantenuti in VRAM per tutta la sessione.
"""

from __future__ import annotations

import asyncio
import io
import re
import subprocess
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Coroutine, Optional, Union

import httpx
import numpy as np

from config.settings import settings
from core.logger import logger

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

OUTPUT_SAMPLE_RATE: int = 24_000
_INT16_MAX: float = 32_768.0

PROJECT_ROOT   = Path(__file__).parent.parent.parent
_VENV_TTS      = PROJECT_ROOT / "venv-tts"
_SERVER_SCRIPT = Path(__file__).parent / "server.py"


def _die_with_parent() -> None:
    """
    Rete di sicurezza per il server TTS: se il processo padre muore in modo
    brutale (crash, kill -9), la chiusura ordinata non gira e il server
    resterebbe orfano a occupare ~3,4 GB di RAM e la VRAM della GPU1.

    PR_SET_PDEATHSIG chiede al kernel di mandargli SIGTERM appena il padre
    termina, qualunque sia la causa. Solo Linux; best-effort, gira nel
    figlio tra fork ed exec.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, 15, 0, 0, 0)   # PR_SET_PDEATHSIG, SIGTERM
    except Exception:
        pass

_LANG_MAP: dict[str, str] = {
    "it":    "italian",
    "en":    "english",
    "en-us": "english",
    "en-gb": "english",
    "fr":    "french",
    "de":    "german",
    "es":    "spanish",
    "pt":    "portuguese",
    "ja":    "japanese",
    "ko":    "korean",
    "zh":    "chinese",
    "ru":    "russian",
    "auto":  "auto",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|(?<=\n)\s*")


# ---------------------------------------------------------------------------
# Dataclasses pubblici
# ---------------------------------------------------------------------------

@dataclass
class TTSChunk:
    """Un chunk audio prodotto durante lo streaming frase-per-frase."""
    text:         str
    audio_bytes:  bytes
    sample_rate:  int
    duration_s:   float
    inference_ms: float
    index:        int  = 0
    is_last:      bool = False


@dataclass
class TTSResult:
    """Risultato completo della sintesi TTS."""
    text:         str
    audio_bytes:  bytes
    sample_rate:  int
    duration_s:   float
    inference_ms: float
    chunks:       list[TTSChunk] = field(default_factory=list)
    voice:        str = ""
    language:     str = ""

    def is_empty(self) -> bool:
        return len(self.audio_bytes) == 0

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "text_len":     len(self.text),
            "duration_s":   round(self.duration_s, 2),
            "inference_ms": round(self.inference_ms, 1),
            "voice":        self.voice,
            "language":     self.language,
            "chunks":       len(self.chunks),
        }


# ---------------------------------------------------------------------------
# Helper audio
# ---------------------------------------------------------------------------

def _wav_bytes_to_float32(wav_bytes: bytes) -> np.ndarray:
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return samples / _INT16_MAX


def _duration_from_wav(wav_bytes: bytes) -> float:
    if not wav_bytes:
        return 0.0
    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def _split_sentences(text: str) -> list[str]:
    raw = _SENTENCE_SPLIT_RE.split(text.strip())
    sentences: list[str] = []
    buf = ""
    for part in raw:
        part = part.strip()
        if not part:
            continue
        buf = (buf + " " + part).strip() if buf else part
        if len(buf.split()) >= 3:
            sentences.append(buf)
            buf = ""
    if buf:
        sentences.append(buf)
    return sentences if sentences else [text.strip()]


def _resolve_language(lang_code: str) -> str:
    return _LANG_MAP.get(lang_code.lower(), "italian")


def _write_wav(path: Path, wav_bytes: bytes) -> None:
    path.write_bytes(wav_bytes)


def _start_playback(wav_bytes: bytes, device: Optional[int]) -> None:
    """Avvia la riproduzione NON bloccante (la durata gira sul thread interno
    di PortAudio). L'attesa/interruzione è gestita in modo asincrono da play()."""
    import sounddevice as sd
    audio_f32 = _wav_bytes_to_float32(wav_bytes)
    sd.play(audio_f32, samplerate=OUTPUT_SAMPLE_RATE, device=device)


# ---------------------------------------------------------------------------
# Gestione server
# ---------------------------------------------------------------------------

async def _wait_for_server(url: str, timeout_s: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"{url}/health")
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(1.0)
    return False


# ---------------------------------------------------------------------------
# Client principale
# ---------------------------------------------------------------------------

class Qwen3TTS:
    """
    Client TTS asincrono per il server Qwen3-TTS locale.

    Args:
        profile:           Nome profilo vocale (default da settings.tts.voice).
        language:          Codice lingua ISO (default "it").
        cuda_device_index: Indice GPU CUDA (default da settings.tts.cuda_device_index).
        port:              Porta del server HTTP (default 8765).
        server_timeout_s:  Timeout avvio server in secondi (default da
                           settings.tts.server_timeout_s, 600). Il modello
                           Qwen3-TTS impiega 60-240s a caricare a freddo
                           sulla 3060 Ti; al primissimo avvio scarica anche
                           i pesi da HuggingFace, da cui il margine largo.
    """

    def __init__(
        self,
        profile:           Optional[str] = None,
        language:          Optional[str] = None,
        cuda_device_index: Optional[int] = None,
        port:              int   = 8765,
        server_timeout_s:  Optional[float] = None,
        speed:             Optional[float] = None,
    ) -> None:
        self._profile           = profile or settings.tts.voice
        self._language          = language or getattr(settings.stt, "language", "it")
        self._cuda_device_index = cuda_device_index if cuda_device_index is not None \
                                  else getattr(settings.tts, "cuda_device_index", 1)
        self._port             = port
        self._server_timeout_s = server_timeout_s if server_timeout_s is not None \
                                 else getattr(settings.tts, "server_timeout_s", 600.0)
        # Velocità del parlato (time-stretch server-side a pitch invariato).
        self._speed            = speed if speed is not None \
                                 else getattr(settings.tts, "speed", 1.0)
        self._base_url         = f"http://127.0.0.1:{port}"
        self._process:   Optional[subprocess.Popen] = None
        self._http:      Optional[httpx.AsyncClient] = None
        self._owns_server: bool = False

    # -- context manager -------------------------------------------------------

    async def __aenter__(self) -> "Qwen3TTS":
        await self._ensure_server()
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(60.0),
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None
        if self._owns_server and self._process:
            logger.info("tts | arresto server")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    # -- avvio server ----------------------------------------------------------

    async def _ensure_server(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{self._base_url}/health")
                if r.status_code == 200:
                    logger.info("tts | server già attivo su porta {}", self._port)
                    return
        except Exception:
            pass

        python = _VENV_TTS / "bin" / "python"
        if not python.exists():
            raise RuntimeError(
                f"venv-tts non trovato in {_VENV_TTS}. "
                "Crea il venv con: python3.12 -m venv venv-tts"
            )

        import os
        env = os.environ.copy()
        nvidia_path = _VENV_TTS / "lib" / "python3.12" / "site-packages" / "nvidia"
        if nvidia_path.exists():
            cuda_libs = ":".join([
                str(nvidia_path / "cublas"       / "lib"),
                str(nvidia_path / "cudnn"        / "lib"),
                str(nvidia_path / "cuda_runtime" / "lib"),
                str(nvidia_path / "cufft"        / "lib"),
                str(nvidia_path / "curand"       / "lib"),
            ])
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{cuda_libs}:{existing}" if existing else cuda_libs

        cmd = [
            str(python), str(_SERVER_SCRIPT),
            "--profile", self._profile,
            "--port",    str(self._port),
            "--gpu",     str(self._cuda_device_index),
            "--model-size", getattr(settings.tts, "model_size", "1.7B"),
        ]

        logger.info(
            "tts | avvio server [profilo={}, gpu={}, porta={}]",
            self._profile, self._cuda_device_index, self._port,
        )

        self._process = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=_die_with_parent,
        )
        self._owns_server = True

        ready = await _wait_for_server(self._base_url, self._server_timeout_s)
        if not ready:
            self._process.kill()
            raise RuntimeError(
                f"Il server TTS non ha risposto entro {self._server_timeout_s}s."
            )
        logger.info("tts | server pronto")

    # -- sintesi ---------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        *,
        language: Optional[str] = None,
    ) -> TTSResult:
        if self._http is None:
            raise RuntimeError("Qwen3TTS non inizializzato — usa 'async with Qwen3TTS()'")

        text = text.strip()
        if not text:
            return TTSResult(
                text="", audio_bytes=b"", sample_rate=OUTPUT_SAMPLE_RATE,
                duration_s=0.0, inference_ms=0.0,
            )

        resolved_lang = _resolve_language(language or self._language)
        t0 = time.perf_counter()
        r  = await self._http.post("/synthesize", json={"text": text, "language": resolved_lang, "speed": self._speed})
        r.raise_for_status()
        inference_ms = (time.perf_counter() - t0) * 1000

        wav_bytes  = r.content
        duration_s = _duration_from_wav(wav_bytes)

        result = TTSResult(
            text=text, audio_bytes=wav_bytes, sample_rate=OUTPUT_SAMPLE_RATE,
            duration_s=duration_s, inference_ms=inference_ms,
            voice=self._profile, language=resolved_lang,
        )
        logger.info("tts | '{:.60s}' → {:.1f}s audio ({:.0f} ms)", text, duration_s, inference_ms)
        return result

    # -- warmup ----------------------------------------------------------------

    async def warmup(self, *, text: str = "ok") -> bool:
        """
        Prima inferenza usa-e-getta per pagare a vuoto il costo della prima
        sintesi (compilazione kernel CUDA / autotuning), così la prima
        risposta reale non lo sconta.

        Il server carica già i pesi in VRAM al boot (atteso da
        _wait_for_server in _ensure_server), quindi qui NON ricarichiamo il
        modello: facciamo solo una synth minima e scartiamo l'audio.

        Best-effort: non solleva mai, ritorna True solo se la sintesi è
        andata a buon fine. No-op se il client non è inizializzato.
        """
        if self._http is None:
            return False
        try:
            t0 = time.perf_counter()
            await self.synthesize(text)
            logger.info("tts | warmup ✓ ({:.0f}ms)", (time.perf_counter() - t0) * 1000)
            return True
        except Exception as exc:
            logger.warning("tts | warmup fallito: {} — continuo", exc)
            return False

    # -- streaming (callback) --------------------------------------------------

    async def stream_sentences(
        self,
        text: str,
        callback: Callable[[TTSChunk], Coroutine[Any, Any, None]],
        *,
        language: Optional[str] = None,
    ) -> None:
        if self._http is None:
            raise RuntimeError("Qwen3TTS non inizializzato — usa 'async with Qwen3TTS()'")

        sentences     = _split_sentences(text)
        total         = len(sentences)
        resolved_lang = _resolve_language(language or self._language)

        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            t0 = time.perf_counter()
            r  = await self._http.post("/synthesize", json={"text": sentence, "language": resolved_lang, "speed": self._speed})
            r.raise_for_status()
            inference_ms = (time.perf_counter() - t0) * 1000
            wav_bytes = r.content
            chunk = TTSChunk(
                text=sentence, audio_bytes=wav_bytes, sample_rate=OUTPUT_SAMPLE_RATE,
                duration_s=_duration_from_wav(wav_bytes), inference_ms=inference_ms,
                index=idx, is_last=(idx == total - 1),
            )
            logger.debug("tts.stream | [{}/{}] '{:.50s}' ({:.0f} ms)", idx+1, total, sentence, inference_ms)
            await callback(chunk)

    # -- streaming (generator) -------------------------------------------------

    async def stream_sentences_gen(
        self,
        text: str,
        *,
        language: Optional[str] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        if self._http is None:
            raise RuntimeError("Qwen3TTS non inizializzato — usa 'async with Qwen3TTS()'")

        sentences     = _split_sentences(text)
        total         = len(sentences)
        resolved_lang = _resolve_language(language or self._language)

        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            t0 = time.perf_counter()
            r  = await self._http.post("/synthesize", json={"text": sentence, "language": resolved_lang, "speed": self._speed})
            r.raise_for_status()
            inference_ms = (time.perf_counter() - t0) * 1000
            wav_bytes = r.content
            yield TTSChunk(
                text=sentence, audio_bytes=wav_bytes, sample_rate=OUTPUT_SAMPLE_RATE,
                duration_s=_duration_from_wav(wav_bytes), inference_ms=inference_ms,
                index=idx, is_last=(idx == total - 1),
            )

    # -- file ------------------------------------------------------------------

    async def synthesize_to_file(
        self,
        text: str,
        path: Union[str, Path],
        *,
        language: Optional[str] = None,
    ) -> Path:
        result = await self.synthesize(text, language=language)
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: _write_wav(output, result.audio_bytes))
        logger.info("tts | salvato WAV: {} ({:.1f}s)", output, result.duration_s)
        return output

    # -- riproduzione ----------------------------------------------------------

    # Intervallo di polling dell'attesa asincrona (s). Compromesso tra
    # reattività dello stop (≤~30ms) e carico CPU trascurabile.
    _PLAY_POLL_S = 0.03

    async def play(
        self,
        audio_bytes: bytes,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        """
        Riproduce un blocco WAV attendendo in modo ASINCRONO e INTERROMPIBILE.

        A differenza di un sd.play(blocking=True) in un thread, qui l'avvio è
        non bloccante e l'attesa è un polling su asyncio.sleep: questo
        garantisce che
          • la riproduzione si possa fermare all'istante (stop_playback →
            sd.stop()), e
          • il task che ci attende sia cancellabile davvero (la cancellazione
            colpisce asyncio.sleep, non resta intrappolata in un sd.wait()
            bloccante dentro un executor).

        Tutte le chiamate sounddevice avvengono sul thread dell'event loop:
        niente accesso cross-thread allo stream → niente deadlock con PortAudio.
        Ritorna quando la riproduzione è finita o è stata interrotta.
        """
        if not audio_bytes:
            return
        import sounddevice as sd
        try:
            _start_playback(audio_bytes, device_index)
        except Exception as exc:
            logger.warning("tts | avvio play fallito: {}", exc)
            return
        try:
            while True:
                # get_stream() solleva RuntimeError se non è mai partito nulla;
                # l'accesso a .active solleva PortAudioError su uno stream già
                # chiuso — è il caso del mute, dove interrupt_tts → stop_playback
                # chiama sd.stop() (che ferma E chiude lo stream) mentre questo
                # loop sta ancora attendendo. In tutti questi casi la
                # riproduzione è di fatto terminata: esci pulito senza loggare
                # un falso "play() fallito".
                try:
                    stream = sd.get_stream()
                    active = stream is not None and stream.active
                except Exception:
                    break
                if not active:
                    break
                await asyncio.sleep(self._PLAY_POLL_S)
        except asyncio.CancelledError:
            # Cancellazione del task (es. stop generazione): taglia l'audio.
            try:
                sd.stop()
            except Exception:
                pass
            raise

    async def say(self, text: str, *, language: Optional[str] = None) -> TTSResult:
        """Shortcut: sintetizza e riproduce immediatamente."""
        result = await self.synthesize(text, language=language)
        await self.play(result.audio_bytes)
        return result

    async def stop_playback(self) -> None:
        """
        Interrompe immediatamente la riproduzione audio in corso.

        Va invocato dal thread dell'event loop (come fa interrupt_tts): chiama
        sd.stop() in modo sincrono — stessa thread di play() — azzerando lo
        stream. Il loop di attesa di play() vede subito stream.active == False
        e ritorna. No-op se non c'è nulla in riproduzione.
        """
        import sounddevice as sd
        try:
            sd.stop()
        except Exception as exc:
            logger.warning("tts | stop_playback fallito: {}", exc)

    # -- info ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return OUTPUT_SAMPLE_RATE

    @property
    def profile(self) -> str:
        return self._profile

    @property
    def language(self) -> str:
        return self._language

    async def server_info(self) -> dict:
        if self._http is None:
            raise RuntimeError("Qwen3TTS non inizializzato")
        r = await self._http.get("/health")
        r.raise_for_status()
        return r.json()

    async def available_profiles(self) -> list[str]:
        if self._http is None:
            raise RuntimeError("Qwen3TTS non inizializzato")
        r = await self._http.get("/profiles")
        r.raise_for_status()
        return r.json().get("profiles", [])
