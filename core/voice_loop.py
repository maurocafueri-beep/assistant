"""
core/voice_loop.py
VoiceLoop — loop vocale Push-To-Talk per local-assistant.

Modalita PTT:
    Tieni premuto SPAZIO -> registra dal microfono
    Rilascia SPAZIO      -> trascrivi istantaneamente -> LLM -> TTS

Pipeline per ogni turno:
    PTT (hold SPAZIO) -> raw audio -> STT.transcribe()
        -> AssistantContext -> Orchestrator.turn() -> stream chunk LLM
        -> sentence accumulator -> TTS.synthesize() + play() in parallelo

Vantaggi rispetto al VAD:
    - Nessun silenzio di attesa (1.5s risparmiati per turno)
    - Nessun eco: il mic e gia fermo quando l assistente parla
    - L utente controlla esattamente inizio e fine della registrazione

Requisito Linux: l utente deve essere nel gruppo input:
    sudo usermod -a -G input $USER   (poi logout/login)

API pubblica:
    async with VoiceLoop() as loop:
        await loop.run()

    await loop.stop()

Uso:
    venv-runtime/bin/python scripts/run_voice.py
    venv-runtime/bin/python scripts/run_voice.py --personality dev
"""

from __future__ import annotations

import asyncio
import io
import re
import sys
import time
import uuid
import wave
from dataclasses import dataclass
from typing import Any, Optional

from config.settings import settings
from core.context import AssistantContext, InputMode, ModelRole, OutputMode
from core.logger import logger
from core.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

_SENTENCE_END_RE    = re.compile(r"(?<=[.!?\u2026])\s+|(?<=\n)\n?")
_MIN_SENTENCE_CHARS = 15
_PTT_MIN_DURATION_S = 0.3
_PTT_SAMPLE_RATE    = 16_000
_PTT_BLOCK_SIZE     = 512
_AUDIO_DRAIN_S      = 0.35   # drain hardware dopo ultimo play() — fix troncamento
_PTT_ECHO_GRACE_S   = 0.5
_QUEUE_TIMEOUT_S    = 0.2


# ---------------------------------------------------------------------------
# Stato del loop
# ---------------------------------------------------------------------------

class LoopState:
    IDLE      = "idle"
    LISTENING = "listening"
    RECORDING = "recording"
    THINKING  = "thinking"
    SPEAKING  = "speaking"


# ---------------------------------------------------------------------------
# Dataclass pubblica
# ---------------------------------------------------------------------------

@dataclass
class VoiceLoopStats:
    turns:           int = 0
    stt_errors:      int = 0
    tts_errors:      int = 0
    total_words_in:  int = 0
    total_words_out: int = 0

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "turns":      self.turns,
            "stt_errors": self.stt_errors,
            "tts_errors": self.tts_errors,
            "words_in":   self.total_words_in,
            "words_out":  self.total_words_out,
        }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract_sentences(buf: str) -> tuple[list[str], str]:
    parts = _SENTENCE_END_RE.split(buf)
    if len(parts) <= 1:
        return [], buf
    return [s.strip() for s in parts[:-1] if s.strip()], parts[-1]


def _strip_markdown(text: str) -> str:
    text = re.sub(r"^\s*[\*\-\+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*",       r"\1", text)
    text = re.sub(r"\*([^*\n]+)\*",           r"\1", text)
    text = re.sub(r"_{2}([^_]+)_{2}",            r"\1", text)
    text = re.sub(r"_([^_\n]+)_",               r"\1", text)
    text = re.sub(r"^#{1,6}\s+",               "",    text, flags=re.MULTILINE)
    text = re.sub(r"`([^`]+)`",                  r"\1", text)
    text = re.sub(r"\n+",  " ", text)
    text = re.sub(r"  +",   " ", text)
    return text.strip()


# Energia minima sotto cui i campioni finali vengono considerati rumore/artefatti.
# Qwen3-TTS con voice cloning genera spesso rumore residuo dopo l'ultima parola.
# Il trimmer taglia tutto ciò che segue l'ultimo picco energetico reale,
# poi aggiunge _SILENCE_PAD_S di silenzio pulito per evitare il troncamento hardware.
_NOISE_THRESHOLD = 0.002   # ampiezza float32 sotto cui = rumore/silenzio
_WINDOW_MS       = 20      # finestra di analisi energetica (ms)
_SILENCE_PAD_S   = 0.25    # silenzio pulito aggiunto dopo il trim (s)


def _trim_wav_audio(wav_bytes: bytes) -> bytes:
    """
    Rimuove artefatti/rumore dal fondo di un WAV PCM int16 e aggiunge
    silenzio pulito finale per evitare il troncamento hardware.

    Algoritmo:
        1. Legge campioni int16 e normalizza in float32
        2. Scansiona dal fondo a finestre di _WINDOW_MS ms
        3. Trova l'ultimo frame con energia > _NOISE_THRESHOLD
        4. Taglia tutto dopo + aggiunge _SILENCE_PAD_S di silenzio pulito
    """
    import struct
    if not wav_bytes:
        return wav_bytes
    try:
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            n_ch      = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            raw       = wf.readframes(wf.getnframes())

        if sampwidth != 2:
            return wav_bytes  # solo int16 supportato

        n_samples = len(raw) // 2
        samples   = [struct.unpack_from("<h", raw, i * 2)[0] / 32768.0
                     for i in range(n_samples)]

        # Scansione dal fondo in finestre di _WINDOW_MS ms
        win = max(1, int(framerate * _WINDOW_MS / 1000))
        last_active = len(samples)  # default: tieni tutto

        for end in range(len(samples), 0, -win):
            start  = max(0, end - win)
            energy = max(abs(s) for s in samples[start:end])
            if energy > _NOISE_THRESHOLD:
                last_active = end
                break

        silence_frames = int(framerate * _SILENCE_PAD_S)
        silence_bytes  = b"\x00" * silence_frames * n_ch * sampwidth
        trimmed_raw    = raw[:last_active * sampwidth] + silence_bytes

        out = io.BytesIO()
        with wave.open(out, "wb") as wf:
            wf.setnchannels(n_ch)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(trimmed_raw)
        return out.getvalue()

    except Exception:
        return wav_bytes   # fallback sicuro


# ---------------------------------------------------------------------------
# VoiceLoop
# ---------------------------------------------------------------------------

class VoiceLoop:

    def __init__(
        self,
        personality:   Optional[str] = None,
        tts_profile:   Optional[str] = None,
        session_id:    Optional[str] = None,
        model_role:    ModelRole     = ModelRole.CHAT,
        enable_memory: bool          = True,
    ) -> None:
        self._personality   = personality
        self._tts_profile   = tts_profile
        self._session_id    = session_id or str(uuid.uuid4())[:8]
        self._model_role    = model_role
        self._enable_memory = enable_memory

        self._orch: Optional[Orchestrator] = None
        self._stt:  Optional[Any]          = None
        self._tts:  Optional[Any]          = None

        self._state:              str            = LoopState.IDLE
        self._is_speaking:        bool           = False
        self._echo_block_until:   float          = 0.0
        self._tts_last_play_end:  float          = 0.0
        self._tts_speaking_start: float          = 0.0
        self._stop_event:         asyncio.Event  = asyncio.Event()
        self._turn_queue:         asyncio.Queue  = asyncio.Queue(maxsize=1)
        self._stats:              VoiceLoopStats = VoiceLoopStats()
        self._loaded:             bool           = False

    # context manager

    async def __aenter__(self) -> "VoiceLoop":
        await self.load()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        self._stop_event.set()
        tasks = []
        if self._orch:
            tasks.append(self._orch.aclose())
        if self._tts:
            tasks.append(self._tts.__aexit__(None, None, None))
        if self._stt:
            tasks.append(self._stt.__aexit__(None, None, None))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._orch = self._stt = self._tts = None
        self._loaded = False
        logger.info("voice_loop | chiuso | stats={}", self._stats.to_log_dict())

    # inizializzazione

    async def load(self) -> None:
        logger.info("voice_loop | avvio | session={}", self._session_id)
        t0 = time.monotonic()

        self._orch = Orchestrator(
            personality = self._personality,
            enable_stt  = False,
            enable_tts  = False,
        )
        await self._orch.load()
        if not self._enable_memory:
            self._orch._memory = None

        await self._load_stt()
        await self._load_tts()

        self._loaded = True
        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "voice_loop | pronto in {:.0f}ms | stt={} tts={} memory={}",
            elapsed,
            "\u2713" if self._stt else "\u2717",
            "\u2713" if self._tts else "\u2717",
            "\u2713" if self._orch._memory else "\u2717",
        )
        self._print_status(elapsed)

    async def _load_stt(self) -> None:
        from modules.stt import WhisperSTT
        stt = WhisperSTT()
        await stt._load_models()
        self._stt = stt
        logger.debug("voice_loop | STT caricato")

    async def _load_tts(self) -> None:
        from modules.tts import Qwen3TTS
        tts = Qwen3TTS(profile=self._tts_profile)
        await tts.__aenter__()
        self._tts = tts
        logger.debug("voice_loop | TTS caricato (profilo={})", tts.profile)

    # API pubblica

    async def run(self) -> None:
        self._require_loaded()
        self._stop_event.clear()
        consumer_task = asyncio.create_task(self._turn_consumer())
        try:
            self._set_state(LoopState.LISTENING)
            await self._run_ptt()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("voice_loop | PTT loop fallito: {}", exc)
        finally:
            self._stop_event.set()
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass

    async def stop(self) -> None:
        logger.info("voice_loop | stop richiesto")
        self._stop_event.set()

    @property
    def stats(self) -> VoiceLoopStats:
        return self._stats

    @property
    def state(self) -> str:
        return self._state

    @property
    def session_id(self) -> str:
        return self._session_id

    # PTT

    def _ptt_matches(self, key: Any) -> bool:
        """
        Ritorna True se il tasto corrisponde al trigger PTT.
        Sovrascrivibile nelle sottoclassi per cambiare il tasto senza
        duplicare _run_ptt.
        """
        from pynput import keyboard as pynput_kb
        return key == pynput_kb.Key.space

    async def _run_ptt(self) -> None:
        """
        Loop PTT basato su pynput (funziona senza root su X11/Wayland).

        Un asyncio.Event thread-safe traccia lo stato dello spazio:
          on_press  → space_held.set()
          on_release → space_held.clear()

        Il loop attende il set, registra mentre e set, al clear trascrive.
        """
        import numpy as np
        import sounddevice as sd
        from pynput import keyboard as pynput_kb

        loop = asyncio.get_running_loop()

        # Event thread-safe: True = spazio premuto, False = rilasciato
        space_held = asyncio.Event()

        def on_press(key):
            if self._ptt_matches(key):
                loop.call_soon_threadsafe(space_held.set)

        def on_release(key):
            if self._ptt_matches(key):
                loop.call_soon_threadsafe(space_held.clear)

        listener = pynput_kb.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        logger.info("voice_loop | PTT attivo — tieni SPAZIO per parlare")

        try:
            while not self._stop_event.is_set():

                # Attendi pressione spazio
                try:
                    await asyncio.wait_for(space_held.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                if self._stop_event.is_set():
                    break

                # Ignora se assistente sta parlando
                if self._is_speaking or time.monotonic() < self._echo_block_until:
                    logger.debug("voice_loop | PTT ignorato (assistente sta parlando)")
                    # Aspetta rilascio prima di ricontrollare
                    while space_held.is_set():
                        await asyncio.sleep(0.02)
                    continue

                # --- Registrazione ---
                self._set_state(LoopState.RECORDING)
                print("\r\U0001f534 Registrazione... (rilascia SPAZIO per inviare)",
                      end="", flush=True)

                frames: list = []
                t_start = time.monotonic()

                stream = sd.InputStream(
                    samplerate = _PTT_SAMPLE_RATE,
                    channels   = 1,
                    dtype      = "int16",
                    blocksize  = _PTT_BLOCK_SIZE,
                )
                stream.start()

                try:
                    while space_held.is_set() and not self._stop_event.is_set():
                        data, _ = stream.read(_PTT_BLOCK_SIZE)
                        frames.append(data.copy())
                        await asyncio.sleep(0.005)
                finally:
                    stream.stop()
                    stream.close()

                duration = time.monotonic() - t_start
                print("\r" + " " * 58 + "\r", end="", flush=True)
                self._set_state(LoopState.LISTENING)

                if duration < _PTT_MIN_DURATION_S or not frames:
                    logger.debug("voice_loop | PTT troppo breve ({:.2f}s) — scartato", duration)
                    continue

                audio_bytes = np.concatenate(frames, axis=0).tobytes()
                logger.debug("voice_loop | PTT {:.2f}s registrati", duration)

                try:
                    result = await self._stt.transcribe(audio_bytes)
                    if not result.is_empty():
                        logger.debug("voice_loop | STT: \'{}\'", result.text[:80])
                        try:
                            self._turn_queue.put_nowait(result)
                        except asyncio.QueueFull:
                            logger.debug("voice_loop | queue piena — scartato")
                    else:
                        logger.debug("voice_loop | STT: nessun parlato")
                except Exception as exc:
                    self._stats.stt_errors += 1
                    logger.error("voice_loop | STT fallito: {}", exc)

        finally:
            listener.stop()

    # consumer

    async def _turn_consumer(self) -> None:
        while not self._stop_event.is_set():
            try:
                stt_result = await asyncio.wait_for(
                    self._turn_queue.get(), timeout=_QUEUE_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            await self._process_turn(stt_result)

    async def _process_turn(self, stt_result: Any) -> None:
        user_text = stt_result.text.strip()
        if not user_text:
            return

        self._is_speaking = True
        self._set_state(LoopState.THINKING)
        self._stats.turns += 1
        self._stats.total_words_in += len(user_text.split())

        print(f"\n{chr(8212)*50}")
        print(f"\U0001f3a4  Tu:  {user_text}")
        print(f"\U0001f916  Assistente: ", end="", flush=True)

        ctx = AssistantContext(
            user_text   = user_text,
            session_id  = self._session_id,
            input_mode  = InputMode.TEXT,
            output_mode = OutputMode.TEXT,
            model_role  = self._model_role,
        )

        try:
            await self._stream_and_speak(ctx)
        except Exception as exc:
            logger.error("voice_loop | _process_turn fallito: {}", exc)
            print(f"\n  \u26a0 Errore: {exc}")
        finally:
            self._is_speaking = False
            if self._tts_last_play_end > 0:
                self._echo_block_until = self._tts_last_play_end + _PTT_ECHO_GRACE_S
                logger.debug(
                    "voice_loop | echo grace {:.1f}s",
                    self._echo_block_until - time.monotonic(),
                )
            self._tts_last_play_end   = 0.0
            self._tts_speaking_start  = 0.0
            self._set_state(LoopState.LISTENING)
            self._stats.total_words_out += len(ctx.assistant_text.split())
            print()
            logger.info(
                "voice_loop | turno {} completato | {}",
                self._stats.turns,
                ctx.to_log_dict(),
            )

    # pipeline LLM->TTS

    async def _stream_and_speak(self, ctx: AssistantContext) -> None:
        self._set_state(LoopState.THINKING)
        audio_queue: asyncio.Queue = asyncio.Queue(maxsize=2)

        async def _player() -> None:
            while True:
                audio = await audio_queue.get()
                if audio is None:
                    break
                if self._tts:
                    try:
                        if self._tts_speaking_start == 0.0:
                            self._tts_speaking_start = time.monotonic()
                        await self._tts.play(_trim_wav_audio(audio))
                        self._tts_last_play_end = time.monotonic()
                    except Exception as exc:
                        logger.warning("voice_loop | play() fallito: {}", exc)

        player_task = asyncio.create_task(_player())

        async def _synth(text: str) -> None:
            if not self._tts:
                return
            clean = _strip_markdown(text)
            if not clean:
                return
            try:
                result = await self._tts.synthesize(clean)
                await audio_queue.put(result.audio_bytes)
            except Exception as exc:
                self._stats.tts_errors += 1
                detail = ""
                if hasattr(exc, "response"):
                    try:
                        detail = exc.response.text[:200]
                    except Exception:
                        pass
                logger.warning("voice_loop | synth fallito: {} {} — skip", exc, detail)

        buf, full_text, first_chunk = "", "", True

        try:
            async for chunk in self._orch.turn(ctx):
                full_text += chunk
                buf       += chunk
                if first_chunk:
                    first_chunk = False
                    self._set_state(LoopState.SPEAKING)
                print(chunk, end="", flush=True)
                sentences, buf = _extract_sentences(buf)
                # Frasi troppo corte: fondile con la successiva
                # (es. "Tu invece?" → prepend alla frase che segue)
                merged_sentences = list(sentences)
                i = 0
                while i < len(merged_sentences):
                    s = merged_sentences[i]
                    if len(s) < _MIN_SENTENCE_CHARS:
                        if i + 1 < len(merged_sentences):
                            merged_sentences[i + 1] = s + " " + merged_sentences[i + 1]
                            merged_sentences.pop(i)
                            continue
                        elif i > 0:
                            merged_sentences[i - 1] = merged_sentences[i - 1] + " " + s
                            merged_sentences.pop(i)
                            continue
                    i += 1
                for sentence in merged_sentences:
                    if len(sentence) >= _MIN_SENTENCE_CHARS:
                        await _synth(sentence)
                    else:
                        buf = sentence + " " + buf
            if buf.strip():
                await _synth(buf.strip())
        except Exception:
            audio_queue.put_nowait(None)
            player_task.cancel()
            raise
        finally:
            try:
                await audio_queue.put(None)
            except Exception:
                pass
            await player_task
            # Drain hardware: senza questo sleep l ultima sillaba viene troncata
            # perche il driver audio non ha ancora svuotato il suo buffer interno.
            await asyncio.sleep(_AUDIO_DRAIN_S)

        ctx.assistant_text = full_text

    # UI

    def _set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            icons = {
                LoopState.IDLE:      "\u23f8",
                LoopState.LISTENING: "\U0001f7e2",
                LoopState.RECORDING: "\U0001f534",
                LoopState.THINKING:  "\U0001f914",
                LoopState.SPEAKING:  "\U0001f50a",
            }
            logger.debug("voice_loop | stato -> {}", state)
            print(
                f"\r{icons.get(state, chr(183))} {state:<12}",
                end="", flush=True, file=sys.stderr,
            )

    def _print_status(self, elapsed_ms: float) -> None:
        print(f"\n{chr(9552)*50}")
        print(f"  \U0001f916  local-assistant  |  sessione: {self._session_id}")
        print(f"{chr(9552)*50}")
        print(f"  LLM:         {self._orch._personality.active.display_name}")
        print(f"  Personalita: {self._orch._personality.active.name}")
        print(f"  Memoria RAG: {chr(10003) if self._orch._memory else chr(10007)}")
        print(f"  STT:         {chr(10003) + ' Whisper' if self._stt else chr(10007) + ' non disponibile'}")
        print(f"  TTS:         {chr(10003) + ' ' + self._tts.profile if self._tts else chr(10007) + ' non disponibile'}")
        print(f"  Avvio:       {elapsed_ms:.0f}ms")
        print(f"{chr(9472)*50}")
        print("  Tieni SPAZIO per parlare, rilascia per inviare.")
        print("  Ctrl-C per uscire.")
        print(f"{chr(9552)*50}\n")

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "VoiceLoop non inizializzato: "
                "usa async with VoiceLoop() oppure await loop.load()"
            )

    def __repr__(self) -> str:
        if self._loaded:
            return (
                f"<VoiceLoop session='{self._session_id}' "
                f"state='{self._state}' "
                f"stt={chr(10003) if self._stt else chr(10007)} "
                f"tts={chr(10003) if self._tts else chr(10007)}>"
            )
        return "<VoiceLoop [non caricato]>"
