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
    WARMUP    = "warmup"


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
        self._wake_detector: Optional[Any] = None  # WakeWordDetector (lazy)

        self._state:              str            = LoopState.IDLE
        self._tts_enabled:        bool           = True
        self._is_speaking:        bool           = False
        self._echo_block_until:   float          = 0.0
        self._tts_last_play_end:  float          = 0.0
        self._tts_speaking_start: float          = 0.0

        # Latenze ultimo turno (popolate da _stream_and_speak / _run_ptt).
        # Misurate qui nel loop base, senza patchare i metodi a runtime.
        self._last_stt_ms:        float          = 0.0   # trascrizione (da _run_ptt)
        self._last_llm_ms:        float          = 0.0   # TTFT: start → primo chunk
        self._last_tts_ms:        float          = 0.0   # primo synth → primo audio
        self._stop_event:         asyncio.Event  = asyncio.Event()
        # Serializza i warmup (avvio + switch) così non si sovrappongono e
        # non sporcano il segnalino con transizioni concorrenti.
        self._warmup_lock:        asyncio.Lock   = asyncio.Lock()
        # Cancellazione del singolo turno (stop dell'output LLM senza
        # spegnere il loop). Settato da cancel_generation(), consumato e
        # ripulito da _stream_and_speak a ogni turno.
        self._cancel_event:       asyncio.Event  = asyncio.Event()
        # Coda audio del turno in corso (riferimento usato da interrupt_tts
        # per scartare il pendente quando si muta il TTS mentre parla).
        self._active_audio_queue: Optional[asyncio.Queue] = None
        # Task fire-and-forget per fermare la riproduzione: tenuti referenziati
        # finché non completano (altrimenti l'event loop li può GC-are).
        self._audio_tasks:        set            = set()
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
        await self._load_wake_word()

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

    async def warmup(self) -> None:
        """
        Warmup d'avvio: scalda il modello chat attivo + l'embedding + il TTS,
        mostrando lo stato "warmup" sul segnalino. Pensato per girare in
        background dopo load(). Gated su settings.ollama.warmup; best-effort
        (non solleva mai). Serializzato da _warmup_lock.
        """
        if not getattr(settings.ollama, "warmup", True):
            logger.debug("voice_loop | warmup disabilitato da settings")
            return
        await self._do_warmup(models=None, embed=True, include_tts=True)

    async def warmup_switch(self, model: str, *, include_tts: bool = False) -> None:
        """
        Warmup mirato dopo uno switch a runtime (cambio modello dalla tendina
        chat/terminale, oppure cambio modalità chat↔terminale).

        Scalda SOLO il modello di destinazione `model` (l'override vale dal
        turno successivo, quindi questo non deve bloccare nulla: il chiamante
        lo lancia fire-and-forget). Il segnalino mostra "warmup" mentre scalda
        e torna allo stato precedente alla fine.

        Gated su settings.ollama.warmup (master) E warmup_on_switch.
        Best-effort: non solleva mai. Serializzato da _warmup_lock.
        """
        if not getattr(settings.ollama, "warmup", True):
            return
        if not getattr(settings.ollama, "warmup_on_switch", True):
            return
        if not model:
            return
        await self._do_warmup(models=[model], embed=False, include_tts=include_tts)

    async def _do_warmup(
        self,
        *,
        models:      Optional[list[str]],
        embed:       bool,
        include_tts: bool,
    ) -> None:
        """
        Primitiva comune di warmup con segnalino.

        - models=None → scalda il modello chat attivo + embed via
          Orchestrator.warmup() (caso d'avvio).
        - models=[...] → scalda per nome i modelli indicati via
          Orchestrator.warmup_model() (caso switch).

        Imposta lo stato "warmup" all'inizio e lo ripristina alla fine SOLO
        se nel frattempo non è cambiato (per non calpestare un turno che
        l'utente abbia avviato durante il warmup). Mai solleva.
        """
        if self._orch is None:
            return
        async with self._warmup_lock:
            prev = self._state
            self._set_state(LoopState.WARMUP)
            try:
                if models is None:
                    # Avvio: modello chat attivo + embedding (gestiti dall'orch).
                    await self._orch.warmup()
                else:
                    ka = getattr(settings.ollama, "warmup_keep_alive", None)
                    for m in models:
                        await self._orch.warmup_model(m, keep_alive=ka)
                    if embed and self._orch._memory is not None:
                        try:
                            await self._orch._llm.embed(["warmup"])
                        except Exception as exc:
                            logger.warning("voice_loop | warmup embed: {}", exc)
                if include_tts:
                    await self._warmup_tts()
            except Exception as exc:
                logger.warning("voice_loop | warmup fallito: {}", exc)
            finally:
                self._restore_state_after_warmup(prev)

    async def _warmup_tts(self) -> None:
        """
        Warmup del TTS (prima inferenza usa-e-getta). Gated su
        settings.tts.warmup; no-op se il TTS non è caricato. Best-effort.
        """
        if self._tts is None or not getattr(settings.tts, "warmup", True):
            return
        try:
            await self._tts.warmup()
        except Exception as exc:
            logger.warning("voice_loop | warmup TTS fallito: {}", exc)

    def _restore_state_after_warmup(self, prev: str) -> None:
        """
        Ripristina lo stato dopo un warmup, ma SOLO se è ancora "warmup":
        se l'utente ha iniziato a registrare/parlare nel frattempo, lo stato
        è già cambiato e non lo tocchiamo. Decisione di design: il warmup NON
        inibisce il PTT, quindi una transizione concorrente ha la precedenza.
        """
        if self._state != LoopState.WARMUP:
            return
        target = LoopState.LISTENING if self._loaded else LoopState.IDLE
        # Evita di "tornare" a warmup se prev era già warmup (caso degenere).
        if prev == LoopState.WARMUP:
            prev = target
        self._set_state(prev if prev in (
            LoopState.LISTENING, LoopState.IDLE,
        ) else target)

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

    async def _load_wake_word(self) -> None:
        """
        Carica il detector openWakeWord (CPU) se abilitato. Best-effort:
        senza pacchetto/modello il loop resta solo-PTT, nessun errore fatale.
        Richiede lo STT (la frase post-trigger va trascritta).
        """
        if not getattr(settings.wake_word, "enabled", False):
            return
        if not self._stt:
            logger.info("voice_loop | wake word saltato: STT non disponibile")
            return
        try:
            from modules.wake_word import WakeWordDetector
            det = WakeWordDetector()
            await asyncio.get_running_loop().run_in_executor(None, det.load)
            self._wake_detector = det
        except Exception as exc:
            logger.warning(
                "voice_loop | wake word non disponibile ({}) — solo PTT", exc
            )

    # API pubblica

    async def run(self) -> None:
        self._require_loaded()
        self._stop_event.clear()
        consumer_task = asyncio.create_task(self._turn_consumer())
        wake_task: Optional[asyncio.Task] = None
        if self._wake_detector is not None:
            wake_task = asyncio.create_task(self._wake_producer())
        try:
            self._set_state(LoopState.LISTENING)
            await self._run_ptt()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("voice_loop | PTT loop fallito: {}", exc)
        finally:
            self._stop_event.set()
            for t in (consumer_task, wake_task):
                if t is None:
                    continue
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

    async def stop(self) -> None:
        logger.info("voice_loop | stop richiesto")
        self._stop_event.set()

    def cancel_generation(self) -> bool:
        """
        Interrompe la generazione LLM del turno in corso (come il tasto
        "stop" di Claude/Gemini): lo stream viene troncato, la connessione
        verso Ollama chiusa e l'eventuale TTS pendente abortito. Il testo
        prodotto fino a quel momento viene comunque conservato.

        Cooperativo e non distruttivo: NON spegne il loop né svuota lo
        storico. È un no-op se non c'è nulla in generazione.

        Returns:
            True se un turno era effettivamente in corso (thinking/speaking),
            False altrimenti.
        """
        if self._state in (LoopState.THINKING, LoopState.SPEAKING):
            self._cancel_event.set()
            logger.info("voice_loop | cancellazione generazione richiesta")
            return True
        return False

    def interrupt_tts(self) -> bool:
        """
        Zittisce immediatamente la voce: taglia la riproduzione in corso e
        scarta l'audio già accodato del turno attivo. A differenza di
        cancel_generation() NON ferma l'LLM — il testo continua a scorrere,
        solo la voce tace (utile quando si muta il TTS mentre l'assistente
        sta parlando).

        Le sintesi successive del turno sono già inibite dal flag
        _tts_enabled in _stream_and_speak. Ritorna True se c'era una coda
        audio attiva su cui agire.
        """
        acted = False
        q = self._active_audio_queue
        if q is not None:
            try:
                while True:
                    q.get_nowait()
                    acted = True
            except asyncio.QueueEmpty:
                pass
        if self._tts is not None:
            try:
                loop = asyncio.get_running_loop()
                t = loop.create_task(self._tts.stop_playback())
                self._audio_tasks.add(t)
                t.add_done_callback(self._audio_tasks.discard)
                acted = True
            except RuntimeError:
                # Nessun event loop in esecuzione: niente da interrompere.
                pass
        if acted:
            logger.info("voice_loop | riproduzione TTS interrotta (mute)")
        return acted

    async def _iter_cancellable(self, agen):
        """
        Itera un async-generator rendendolo interrompibile tramite
        ``self._cancel_event``. A ogni passo corre l'attesa del prossimo
        chunk contro l'evento di cancel: se l'evento scatta per primo —
        anche mentre il modello è bloccato in attesa del token successivo —
        l'iterazione del chunk viene annullata e il generatore chiuso, così
        la connessione HTTP verso Ollama si libera subito.

        Garantisce sempre ``agen.aclose()`` in uscita (break, fine naturale
        o eccezione), evitando connessioni appese.
        """
        cancel_wait = asyncio.ensure_future(self._cancel_event.wait())
        try:
            while True:
                nxt = asyncio.ensure_future(agen.__anext__())
                done, _ = await asyncio.wait(
                    {nxt, cancel_wait}, return_when=asyncio.FIRST_COMPLETED
                )
                # Cancel ha vinto la corsa (il chunk non è ancora arrivato).
                if cancel_wait in done and nxt not in done:
                    nxt.cancel()
                    try:
                        await nxt
                    except (asyncio.CancelledError, StopAsyncIteration):
                        pass
                    except Exception as exc:
                        logger.debug("voice_loop | chunk annullato: {}", exc)
                    return
                try:
                    chunk = nxt.result()
                except StopAsyncIteration:
                    return
                yield chunk
                # Cancel arrivato tra un chunk e l'altro: esci pulito.
                if self._cancel_event.is_set():
                    return
        finally:
            if not cancel_wait.done():
                cancel_wait.cancel()
            await agen.aclose()

    @property
    def stats(self) -> VoiceLoopStats:
        return self._stats

    @property
    def state(self) -> str:
        return self._state

    @property
    def tts_enabled(self) -> bool:
        """True se la sintesi vocale è attiva per i turni successivi."""
        return self._tts_enabled

    @tts_enabled.setter
    def tts_enabled(self, value: bool) -> None:
        self._tts_enabled = bool(value)

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

    # wake word (hands-free)

    async def _wake_producer(self) -> None:
        """
        Produttore hands-free: microfono sempre aperto su un flusso dedicato,
        ogni frame (80ms) passa dal WakeWordDetector (CPU). Al trigger
        registra la frase con endpointing a energia (si ferma da sola al
        silenzio), trascrive e accoda il turno nella stessa coda del PTT.

        Convive col PTT: mentre l'assistente parla (o subito dopo, finestra
        anti-eco) i frame vengono letti e scartati, così la voce del TTS
        non può auto-attivare l'assistente. Le letture bloccanti girano in
        thread (asyncio.to_thread) per non fermare l'event loop.
        """
        import numpy as np
        import sounddevice as sd

        from modules.wake_word import FRAME_SAMPLES, SAMPLE_RATE, Endpointer

        det = self._wake_detector
        try:
            stream = sd.InputStream(
                samplerate = SAMPLE_RATE,
                channels   = 1,
                dtype      = "int16",
                blocksize  = FRAME_SAMPLES,
            )
            stream.start()
        except Exception as exc:
            logger.warning("voice_loop | wake word: microfono non apribile ({})", exc)
            return

        logger.info(
            "voice_loop | wake word attivo — di' '{}' per parlare",
            settings.wake_word.model.replace("_", " "),
        )
        print(f"\U0001f44b  Wake word attiva: di' «{settings.wake_word.model.replace('_', ' ')}»")

        muted_prev = False
        try:
            while not self._stop_event.is_set():
                data, _ = await asyncio.to_thread(stream.read, FRAME_SAMPLES)
                frame = data[:, 0] if data.ndim > 1 else data

                # Assistente che parla / anti-eco / PTT in corso: scarta.
                muted = (
                    self._is_speaking
                    or time.monotonic() < self._echo_block_until
                    or self._state == LoopState.RECORDING
                )
                if muted:
                    muted_prev = True
                    continue
                if muted_prev:
                    # Uscita dalla finestra muta: azzera i buffer del
                    # detector, contengono la voce del TTS.
                    det.reset()
                    muted_prev = False

                if not det.process(frame):
                    continue

                # --- Trigger: registra la frase fino al silenzio ---
                logger.debug("voice_loop | wake word (score={:.2f})", det.last_score)
                self._set_state(LoopState.RECORDING)
                print("\r\U0001f399️  Ti ascolto...", end="", flush=True)

                ep = Endpointer()
                frames: list = []
                while not self._stop_event.is_set():
                    data, _ = await asyncio.to_thread(stream.read, FRAME_SAMPLES)
                    frame = data[:, 0] if data.ndim > 1 else data
                    frames.append(frame.copy())
                    if ep.update(frame):
                        break

                print("\r" + " " * 24 + "\r", end="", flush=True)
                self._set_state(LoopState.LISTENING)
                det.reset()

                if not frames:
                    continue
                audio_bytes = np.concatenate(frames, axis=0).tobytes()

                try:
                    result = await self._stt.transcribe_with_vad(audio_bytes)
                except Exception as exc:
                    self._stats.stt_errors += 1
                    logger.error("voice_loop | wake word STT fallito: {}", exc)
                    continue
                if result.is_empty():
                    logger.debug("voice_loop | wake word: nessun parlato dopo il trigger")
                    continue
                try:
                    self._turn_queue.put_nowait(result)
                except asyncio.QueueFull:
                    logger.debug("voice_loop | wake word: queue piena — scartato")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("voice_loop | wake word producer fallito: {}", exc)
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

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
        # Esponi la coda così interrupt_tts() può svuotarla se il TTS viene
        # mutato mentre l'assistente parla.
        self._active_audio_queue = audio_queue

        # Strumentazione latenze (senza patchare metodi condivisi):
        #   _last_llm_ms = stream_start → primo chunk LLM
        #   _last_tts_ms = primo synth() → inizio riproduzione audio
        stream_start             = time.monotonic()
        first_synth_time: float  = 0.0
        self._last_llm_ms        = 0.0
        self._last_tts_ms        = 0.0

        async def _player() -> None:
            while True:
                audio = await audio_queue.get()
                if audio is None:
                    break
                # TTS mutato a metà turno: scarta l'audio senza riprodurlo
                # (doppia guardia oltre allo svuotamento fatto da interrupt_tts).
                if not self._tts_enabled:
                    continue
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
            nonlocal first_synth_time
            if not self._tts or not self._tts_enabled:
                return
            clean = _strip_markdown(text)
            if not clean:
                return
            if first_synth_time == 0.0:
                first_synth_time = time.monotonic()
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
        # Ogni turno parte senza richieste di stop pendenti.
        self._cancel_event.clear()
        cancelled = False

        try:
            async for chunk in self._iter_cancellable(self._orch.turn(ctx)):
                full_text += chunk
                buf       += chunk
                if first_chunk:
                    first_chunk = False
                    self._last_llm_ms = round((time.monotonic() - stream_start) * 1000, 1)
                    self._set_state(LoopState.SPEAKING)
                print(chunk, end="", flush=True)
                self._on_llm_chunk(chunk)
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
                # Stop richiesto a metà turno: smetti di sintetizzare nuove frasi.
                if self._cancel_event.is_set():
                    cancelled = True
                    break
            # Coda finale: sintetizzata solo se il turno è terminato da sé.
            cancelled = cancelled or self._cancel_event.is_set()
            if not cancelled and buf.strip():
                await _synth(buf.strip())
        except Exception:
            audio_queue.put_nowait(None)
            player_task.cancel()
            raise
        finally:
            if cancelled:
                # Interruzione: scarta l'audio ancora in coda e taglia la
                # riproduzione corrente, senza attendere il drain naturale.
                try:
                    while True:
                        audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                player_task.cancel()
                try:
                    await player_task
                except asyncio.CancelledError:
                    pass
                logger.info("voice_loop | generazione interrotta dall'utente")
            else:
                try:
                    await audio_queue.put(None)
                except Exception:
                    pass
                await player_task
                # TTS first-audio: primo synth() → inizio riproduzione.
                if first_synth_time > 0.0 and self._tts_speaking_start > 0.0:
                    self._last_tts_ms = round(
                        (self._tts_speaking_start - first_synth_time) * 1000, 1
                    )
                # Drain hardware: senza questo sleep l ultima sillaba viene troncata
                # perche il driver audio non ha ancora svuotato il suo buffer interno.
                await asyncio.sleep(_AUDIO_DRAIN_S)
            # Turno concluso: nessuna coda audio su cui possa agire interrupt_tts.
            self._active_audio_queue = None

        ctx.assistant_text = full_text

    # UI

    def _on_llm_chunk(self, chunk: str) -> None:
        """
        Hook chiamato per ogni chunk di testo prodotto dall'LLM durante lo
        streaming. Default: no-op (la console stampa già il chunk).
        Le sottoclassi (es. UIBridge) lo sovrascrivono per inoltrarlo
        altrove, senza dover patchare i metodi dell'orchestratore.
        """
        pass

    def _set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            icons = {
                LoopState.IDLE:      "\u23f8",
                LoopState.LISTENING: "\U0001f7e2",
                LoopState.RECORDING: "\U0001f534",
                LoopState.THINKING:  "\U0001f914",
                LoopState.SPEAKING:  "\U0001f50a",
                LoopState.WARMUP:    "\U0001f525",
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
