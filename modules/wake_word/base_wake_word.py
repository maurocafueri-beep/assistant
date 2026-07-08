"""
modules/wake_word/base_wake_word.py
WakeWordDetector — rilevamento della parola di attivazione con openWakeWord.

Gira interamente su CPU (ONNX, ~ms per frame): nessun impatto sulle GPU.
Il modello di default è "hey_jarvis" (pretrained, incluso nel pacchetto);
se ne può addestrare uno custom con openwakeword e puntarlo via
WAKE_WORD_MODEL (nome pretrained o path a un .onnx).

Formato audio atteso: PCM int16 mono a 16 000 Hz, frame da 1280 campioni
(80 ms) — la dimensione nativa di openwakeword.

Uso:
    det = WakeWordDetector()
    det.load()
    for frame in frames:            # np.ndarray int16, 1280 campioni
        if det.process(frame):
            ...                     # parola di attivazione rilevata

L'Endpointer decide quando la frase post-trigger è finita (energia RMS:
parla → silenzio prolungato → stop), così la registrazione si chiude da
sola senza tenere il microfono aperto oltre il necessario.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from config.settings import settings
from core.logger import logger

SAMPLE_RATE   = 16_000
FRAME_SAMPLES = 1_280   # 80 ms — chunk nativo di openwakeword


class WakeWordDetector:
    """
    Wrapper minimale attorno a openwakeword.Model.

    Args:
        model:      nome del modello pretrained (es. "hey_jarvis") o path a
                    un .onnx custom. Default da settings.wake_word.model.
        threshold:  soglia di attivazione sullo score [0..1].
        cooldown_s: dopo un trigger, ignora nuovi trigger per questo tempo
                    (evita doppie attivazioni sulla stessa pronuncia).
    """

    def __init__(
        self,
        model:      Optional[str]   = None,
        threshold:  Optional[float] = None,
        cooldown_s: Optional[float] = None,
    ) -> None:
        cfg = settings.wake_word
        self._model_name = model      if model      is not None else cfg.model
        self._threshold  = threshold  if threshold  is not None else cfg.threshold
        self._cooldown_s = cooldown_s if cooldown_s is not None else cfg.cooldown_s

        self._oww: Any = None          # openwakeword.Model
        self._key: str = ""            # chiave dello score nel dict di predict()
        self._last_trigger: float = 0.0
        self.last_score: float = 0.0

    @property
    def loaded(self) -> bool:
        return self._oww is not None

    def load(self) -> None:
        """Carica il modello openwakeword (CPU). Import lazy: il pacchetto
        serve solo se il wake word è abilitato."""
        from openwakeword.model import Model

        kwargs: dict = {}
        if self._model_name.endswith(".onnx"):
            kwargs["wakeword_model_paths"] = [self._model_name]
        self._oww = Model(**kwargs)

        keys = list(self._oww.models.keys())
        if self._model_name in keys:
            self._key = self._model_name
        elif len(keys) == 1:
            self._key = keys[0]
        else:
            # Modello pretrained non tra quelli caricati: errore chiaro subito,
            # non uno score sempre a zero in produzione.
            raise ValueError(
                f"wake word model '{self._model_name}' non trovato; "
                f"disponibili: {keys}"
            )
        logger.info(
            "wake_word | modello '{}' pronto (soglia={}, cooldown={}s)",
            self._key, self._threshold, self._cooldown_s,
        )

    def process(self, frame: np.ndarray) -> bool:
        """
        Analizza un frame (int16, 1280 campioni). True se la parola di
        attivazione è stata rilevata (soglia superata, fuori cooldown).
        """
        if self._oww is None:
            raise RuntimeError("WakeWordDetector non caricato — chiama load()")
        scores = self._oww.predict(frame)
        # predict() può esporre chiavi diverse dal nome modello (es. "timer"
        # → "1_minute_timer"): self._key è risolta in load().
        self.last_score = float(scores.get(self._key, 0.0))
        if self.last_score < self._threshold:
            return False
        now = time.monotonic()
        if now - self._last_trigger < self._cooldown_s:
            return False
        self._last_trigger = now
        return True

    def reset(self) -> None:
        """Svuota i buffer interni (da chiamare dopo ogni turno gestito,
        così l'audio residuo della pronuncia non ri-triggera)."""
        if self._oww is not None and hasattr(self._oww, "reset"):
            self._oww.reset()


class Endpointer:
    """
    Decide quando la frase post-trigger è finita, su base energia RMS.

    Logica: il rumore di fondo viene stimato dai primi frame; si considera
    "parlato" un frame con RMS > max(min_rms, ratio × rumore). La frase è
    conclusa quando, DOPO aver sentito parlato, passano `silence_stop_s`
    di frame consecutivi sotto soglia — oppure al raggiungimento di
    `max_command_s` totali (parlato o no).
    """

    def __init__(
        self,
        *,
        silence_stop_s: Optional[float] = None,
        max_command_s:  Optional[float] = None,
        min_rms:        float = 250.0,
        noise_ratio:    float = 2.5,
        abs_speech_rms: float = 1500.0,
        frame_s:        float = FRAME_SAMPLES / SAMPLE_RATE,
    ) -> None:
        cfg = settings.wake_word
        self._silence_stop_s = silence_stop_s if silence_stop_s is not None \
                               else cfg.silence_stop_s
        self._max_command_s  = max_command_s if max_command_s is not None \
                               else cfg.max_command_s
        self._min_rms        = min_rms
        self._noise_ratio    = noise_ratio
        # Tetto assoluto: sopra questa RMS è SEMPRE parlato, qualunque sia il
        # rumore stimato. Cruciale quando l'utente parla subito dopo il
        # trigger: senza, il primo frame di voce gonfierebbe la stima del
        # rumore e la soglia adattiva non verrebbe mai superata.
        self._abs_speech_rms = abs_speech_rms
        self._frame_s        = frame_s

        self._elapsed_s     = 0.0
        self._silence_run_s = 0.0
        self._speech_seen   = False
        # Rumore di fondo = minimo scorrevole delle RMS viste (converge al
        # silenzio reale anche se i primi frame contengono voce).
        self._noise_floor: float = float("inf")

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        if len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(frame.astype(np.float64)))))

    def update(self, frame: np.ndarray) -> bool:
        """Consuma un frame. True = la frase è finita, ferma la registrazione."""
        self._elapsed_s += self._frame_s
        if self._elapsed_s >= self._max_command_s:
            return True

        rms = self._rms(frame)
        self._noise_floor = min(self._noise_floor, rms)

        threshold = max(self._min_rms, self._noise_floor * self._noise_ratio)
        threshold = min(threshold, self._abs_speech_rms)
        if rms > threshold:
            self._speech_seen   = True
            self._silence_run_s = 0.0
            return False

        if self._speech_seen:
            self._silence_run_s += self._frame_s
            if self._silence_run_s >= self._silence_stop_s:
                return True
        return False
