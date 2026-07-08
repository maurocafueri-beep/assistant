"""
tests/test_wake_word.py
WakeWordDetector (soglia, cooldown, reset, risoluzione chiave) ed Endpointer
(endpointing a energia della frase post-trigger). Il modello openwakeword è
mockato: niente ONNX nei test.
"""
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from modules.wake_word import FRAME_SAMPLES, SAMPLE_RATE, Endpointer, WakeWordDetector


def _frame(amplitude: int = 0) -> np.ndarray:
    if amplitude == 0:
        return np.zeros(FRAME_SAMPLES, dtype=np.int16)
    rng = np.random.default_rng(42)
    return (rng.standard_normal(FRAME_SAMPLES) * amplitude).astype(np.int16)


def _det(score: float, threshold: float = 0.5, cooldown_s: float = 2.0) -> WakeWordDetector:
    det = WakeWordDetector(model="hey_jarvis", threshold=threshold, cooldown_s=cooldown_s)
    oww = MagicMock()
    oww.models = {"hey_jarvis": object()}
    oww.predict = MagicMock(return_value={"hey_jarvis": score})
    det._oww = oww
    det._key = "hey_jarvis"
    return det


class TestWakeWordDetector:
    def test_non_caricato_solleva(self):
        det = WakeWordDetector(model="hey_jarvis")
        with pytest.raises(RuntimeError, match="non caricato"):
            det.process(_frame())

    def test_trigger_sopra_soglia(self):
        det = _det(score=0.9)
        assert det.process(_frame()) is True
        assert det.last_score == pytest.approx(0.9)

    def test_niente_trigger_sotto_soglia(self):
        det = _det(score=0.3)
        assert det.process(_frame()) is False

    def test_cooldown_sopprime_doppio_trigger(self):
        det = _det(score=0.9, cooldown_s=60.0)
        assert det.process(_frame()) is True
        assert det.process(_frame()) is False   # entro il cooldown

    def test_cooldown_scaduto_ritriggera(self):
        det = _det(score=0.9, cooldown_s=0.01)
        assert det.process(_frame()) is True
        time.sleep(0.02)
        assert det.process(_frame()) is True

    def test_reset_inoltra_al_modello(self):
        det = _det(score=0.0)
        det.reset()
        det._oww.reset.assert_called_once()

    def test_chiave_score_mancante_da_zero(self):
        det = _det(score=0.9)
        det._oww.predict = MagicMock(return_value={"altro_modello": 0.9})
        assert det.process(_frame()) is False
        assert det.last_score == 0.0


class TestEndpointer:
    def _ep(self, **kw):
        kw.setdefault("silence_stop_s", 0.4)
        kw.setdefault("max_command_s", 5.0)
        return Endpointer(**kw)

    def test_silenzio_iniziale_non_ferma(self):
        # Senza parlato non deve chiudere (aspetta fino a max_command_s).
        ep = self._ep()
        for _ in range(20):                      # 1.6s di silenzio
            assert ep.update(_frame(0)) is False

    def test_parla_poi_silenzio_ferma(self):
        ep = self._ep()
        for _ in range(5):
            assert ep.update(_frame(3000)) is False   # parlato
        done = False
        for _ in range(10):                           # 0.8s di silenzio
            if ep.update(_frame(0)):
                done = True
                break
        assert done

    def test_pausa_breve_non_ferma(self):
        ep = self._ep(silence_stop_s=1.0)
        for _ in range(5):
            ep.update(_frame(3000))
        for _ in range(3):                            # ~0.24s di pausa
            assert ep.update(_frame(0)) is False
        assert ep.update(_frame(3000)) is False       # riprende a parlare

    def test_durata_massima_ferma_comunque(self):
        ep = self._ep(max_command_s=0.5)
        done = False
        for _ in range(10):                           # 0.8s totali
            if ep.update(_frame(3000)):               # sempre parlato
                done = True
                break
        assert done

    def test_rumore_di_fondo_alza_la_soglia(self):
        # Con rumore costante ad ampiezza media, il parlato deve superare
        # la soglia adattiva ma il rumore stesso no.
        ep = self._ep()
        for _ in range(4):
            ep.update(_frame(400))                    # calibrazione rumore
        for _ in range(5):
            ep.update(_frame(5000))                   # parlato netto
        done = False
        for _ in range(10):
            if ep.update(_frame(400)):                # torna solo rumore
                done = True
                break
        assert done


class TestWakeWordSettings:
    def test_defaults(self):
        from config.settings import settings
        cfg = settings.wake_word
        assert cfg.model == "hey_jarvis"
        assert 0.0 <= cfg.threshold <= 1.0
        assert cfg.cooldown_s >= 0.0
        assert cfg.silence_stop_s >= 0.2
        assert cfg.max_command_s >= 1.0

    def test_frame_costanti(self):
        assert SAMPLE_RATE == 16_000
        assert FRAME_SAMPLES == 1_280
