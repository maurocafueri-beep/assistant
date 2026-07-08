"""
modules/wake_word
=================
Rilevamento della parola di attivazione (hands-free) con openWakeWord.

    from modules.wake_word import WakeWordDetector, Endpointer

Il detector gira su CPU (ONNX); il voice loop lo usa come produttore di
turni accanto al PTT. Vedi base_wake_word.py per formato audio e uso.
"""

from modules.wake_word.base_wake_word import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    Endpointer,
    WakeWordDetector,
)

__all__ = ["WakeWordDetector", "Endpointer", "SAMPLE_RATE", "FRAME_SAMPLES"]
