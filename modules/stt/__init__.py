"""
modules/stt/__init__.py
Espone l'API pubblica del modulo STT.

    from modules.stt import WhisperSTT, STTResult, STTSegment, SAMPLE_RATE
"""
from modules.stt.base_stt import (
    WhisperSTT,
    STTResult,
    STTSegment,
    SAMPLE_RATE,
)

__all__ = ["WhisperSTT", "STTResult", "STTSegment", "SAMPLE_RATE"]
