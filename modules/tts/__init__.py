"""
modules/tts
===========
Client Qwen3-TTS con voice cloning per tutto il progetto.

Importare sempre da qui:

    from modules.tts import Qwen3TTS, TTSResult, TTSChunk
"""

from .base_tts import Qwen3TTS, TTSChunk, TTSResult

__all__ = [
    "Qwen3TTS",
    "TTSResult",
    "TTSChunk",
]
