"""
scripts/smoke_stt.py
Test interattivo STT da microfono.
Parla, fai una pausa, e vedi la trascrizione in tempo reale.
Ctrl+C per uscire.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.stt import WhisperSTT, STTResult


async def on_transcript(result: STTResult) -> None:
    print(f"\n🎤 [{result.language}] {result.text}")
    print(f"   (VAD: {result.vad_kept_s:.1f}s | inference: {result.inference_ms:.0f}ms)")


async def main():
    print("Caricamento modelli...")
    async with WhisperSTT() as stt:
        print("✓ Pronto — parla! (Ctrl+C per uscire)\n")
        stop = asyncio.Event()
        try:
            await stt.stream_mic(on_transcript, stop_event=stop, chunk_ms=32)
        except KeyboardInterrupt:
            stop.set()
            print("\nArrestato.")


asyncio.run(main())
