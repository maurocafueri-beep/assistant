"""
scripts/smoke_tts.py
Test interattivo TTS con Qwen3-TTS e voice cloning.
Verifica avvio server, sintesi, streaming, cambio profilo e salvataggio WAV.
Ctrl+C per uscire.

Uso:
    venv-runtime/bin/python scripts/smoke_tts.py
    venv-runtime/bin/python scripts/smoke_tts.py --no-play     # senza audio
    venv-runtime/bin/python scripts/smoke_tts.py --profile squib
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.tts import Qwen3TTS, TTSChunk


async def main(play_audio: bool = True, profile: str = "mercoledì") -> None:
    print(f"Avvio server Qwen3-TTS [profilo={profile}]...")
    t0 = time.time()

    async with Qwen3TTS(profile=profile) as tts:
        info = await tts.server_info()
        print(f"✓ Server pronto ({time.time() - t0:.1f}s)\n")
        print(f"  Profilo attivo:      {info['profile']}")
        print(f"  Profili disponibili: {', '.join(info['profiles'])}")
        print(f"  Modello:             {info['model']}")
        print(f"  Sample rate:         {tts.sample_rate} Hz\n")

        # ------------------------------------------------------------------
        # Test 1 — latenza sintesi singola frase
        # ------------------------------------------------------------------
        text1  = "Ciao, sono il tuo assistente AI locale."
        start  = time.time()
        result = await tts.synthesize(text1)

        print("[1] Sintesi singola frase")
        print(f"    Testo:        '{text1}'")
        print(f"    Durata audio: {result.duration_s:.2f}s")
        print(f"    Inference:    {result.inference_ms:.0f}ms  "
              f"(RTF: {result.inference_ms / 1000 / result.duration_s:.2f})")
        print(f"    Bytes:        {len(result.audio_bytes):,}")
        if play_audio:
            print("    ▶ Riproduzione...")
            await tts.play(result.audio_bytes)
        print()

        # ------------------------------------------------------------------
        # Test 2 — testo lungo, throughput
        # ------------------------------------------------------------------
        text2 = (
            "Qwen3-TTS è un modello text-to-speech con voice cloning integrato. "
            "Supporta oltre dieci lingue tra cui l'italiano e permette di clonare "
            "qualsiasi voce a partire da pochi secondi di audio di riferimento. "
            "Il modello da un virgola sette miliardi di parametri gira interamente "
            "in locale senza inviare dati a servizi esterni."
        )
        result2 = await tts.synthesize(text2)
        rtf     = result2.inference_ms / 1000 / result2.duration_s

        print(f"[2] Testo lungo ({len(text2)} caratteri)")
        print(f"    Durata audio: {result2.duration_s:.1f}s")
        print(f"    Inference:    {result2.inference_ms:.0f}ms  (RTF: {rtf:.2f})")
        if play_audio:
            print("    ▶ Riproduzione...")
            await tts.play(result2.audio_bytes)
        print()

        # ------------------------------------------------------------------
        # Test 3 — streaming frase per frase (TTFC critico per pipeline LLM)
        # ------------------------------------------------------------------
        text3 = (
            "Prima frase: il sistema è operativo. "
            "Seconda frase: lo streaming funziona correttamente. "
            "Terza frase: la latenza è nella norma."
        )
        print("[3] Streaming frase per frase")

        chunks_received: list[TTSChunk] = []
        first_chunk_time: float = 0.0
        stream_start = time.time()

        async def on_chunk(chunk: TTSChunk) -> None:
            nonlocal first_chunk_time
            if chunk.index == 0:
                first_chunk_time = time.time() - stream_start
            print(f"    Chunk [{chunk.index + 1}] '{chunk.text[:50]}' "
                  f"— {chunk.duration_s:.2f}s audio, {chunk.inference_ms:.0f}ms inference"
                  f"{'  ← ULTIMO' if chunk.is_last else ''}")
            chunks_received.append(chunk)
            if play_audio:
                await tts.play(chunk.audio_bytes)

        await tts.stream_sentences(text3, on_chunk)
        print(f"    Time-to-first-chunk: {first_chunk_time:.2f}s")
        print(f"    Chunk totali: {len(chunks_received)}")
        print()

        # ------------------------------------------------------------------
        # Test 4 — async generator
        # ------------------------------------------------------------------
        text4 = "Test con async generator. Funziona esattamente come il callback."
        print("[4] Async generator")
        count = 0
        async for chunk in tts.stream_sentences_gen(text4):
            count += 1
            print(f"    yield [{chunk.index + 1}] {chunk.duration_s:.2f}s")
            if play_audio:
                await tts.play(chunk.audio_bytes)
        print(f"    Chunk totali: {count}")
        print()

        # ------------------------------------------------------------------
        # Test 5 — cambio profilo vocale
        # ------------------------------------------------------------------
        profiles    = info["profiles"]
        alt_profile = next((p for p in profiles if p != profile), None)
        if alt_profile:
            import httpx
            print(f"[5] Cambio profilo: {profile} → {alt_profile}")
            async with httpx.AsyncClient() as c:
                r = await c.post(f"http://127.0.0.1:8765/switch/{alt_profile}")
                print(f"    Switch: {r.json()}")
            result5 = await tts.synthesize(f"Ciao, questo è il profilo {alt_profile}.")
            print(f"    Inference: {result5.inference_ms:.0f}ms, durata: {result5.duration_s:.2f}s")
            if play_audio:
                print("    ▶ Riproduzione...")
                await tts.play(result5.audio_bytes)
            async with httpx.AsyncClient() as c:
                await c.post(f"http://127.0.0.1:8765/switch/{profile}")
            print()

        # ------------------------------------------------------------------
        # Test 6 — salvataggio WAV
        # ------------------------------------------------------------------
        out_path = Path("data/tts_smoke_output.wav")
        text6    = "File WAV generato dallo smoke test di Qwen3-TTS."
        saved    = await tts.synthesize_to_file(text6, out_path)
        size_kb  = saved.stat().st_size / 1024
        print("[6] Salvataggio WAV")
        print(f"    File: {saved}  ({size_kb:.1f} KB)")
        print()

        # ------------------------------------------------------------------
        # Riepilogo
        # ------------------------------------------------------------------
        print("─" * 52)
        print("✓ Smoke test completato — Qwen3-TTS operativo")
        if not play_audio:
            print("  (audio non riprodotto — rimuovi --no-play per sentirlo)")


if __name__ == "__main__":
    play    = "--no-play" not in sys.argv
    profile = "mercoledì"
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith("--profile="):
            profile = arg.split("=", 1)[1]
        elif arg == "--profile" and i < len(sys.argv) - 1:
            profile = sys.argv[i + 1]

    try:
        asyncio.run(main(play_audio=play, profile=profile))
    except KeyboardInterrupt:
        print("\nInterrotto.")
