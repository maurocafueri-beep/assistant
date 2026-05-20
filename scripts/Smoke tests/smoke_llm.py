import asyncio
import time
from core.context import ModelRole
from modules.llm import OllamaClient, Message, Role

async def main():
    async with OllamaClient() as llm:
        # Test 1 — latenza breve
        start = time.time()
        resp = await llm.chat(
            [Message(Role.USER, "Rispondi solo: OK")],
            role=ModelRole.CHAT,
        )
        print(f"[1] Latenza breve: {time.time()-start:.1f}s | token: {resp.tokens}")

        # Test 2 — throughput
        start = time.time()
        resp = await llm.chat(
            [Message(Role.USER, "Scrivi una lista di 20 animali con una curiosità per ognuno.")],
            role=ModelRole.CHAT,
        )
        elapsed = time.time() - start
        tps = resp.completion_tokens / elapsed
        print(f"[2] Throughput: {tps:.1f} tok/s | {resp.completion_tokens} token in {elapsed:.1f}s")

        # Test 3 — TTFC streaming (critico per TTS)
        start = time.time()
        first_chunk = True
        full_response = []
        async for chunk in llm.stream(
            [Message(Role.USER, "Raccontami una barzelletta.")],
            role=ModelRole.CHAT,
        ):
            if first_chunk:
                print(f"[3] Time-to-first-chunk: {time.time()-start:.2f}s")
                first_chunk = False
            full_response.append(chunk)
        print(f"[3] Risposta completa: {''.join(full_response)[:120]}...")

        # Test 4 — italiano (lingua principale del progetto)
        start = time.time()
        resp = await llm.chat(
            [Message(Role.USER, "Chi sei e cosa sai fare? Rispondi in massimo 3 frasi.")],
            role=ModelRole.CHAT,
            system="Sei un assistente AI locale. Parli sempre in italiano.",
        )
        print(f"[4] Italiano: {resp.content[:150]}...")
        print(f"[4] Tempo: {time.time()-start:.1f}s")

asyncio.run(main())
