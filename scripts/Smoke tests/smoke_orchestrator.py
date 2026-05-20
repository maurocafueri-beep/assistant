"""
scripts/smoke_orchestrator.py
Test interattivo dell'orchestratore completo.

Richiede: Ollama attivo con il modello chat configurato.
TTS e STT sono disabilitati di default (usa --tts / --stt per abilitarli).

Uso:
    venv-runtime/bin/python scripts/smoke_orchestrator.py
    venv-runtime/bin/python scripts/smoke_orchestrator.py --personality dev
    venv-runtime/bin/python scripts/smoke_orchestrator.py --tts --stt
    venv-runtime/bin/python scripts/smoke_orchestrator.py --no-memory
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.context import AssistantContext, InputMode, ModelRole, OutputMode
from core.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Formattazione
# ---------------------------------------------------------------------------

def _sep(char: str = "─", n: int = 60) -> None:
    print(char * n)


def _header(title: str) -> None:
    _sep("═")
    print(f"  {title}")
    _sep("═")


def _section(title: str) -> None:
    print()
    _sep()
    print(f"  {title}")
    _sep()


# ---------------------------------------------------------------------------
# Scenari di test
# ---------------------------------------------------------------------------

TURNS = [
    ("sess-a", "Chi sei e cosa sai fare?"),
    ("sess-a", "Puoi ricordarti il mio nome se te lo dico?"),
    ("sess-a", "Mi chiamo Mauro, sono un ingegnere software."),
    ("sess-a", "Come mi chiamo?"),
    ("sess-b", "Questa è una sessione separata. Chi sei?"),
]


async def smoke_text_turns(orch: Orchestrator) -> None:
    _section("1. TURNI DI TESTO (streaming)")

    for session_id, user_text in TURNS:
        ctx = AssistantContext(
            user_text=user_text,
            session_id=session_id,
            input_mode=InputMode.TEXT,
            output_mode=OutputMode.TEXT,
            model_role=ModelRole.CHAT,
        )

        print(f"\n[{session_id}] Utente: {user_text}")
        print(f"[{session_id}] Assistente: ", end="", flush=True)

        t0     = time.monotonic()
        chunks = []
        async for chunk in orch.turn(ctx):
            print(chunk, end="", flush=True)
            chunks.append(chunk)

        elapsed = (time.monotonic() - t0) * 1000
        print()  # a capo dopo lo streaming

        if ctx.error:
            print(f"  ⚠ ERRORE: {ctx.error}")
        else:
            print(
                f"  ✓ {len(''.join(chunks))} caratteri | "
                f"llm={ctx.timings.get('llm', 0):.0f}ms | "
                f"mem={ctx.timings.get('memory', 0):.0f}ms | "
                f"tot={elapsed:.0f}ms"
            )
            if ctx.retrieved_memories:
                print(f"  📚 {len(ctx.retrieved_memories)} chunk di memoria recuperati")


async def smoke_turn_sync(orch: Orchestrator) -> None:
    _section("2. TURN_SYNC (risposta completa)")

    ctx = AssistantContext(
        user_text="Riassumi cosa sai di me finora.",
        session_id="sess-a",
        input_mode=InputMode.TEXT,
        output_mode=OutputMode.TEXT,
        model_role=ModelRole.CHAT,
    )

    t0     = time.monotonic()
    result = await orch.turn_sync(ctx)
    elapsed = (time.monotonic() - t0) * 1000

    if result.error:
        print(f"  ⚠ ERRORE: {result.error}")
    else:
        print(f"Risposta ({len(result.assistant_text)} car, {elapsed:.0f}ms):")
        print(result.assistant_text[:300])
        if len(result.assistant_text) > 300:
            print("  [...]")


async def smoke_add_memory(orch: Orchestrator) -> None:
    _section("3. ADD_MEMORY manuale")

    facts = [
        ("Il progetto si chiama local-assistant.", {"source": "smoke", "type": "fact"}),
        ("L'assistente usa Ollama come backend LLM.",   {"source": "smoke", "type": "fact"}),
        ("La persistenza vettoriale è gestita da ChromaDB.", {"source": "smoke", "type": "fact"}),
    ]

    for text, meta in facts:
        result = await orch.add_memory(text, meta)
        if result:
            print(f"  ✓ Salvato: '{text[:50]}' → id={result.chunk_id[:8]}…")
        else:
            print(f"  ✗ Memoria non disponibile per: '{text[:50]}'")


async def smoke_personality(orch: Orchestrator, personalities: list[str]) -> None:
    _section("4. SWITCH PERSONALITÀ")

    for pname in personalities:
        try:
            orch.switch_personality(pname)
            ctx = AssistantContext(
                user_text="Presentati brevemente.",
                session_id=f"sess-{pname}",
                input_mode=InputMode.TEXT,
                output_mode=OutputMode.TEXT,
            )
            result = await orch.turn_sync(ctx)
            print(f"  [{pname}] {result.assistant_text[:120]}")
        except KeyError as e:
            print(f"  ⚠ Profilo '{pname}' non trovato: {e}")


async def smoke_session_isolation(orch: Orchestrator) -> None:
    _section("5. ISOLAMENTO SESSIONI")

    # sessB non deve avere memoria di sessA
    ctx = AssistantContext(
        user_text="Cosa sai di Mauro?",
        session_id="sess-b",
        input_mode=InputMode.TEXT,
        output_mode=OutputMode.TEXT,
    )
    result = await orch.turn_sync(ctx)
    history_b = orch._session_histories.get("sess-b", [])
    history_a = orch._session_histories.get("sess-a", [])

    print(f"  sessA: {len(history_a)} messaggi in storia")
    print(f"  sessB: {len(history_b)} messaggi in storia")
    print(f"  Le sessioni sono separate: {history_a is not history_b}")

    # Clear sessione
    orch.clear_session("sess-b")
    assert "sess-b" not in orch._session_histories
    print("  ✓ clear_session() funziona correttamente")


async def smoke_status(orch: Orchestrator) -> None:
    _section("6. STATUS")
    s = orch.status
    print(f"  llm:         {'✓' if s.llm_ok else '✗'}")
    print(f"  memory:      {'✓' if s.memory_ok else '✗'}")
    print(f"  personality: {'✓' if s.personality_ok else '✗'}")
    print(f"  stt:         {'✓' if s.stt_ok else '✗'}")
    print(f"  tts:         {'✓' if s.tts_ok else '✗'}")
    print(f"  sessioni:    {s.active_sessions}")
    print(f"  repr:        {orch!r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    _header("smoke_orchestrator.py — test integrazione orchestratore")

    print(f"  personality : {args.personality}")
    print(f"  enable_tts  : {args.tts}")
    print(f"  enable_stt  : {args.stt}")
    print(f"  enable_memory: {not args.no_memory}")

    t_start = time.monotonic()

    async with Orchestrator(
        personality  = args.personality,
        enable_tts   = args.tts,
        enable_stt   = args.stt,
    ) as orch:
        # Disabilita memoria se richiesto
        if args.no_memory:
            orch._memory = None

        await smoke_status(orch)
        await smoke_add_memory(orch)
        await smoke_text_turns(orch)
        await smoke_turn_sync(orch)
        await smoke_personality(orch, ["default", "dev"])
        await smoke_session_isolation(orch)

    _sep("═")
    total = (time.monotonic() - t_start)
    print(f"  ✅ Smoke completato in {total:.1f}s")
    _sep("═")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test orchestratore")
    parser.add_argument(
        "--personality", default="default",
        help="Nome del profilo personalità da attivare (default: default)",
    )
    parser.add_argument(
        "--tts", action="store_true",
        help="Abilita TTS (richiede server Qwen3-TTS attivo)",
    )
    parser.add_argument(
        "--stt", action="store_true",
        help="Abilita STT (richiede Whisper e GPU)",
    )
    parser.add_argument(
        "--no-memory", action="store_true",
        help="Disabilita ChromaDB/RAG per questo smoke",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n  Interrotto dall'utente.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n  ❌ Errore fatale: {exc}")
        raise
