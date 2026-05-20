"""
scripts/smoke_personality.py
Test interattivo del modulo modules/personality.
Carica i profili reali, li ispeziona e mostra il cambio di personalità.

Uso:
    venv-runtime/bin/python scripts/smoke_personality.py
    venv-runtime/bin/python scripts/smoke_personality.py --profile dev
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.personality import PersonalityManager, PersonalityProfile


def _print_profile(p: PersonalityProfile, prefix: str = "") -> None:
    print(f"{prefix}  Nome:          {p.name}")
    print(f"{prefix}  Display name:  {p.display_name}")
    print(f"{prefix}  Lingua:        {p.language}")
    print(f"{prefix}  Voce TTS:      {p.tts_voice}")
    print(f"{prefix}  LoRA adapter:  {p.lora_adapter}")
    print(f"{prefix}  Tool abilitati: {p.allowed_tools or '(tutti)'}")
    print(f"{prefix}  System prompt ({len(p.system_prompt)} char):")
    for line in p.system_prompt.splitlines()[:4]:
        print(f"{prefix}    {line}")
    if p.system_prompt.count("\n") >= 4:
        print(f"{prefix}    ...")


async def main(initial_profile: str = "default") -> None:
    print("Caricamento profili di personalità...")
    t0 = time.time()

    async with PersonalityManager() as pm:
        elapsed = time.time() - t0
        profiles = pm.list_profiles()
        print(f"✓ {len(profiles)} profili caricati ({elapsed * 1000:.0f}ms)\n")

        # ------------------------------------------------------------------
        # Test 1 — elenco profili
        # ------------------------------------------------------------------
        print("[1] Profili disponibili:")
        for name in profiles:
            marker = " ← attivo" if name == pm.active.name else ""
            print(f"    • {name}{marker}")
        print()

        # ------------------------------------------------------------------
        # Test 2 — ispezione profilo default
        # ------------------------------------------------------------------
        print(f"[2] Profilo attivo: '{pm.active.name}'")
        _print_profile(pm.active, prefix="")
        print()

        # ------------------------------------------------------------------
        # Test 3 — cambio profilo
        # ------------------------------------------------------------------
        alt = initial_profile if initial_profile in profiles else next(
            (p for p in profiles if p != pm.active.name), None
        )
        if alt and alt != pm.active.name:
            print(f"[3] Switch → '{alt}'")
            pm.switch(alt)
            _print_profile(pm.active, prefix="")
            print()
        else:
            print(f"[3] Solo un profilo disponibile — skip switch\n")

        # ------------------------------------------------------------------
        # Test 4 — apply_to_context
        # ------------------------------------------------------------------
        from core.context import AssistantContext
        ctx = AssistantContext()
        pm.apply_to_context(ctx)
        print("[4] AssistantContext dopo apply_to_context():")
        print(f"    ctx.personality_name = '{ctx.personality_name}'")
        print(f"    ctx.system_prompt    = '{ctx.system_prompt[:80]}...'")
        print()

        # ------------------------------------------------------------------
        # Test 5 — apply_to_context non sovrascrive prompt esistente
        # ------------------------------------------------------------------
        ctx2 = AssistantContext()
        ctx2.system_prompt = "Prompt custom iniettato dall'orchestratore."
        pm.apply_to_context(ctx2)
        assert ctx2.system_prompt == "Prompt custom iniettato dall'orchestratore."
        print("[5] apply_to_context non sovrascrive system_prompt esistente: ✓")
        print()

        # ------------------------------------------------------------------
        # Test 6 — allows_tool
        # ------------------------------------------------------------------
        print("[6] Verifica allows_tool sul profilo attivo:")
        for tool in ["web_search", "code_execution", "file_write", "pc_control"]:
            allowed = pm.active.allows_tool(tool)
            symbol  = "✓" if allowed else "✗"
            print(f"    [{symbol}] {tool}")
        print()

        # ------------------------------------------------------------------
        # Test 7 — reload (pickup di nuovi file)
        # ------------------------------------------------------------------
        print("[7] Reload profili (simula aggiunta di nuovi YAML a runtime)...")
        before = set(pm.list_profiles())
        await pm.reload()
        after  = set(pm.list_profiles())
        added  = after - before
        print(f"    Profili prima: {sorted(before)}")
        print(f"    Profili dopo:  {sorted(after)}")
        print(f"    Nuovi aggiunti: {sorted(added) or 'nessuno'}")
        print()

        # ------------------------------------------------------------------
        # Riepilogo
        # ------------------------------------------------------------------
        print("─" * 52)
        print("✓ Smoke test completato — PersonalityManager operativo")
        print(f"  Profilo finale attivo: '{pm.active.name}'")


if __name__ == "__main__":
    profile = "default"
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith("--profile="):
            profile = arg.split("=", 1)[1]
        elif arg == "--profile" and i < len(sys.argv) - 1:
            profile = sys.argv[i + 1]

    try:
        asyncio.run(main(initial_profile=profile))
    except KeyboardInterrupt:
        print("\nInterrotto.")
