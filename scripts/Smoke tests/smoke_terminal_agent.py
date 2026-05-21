"""
scripts/Smoke tests/smoke_terminal_agent.py
Smoke test manuale per modules/terminal_agent.

Richiede:
    - Ollama attivo su settings.ollama.base_url
    - SearXNG opzionale (per i test di ricerca web). Se assente, i test
      della ricerca falliscono in modo controllato senza crashare.

Esecuzione:
    cd ~/assistant
    venv-runtime/bin/python "scripts/Smoke tests/smoke_terminal_agent.py"

Per eseguire SOLO alcuni gruppi di test:
    TA_ONLY=propose venv-runtime/bin/python "scripts/Smoke tests/smoke_terminal_agent.py"
    TA_ONLY=execute,run venv-runtime/bin/python "scripts/Smoke tests/smoke_terminal_agent.py"

Gruppi disponibili: classify, propose, execute, run, analyze, search

I test scrivono solo in una cwd temporanea sotto /tmp, niente tocca la $HOME.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from modules.terminal_agent import (
    AgentTurn,
    CommandProposal,
    CommandResult,
    RiskLevel,
    TerminalAgent,
)
from modules.terminal_agent.base_terminal_agent import _classify


# ---------------------------------------------------------------------------
# Helpers di stampa
# ---------------------------------------------------------------------------

W = 72  # larghezza header


def header(title: str) -> None:
    print()
    print("=" * W)
    print(f"  {title}")
    print("=" * W)


def section(title: str) -> None:
    print()
    print(f"--- {title} ".ljust(W, "-"))


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


def fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def kv(k: str, v: object) -> None:
    print(f"    {k:<18} {v}")


def show_proposal(p: CommandProposal) -> None:
    kv("command",   p.command)
    kv("rationale", (p.rationale or "")[:120])
    kv("risk",      p.risk_level.value)
    kv("confirm?",  p.needs_confirmation)
    kv("cwd",       p.cwd)
    kv("search?",   p.search_used)
    if p.sources:
        kv("sources",   f"{len(p.sources)} url")
        for s in p.sources[:3]:
            print(f"                       - {s}")


def show_result(r: CommandResult) -> None:
    kv("exit_code",   r.exit_code)
    kv("duration",    f"{r.duration_ms:.0f} ms")
    kv("cwd_after",   r.cwd_after)
    kv("truncated",   r.truncated)
    if r.stdout:
        kv("stdout",   "(primi 200 char)")
        for ln in r.stdout[:200].splitlines()[:6]:
            print(f"                       │ {ln}")
    if r.stderr:
        kv("stderr",   "(primi 200 char)")
        for ln in r.stderr[:200].splitlines()[:6]:
            print(f"                       │ {ln}")


def show_turn(t: AgentTurn) -> None:
    if t.error:
        fail(f"errore: {t.error}")
    if t.proposal:
        section("proposal")
        show_proposal(t.proposal)
    if t.skipped:
        warn("turno SKIPPED — richiede conferma, auto_confirm=False")
        return
    if t.result:
        section("result")
        show_result(t.result)
    if t.analysis:
        section("analysis (LLM)")
        for ln in t.analysis.splitlines():
            print(f"    {ln}")


# ---------------------------------------------------------------------------
# Test 1 — classificazione del rischio (offline, niente LLM)
# ---------------------------------------------------------------------------

def test_classify() -> None:
    header("1. Classificazione rischio (offline)")
    cases = [
        ("ls -la",                 RiskLevel.SAFE),
        ("git status",             RiskLevel.SAFE),
        ("find . -name '*.py'",    RiskLevel.SAFE),
        ("pwd && whoami",          RiskLevel.SAFE),
        ("mkdir foo",              RiskLevel.MODERATE),
        ("git commit -m 'x'",      RiskLevel.MODERATE),
        ("echo hi > /tmp/f",       RiskLevel.MODERATE),
        ("rm file.txt",            RiskLevel.DANGEROUS),
        ("chmod 777 file",         RiskLevel.DANGEROUS),
        ("rm -rf /",               RiskLevel.BLOCKED),
        ("sudo apt update",        RiskLevel.BLOCKED),
        (":(){ :|:& };:",          RiskLevel.BLOCKED),
    ]
    n_ok = 0
    for cmd, expected in cases:
        got = _classify(cmd, allow_sudo=False)
        mark = "✓" if got == expected else "✗"
        if got == expected:
            n_ok += 1
        print(f"    {mark}  [{got.value:<9}] {cmd}")
    print()
    if n_ok == len(cases):
        ok(f"{n_ok}/{len(cases)} classificazioni corrette")
    else:
        fail(f"{n_ok}/{len(cases)} classificazioni corrette")


# ---------------------------------------------------------------------------
# Test 2 — propose() contro LLM reale
# ---------------------------------------------------------------------------

async def test_propose(workdir: str) -> None:
    header("2. propose() — LLM propone il comando")

    requests = [
        "Mostrami i file in questa cartella",
        "Quanti file Python ci sono qui dentro? Conta solo i .py",
        "Dimmi quanto spazio occupa la mia cartella corrente",
    ]

    async with TerminalAgent(
        initial_cwd=workdir, safe_dirs=[workdir, "/tmp"],
    ) as ta:
        for i, req in enumerate(requests, 1):
            section(f"[{i}] richiesta: {req!r}")
            t0 = time.monotonic()
            try:
                prop = await ta.propose(req)
            except Exception as exc:
                fail(f"propose() fallito: {exc}")
                continue
            elapsed = time.monotonic() - t0
            kv("elapsed", f"{elapsed:.1f}s")
            show_proposal(prop)


# ---------------------------------------------------------------------------
# Test 3 — execute() su proposte costruite a mano
# ---------------------------------------------------------------------------

async def test_execute(workdir: str) -> None:
    header("3. execute() — esecuzione di comandi pre-costruiti")

    async with TerminalAgent(
        initial_cwd=workdir, safe_dirs=[workdir],
    ) as ta:
        # [3.1] comando safe semplice
        section("[3.1] echo (safe, auto-run)")
        p = CommandProposal(
            command="echo 'ciao dal terminale agente'",
            rationale="saluto", risk_level=RiskLevel.SAFE,
            needs_confirmation=False, cwd=workdir,
        )
        r = await ta.execute(p)
        show_result(r)

        # [3.2] cd persistente
        section("[3.2] cd in subdir e pwd")
        sub = Path(workdir) / "sandbox_dir"
        sub.mkdir(exist_ok=True)
        kv("before cwd", ta.cwd)
        p2 = CommandProposal(
            command=f"cd {sub.name} && pwd && touch marker.txt",
            rationale="test cwd", risk_level=RiskLevel.MODERATE,
            needs_confirmation=True, cwd=workdir,
        )
        r2 = await ta.execute(p2, confirmed=True)
        show_result(r2)
        kv("after cwd",  ta.cwd)
        kv("marker exists?", (sub / "marker.txt").exists())

        # [3.3] comando con stderr ed exit != 0
        section("[3.3] ls su path inesistente")
        p3 = CommandProposal(
            command="ls /this/does/not/exist/123",
            rationale="forza errore", risk_level=RiskLevel.SAFE,
            needs_confirmation=False, cwd=ta.cwd,
        )
        r3 = await ta.execute(p3)
        show_result(r3)

        # [3.4] tentativo di eseguire un BLOCKED
        section("[3.4] tentativo rm -rf / (deve essere bloccato)")
        p4 = CommandProposal(
            command="rm -rf /", rationale="cattiveria",
            risk_level=RiskLevel.BLOCKED, needs_confirmation=True,
            cwd=ta.cwd,
        )
        try:
            await ta.execute(p4, confirmed=True)
            fail("non ha bloccato! questo è un BUG di sicurezza")
        except ValueError as exc:
            ok(f"bloccato come previsto: {exc}")


# ---------------------------------------------------------------------------
# Test 4 — analyze() su un risultato reale
# ---------------------------------------------------------------------------

async def test_analyze(workdir: str) -> None:
    header("4. analyze() — l'LLM legge l'output e diagnostica")

    async with TerminalAgent(
        initial_cwd=workdir, safe_dirs=[workdir],
    ) as ta:
        # esegui un comando che fallisce in modo "interessante"
        section("[4.1] python -c con SyntaxError → analisi")
        p = CommandProposal(
            command="python3 -c 'print(hello'",
            rationale="syntax error apposta",
            risk_level=RiskLevel.MODERATE,
            needs_confirmation=True, cwd=workdir,
        )
        r = await ta.execute(p, confirmed=True)
        show_result(r)

        t0 = time.monotonic()
        analysis = await ta.analyze(
            r,
            user_request="esegui un piccolo snippet Python per testare",
        )
        elapsed = time.monotonic() - t0
        kv("analyze elapsed", f"{elapsed:.1f}s")
        section("analysis")
        for ln in analysis.splitlines():
            print(f"    {ln}")


# ---------------------------------------------------------------------------
# Test 5 — run() end-to-end (propose → execute → analyze)
# ---------------------------------------------------------------------------

async def test_run(workdir: str) -> None:
    header("5. run() — flusso completo end-to-end")

    async with TerminalAgent(
        initial_cwd=workdir, safe_dirs=[workdir, "/tmp"],
    ) as ta:
        # [5.1] flusso safe completo, niente conferma
        section("[5.1] richiesta semplice (safe, auto-run)")
        t0 = time.monotonic()
        turn = await ta.run("In che directory mi trovo?")
        kv("total elapsed", f"{time.monotonic() - t0:.1f}s")
        show_turn(turn)

        # [5.2] richiesta che (probabilmente) genera un comando moderate
        section("[5.2] richiesta che richiede conferma (auto_confirm=False)")
        turn2 = await ta.run(
            "Crea una cartella chiamata 'esperimento_smoke' qui",
            auto_confirm=False,
        )
        show_turn(turn2)

        # [5.3] stessa richiesta con auto_confirm=True
        section("[5.3] stessa richiesta con auto_confirm=True")
        turn3 = await ta.run(
            "Crea una cartella chiamata 'esperimento_smoke2' qui",
            auto_confirm=True,
        )
        show_turn(turn3)


# ---------------------------------------------------------------------------
# Test 6 — search() — solo se SearXNG è disponibile
# ---------------------------------------------------------------------------

async def test_search(workdir: str) -> None:
    header("6. ricerca web — richiede SearXNG")

    # health-check di SearXNG prima
    try:
        from modules.web_search import SearXNGClient  # type: ignore
        async with SearXNGClient() as ws:
            available = await ws.is_available()
    except Exception as exc:
        warn(f"impossibile testare SearXNG: {exc}")
        return

    if not available:
        warn("SearXNG offline — saltato. Avvialo con:")
        print(f"     docker compose -f docker/docker-compose.yml up -d searxng")
        return

    ok("SearXNG raggiungibile")

    async with TerminalAgent(
        initial_cwd=workdir, safe_dirs=[workdir],
    ) as ta:
        # richiesta che dovrebbe spingere l'LLM a cercare
        # (sintassi non triviale di un tool che potrebbe non conoscere bene)
        section("[6.1] richiesta che invita alla ricerca")
        t0 = time.monotonic()
        prop = await ta.propose(
            "Trova tutti i file modificati negli ultimi 7 giorni nella mia "
            "cartella corrente, esclusi quelli dentro .git, e mostra solo "
            "quelli più grandi di 1 MB",
        )
        kv("elapsed", f"{time.monotonic() - t0:.1f}s")
        show_proposal(prop)
        if prop.search_used:
            ok("l'LLM ha consultato il web prima di proporre")
        else:
            warn(
                "l'LLM non ha cercato — può essere ok se conosceva già la sintassi"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_GROUPS = {
    "classify": (test_classify, False),  # sync, no workdir
    "propose":  (test_propose,  True),
    "execute":  (test_execute,  True),
    "analyze":  (test_analyze,  True),
    "run":      (test_run,      True),
    "search":   (test_search,   True),
}


async def amain() -> int:
    only = os.environ.get("TA_ONLY", "").strip()
    groups = [g.strip() for g in only.split(",") if g.strip()] if only else list(ALL_GROUPS)

    invalid = [g for g in groups if g not in ALL_GROUPS]
    if invalid:
        print(f"Gruppi non validi: {invalid}")
        print(f"Disponibili: {list(ALL_GROUPS)}")
        return 1

    # working dir temporanea — tutto isolato in /tmp
    workdir = tempfile.mkdtemp(prefix="terminal_agent_smoke_")
    print(f"\n  → workdir temporanea: {workdir}\n")

    overall_start = time.monotonic()
    try:
        for g in groups:
            fn, needs_wd = ALL_GROUPS[g]
            try:
                if asyncio.iscoroutinefunction(fn):
                    if needs_wd:
                        await fn(workdir)
                    else:
                        await fn()
                else:
                    fn()
            except Exception as exc:
                fail(f"gruppo '{g}' è esploso: {exc}")
                import traceback
                traceback.print_exc()
    finally:
        # pulizia
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass

    header(f"FINE — tempo totale: {time.monotonic() - overall_start:.1f}s")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
