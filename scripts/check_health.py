#!/usr/bin/env python3
"""
scripts/check_health.py
Verifica che tutti i componenti del sistema siano operativi.
Uso: python scripts/check_health.py  oppure: make check
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich import box as rich_box

console = Console()
results = []

def ok(cat, name, detail=""): results.append((cat, name, "✓", detail, "green"))
def fail(cat, name, detail=""): results.append((cat, name, "✗", detail, "red"))
def warn(cat, name, detail=""): results.append((cat, name, "⚠", detail, "yellow"))

import platform
py = platform.python_version()
(ok if tuple(int(x) for x in py.split(".")[:2]) >= (3,11) else fail)(
    "Python", f"Python {py}", "≥ 3.11 richiesto")

try:
    import torch
    if torch.cuda.is_available():
        gpus = [f"{torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory//1024**3}GB)"
                for i in range(torch.cuda.device_count())]
        ok("Hardware", "CUDA", " | ".join(gpus))
    else:
        warn("Hardware", "CUDA", "nessuna GPU — STT su CPU")
except ImportError:
    warn("Hardware", "PyTorch", "non installato")

for cat, mod, pkg in [
    ("STT","faster_whisper","faster-whisper"),
    ("STT","sounddevice","sounddevice"),
    ("Speaker","pyannote.audio","pyannote.audio"),  # type: ignore[import]
    ("TTS","soundfile","soundfile"),
    ("Memoria","chromadb","chromadb"),
    ("API","fastapi","fastapi"),
    ("Web","playwright","playwright"),
    ("File","unstructured","unstructured"),
    ("Config","pydantic_settings","pydantic-settings"),
    ("Log","loguru","loguru"),
]:
    try: __import__(mod); ok(cat, pkg)
    except ImportError: fail(cat, pkg, f"pip install {pkg}")

import httpx
for name, url, err_msg in [
    ("Ollama", "http://localhost:11434/api/tags", "esegui: systemctl start ollama"),
    ("SearXNG", "http://localhost:8080/search?q=test&format=json", "esegui: make setup-docker"),
    ("Open WebUI", "http://localhost:3000", "esegui: make setup-docker"),
]:
    try:
        r = httpx.get(url, timeout=3)
        if name == "Ollama":
            models = [m["name"] for m in r.json().get("models", [])]
            ok("Servizi", name, f"{len(models)} modelli" if models else "attivo, nessun modello")
        else:
            ok("Servizi", name, f"HTTP {r.status_code}")
    except Exception as e:
        warn("Servizi", name, err_msg)

root = Path(__file__).parent.parent
for f, desc in [
    (".env", "variabili ambiente"),
    ("config/settings.py", "configurazione"),
    ("config/personalities/default.yaml", "personalità default"),
    ("core/context.py", "contesto"),
    ("core/logger.py", "logger"),
]:
    (ok if (root/f).exists() else fail)("Files", f, desc)

table = Table(title="Health Check — Assistente AI Locale",
              box=rich_box.ROUNDED, show_header=True, header_style="bold")
table.add_column("Categoria", style="dim", width=14)
table.add_column("Componente", width=26)
table.add_column("", width=3)
table.add_column("Dettaglio", style="dim")

prev = ""
for cat, name, status, detail, color in results:
    table.add_row(cat if cat != prev else "", name,
                  f"[{color}]{status}[/{color}]", detail)
    prev = cat

console.print()
console.print(table)
ok_n = sum(1 for *_,s,_ in results if s=="✓")
fail_n = sum(1 for *_,s,_ in results if s=="✗")
warn_n = sum(1 for *_,s,_ in results if s=="⚠")
console.print(f"\n  [green]{ok_n} ok[/green]  [yellow]{warn_n} avvisi[/yellow]  [red]{fail_n} errori[/red]\n")
if fail_n > 0:
    console.print("[red]  ✗ Risolvi gli errori prima di avviare.[/red]\n"); sys.exit(1)
elif warn_n > 0:
    console.print("[yellow]  ⚠ Sistema avviabile, alcune funzionalità mancanti.[/yellow]\n")
else:
    console.print("[green]  ✓ Tutto OK — avvia con: make start[/green]\n")
