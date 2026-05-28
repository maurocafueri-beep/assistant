"""
scripts/run_ui.py
Launcher per l'interfaccia web di local-assistant.

Sequenza di avvio:
    1. Avvia uvicorn subito  → browser può connettersi immediatamente
    2. Carica l'assistente   → Whisper, TTS, ChromaDB (~30-60s)
    3. Avvia il loop PTT     → tutto operativo

Uso:
    venv-runtime/bin/python scripts/run_ui.py
    venv-runtime/bin/python scripts/run_ui.py --personality dev
    venv-runtime/bin/python scripts/run_ui.py --ptt-key f4
    venv-runtime/bin/python scripts/run_ui.py --no-browser
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn

from config.settings import settings
from core.logger import logger
from ui.bridge import UIBridge, WSManager
from ui.server import create_app


async def _main(personality: str | None, ptt_key: str, open_browser: bool) -> None:
    ws_mgr     = WSManager()
    app, state = create_app(ws_mgr)

    host = settings.api.host
    port = settings.api.port

    # ── 1. Avvia il server subito ─────────────────────────────────────────
    server = uvicorn.Server(uvicorn.Config(
        app,
        host      = host,
        port      = port,
        log_level = "warning",
        loop      = "none",
    ))
    server_task = asyncio.create_task(server.serve())

    # Breve pausa per garantire che uvicorn sia in ascolto prima di aprire il browser
    await asyncio.sleep(0.5)

    if open_browser:
        webbrowser.open(f"http://{host}:{port}")

    logger.info("run_ui | server avviato → http://{}:{}", host, port)

    # ── 2. Carica e avvia l'assistente ───────────────────────────────────
    ui_loop = UIBridge(
        ws_manager  = ws_mgr,
        ptt_key     = ptt_key,
        personality = personality,
    )
    state["loop"] = ui_loop

    # Notifica la UI che il caricamento è in corso
    await ws_mgr.broadcast({"type": "state", "value": "loading"})

    async with ui_loop:
        # Caricamento completato — invia lo stato iniziale a eventuali client già connessi
        await ws_mgr.broadcast({
            "type":    "init",
            "state":   ui_loop.state,
            "ptt_key": ui_loop.ptt_key,
            "session": ui_loop.session_id,
            "stats":   ui_loop.stats.to_log_dict(),
        })
        logger.info("run_ui | assistente pronto | PTT={}", ptt_key)

        # Warmup modelli (chat + embed) e TTS in background: scalda DOPO che
        # l'assistente è pronto, senza bloccare il loop. Il segnalino mostra
        # lo stato "warmup" mentre scalda. Riferimento tenuto per evitare il
        # GC del task. Best-effort (warmup() non solleva mai).
        def _warmup_done(t: "asyncio.Task") -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.warning("run_ui | warmup task: {}", exc)

        warmup_task = asyncio.create_task(ui_loop.warmup())
        warmup_task.add_done_callback(_warmup_done)

        # ── 3. Loop PTT + server in parallelo ─────────────────────────────
        try:
            await asyncio.gather(ui_loop.run(), server_task)
        except asyncio.CancelledError:
            pass

    server.should_exit = True


def main() -> None:
    p = argparse.ArgumentParser(description="local-assistant — UI mode")
    p.add_argument("--personality", default=None,    help="Profilo personalità")
    p.add_argument("--ptt-key",     default="space", help="Tasto PTT (space, f4, a, …)")
    p.add_argument("--no-browser",  action="store_true", help="Non aprire il browser")
    args = p.parse_args()

    try:
        asyncio.run(_main(args.personality, args.ptt_key, not args.no_browser))
    except KeyboardInterrupt:
        logger.info("run_ui | uscita")


if __name__ == "__main__":
    main()
