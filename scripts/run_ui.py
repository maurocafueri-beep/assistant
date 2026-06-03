"""
scripts/run_ui.py
Launcher headless per l'interfaccia web di local-assistant (dev, senza
finestra nativa Qt — per quella usare scripts/run_app.py).

La sequenza di avvio completa (server uvicorn + UIBridge + TerminalBridge +
restore sessioni/impostazioni + warmup + cleanup) vive in ui.bootstrap.serve(),
condivisa con l'app Qt: qui passiamo solo l'hook che apre il browser quando il
server è pronto.

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

from config.settings import settings
from core.logger import logger
from ui.bootstrap import serve


async def _main(personality: str | None, ptt_key: str, open_browser: bool) -> None:
    host = settings.api.host
    port = settings.api.port

    def _on_ready() -> None:
        if open_browser:
            webbrowser.open(f"http://{host}:{port}")
        logger.info("run_ui | server pronto | PTT={}", ptt_key)

    await serve(
        personality     = personality,
        ptt_key         = ptt_key,
        on_server_ready = _on_ready,
    )


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
