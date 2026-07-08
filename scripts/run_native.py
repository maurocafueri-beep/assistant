"""
scripts/run_native.py
Launcher della UI nativa Qt Quick (QML) di local-assistant.

A differenza di run_app.py (webview QWebEngine + FastAPI) qui non c'è
alcun server HTTP: un solo processo, backend in-process.

Uso:
    venv-runtime/bin/python scripts/run_native.py
    venv-runtime/bin/python scripts/run_native.py --personality dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    p = argparse.ArgumentParser(description="local-assistant — UI nativa Qt Quick")
    p.add_argument("--personality", default=None,    help="Profilo personalità")
    p.add_argument("--ptt-key",     default="space", help="Tasto PTT (space, f4, …)")
    args = p.parse_args()

    # Impostazioni UI salvate: la CLI ha priorità (stessa logica di run_app).
    cfg_file = Path.home() / ".config" / "local-assistant" / "ui-settings.json"
    cfg: dict = {}
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text())
        except Exception:
            pass

    personality = args.personality or cfg.get("personality")
    ptt_key = args.ptt_key if args.ptt_key != "space" else cfg.get("ptt_key", "space")

    from ui.native_app import run
    sys.exit(run(personality=personality, ptt_key=ptt_key))


if __name__ == "__main__":
    main()
