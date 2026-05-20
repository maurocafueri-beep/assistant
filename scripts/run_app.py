"""
scripts/run_app.py
Launcher per local-assistant come applicazione Ubuntu nativa.

Uso:
    venv-runtime/bin/python scripts/run_app.py
    venv-runtime/bin/python scripts/run_app.py --personality dev
    venv-runtime/bin/python scripts/run_app.py --ptt-key f4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Assicura che il progetto sia nel path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Required per QWebEngine su alcuni sistemi Linux prima di qualsiasi import Qt
import os
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu-sandbox")
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")


def main() -> None:
    p = argparse.ArgumentParser(description="local-assistant — App mode")
    p.add_argument("--personality", default=None,    help="Profilo personalità (default: default)")
    p.add_argument("--ptt-key",     default="space", help="Tasto PTT (space, f4, a, …)")
    args = p.parse_args()

    try:
        # Carica impostazioni salvate, CLI ha priorità
        import json
        from pathlib import Path
        cfg_file = Path.home() / ".config" / "local-assistant" / "ui-settings.json"
        cfg = {}
        if cfg_file.exists():
            try: cfg = json.loads(cfg_file.read_text())
            except: pass

        personality = args.personality or cfg.get("personality")
        ptt_key     = args.ptt_key if args.ptt_key != "space" else cfg.get("ptt_key", "space")

        from ui.app import run
        exit_code = run(
            personality = personality,
            ptt_key     = ptt_key,
        )
        sys.exit(exit_code)
    except ImportError as e:
        print(f"\n✗ Dipendenze mancanti: {e}")
        print("\nInstalla con:")
        print("  venv-runtime/bin/pip install PyQt6 PyQt6-WebEngine")
        print("\nOppure su Ubuntu (più veloce):")
        print("  sudo apt install python3-pyqt6 python3-pyqt6.qtwebengine")
        print("  # poi collega al venv:")
        print("  ln -sf /usr/lib/python3/dist-packages/PyQt6 venv-runtime/lib/python3.12/site-packages/")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Errore: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
