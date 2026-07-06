#!/usr/bin/env bash
# scripts/launch_app.sh
# Launcher per local-assistant quando parte dal menu applicazioni (non da un
# terminale già dentro la sessione). Garantisce l'ambiente minimo che l'app si
# aspetta e poi passa a run_app.py.
#
# Perché serve un wrapper:
#   - ui/bridge.py e core/voice_loop.py fanno `from pynput import keyboard`, che
#     carica il backend X di Xlib: senza DISPLAY e senza un file XAUTHORITY
#     leggibile crasha. Su Hyprland/Wayland puntiamo a Xwayland (:1) e a un
#     ~/.Xauthority vuoto (Xwayland accetta la connessione locale senza cookie).
#   - il file .desktop non può fare `cd`; qui ci spostiamo nella root del progetto.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# X per pynput (usa quello ereditato dalla sessione se c'è, altrimenti fallback)
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
[ -f "$XAUTHORITY" ] || touch "$XAUTHORITY"

exec venv-runtime/bin/python scripts/run_app.py "$@"
