#!/bin/bash
# tts_studio.sh — Avvia TTS Studio con un doppio click
# Posizionalo nella root del progetto: ~/assistant/tts_studio.sh

cd "$(dirname "$0")"

# Lo Studio è un orchestratore web leggero (fastapi/httpx/uvicorn) e gira in
# venv-runtime. NON carica CUDA: il vero server TTS è lanciato come
# sottoprocesso con venv-tts/bin/python da tools/tts_studio.py, che imposta lì
# LD_LIBRARY_PATH sulle lib CUDA di venv-tts. Qui non serve esportarle.
echo "Avvio TTS Studio..."
venv-runtime/bin/python tools/tts_studio.py
