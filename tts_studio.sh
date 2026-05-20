#!/bin/bash
# tts_studio.sh — Avvia TTS Studio con un doppio click
# Posizionalo nella root del progetto: ~/assistant/tts_studio.sh

cd "$(dirname "$0")"

VENV_NVIDIA="$(pwd)/venv-tts/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="\
$VENV_NVIDIA/cublas/lib:\
$VENV_NVIDIA/cudnn/lib:\
$VENV_NVIDIA/cuda_runtime/lib:\
${LD_LIBRARY_PATH}"

echo "Avvio TTS Studio..."
venv-runtime/bin/python tools/tts_studio.py
