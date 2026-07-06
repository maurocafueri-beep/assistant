#!/usr/bin/env bash
# scripts/setup_arch.sh
# ─────────────────────────────────────────────────────────────────────────────
# Setup di local-assistant su Arch Linux a partire dalla cartella del progetto
# appena trasferita da Ubuntu.
#
# Idempotente: ogni step verifica lo stato prima di agire, puoi rilanciarlo
# tutte le volte che vuoi. Va eseguito come utente normale (NON root); chiede
# `sudo` solo dove serve davvero (pacman, gruppi, systemd, docker).
#
# Step coperti:
#   1.  pacman -Syu base e dipendenze di sistema
#   2.  NVIDIA + CUDA (opzionale, gated su NVIDIA_INSTALL=1)
#   3.  Ollama (cuda se NVIDIA disponibile) + drop-in systemd opzionale
#   4.  Pull dei modelli usati dal progetto
#   5.  Ricreazione di venv-runtime + venv-training da requirements-*.txt
#   6.  Aggiunta dell'utente al gruppo `input` (per il PTT via pynput)
#   7.  Docker + abilitazione servizio (SearXNG via docker-compose, on-demand)
#   8.  Note finali su venv-tts e ripristino delle impostazioni UI
#
# Uso:
#   bash scripts/setup_arch.sh             # interattivo, chiede conferme
#   ASSUME_YES=1 bash scripts/setup_arch.sh  # tutto sì, niente prompt
#   NVIDIA_INSTALL=1 bash scripts/setup_arch.sh   # installa anche driver+CUDA
#
# Personalizza in cima al file la lista MODELS / pacchetti se serve.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configurazione ─────────────────────────────────────────────────────────
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY_BIN="${PY_BIN:-python}"     # python di sistema da usare per i venv

# Modelli Ollama da pre-scaricare. Modifica liberamente.
# Tenere allineato a .env (OLLAMA_CHAT_MODEL / OLLAMA_EMBED_MODEL /
# OLLAMA_VISION_MODEL): se qui manca il modello configurato, i turni chat
# e i test d'integrazione falliscono con 404 da /api/chat.
MODELS=(
    "VladimirGav/gemma4-26b-16GB-VRAM-Uncensored:latest"
    "gemma4:12b"
    "nomic-embed-text"
    "qwen3-vl:8b"
)

# Pacchetti pacman richiesti (system-wide, non vivono nei venv).
PACMAN_PKGS=(
    base-devel git curl wget unzip
    python python-pip python-virtualenv
    ffmpeg sox libsndfile portaudio
    qt6-base qt6-webengine qt6-multimedia
    polkit pipewire-pulse alsa-utils
)

# Pacchetti NVIDIA (gated su NVIDIA_INSTALL=1)
NVIDIA_PKGS=(nvidia nvidia-utils cuda cudnn)

# Drop-in systemd per Ollama (KV cache q8 + flash attention).
ENABLE_OLLAMA_TUNING="${ENABLE_OLLAMA_TUNING:-1}"

# ── Helpers ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'; BLU='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${BLU}[setup]${NC} $*"; }
ok()   { echo -e "${GRN}  ✓${NC} $*"; }
warn() { echo -e "${YLW}  ⚠${NC} $*"; }
err()  { echo -e "${RED}  ✗${NC} $*" >&2; }
section() { echo; echo -e "${BLU}══ $* ══${NC}"; }

confirm() {
    [[ "${ASSUME_YES:-0}" == "1" ]] && return 0
    read -r -p "$1 [y/N] " ans
    [[ "$ans" =~ ^[yYsS]$ ]]
}

require_not_root() {
    if [[ $EUID -eq 0 ]]; then
        err "Esegui questo script come utente normale, non root."
        err "Verrà chiesto sudo dove serve."
        exit 1
    fi
}

# ── 0. Sanity check ────────────────────────────────────────────────────────
require_not_root

if [[ ! -f "${PROJECT_DIR}/pyproject.toml" || ! -d "${PROJECT_DIR}/config" ]]; then
    err "PROJECT_DIR='${PROJECT_DIR}' non sembra la root del progetto."
    exit 1
fi
log "Progetto: ${PROJECT_DIR}"
log "Utente:   ${USER}  (UID ${EUID})"

if ! command -v pacman >/dev/null; then
    err "pacman non trovato. Questo script è per Arch Linux."
    exit 1
fi

# ── 1. Pacchetti di sistema ────────────────────────────────────────────────
section "1/8  Pacchetti di sistema"
if confirm "Aggiorno il sistema e installo ${#PACMAN_PKGS[@]} pacchetti?"; then
    sudo pacman -Syu --needed --noconfirm "${PACMAN_PKGS[@]}"
    ok "Pacchetti base installati"
else
    warn "Salto installazione pacman (potresti avere problemi a seguire)."
fi

# ── 2. NVIDIA + CUDA (opzionale) ───────────────────────────────────────────
section "2/8  NVIDIA + CUDA"
if [[ "${NVIDIA_INSTALL:-0}" == "1" ]]; then
    if confirm "Installo driver NVIDIA proprietari + CUDA (richiede reboot)?"; then
        sudo pacman -S --needed --noconfirm "${NVIDIA_PKGS[@]}"
        ok "NVIDIA + CUDA installati (reboot consigliato a fine setup)"
    fi
else
    warn "Salto NVIDIA (NVIDIA_INSTALL=0). Rilancia con NVIDIA_INSTALL=1 per installarli."
    if command -v nvidia-smi >/dev/null; then
        ok "nvidia-smi già presente:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/      /'
    fi
fi

# ── 3. Ollama ──────────────────────────────────────────────────────────────
section "3/8  Ollama"
OLLAMA_PKG="ollama"
if command -v nvidia-smi >/dev/null 2>&1; then
    OLLAMA_PKG="ollama-cuda"
fi
if ! command -v ollama >/dev/null; then
    if confirm "Installo ${OLLAMA_PKG}?"; then
        sudo pacman -S --needed --noconfirm "${OLLAMA_PKG}"
    fi
else
    ok "ollama già installato ($(ollama --version 2>/dev/null | head -1))"
fi

# Drop-in systemd: KV cache q8 + Flash Attention. Riduce l'impronta VRAM del
# context senza degradare la qualità in modo percepibile.
if [[ "${ENABLE_OLLAMA_TUNING}" == "1" ]] && systemctl list-unit-files ollama.service &>/dev/null; then
    DROPIN_DIR="/etc/systemd/system/ollama.service.d"
    DROPIN_FILE="${DROPIN_DIR}/override.conf"
    if [[ ! -f "${DROPIN_FILE}" ]] && confirm "Aggiungo drop-in systemd (FLASH_ATTENTION + KV_CACHE q8) a ollama?"; then
        sudo mkdir -p "${DROPIN_DIR}"
        sudo tee "${DROPIN_FILE}" >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
EOF
        sudo systemctl daemon-reload
        sudo systemctl restart ollama || true
        ok "Drop-in scritto: ${DROPIN_FILE}"
    fi
fi

if systemctl is-enabled ollama.service &>/dev/null; then
    ok "ollama.service abilitato"
else
    if confirm "Abilito e avvio ollama.service?"; then
        sudo systemctl enable --now ollama
    fi
fi

# ── 4. Pull modelli ────────────────────────────────────────────────────────
section "4/8  Pull modelli Ollama"
if command -v ollama >/dev/null; then
    # Aspetta che il servizio risponda (massimo 30s)
    for _ in $(seq 1 30); do
        ollama list >/dev/null 2>&1 && break
        sleep 1
    done
    INSTALLED="$(ollama list 2>/dev/null | awk 'NR>1 {print $1}')"
    for m in "${MODELS[@]}"; do
        if grep -Fxq "$m" <<<"${INSTALLED}"; then
            ok "${m} già presente"
        elif confirm "Scarico ${m}? (può essere grande)"; then
            ollama pull "$m"
        fi
    done
else
    warn "ollama non disponibile, salto il pull dei modelli."
fi

# ── 5. Virtualenv runtime + training ───────────────────────────────────────
section "5/8  Virtualenv Python"
make_venv() {
    local name="$1" req="$2"
    local path="${PROJECT_DIR}/${name}"
    if [[ -x "${path}/bin/python" ]] && "${path}/bin/python" -c 'import sys' &>/dev/null; then
        ok "${name} già funzionante — salto"
        return
    fi
    if [[ -d "${path}" ]]; then
        warn "${name} esiste ma non parte: lo rimuovo e ricreo"
        rm -rf "${path}"
    fi
    log "Creo ${name} con ${PY_BIN}…"
    "${PY_BIN}" -m venv "${path}"
    "${path}/bin/pip" install --upgrade pip wheel
    log "Installo da ${req} (può richiedere qualche minuto)…"
    # I pin torch/torchaudio sono build +cu128: vivono solo sull'index PyTorch,
    # non su PyPI. Senza questo extra-index l'install fallisce su una macchina pulita.
    "${path}/bin/pip" install --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r "${PROJECT_DIR}/${req}"
    # torchcodec/nvidia-* non hanno l'RPATH verso le dir sorelle nvidia/*/lib:
    # senza questo, dlopen di torchcodec non trova libnvrtc.so.12 / le NPP e
    # rompe sentence_transformers. Un sitecustomize pre-carica quelle lib CUDA
    # all'avvio dell'interprete, senza dover impostare LD_LIBRARY_PATH.
    local sp
    sp="$(echo "${path}"/lib/python*/site-packages)"
    if [[ -d "${sp}/nvidia/cuda_nvrtc" ]]; then
        cat > "${sp}/sitecustomize.py" <<'PYEOF'
import ctypes, glob, os
_sp = os.path.dirname(__file__)
for _sub in ("cuda_nvrtc", "npp"):
    for _lib in sorted(glob.glob(os.path.join(_sp, "nvidia", _sub, "lib", "*.so*"))):
        try:
            ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
PYEOF
        ok "sitecustomize.py (preload CUDA) scritto in ${name}"
    fi
    ok "${name} pronto"
}

if confirm "Ricreo venv-runtime da requirements-runtime.txt?"; then
    make_venv "venv-runtime" "requirements-runtime.txt"
fi
if confirm "Ricreo venv-training da requirements-training.txt? (pesante, opzionale)"; then
    make_venv "venv-training" "requirements-training.txt"
fi

# ── 6. Gruppo input (per PTT pynput) ───────────────────────────────────────
section "6/8  Gruppo input (PTT)"
if id -nG "${USER}" | grep -qw input; then
    ok "${USER} già nel gruppo input"
else
    if confirm "Aggiungo ${USER} al gruppo input? (servirà logout/login)"; then
        sudo usermod -aG input "${USER}"
        warn "Devi fare logout e rilogin perché pynput veda la tastiera senza root."
    fi
fi

# ── 7. Docker (per SearXNG) ────────────────────────────────────────────────
section "7/8  Docker (SearXNG)"
if ! command -v docker >/dev/null; then
    if confirm "Installo docker + docker-compose?"; then
        sudo pacman -S --needed --noconfirm docker docker-compose
    fi
fi
if command -v docker >/dev/null; then
    if ! systemctl is-enabled docker.service &>/dev/null; then
        if confirm "Abilito e avvio docker.service?"; then
            sudo systemctl enable --now docker
        fi
    else
        ok "docker.service abilitato"
    fi
    if ! id -nG "${USER}" | grep -qw docker; then
        if confirm "Aggiungo ${USER} al gruppo docker? (per usare docker senza sudo)"; then
            sudo usermod -aG docker "${USER}"
        fi
    fi
    log "SearXNG: avvialo on-demand con 'cd docker && docker compose up -d'"
fi

# ── 8. Note finali ─────────────────────────────────────────────────────────
section "8/8  Note finali"

# venv-tts: server Qwen3-TTS (transformers 4.57.3), separato dal runtime. Lo
# richiede l'app stessa (modules/tts/base_tts.py l'avvia come sottoprocesso),
# non solo TTS Studio. La path è hardcoded a python3.12 → PY_BIN dev'essere 3.12.
# torch/torchaudio pinnati e allineati a 2.11.0 (+cu128 via l'extra-index).
TTS_VENV="${PROJECT_DIR}/venv-tts"
if [[ -x "${TTS_VENV}/bin/python" ]] && "${TTS_VENV}/bin/python" -c 'import qwen_tts' &>/dev/null; then
    ok "venv-tts già funzionante — salto"
elif confirm "Creo venv-tts (server Qwen3-TTS da vendor/Qwen3-TTS, ~qualche GB)?"; then
    [[ -d "${TTS_VENV}" ]] && rm -rf "${TTS_VENV}"
    "${PY_BIN}" -m venv "${TTS_VENV}"
    "${TTS_VENV}/bin/pip" install --upgrade pip wheel
    "${TTS_VENV}/bin/pip" install --extra-index-url https://download.pytorch.org/whl/cu128 \
        -e "${PROJECT_DIR}/vendor/Qwen3-TTS" fastapi uvicorn torch==2.11.0 torchaudio==2.11.0
    ok "venv-tts pronto"
fi

# Impostazioni UI personali (modello/voce/personalità/sessioni attive)
UI_CFG="${HOME}/.config/local-assistant"
if [[ ! -d "${UI_CFG}" ]]; then
    warn "Nessun ${UI_CFG}: se hai un backup da Ubuntu, copialo qui per ripristinare"
    warn "lo stato della UI (modello attivo, sessioni di chat, voce, personalità)."
else
    ok "Trovato ${UI_CFG} — stato UI già presente"
fi

# Test rapidi di salute
section "Health check rapido"
if [[ -x "${PROJECT_DIR}/venv-runtime/bin/python" ]]; then
    if "${PROJECT_DIR}/venv-runtime/bin/python" -c \
        "import sounddevice, torch, fastapi; print('  sounddevice', sounddevice.__version__);\
         print('  torch', torch.__version__, 'cuda?', torch.cuda.is_available());\
         print('  fastapi', fastapi.__version__)" 2>&1; then
        ok "venv-runtime importa le librerie chiave"
    else
        warn "venv-runtime ha qualche import rotto, controlla l'output sopra"
    fi
fi
if command -v ollama >/dev/null && ollama list &>/dev/null; then
    ok "Ollama risponde ($(ollama list | wc -l) righe in 'ollama list')"
fi

echo
log "Tutto fatto. Per partire:"
echo "    cd ${PROJECT_DIR} && venv-runtime/bin/python scripts/run_app.py"
echo
if [[ "${NVIDIA_INSTALL:-0}" == "1" ]]; then
    warn "Hai installato i driver NVIDIA: fai un reboot prima di lanciare l'app."
fi
