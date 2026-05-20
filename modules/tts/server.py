"""
modules/tts/server.py
Server FastAPI per Qwen3-TTS con voice cloning.

Gira nel venv-tts (Python 3.12, transformers 4.57.3).
Viene avviato automaticamente da Qwen3TTS (base_tts.py) come sottoprocesso.

Avvio manuale (debug):
    venv-tts/bin/python modules/tts/server.py \
        --profile mercoledì \
        --port 8765 \
        --gpu 1

Endpoint:
    GET  /health        → {"status": "ok", "profile": ..., "model": ...}
    GET  /profiles      → {"profiles": [...]}
    POST /synthesize    → WAV bytes  (body: {"text": str, "language": str})

Profili vocali in data/voice_profiles/<nome>/:
    profile.yaml        → metadata (ref_audio, ref_text, language)
    ref_audio.mp3/.wav  → audio di riferimento per il cloning
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

PROJECT_ROOT    = Path(__file__).parent.parent.parent
PROFILES_DIR    = PROJECT_ROOT / "data" / "voice_profiles"
OUTPUT_SR       = 24_000
DEFAULT_PROFILE = "mercoledì"

# Parametri di generazione ottimali (testati)
_GEN_KWARGS = dict(
    non_streaming_mode=True,
    do_sample=False,
    temperature=1.0,
    repetition_penalty=1.3,
)


# ---------------------------------------------------------------------------
# Modelli pydantic per le richieste
# ---------------------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    text:     str
    language: str = "italian"


# ---------------------------------------------------------------------------
# Stato globale del server
# ---------------------------------------------------------------------------

_model:          Any = None
_voice_prompts:  dict[str, Any] = {}
_active_profile: str = DEFAULT_PROFILE
_model_name:     str = ""


# ---------------------------------------------------------------------------
# Helpers audio
# ---------------------------------------------------------------------------

def _float32_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """numpy float32 → WAV bytes PCM int16."""
    clipped = np.clip(samples * 32768, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(clipped.tobytes())
    return buf.getvalue()


def _load_ref_audio(path: Path, max_seconds: int = 16) -> tuple[np.ndarray, int]:
    """Carica e tronca l'audio di riferimento."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    max_samples = sr * max_seconds
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    return audio, sr


# ---------------------------------------------------------------------------
# Caricamento profili
# ---------------------------------------------------------------------------

def _load_profile(model, profile_name: str) -> Any:
    """
    Carica un profilo vocale e calcola il voice_clone_prompt.

    Struttura attesa in data/voice_profiles/<nome>/profile.yaml:
        ref_audio:   nome_file.mp3    # relativo alla cartella profilo
        ref_text:    "trascrizione"   # testo dell'audio (opzionale)
        language:    italian
        max_seconds: 16              # secondi da usare (default 16)
    """
    import yaml

    profile_dir = PROFILES_DIR / profile_name
    yaml_path   = profile_dir / "profile.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Profilo '{profile_name}' non trovato in {profile_dir}.\n"
            f"Crea {yaml_path} con i campi: ref_audio, ref_text, language"
        )

    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ref_audio_path = profile_dir / cfg["ref_audio"]
    if not ref_audio_path.exists():
        raise FileNotFoundError(f"File audio '{ref_audio_path}' non trovato")

    ref_text = cfg.get("ref_text", "")
    max_s    = int(cfg.get("max_seconds", 16))

    print(f"[tts-server] caricamento profilo '{profile_name}'...")
    t0 = time.time()

    audio, sr = _load_ref_audio(ref_audio_path, max_seconds=max_s)

    if ref_text:
        voice_prompt = model.create_voice_clone_prompt(
            ref_audio=(audio, sr),
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        mode = "ICL"
    else:
        voice_prompt = model.create_voice_clone_prompt(
            ref_audio=(audio, sr),
            x_vector_only_mode=True,
        )
        mode = "x-vector"

    elapsed = time.time() - t0
    print(f"[tts-server] profilo '{profile_name}' pronto [{mode}] ({elapsed:.2f}s)")
    return voice_prompt


def _load_all_profiles(model) -> None:
    """Carica tutti i profili presenti in PROFILES_DIR."""
    if not PROFILES_DIR.exists():
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        return

    for profile_dir in sorted(PROFILES_DIR.iterdir()):
        if not profile_dir.is_dir():
            continue
        name = profile_dir.name
        try:
            _voice_prompts[name] = _load_profile(model, name)
        except Exception as exc:
            print(f"[tts-server] ⚠ profilo '{name}' non caricato: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Inizializzazione modello
# ---------------------------------------------------------------------------

def _init_model(profile: str, gpu: int) -> None:
    global _model, _active_profile, _model_name

    from qwen_tts import Qwen3TTSModel

    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    device   = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

    print(f"[tts-server] caricamento {model_id} su {device}...")
    t0 = time.time()

    _model = Qwen3TTSModel.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map=device,
    )
    _model_name     = model_id
    _active_profile = profile

    print(f"[tts-server] modello pronto ({time.time()-t0:.1f}s)")

    _load_all_profiles(_model)

    if profile not in _voice_prompts:
        print(
            f"[tts-server] ⚠ profilo default '{profile}' non trovato — "
            "sintesi senza voice cloning",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="Qwen3-TTS Server", version="1.0.0")


@app.get("/health")
async def health() -> dict:
    return {
        "status":   "ok",
        "profile":  _active_profile,
        "profiles": list(_voice_prompts.keys()),
        "model":    _model_name,
    }


@app.get("/profiles")
async def list_profiles() -> dict:
    return {"profiles": list(_voice_prompts.keys())}


@app.post("/reload")
async def reload_profiles() -> dict:
    """Ricarica tutti i profili vocali senza riavviare il server."""
    if _model is None:
        raise HTTPException(503, "Modello non inizializzato")
    old_profiles = set(_voice_prompts.keys())
    _voice_prompts.clear()
    _load_all_profiles(_model)
    new_profiles = set(_voice_prompts.keys())
    return {
        "status":   "ok",
        "profiles": sorted(new_profiles),
        "added":    sorted(new_profiles - old_profiles),
        "removed":  sorted(old_profiles - new_profiles),
    }


@app.post("/switch/{profile_name}")
async def switch_profile(profile_name: str) -> dict:
    """Cambia il profilo vocale attivo."""
    global _active_profile
    if profile_name not in _voice_prompts:
        raise HTTPException(404, f"Profilo '{profile_name}' non trovato. Disponibili: {list(_voice_prompts.keys())}")
    _active_profile = profile_name
    return {"status": "ok", "profile": _active_profile}


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest) -> Response:
    if _model is None:
        raise HTTPException(503, "Modello non inizializzato")

    text = req.text.strip()
    if not text:
        empty = _float32_to_wav_bytes(np.zeros(100, dtype=np.float32), OUTPUT_SR)
        return Response(content=empty, media_type="audio/wav")

    voice_prompt = _voice_prompts.get(_active_profile)

    try:
        if voice_prompt is not None:
            audios, sr = _model.generate_voice_clone(
                text=text,
                language=req.language,
                voice_clone_prompt=voice_prompt,
                **_GEN_KWARGS,
            )
        else:
            # Fallback senza voice cloning
            audios, sr = _model.generate_custom_voice(
                text=text,
                speaker="aiden",
                language=req.language,
                non_streaming_mode=True,
            )
    except Exception as exc:
        raise HTTPException(500, f"Errore sintesi: {exc}") from exc

    wav_bytes = _float32_to_wav_bytes(np.asarray(audios[0], dtype=np.float32), sr)
    return Response(content=wav_bytes, media_type="audio/wav")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS Server")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Profilo vocale default")
    parser.add_argument("--port",    type=int, default=8765,   help="Porta HTTP")
    parser.add_argument("--gpu",     type=int, default=1,      help="Indice GPU CUDA")
    args = parser.parse_args()

    _init_model(args.profile, args.gpu)

    print(f"[tts-server] in ascolto su http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
