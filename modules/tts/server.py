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
import os
import sys
import threading
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

# Dimensione modello: 1.7B (qualità) o 0.6B (~2-3x più veloce in generazione,
# stesso voice cloning). Default da env TTS_MODEL_SIZE, sovrascrivibile con
# --model-size. Il client (base_tts.py) passa il valore di settings allo spawn.
MODEL_IDS = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

# Parametri di generazione ottimali (testati SULL'1.7B). Sul 0.6B questa
# combinazione manda il sampler in NaN (device-side assert): lì si usano i
# default del vendor, che funzionano anche in ICL. Vedi _gen_kwargs().
_GEN_KWARGS = dict(
    non_streaming_mode=True,
    do_sample=False,
    temperature=1.0,
    repetition_penalty=1.3,
)


def _gen_kwargs() -> dict:
    if _model_size == "0.6B":
        return {"non_streaming_mode": True}
    return dict(_GEN_KWARGS)


# ---------------------------------------------------------------------------
# Modelli pydantic per le richieste
# ---------------------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    text:     str
    language: str = "italian"
    # Velocità del parlato (time-stretch a pitch invariato, applicato dopo la
    # sintesi): 1.0 = naturale, 1.2 = 20% più veloce. Clampata a [0.5, 2.0].
    speed:    float = 1.0


# ---------------------------------------------------------------------------
# Stato globale del server
# ---------------------------------------------------------------------------

_model:          Any = None
_voice_prompts:  dict[str, Any] = {}
_active_profile: str = DEFAULT_PROFILE
_model_name:     str = ""
_model_size:     str = "1.7B"

# La generazione satura la GPU: serializzarla evita OOM da richieste
# concorrenti. /synthesize è un endpoint sync (FastAPI lo esegue in
# threadpool), quindi /health e /profiles restano reattivi anche a sintesi
# in corso — prima l'endpoint async bloccava l'intero event loop.
_gen_lock = threading.Lock()


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


def _apply_speed(samples: np.ndarray, speed: float) -> np.ndarray:
    """
    Time-stretch a pitch invariato. speed>1 accorcia l'audio (parlato più
    veloce). No-op per speed≈1 (audio bit-identico, nessun processing).

    Percorso primario: sox `tempo -s` (WSOLA ottimizzato per il parlato,
    qualità nettamente superiore). Fallback: phase vocoder di librosa, che
    però sul parlato introduce artefatti udibili — meglio evitarlo.
    """
    speed = max(0.5, min(2.0, float(speed)))
    if abs(speed - 1.0) < 1e-3 or len(samples) == 0:
        return samples

    import shutil
    import subprocess
    if shutil.which("sox"):
        try:
            raw_fmt = ["-t", "raw", "-r", str(OUTPUT_SR), "-e", "floating-point",
                       "-b", "32", "-c", "1"]
            proc = subprocess.run(
                ["sox", *raw_fmt, "-", *raw_fmt, "-", "tempo", "-s", f"{speed}"],
                input=samples.astype(np.float32).tobytes(),
                capture_output=True, timeout=30, check=True,
            )
            return np.frombuffer(proc.stdout, dtype=np.float32).copy()
        except Exception as exc:
            print(f"[tts-server] ⚠ sox tempo fallito ({exc}), fallback librosa",
                  file=sys.stderr)

    import librosa
    return librosa.effects.time_stretch(samples.astype(np.float32), rate=speed)


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

def _init_model(profile: str, gpu: int, model_size: str = "1.7B") -> None:
    global _model, _active_profile, _model_name, _model_size
    _model_size = model_size

    from qwen_tts import Qwen3TTSModel

    model_id = MODEL_IDS.get(model_size, MODEL_IDS["1.7B"])
    device   = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    # Lo 0.6B in float16 va in overflow (NaN nel tensore di probabilità →
    # device-side assert alla prima sintesi): serve bfloat16. L'1.7B resta
    # in float16, verificato stabile.
    dtype = torch.bfloat16 if model_size == "0.6B" else torch.float16

    # NB: NIENTE cudnn.benchmark qui. Con le shape variabili della
    # generazione autoregressiva (ogni frase ha lunghezza diversa)
    # l'autotuner ri-tara di continuo: misurato RTF 1.6-1.9 con benchmark
    # attivo contro 0.49 senza, sullo stesso hardware e modello.

    print(f"[tts-server] caricamento {model_id} su {device}...")
    t0 = time.time()

    _model = Qwen3TTSModel.from_pretrained(
        model_id,
        dtype=dtype,
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

    # Warmup: la prima generazione paga kernel CUDA/autotuning (secondi).
    # Una sintesi usa-e-getta al boot sposta quel costo fuori dal primo
    # turno reale. Disattivabile con TTS_WARMUP=0.
    if os.environ.get("TTS_WARMUP", "1") != "0":
        try:
            t0 = time.time()
            # Frase di lunghezza realistica: l'autotuner cudnn si tara sulle
            # shape effettive — con un "Ok." corto le prime frasi vere
            # ripagherebbero il tuning (RTF ~1.8 invece di ~0.5 sulla 5080).
            _generate(
                "Questa è una frase di riscaldamento abbastanza lunga da "
                "preparare i kernel per le frasi di una conversazione reale.",
                "italian",
            )
            print(f"[tts-server] warmup completato ({time.time()-t0:.1f}s)")
        except Exception as exc:
            print(f"[tts-server] ⚠ warmup fallito: {exc}", file=sys.stderr)


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


def _generate(text: str, language: str) -> tuple[Any, int]:
    """Sintesi serializzata sulla GPU (vedi _gen_lock)."""
    voice_prompt = _voice_prompts.get(_active_profile)
    with _gen_lock:
        if voice_prompt is not None:
            return _model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_prompt,
                **_gen_kwargs(),
            )
        # I modelli *-Base non hanno voci predefinite (generate_custom_voice
        # è della variante CustomVoice): senza un profilo caricato non si può
        # sintetizzare. Errore chiaro invece del crash vendor.
        raise RuntimeError(
            f"nessun profilo vocale caricato (attivo: '{_active_profile}') — "
            "il modello Base richiede un profilo per il voice cloning"
        )


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest) -> Response:
    if _model is None:
        raise HTTPException(503, "Modello non inizializzato")

    text = req.text.strip()
    if not text:
        empty = _float32_to_wav_bytes(np.zeros(100, dtype=np.float32), OUTPUT_SR)
        return Response(content=empty, media_type="audio/wav")

    try:
        audios, sr = _generate(text, req.language)
    except Exception as exc:
        raise HTTPException(500, f"Errore sintesi: {exc}") from exc

    samples = _apply_speed(np.asarray(audios[0], dtype=np.float32), req.speed)
    wav_bytes = _float32_to_wav_bytes(samples, sr)
    return Response(content=wav_bytes, media_type="audio/wav")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS Server")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Profilo vocale default")
    parser.add_argument("--port",    type=int, default=8765,   help="Porta HTTP")
    parser.add_argument("--gpu",     type=int,
                        default=int(os.environ.get("TTS_CUDA_DEVICE", "1")),
                        help="Indice GPU CUDA (default da env TTS_CUDA_DEVICE)")
    parser.add_argument("--model-size", choices=sorted(MODEL_IDS),
                        default=os.environ.get("TTS_MODEL_SIZE", "1.7B"),
                        help="Dimensione modello (default da env TTS_MODEL_SIZE)")
    args = parser.parse_args()

    _init_model(args.profile, args.gpu, args.model_size)

    print(f"[tts-server] in ascolto su http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
