"""
tools/tts_studio.py
TTS Studio — interfaccia web locale per sintetizzare audio con voice cloning.

Avvio:
    ./tts_studio.sh
    oppure:
    venv-runtime/bin/python tools/tts_studio.py

Apre automaticamente il browser su http://127.0.0.1:7860
"""

from __future__ import annotations

import asyncio
import io
import shutil
import subprocess
import time
import wave
import webbrowser
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

PROJECT_ROOT   = Path(__file__).parent.parent
TTS_SERVER_URL = "http://127.0.0.1:8765"
STUDIO_PORT    = 7860
VENV_TTS       = PROJECT_ROOT / "venv-tts"
SERVER_SCRIPT  = PROJECT_ROOT / "modules" / "tts" / "server.py"
PROFILES_DIR   = PROJECT_ROOT / "data" / "voice_profiles"

_tts_process: subprocess.Popen | None = None

# ---------------------------------------------------------------------------
# Gestione server TTS
# ---------------------------------------------------------------------------

async def _ensure_tts_server(profile: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"{TTS_SERVER_URL}/health")
            if r.status_code == 200:
                return True
    except Exception:
        pass

    global _tts_process
    python = VENV_TTS / "bin" / "python"
    if not python.exists():
        return False

    import os
    env = os.environ.copy()
    nvidia = VENV_TTS / "lib" / "python3.12" / "site-packages" / "nvidia"
    if nvidia.exists():
        cuda_libs = ":".join([
            str(nvidia / "cublas" / "lib"),
            str(nvidia / "cudnn" / "lib"),
            str(nvidia / "cuda_runtime" / "lib"),
        ])
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{cuda_libs}:{existing}" if existing else cuda_libs

    _tts_process = subprocess.Popen(
        [str(python), str(SERVER_SCRIPT), "--profile", profile, "--port", "8765", "--gpu", "1"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    deadline = time.monotonic() + 120
    async with httpx.AsyncClient(timeout=2.0) as c:
        while time.monotonic() < deadline:
            try:
                r = await c.get(f"{TTS_SERVER_URL}/health")
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(1.0)
    return False


# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.get("/api/status")
async def status():
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"{TTS_SERVER_URL}/health")
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return {"status": "offline", "profiles": [], "profile": ""}


@app.post("/api/start")
async def start_server(body: dict):
    profile = body.get("profile", "mercoledì")
    ok = await _ensure_tts_server(profile)
    if not ok:
        return JSONResponse({"ok": False, "error": "Server TTS non avviato"}, status_code=500)
    async with httpx.AsyncClient(timeout=5.0) as c:
        info = (await c.get(f"{TTS_SERVER_URL}/health")).json()
    return {"ok": True, **info}


@app.post("/api/switch")
async def switch_profile(body: dict):
    profile = body.get("profile", "mercoledì")
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.post(f"{TTS_SERVER_URL}/switch/{profile}")
            return r.json()
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/synthesize")
async def synthesize(body: dict):
    text     = body.get("text", "").strip()
    language = body.get("language", "italian")

    if not text:
        return JSONResponse({"error": "Testo vuoto"}, status_code=400)

    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            t0 = time.perf_counter()
            r  = await c.post(
                f"{TTS_SERVER_URL}/synthesize",
                json={"text": text, "language": language},
            )
            r.raise_for_status()
            elapsed_ms = (time.perf_counter() - t0) * 1000

        wav_bytes = r.content
        duration  = _duration_from_wav(wav_bytes)

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Inference-Ms": str(int(elapsed_ms)),
                "X-Duration-S":   f"{duration:.2f}",
            },
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/create_profile")
async def create_profile(
    name:        str        = Form(...),
    ref_text:    str        = Form(...),
    max_seconds: int        = Form(16),
    audio:       UploadFile = File(...),
):
    name = name.strip().lower()
    if not name:
        return JSONResponse({"ok": False, "error": "Nome profilo vuoto"}, status_code=400)

    profile_dir = PROFILES_DIR / name
    profile_dir.mkdir(parents=True, exist_ok=True)

    suffix     = Path(audio.filename).suffix or ".wav"
    audio_path = profile_dir / f"ref_audio{suffix}"
    data       = await audio.read()
    audio_path.write_bytes(data)

    import yaml
    cfg = {
        "ref_audio":   audio_path.name,
        "ref_text":    ref_text.strip(),
        "language":    "italian",
        "max_seconds": max_seconds,
    }
    with open(profile_dir / "profile.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(f"{TTS_SERVER_URL}/reload")
            reload_info = r.json()
    except Exception:
        reload_info = {"status": "server_offline"}

    return {"ok": True, "profile": name, "file": audio_path.name, "reload": reload_info}


@app.delete("/api/profile/{name}")
async def delete_profile(name: str):
    profile_dir = PROFILES_DIR / name
    if not profile_dir.exists():
        return JSONResponse({"ok": False, "error": "Profilo non trovato"}, status_code=404)
    shutil.rmtree(profile_dir)
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            await c.post(f"{TTS_SERVER_URL}/reload")
    except Exception:
        pass
    return {"ok": True, "deleted": name}


@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Trascrive un file audio usando WhisperSTT."""
    import sys
    import tempfile
    import numpy as np
    import scipy.signal as sps
    import soundfile as sf

    sys.path.insert(0, str(PROJECT_ROOT))
    from modules.stt import WhisperSTT

    data   = await audio.read()
    suffix = Path(audio.filename).suffix or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        samples, sr = sf.read(tmp_path, dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if sr != 16000:
            n       = int(len(samples) * 16000 / sr)
            samples = sps.resample(samples, n)
        pcm = (np.clip(samples, -1.0, 1.0) * 32768).astype(np.int16).tobytes()

        async with WhisperSTT(language="it") as stt:
            result = await stt.transcribe_with_vad(pcm)

        return {"ok": True, "text": result.text.strip(), "language": result.language}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _duration_from_wav(wav_bytes: bytes) -> float:
    if not wav_bytes:
        return 0.0
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# HTML embedded
# ---------------------------------------------------------------------------

HTML_PAGE = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TTS Studio</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:      #0d0d0f;
    --surface: #141418;
    --border:  #2a2a32;
    --accent:  #c8a96e;
    --accent2: #7c6fcd;
    --text:    #e8e4dc;
    --muted:   #6b6775;
    --success: #6db98a;
    --error:   #c96e6e;
    --radius:  12px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: 'DM Mono', monospace;
    min-height: 100vh; display: flex;
    align-items: center; justify-content: center; padding: 24px;
  }
  .grain {
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    opacity: 0.4;
  }
  .container { position: relative; z-index: 1; width: 100%; max-width: 680px; }
  header { margin-bottom: 40px; }
  .title { font-family: 'DM Serif Display', serif; font-size: 2.8rem; letter-spacing: -0.02em; line-height: 1; }
  .title em { font-style: italic; color: var(--accent); }
  .subtitle { margin-top: 8px; font-size: 0.78rem; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 28px; margin-bottom: 16px; }
  .card-label { font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 16px; }
  .status-row { display: flex; align-items: center; gap: 12px; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); flex-shrink: 0; transition: background 0.3s; }
  .status-dot.online  { background: var(--success); box-shadow: 0 0 8px var(--success); }
  .status-dot.loading { background: var(--accent); animation: pulse 1s infinite; }
  .status-dot.error   { background: var(--error); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
  .status-text { font-size: 0.82rem; color: var(--text); flex: 1; }
  .status-model { font-size: 0.72rem; color: var(--muted); }
  .profile-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }
  .profile-btn { padding: 8px 18px; border-radius: 40px; border: 1px solid var(--border); background: transparent; color: var(--muted); font-family: 'DM Mono', monospace; font-size: 0.80rem; cursor: pointer; transition: all 0.2s; letter-spacing: 0.04em; }
  .profile-btn:hover  { border-color: var(--accent); color: var(--accent); }
  .profile-btn.active { background: var(--accent); border-color: var(--accent); color: var(--bg); font-weight: 500; }
  .start-btn { margin-top: 16px; padding: 10px 24px; border-radius: 8px; border: 1px solid var(--accent2); background: transparent; color: var(--accent2); font-family: 'DM Mono', monospace; font-size: 0.80rem; cursor: pointer; transition: all 0.2s; letter-spacing: 0.06em; text-transform: uppercase; }
  .start-btn:hover { background: var(--accent2); color: var(--bg); }
  .start-btn:disabled { opacity: 0.4; cursor: default; }
  textarea, input[type=text], input[type=number] {
    width: 100%; background: #0d0d0f; border: 1px solid var(--border); border-radius: 8px;
    color: var(--text); font-family: 'DM Mono', monospace; font-size: 0.88rem; padding: 12px 16px;
    outline: none; transition: border-color 0.2s;
  }
  textarea { min-height: 140px; line-height: 1.7; resize: vertical; }
  textarea:focus, input[type=text]:focus, input[type=number]:focus { border-color: var(--accent); }
  textarea::placeholder, input::placeholder { color: var(--muted); }
  .char-count { text-align: right; font-size: 0.68rem; color: var(--muted); margin-top: 8px; }
  .generate-btn { width: 100%; padding: 16px; margin-top: 16px; border-radius: var(--radius); border: none; background: linear-gradient(135deg, var(--accent) 0%, #e8c48a 100%); color: var(--bg); font-family: 'DM Mono', monospace; font-size: 0.88rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; cursor: pointer; transition: all 0.2s; }
  .generate-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(200,169,110,0.3); }
  .generate-btn:disabled { opacity: 0.5; cursor: default; }
  .result-card { display: none; }
  .result-card.visible { display: block; }
  .result-meta { display: flex; gap: 24px; margin-bottom: 16px; font-size: 0.72rem; color: var(--muted); }
  .result-meta strong { color: var(--accent); }
  audio { width: 100%; height: 40px; accent-color: var(--accent); }
  .error-msg { display: none; padding: 12px 16px; border-radius: 8px; border: 1px solid var(--error); background: rgba(201,110,110,0.08); color: var(--error); font-size: 0.80rem; margin-top: 12px; }
  .error-msg.visible { display: block; }
  .success-msg { display: none; padding: 12px 16px; border-radius: 8px; border: 1px solid var(--success); background: rgba(109,185,138,0.08); color: var(--success); font-size: 0.80rem; margin-top: 12px; }
  .success-msg.visible { display: block; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(13,13,15,0.3); border-top-color: var(--bg); border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .history { margin-top: 8px; display: flex; flex-direction: column; gap: 8px; max-height: 280px; overflow-y: auto; }
  .history-item { padding: 10px 14px; border: 1px solid var(--border); border-radius: 8px; cursor: pointer; transition: border-color 0.2s; font-size: 0.78rem; color: var(--muted); display: flex; justify-content: space-between; align-items: center; gap: 12px; }
  .history-item:hover { border-color: var(--accent); color: var(--text); }
  .history-item .hist-text { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .history-item .hist-meta { flex-shrink: 0; font-size: 0.68rem; }
  /* Profile manager */
  .collapsible-header { cursor: pointer; display: flex; align-items: center; justify-content: space-between; user-select: none; }
  .collapsible-arrow { transition: transform 0.2s; font-size: 0.70rem; color: var(--muted); }
  .collapsible-body { display: none; margin-top: 20px; }
  .collapsible-body.open { display: block; }
  .form-label { font-size: 0.68rem; letter-spacing: 0.10em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
  .form-row { display: grid; grid-template-columns: 1fr 90px; gap: 10px; margin-bottom: 14px; align-items: end; }
  .drop-zone { border: 2px dashed var(--border); border-radius: 8px; padding: 28px 20px; text-align: center; cursor: pointer; transition: all 0.2s; color: var(--muted); font-size: 0.80rem; position: relative; margin-bottom: 14px; }
  .drop-zone:hover { border-color: var(--accent2); color: var(--accent2); background: rgba(124,111,205,0.06); }
  .drop-zone.dragover { border-color: var(--accent2); color: var(--accent2); background: rgba(124,111,205,0.06); }
  .drop-zone input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
  .file-name { margin-top: 6px; font-size: 0.72rem; color: var(--success); }
  .create-btn { width: 100%; padding: 13px; margin-top: 14px; border-radius: 8px; border: none; background: linear-gradient(135deg, var(--accent2) 0%, #9d93e0 100%); color: white; font-family: 'DM Mono', monospace; font-size: 0.82rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; cursor: pointer; transition: all 0.2s; }
  .create-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(124,111,205,0.4); }
  .create-btn:disabled { opacity: 0.5; cursor: default; }
  .profile-list-item { display: flex; align-items: center; gap: 10px; padding: 8px 12px; border: 1px solid var(--border); border-radius: 8px; margin-bottom: 6px; font-size: 0.78rem; }
  .profile-list-item .pname { flex: 1; color: var(--text); }
  .del-btn { background: none; border: none; color: var(--error); cursor: pointer; font-size: 0.80rem; padding: 2px 8px; opacity: 0.5; transition: opacity 0.2s; }
  .del-btn:hover { opacity: 1; }
  .section-divider { border: none; border-top: 1px solid var(--border); margin: 20px 0; }
</style>
</head>
<body>
<div class="grain"></div>
<div class="container">

  <header>
    <h1 class="title">TTS <em>Studio</em></h1>
    <p class="subtitle">Assistente AI Locale — Voice Synthesis</p>
  </header>

  <!-- Server status -->
  <div class="card">
    <div class="card-label">Server</div>
    <div class="status-row">
      <div class="status-dot" id="dot"></div>
      <div class="status-text" id="statusText">Connessione...</div>
      <div class="status-model" id="statusModel"></div>
    </div>
    <div class="profile-row" id="profileRow"></div>
    <button class="start-btn" id="startBtn" onclick="startServer()" style="display:none">
      ▶ Avvia Server
    </button>
  </div>

  <!-- Input -->
  <div class="card">
    <div class="card-label">Testo</div>
    <textarea id="textInput" placeholder="Scrivi il testo da sintetizzare…" oninput="updateCount()"></textarea>
    <div class="char-count"><span id="charCount">0</span> caratteri</div>
    <button class="generate-btn" id="genBtn" onclick="generate()" disabled>
      Genera Audio
    </button>
    <div class="error-msg" id="errorMsg"></div>
  </div>

  <!-- Result -->
  <div class="card result-card" id="resultCard">
    <div class="card-label">Output</div>
    <div class="result-meta">
      <span>Durata <strong id="resDuration">—</strong></span>
      <span>Inference <strong id="resInference">—</strong></span>
      <span>Profilo <strong id="resProfile">—</strong></span>
    </div>
    <audio id="audioPlayer" controls></audio>
  </div>

  <!-- History -->
  <div class="card" id="historyCard" style="display:none">
    <div class="card-label">Generazioni recenti</div>
    <div class="history" id="historyList"></div>
  </div>

  <!-- Profile Manager -->
  <div class="card">
    <div class="collapsible-header" id="pmHeader" onclick="toggleProfileManager()">
      <div class="card-label" style="margin-bottom:0">Gestione profili vocali</div>
      <span class="collapsible-arrow" id="pmArrow">▶</span>
    </div>
    <div class="collapsible-body" id="pmBody">

      <div class="form-label" style="margin-top:4px">Profili attivi</div>
      <div id="profileManagerList"></div>

      <hr class="section-divider">

      <div class="form-label">Nuovo profilo</div>

      <div class="form-row">
        <div>
          <div class="form-label">Nome</div>
          <input type="text" id="newProfileName" placeholder="es. mario" />
        </div>
        <div>
          <div class="form-label">Max sec.</div>
          <input type="number" id="newProfileMaxSec" value="16" min="5" max="30" />
        </div>
      </div>

      <div class="form-label">File audio (wav, mp3, ogg)</div>
      <div class="drop-zone" id="dropZone">
        <input type="file" id="audioFile" accept=".wav,.mp3,.ogg,.flac,.m4a" />
        <div>Trascina un file audio qui<br>o clicca per selezionare</div>
        <div class="file-name" id="fileName"></div>
      </div>

      <div class="form-label">Trascrizione esatta dell'audio</div>
      <textarea id="newProfileText" style="min-height:80px; margin-bottom:0"
                placeholder="Scrivi esattamente quello che viene detto nell'audio..."></textarea>

      <button class="create-btn" id="createBtn" onclick="createProfile()">
        Crea profilo
      </button>

      <div class="success-msg" id="createSuccess"></div>
      <div class="error-msg" id="createError"></div>
    </div>
  </div>

</div>

<script>
let currentProfile = '';
let audioBlobs = [];
const MAX_HISTORY = 6;

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

async function checkStatus() {
  const dot   = document.getElementById('dot');
  const text  = document.getElementById('statusText');
  const model = document.getElementById('statusModel');
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    if (d.status === 'ok') {
      dot.className = 'status-dot online';
      text.textContent = 'Server attivo';
      model.textContent = d.model ? d.model.split('/').pop() : '';
      document.getElementById('startBtn').style.display = 'none';
      document.getElementById('genBtn').disabled = false;
      renderProfiles(d.profiles, d.profile);
    } else {
      throw new Error('offline');
    }
  } catch {
    dot.className = 'status-dot error';
    text.textContent = 'Server non attivo';
    model.textContent = '';
    document.getElementById('startBtn').style.display = '';
    document.getElementById('genBtn').disabled = true;
    renderProfiles([], '');
  }
}

function renderProfiles(profiles, active) {
  const row = document.getElementById('profileRow');
  row.innerHTML = '';
  currentProfile = active;
  profiles.forEach(p => {
    const btn = document.createElement('button');
    btn.className = 'profile-btn' + (p === active ? ' active' : '');
    btn.textContent = p;
    btn.onclick = () => switchProfile(p);
    row.appendChild(btn);
  });
}

async function startServer() {
  const btn  = document.getElementById('startBtn');
  const dot  = document.getElementById('dot');
  const text = document.getElementById('statusText');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Avvio in corso...';
  dot.className = 'status-dot loading';
  text.textContent = 'Caricamento modello (~10s)…';
  try {
    const r = await fetch('/api/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({profile: 'mercoledì'}),
    });
    const d = await r.json();
    if (d.ok) { await checkStatus(); }
    else { throw new Error(d.error || 'Errore avvio'); }
  } catch (e) {
    showError(e.message);
    dot.className = 'status-dot error';
    text.textContent = 'Errore avvio';
    btn.disabled = false;
    btn.textContent = '▶ Avvia Server';
  }
}

async function switchProfile(profile) {
  if (profile === currentProfile) return;
  const r = await fetch('/api/switch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({profile}),
  });
  const d = await r.json();
  if (d.profile) {
    currentProfile = d.profile;
    document.querySelectorAll('.profile-btn').forEach(b => {
      b.className = 'profile-btn' + (b.textContent === currentProfile ? ' active' : '');
    });
  }
}

// ---------------------------------------------------------------------------
// Synthesize
// ---------------------------------------------------------------------------

async function generate() {
  const text = document.getElementById('textInput').value.trim();
  if (!text) return;
  const btn = document.getElementById('genBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Generazione...';
  hideError();
  try {
    const r = await fetch('/api/synthesize', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, language: 'italian'}),
    });
    if (!r.ok) { const e = await r.json(); throw new Error(e.error || 'Errore sintesi'); }
    const blob    = await r.blob();
    const url     = URL.createObjectURL(blob);
    const dur     = r.headers.get('X-Duration-S') || '?';
    const infMs   = r.headers.get('X-Inference-Ms') || '?';
    const card    = document.getElementById('resultCard');
    card.classList.add('visible');
    document.getElementById('audioPlayer').src = url;
    document.getElementById('audioPlayer').play();
    document.getElementById('resDuration').textContent  = dur + 's';
    document.getElementById('resInference').textContent = infMs > 0 ? (infMs/1000).toFixed(1)+'s' : '?';
    document.getElementById('resProfile').textContent   = currentProfile;
    addHistory(text, url, dur, currentProfile);
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Genera Audio';
  }
}

function addHistory(text, url, duration, profile) {
  audioBlobs.unshift({text, url, duration, profile});
  if (audioBlobs.length > MAX_HISTORY) audioBlobs.pop();
  const list = document.getElementById('historyList');
  const card = document.getElementById('historyCard');
  card.style.display = '';
  list.innerHTML = '';
  audioBlobs.forEach(item => {
    const div = document.createElement('div');
    div.className = 'history-item';
    div.innerHTML = `
      <span class="hist-text">${item.text.substring(0,80)}${item.text.length>80?'…':''}</span>
      <span class="hist-meta">${item.profile} · ${item.duration}s</span>
    `;
    div.onclick = () => {
      document.getElementById('audioPlayer').src = item.url;
      document.getElementById('audioPlayer').play();
      document.getElementById('resultCard').classList.add('visible');
      document.getElementById('resDuration').textContent = item.duration+'s';
      document.getElementById('resProfile').textContent  = item.profile;
    };
    list.appendChild(div);
  });
}

function updateCount() {
  document.getElementById('charCount').textContent = document.getElementById('textInput').value.length;
}
function showError(msg)  { const e=document.getElementById('errorMsg'); e.textContent=msg; e.classList.add('visible'); }
function hideError()     { document.getElementById('errorMsg').classList.remove('visible'); }

// ---------------------------------------------------------------------------
// Profile Manager
// ---------------------------------------------------------------------------

function toggleProfileManager() {
  const body  = document.getElementById('pmBody');
  const arrow = document.getElementById('pmArrow');
  const open  = body.classList.toggle('open');
  arrow.style.transform = open ? 'rotate(90deg)' : '';
  if (open) renderProfileManager();
}

async function renderProfileManager() {
  const list = document.getElementById('profileManagerList');
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    list.innerHTML = '';
    if (!d.profiles || d.profiles.length === 0) {
      list.innerHTML = '<div style="color:var(--muted);font-size:0.78rem;margin-bottom:4px">Nessun profilo attivo.</div>';
      return;
    }
    d.profiles.forEach(p => {
      const div = document.createElement('div');
      div.className = 'profile-list-item';
      const active = p === d.profile ? '<span style="color:var(--accent);font-size:0.68rem">● attivo</span>' : '';
      div.innerHTML = `<span class="pname">${p}</span>${active}<button class="del-btn" title="Elimina" onclick="deleteProfile('${p}')">✕</button>`;
      list.appendChild(div);
    });
  } catch {
    list.innerHTML = '<div style="color:var(--muted);font-size:0.78rem">Server non raggiungibile.</div>';
  }
}

async function createProfile() {
  const name   = document.getElementById('newProfileName').value.trim().toLowerCase();
  const text   = document.getElementById('newProfileText').value.trim();
  const maxSec = document.getElementById('newProfileMaxSec').value;
  const fileIn = document.getElementById('audioFile');
  hideCreateMessages();
  if (!name)              return showCreateError('Inserisci un nome per il profilo.');
  if (!text)              return showCreateError('Inserisci la trascrizione.');
  if (!fileIn.files.length) return showCreateError('Seleziona un file audio.');
  const btn = document.getElementById('createBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Creazione in corso...';
  const fd = new FormData();
  fd.append('name', name);
  fd.append('ref_text', text);
  fd.append('max_seconds', maxSec);
  fd.append('audio', fileIn.files[0]);
  try {
    const r = await fetch('/api/create_profile', {method:'POST', body:fd});
    const d = await r.json();
    if (!r.ok || !d.ok) throw new Error(d.error || 'Errore creazione profilo');
    showCreateSuccess(`Profilo "${d.profile}" creato!`);
    document.getElementById('newProfileName').value = '';
    document.getElementById('newProfileText').value = '';
    fileIn.value = '';
    document.getElementById('fileName').textContent = '';
    await checkStatus();
    renderProfileManager();
  } catch (e) {
    showCreateError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Crea profilo';
  }
}

async function deleteProfile(name) {
  if (!confirm(`Eliminare il profilo "${name}"?`)) return;
  const r = await fetch('/api/profile/' + encodeURIComponent(name), {method:'DELETE'});
  const d = await r.json();
  if (d.ok) { await checkStatus(); renderProfileManager(); }
  else alert(d.error || 'Errore eliminazione');
}

function showCreateError(msg)   { const e=document.getElementById('createError'); e.textContent=msg; e.classList.add('visible'); }
function showCreateSuccess(msg) { const e=document.getElementById('createSuccess'); e.textContent=msg; e.classList.add('visible'); }
function hideCreateMessages()   { document.getElementById('createError').classList.remove('visible'); document.getElementById('createSuccess').classList.remove('visible'); }

// Drag & drop e trascrizione automatica
async function handleAudioFile(file) {
  if (!file) return;
  document.getElementById('fileName').textContent = file.name;

  // Avvia trascrizione automatica
  const btn  = document.getElementById('createBtn');
  const area = document.getElementById('newProfileText');
  const orig = area.placeholder;
  area.placeholder = '⏳ Trascrizione in corso...';
  area.disabled    = true;

  try {
    const fd = new FormData();
    fd.append('audio', file);
    const r = await fetch('/api/transcribe', { method: 'POST', body: fd });
    const d = await r.json();
    if (d.ok && d.text) {
      area.value = d.text;
      area.placeholder = orig;
    } else {
      area.placeholder = orig;
      showCreateError('Trascrizione fallita: ' + (d.error || 'errore sconosciuto'));
    }
  } catch (e) {
    area.placeholder = orig;
    showCreateError('Trascrizione fallita: ' + e.message);
  } finally {
    area.disabled = false;
  }
}

document.getElementById('audioFile').addEventListener('change', function() {
  handleAudioFile(this.files[0]);
});

const dz = document.getElementById('dropZone');
dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (!f) return;
  const dt = new DataTransfer(); dt.items.add(f);
  document.getElementById('audioFile').files = dt.files;
  handleAudioFile(f);
});

// Ctrl+Enter genera
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') generate();
});

// Avvio
checkStatus();
setInterval(checkStatus, 5000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import threading
    def _open():
        time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:7860")
    threading.Thread(target=_open, daemon=True).start()
    print("TTS Studio — http://127.0.0.1:7860")
    uvicorn.run(app, host="127.0.0.1", port=STUDIO_PORT, log_level="warning")
