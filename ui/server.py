"""ui/server.py — FastAPI app con tutti gli endpoint inclusi model switching."""
from __future__ import annotations
import json
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from config.settings import settings
from core.logger import logger
from modules.file_analysis import FileAnalyzer
import json
from pathlib import Path

_SETTINGS_FILE = Path.home() / ".config" / "local-assistant" / "ui-settings.json"

def _load_ui_settings() -> dict:
    try:
        if _SETTINGS_FILE.exists():
            return json.loads(_SETTINGS_FILE.read_text())
    except Exception: pass
    return {}

_SESSIONS_FILE = Path.home() / ".config" / "local-assistant" / "sessions.json"

def _load_sessions_disk() -> dict:
    try:
        if _SESSIONS_FILE.exists():
            return json.loads(_SESSIONS_FILE.read_text())
    except Exception: pass
    return {}

def _save_sessions_disk(sessions: dict) -> None:
    try:
        _SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SESSIONS_FILE.write_text(json.dumps(sessions, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.warning("ui.server | save sessions: {}", e)

def _save_ui_settings(data: dict) -> None:
    try:
        _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        current = _load_ui_settings()
        current.update(data)
        _SETTINGS_FILE.write_text(json.dumps(current, indent=2))
    except Exception as e:
        logger.warning("ui.server | save settings: {}", e)
if TYPE_CHECKING:
    from ui.bridge import UIBridge, WSManager

_STATIC = Path(__file__).parent / "static"

class SessNameBody(BaseModel):
    name: str = ""

class SessIdBody(BaseModel):
    id: str

class SessRenameBody(BaseModel):
    id: str
    name: str

class SendBody(BaseModel):   text: str
class PTTBody(BaseModel):    key: str
class PersBody(BaseModel):   name: str
class ModelBody(BaseModel):  name: str
class VoiceBody(BaseModel):  name: str
class TTSBody(BaseModel):    enabled: bool
class ModeBody(BaseModel):   mode: str
class TermProposeBody(BaseModel):       text: str
class TermProposalIdBody(BaseModel):    proposal_id: str
class TermModelVisibilityBody(BaseModel):
    name:    str
    visible: bool


# ---------------------------------------------------------------------------
# Helper upload file (per /api/uploads)
# ---------------------------------------------------------------------------

# Directory di destinazione: data/uploads/<session_id>/<timestamp>_<filename>.
# Visibile sul filesystem (utente può ispezionare), non transitoria.
_UPLOADS_DIR = settings.data_dir / "uploads"

# Set di estensioni che FileAnalyzer sa estrarre.
# Lo calcoliamo a partire dal modulo per restare in sync senza duplicare.
_ALLOWED_UPLOAD_EXTS: set[str] = set(FileAnalyzer(safe_dirs=[]).supported_extensions())

# Caratteri permessi nel filename "safe" (oltre a lettere e cifre).
_SAFE_FILENAME_EXTRA = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_filename(name: str) -> str:
    """
    Sanifica il nome file caricato dall'utente:
    - normalizza unicode (rimuove accenti)
    - tiene solo a-zA-Z0-9, '.', '_', '-'
    - tronca a 80 char (max sensato per un filesystem leggibile)
    - garantisce un nome non vuoto

    Esempi:
        "Contratto firmato.PDF"   → "Contratto_firmato.PDF"
        "résumé / 2024.docx"      → "resume_2024.docx"
        "../etc/passwd"           → "etc_passwd"
    """
    if not name:
        return "file"
    # Tieni solo il basename (no path traversal)
    name = Path(name).name
    # Decomposizione unicode + drop dei combining (accenti)
    normalized = unicodedata.normalize("NFKD", name)
    no_accents = "".join(c for c in normalized if not unicodedata.combining(c))
    # Sostituisci ogni run di char non-safe con underscore
    cleaned = _SAFE_FILENAME_EXTRA.sub("_", no_accents)
    cleaned = cleaned.strip("._-") or "file"
    # Truncate
    return cleaned[:80]


def _session_dir_for(session_id: str | None) -> Path:
    """Sub-cartella per la sessione (o 'default' se non disponibile)."""
    safe_sid = _SAFE_FILENAME_EXTRA.sub("_", (session_id or "default"))[:64]
    d = _UPLOADS_DIR / safe_sid
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_app(ws_manager: "WSManager") -> tuple[FastAPI, dict]:
    app   = FastAPI(title="local-assistant", docs_url=None, redoc_url=None)
    state: dict = {"loop": None}

    @app.get("/", response_class=HTMLResponse)
    async def index():
        # Redirect a /chat o /terminal a seconda della modalità corrente.
        # Per il primo load (loop ancora None) → /chat.
        loop: "UIBridge | None" = state["loop"]
        target = "/terminal" if (loop is not None and loop.mode == "terminal") else "/chat"
        return HTMLResponse(
            f'<!DOCTYPE html><meta http-equiv="refresh" content="0;url={target}">'
            f'<title>redirect</title>',
            status_code=200,
        )

    @app.get("/chat", response_class=HTMLResponse)
    async def chat_page():
        return HTMLResponse((_STATIC / "chat.html").read_text(encoding="utf-8"))

    @app.get("/terminal", response_class=HTMLResponse)
    async def terminal_page():
        return HTMLResponse((_STATIC / "terminal.html").read_text(encoding="utf-8"))

    @app.websocket("/ws")
    async def ws_ep(ws: WebSocket):
        await ws_manager.connect(ws)
        loop: "UIBridge | None" = state["loop"]
        if loop is not None:
            term = state.get("terminal")
            term_state = term.state_payload() if term else None
            await ws.send_text(json.dumps({
                "type":"init", "state":loop.state, "ptt_key":loop.ptt_key,
                "session":loop.session_id,
                "sessions":loop.list_sessions(),
                "active_messages":loop._sessions.get(loop.session_id,{}).get("messages",[]),
                "personality":loop.active_personality,
                "personalities":loop.list_personalities(),
                "model":loop.active_model,
                "voice":await loop.active_voice_from_server(),
                "voices":await loop.list_voices(),
                "tts_enabled":loop.tts_enabled,
                "mode":loop.mode,
                "terminal":term_state,
                "stats":loop.stats.to_log_dict(),
            }))
        try:
            while True: await ws.receive_text()
        except (WebSocketDisconnect, Exception): pass
        finally: ws_manager.disconnect(ws)

    @app.get("/api/status")
    async def status():
        loop: "UIBridge | None" = state["loop"]
        if loop is None: return JSONResponse({"state":"loading"})
        term = state.get("terminal")
        term_state = term.state_payload() if term else None
        return JSONResponse({"state":loop.state,"session":loop.session_id,
            "ptt_key":loop.ptt_key,"personality":loop.active_personality,"model":loop.active_model,
                "voice":await loop.active_voice_from_server(),
                "voices":await loop.list_voices(),
                "tts_enabled":loop.tts_enabled,
                "mode":loop.mode,
                "terminal":term_state,
            "stats":loop.stats.to_log_dict()})

    @app.post("/api/send")
    async def send(body: SendBody):
        loop = state["loop"]
        if loop is None: return JSONResponse({"ok":False,"error":"loop non pronto"},status_code=503)
        await loop.send_text(body.text); return JSONResponse({"ok":True})

    @app.post("/api/uploads")
    async def uploads(file: UploadFile = File(...)):
        """
        Carica un file utente, lo salva in data/uploads/<session_id>/ e
        restituisce il path assoluto da iniettare nel messaggio successivo.

        Validazioni:
          - estensione tra quelle supportate da FileAnalyzer (19 totali)
          - dimensione <= settings.file_analysis.max_file_bytes
          - sanitizzazione del filename (no path traversal, no caratteri
            esotici, troncatura a 80 char)

        Risposta:
            { ok, path, name, size, ext }
        oppure 400/413/500 con { ok: false, error }.
        """
        loop = state["loop"]
        session_id = loop.session_id if loop is not None else "default"

        # 1) Validazione estensione
        original_name = file.filename or "file"
        ext = Path(original_name).suffix.lower()
        if ext not in _ALLOWED_UPLOAD_EXTS:
            allowed = ", ".join(sorted(_ALLOWED_UPLOAD_EXTS))
            raise HTTPException(
                status_code=400,
                detail=f"Estensione '{ext}' non supportata. Supportate: {allowed}",
            )

        # 2) Lettura in memoria (streaming sarebbe più gentile ma il limite
        #    è 50MB di default — accettabile per evitare race su scrittura
        #    parziale in caso d'errore).
        max_bytes = settings.file_analysis.max_file_bytes
        try:
            content = await file.read()
        except Exception as exc:
            logger.warning("ui.server | upload read failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"Lettura fallita: {exc}")
        finally:
            await file.close()

        # 3) Limite dimensione
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File troppo grande ({len(content)} byte > {max_bytes}). "
                    "Soglia regolabile via FILE_ANALYSIS_MAX_FILE_BYTES nel .env."
                ),
            )

        # 4) Filename safe + timestamp per evitare collisioni nello stesso secondo
        safe_name = _safe_filename(original_name)
        ts        = datetime.now().strftime("%Y%m%d%H%M%S")
        target    = _session_dir_for(session_id) / f"{ts}_{safe_name}"

        # 5) Scrittura su disco
        try:
            target.write_bytes(content)
        except Exception as exc:
            logger.warning("ui.server | upload write failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"Scrittura fallita: {exc}")

        logger.info(
            "ui.server | upload OK | session='{}' name='{}' size={}B path='{}'",
            session_id, safe_name, len(content), target,
        )
        return JSONResponse({
            "ok":   True,
            "path": str(target),
            "name": safe_name,
            "size": len(content),
            "ext":  ext,
        })


    @app.get("/api/personalities")
    async def personalities():
        loop = state["loop"]
        if loop is None: return JSONResponse({"personalities":[],"active":None})
        return JSONResponse({"personalities":loop.list_personalities(),"active":loop.active_personality})


    @app.get("/api/models")
    async def models():
        loop = state["loop"]
        if loop is None: return JSONResponse({"models":[],"active":None})
        mdls = await loop.list_models()
        return JSONResponse({"models":mdls,"active":loop.active_model})


    @app.get("/api/sessions")
    async def sessions_list():
        loop = state["loop"]
        if loop is None: return JSONResponse({"sessions":[]})
        return JSONResponse({"sessions":loop.list_sessions()})

    @app.post("/api/sessions/new")
    async def sessions_new(body: SessNameBody):
        loop = state["loop"]
        if loop is None: return JSONResponse({"ok":False},status_code=503)
        sid = loop.new_session(body.name or None)
        _save_sessions_disk(loop._sessions)
        return JSONResponse({"ok":True,"id":sid,"sessions":loop.list_sessions()})

    @app.post("/api/sessions/switch")
    async def sessions_switch(body: SessIdBody):
        loop = state["loop"]
        if loop is None: return JSONResponse({"ok":False},status_code=503)
        ok = loop.switch_session(body.id)
        if not ok: return JSONResponse({"ok":False,"error":"sessione non trovata"},status_code=404)
        s = loop._sessions.get(body.id, {})
        return JSONResponse({"ok":True,"id":body.id,
            "messages":s.get("messages",[]),
            "sessions":loop.list_sessions()})

    @app.post("/api/sessions/delete")
    async def sessions_delete(body: SessIdBody):
        loop = state["loop"]
        if loop is None: return JSONResponse({"ok":False},status_code=503)
        ok = loop.delete_session(body.id)
        if not ok: return JSONResponse({"ok":False,"error":"impossibile eliminare"},status_code=400)
        _save_sessions_disk(loop._sessions)
        # Restituisci anche i messaggi della nuova sessione attiva
        new_active = loop.session_id
        new_msgs = loop._sessions.get(new_active,{}).get("messages",[])
        return JSONResponse({
            "ok":True,
            "sessions":loop.list_sessions(),
            "active":new_active,
            "messages":new_msgs,
        })

    @app.post("/api/sessions/rename")
    async def sessions_rename(body: SessRenameBody):
        loop = state["loop"]
        if loop is None: return JSONResponse({"ok":False},status_code=503)
        ok = loop.rename_session(body.id, body.name)
        if not ok: return JSONResponse({"ok":False},status_code=404)
        _save_sessions_disk(loop._sessions)
        return JSONResponse({"ok":True,"sessions":loop.list_sessions()})

    @app.post("/api/new-session")
    async def new_session_legacy():
        loop = state["loop"]
        if loop is None: return JSONResponse({"ok":False},status_code=503)
        sid = loop.new_session()
        _save_sessions_disk(loop._sessions)
        return JSONResponse({"ok":True,"id":sid,"sessions":loop.list_sessions()})

    @app.get("/api/voices")
    async def voices():
        loop: "UIBridge | None" = state["loop"]
        if loop is None: return JSONResponse({"voices":[],"active":None})
        v = await loop.list_voices()
        active = await loop.active_voice_from_server()
        return JSONResponse({"voices":v,"active":active})

    @app.post("/api/voice")
    async def voice(body: VoiceBody):
        loop = state["loop"]
        if loop is None: return JSONResponse({"ok":False},status_code=503)
        ok = await loop.switch_voice(body.name)
        if not ok: return JSONResponse({"ok":False,"error":f"voce non trovata: {body.name!r}"},status_code=404)
        _save_ui_settings({"voice": body.name})
        return JSONResponse({"ok":True,"name":body.name})

    @app.post("/api/tts")
    async def tts_toggle(body: TTSBody):
        loop = state["loop"]
        if loop is None: return JSONResponse({"ok":False,"error":"loop non pronto"},status_code=503)
        loop.set_tts_enabled(body.enabled)
        _save_ui_settings({"tts_enabled": body.enabled})
        return JSONResponse({"ok":True,"enabled":body.enabled})

    @app.post("/api/model")
    async def model(body: ModelBody):
        loop = state["loop"]
        if loop is None: return JSONResponse({"ok":False},status_code=503)
        ok = loop.switch_model(body.name)
        if not ok: return JSONResponse({"ok":False,"error":f"modello non valido"},status_code=400)
        _save_ui_settings({"model": body.name})
        return JSONResponse({"ok":True,"name":body.name})

    @app.post("/api/personality")
    async def personality_save(body: PersBody):
        loop = state["loop"]
        if loop is None: return JSONResponse({"ok":False},status_code=503)
        ok = loop.switch_personality(body.name)
        if not ok: return JSONResponse({"ok":False,"error":f"profilo non trovato: {body.name!r}"},status_code=404)
        _save_ui_settings({"personality": body.name})
        return JSONResponse({"ok":True,"name":body.name})

    @app.post("/api/ptt-key")
    async def ptt_key_save(body: PTTBody):
        loop = state["loop"]
        if loop is None: return JSONResponse({"ok":False},status_code=503)
        ok = loop.set_ptt_key(body.key)
        if not ok: return JSONResponse({"ok":False,"error":f"tasto non valido: {body.key!r}"},status_code=400)
        _save_ui_settings({"ptt_key": body.key})
        return JSONResponse({"ok":True,"key":body.key})

    @app.get("/api/settings")
    async def ui_settings():
        return JSONResponse(_load_ui_settings())

    # ── Mode (chat | terminal) ───────────────────────────────────────
    @app.get("/api/mode")
    async def mode_get():
        loop = state["loop"]
        if loop is None: return JSONResponse({"mode":"chat"})
        return JSONResponse({"mode":loop.mode})

    @app.post("/api/mode")
    async def mode_set(body: ModeBody):
        loop = state["loop"]
        if loop is None: return JSONResponse({"ok":False,"error":"loop non pronto"},status_code=503)
        if body.mode not in ("chat","terminal"):
            return JSONResponse({"ok":False,"error":f"mode non valido: {body.mode!r}"},status_code=400)
        ok = await loop.set_mode(body.mode)
        if not ok: return JSONResponse({"ok":False},status_code=400)
        _save_ui_settings({"mode": body.mode})
        return JSONResponse({"ok":True,"mode":body.mode})

    # ── Terminal endpoints ───────────────────────────────────────────

    def _terminal_or_503():
        """Helper: ritorna (term, None) se pronto, (None, JSONResponse 503) altrimenti."""
        term = state.get("terminal")
        if term is None:
            return None, JSONResponse(
                {"ok":False,"error":"terminal non pronto (bridge mancante)"},
                status_code=503,
            )
        if not getattr(term, "_loaded", False):
            return None, JSONResponse(
                {"ok":False,"error":"terminal non caricato — riavvia l'app"},
                status_code=503,
            )
        return term, None

    @app.get("/api/terminal/state")
    async def terminal_state():
        term, err = _terminal_or_503()
        if err: return err
        try:
            return JSONResponse({"ok":True, **term.state_payload()})
        except Exception as e:
            logger.exception("ui.server | /api/terminal/state: {}", e)
            return JSONResponse({"ok":False,"error":str(e)}, status_code=500)

    @app.post("/api/terminal/propose")
    async def terminal_propose(body: TermProposeBody):
        term, err = _terminal_or_503()
        if err: return err
        try:
            result = await term.propose(body.text)
            if "error" in result:
                return JSONResponse({"ok":False, **result}, status_code=400)
            return JSONResponse({"ok":True, **result})
        except Exception as e:
            logger.exception("ui.server | /api/terminal/propose: {}", e)
            return JSONResponse({"ok":False,"error":str(e)}, status_code=500)

    @app.post("/api/terminal/confirm")
    async def terminal_confirm(body: TermProposalIdBody):
        term, err = _terminal_or_503()
        if err: return err
        try:
            result = await term.confirm(body.proposal_id)
            if "error" in result:
                return JSONResponse({"ok":False, **result}, status_code=404)
            return JSONResponse({"ok":True, **result})
        except Exception as e:
            logger.exception("ui.server | /api/terminal/confirm: {}", e)
            return JSONResponse({"ok":False,"error":str(e)}, status_code=500)

    @app.post("/api/terminal/cancel")
    async def terminal_cancel(body: TermProposalIdBody):
        term, err = _terminal_or_503()
        if err: return err
        try:
            return JSONResponse(await term.cancel(body.proposal_id))
        except Exception as e:
            logger.exception("ui.server | /api/terminal/cancel: {}", e)
            return JSONResponse({"ok":False,"error":str(e)}, status_code=500)

    @app.post("/api/terminal/reset")
    async def terminal_reset():
        term, err = _terminal_or_503()
        if err: return err
        try:
            await term.reset()
            return JSONResponse({"ok":True})
        except Exception as e:
            logger.exception("ui.server | /api/terminal/reset: {}", e)
            return JSONResponse({"ok":False,"error":str(e)}, status_code=500)

    # Modelli del terminale: lista completa con flag hidden/current
    @app.get("/api/terminal/models")
    async def terminal_models():
        term = state.get("terminal")
        if term is None: return JSONResponse({"models":[],"current":None})
        try:
            mdls = await term.list_available_models()
            return JSONResponse({"models":mdls,"current":term.current_model})
        except Exception as e:
            logger.exception("ui.server | /api/terminal/models: {}", e)
            return JSONResponse({"models":[],"current":None,"error":str(e)})

    @app.post("/api/terminal/model")
    async def terminal_model_switch(body: ModelBody):
        term = state.get("terminal")
        if term is None: return JSONResponse({"ok":False,"error":"terminal non pronto"},status_code=503)
        ok = await term.switch_model(body.name)
        if not ok: return JSONResponse({"ok":False,"error":"modello non valido o nascosto"},status_code=400)
        _save_ui_settings({"terminal_model": body.name})
        return JSONResponse({"ok":True,"name":body.name})

    @app.post("/api/terminal/models/visibility")
    async def terminal_models_visibility(body: TermModelVisibilityBody):
        term = state.get("terminal")
        if term is None: return JSONResponse({"ok":False,"error":"terminal non pronto"},status_code=503)
        # Carica lista corrente di nascosti, applica delta, persiste
        s = _load_ui_settings() or {}
        hidden = set(s.get("terminal_hidden_models", []))
        if body.visible:
            hidden.discard(body.name)
        else:
            hidden.add(body.name)
        hidden_list = sorted(hidden)
        term.set_hidden_models(hidden_list)
        _save_ui_settings({"terminal_hidden_models": hidden_list})
        return JSONResponse({"ok":True,"hidden":hidden_list})

    return app, state
