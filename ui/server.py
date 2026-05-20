"""ui/server.py — FastAPI app con tutti gli endpoint inclusi model switching."""
from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from core.logger import logger
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

def create_app(ws_manager: "WSManager") -> tuple[FastAPI, dict]:
    app   = FastAPI(title="local-assistant", docs_url=None, redoc_url=None)
    state: dict = {"loop": None}

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse((_STATIC / "index.html").read_text(encoding="utf-8"))

    @app.websocket("/ws")
    async def ws_ep(ws: WebSocket):
        await ws_manager.connect(ws)
        loop: "UIBridge | None" = state["loop"]
        if loop is not None:
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
        return JSONResponse({"state":loop.state,"session":loop.session_id,
            "ptt_key":loop.ptt_key,"personality":loop.active_personality,"model":loop.active_model,
                "voice":await loop.active_voice_from_server(),
                "voices":await loop.list_voices(),
            "stats":loop.stats.to_log_dict()})

    @app.post("/api/send")
    async def send(body: SendBody):
        loop = state["loop"]
        if loop is None: return JSONResponse({"ok":False,"error":"loop non pronto"},status_code=503)
        await loop.send_text(body.text); return JSONResponse({"ok":True})


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
    async def settings():
        return JSONResponse(_load_ui_settings())

    return app, state
