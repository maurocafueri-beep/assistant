"""
ui/bridge.py  —  WSManager + UIBridge
Aggiunge model listing/switching rispetto alla versione precedente.
"""
from __future__ import annotations
import asyncio, json, time
from dataclasses import dataclass
from typing import Any, Set
from core.logger import logger
from core.voice_loop import (
    VoiceLoop, LoopState,
    _PTT_MIN_DURATION_S, _PTT_SAMPLE_RATE, _PTT_BLOCK_SIZE,
)

_SPECIAL_KEYS = {"space","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12",
                 "ctrl_l","ctrl_r","shift_l","shift_r","alt_l","alt_r","caps_lock","tab",
                 "insert","scroll_lock","pause","num_lock"}

def _resolve_key(key_str: str) -> Any:
    from pynput import keyboard as kb
    k = key_str.lower().strip()
    if k in _SPECIAL_KEYS: return getattr(kb.Key, k, None)
    if len(k) == 1: return kb.KeyCode.from_char(k)
    return None

class WSManager:
    def __init__(self) -> None:
        self._clients: Set[Any] = set()
        self._lock = asyncio.Lock()
    async def connect(self, ws: Any) -> None:
        await ws.accept()
        async with self._lock: self._clients.add(ws)
        logger.debug("ui.bridge | WS connesso ({})", len(self._clients))
    def disconnect(self, ws: Any) -> None:
        self._clients.discard(ws)
    async def broadcast(self, payload: dict) -> None:
        if not self._clients: return
        text = json.dumps(payload, ensure_ascii=False)
        dead = []
        async with self._lock: clients = list(self._clients)
        for ws in clients:
            try: await ws.send_text(text)
            except: dead.append(ws)
        for ws in dead: self.disconnect(ws)
    @property
    def n_clients(self): return len(self._clients)

@dataclass
class _TextMessage:
    text: str
    def is_empty(self): return not self.text.strip()

class UIBridge(VoiceLoop):
    def __init__(self, ws_manager: WSManager, ptt_key: str = "space", **kw):
        super().__init__(**kw)
        self._ws = ws_manager
        self._ptt_key_str = ptt_key
        self._ptt_key_obj = _resolve_key(ptt_key)
        # PTT pilotato dal browser via WebSocket (keydown/keyup) invece del
        # listener globale pynput: quest'ultimo, col backend X11/Xlib, non
        # riceve eventi sotto Wayland. Il frontend sa già qual è il tasto
        # (lo confronta in JS) e ci notifica down/up; qui basta un Event.
        self._ptt_held = asyncio.Event()
        # Le latenze (_last_stt_ms / _last_llm_ms / _last_tts_ms) sono
        # definite e popolate da VoiceLoop; qui le leggiamo soltanto.

        # ── Multi-sessione ──────────────────────────────────────────────
        # Registry: session_id → {name, created, last_active, messages:[{role,text,ts}]}
        self._sessions: dict[str, dict] = {}
        self._persist_cb = None   # set by app.py: callable(sessions_dict)
        self._register_session(self._session_id, name="Chat 1")

        # ── Modalità (chat | terminal) ───────────────────────────────────
        # Switch top-level dell'UI. In "terminal" il flusso turn-by-turn
        # NON arriva a _stream_and_speak: viene deviato a _terminal_bridge.
        # Le primitive STT/TTS restano caricate (PTT continua a funzionare).
        self._mode: str = "chat"
        self._terminal_bridge = None   # set da app.py: TerminalBridge instance
        # Stato del TTS prima di entrare in terminale, per ripristinarlo al
        # ritorno in chat (in terminale la voce è forzata OFF).
        self._tts_before_terminal: bool | None = None

        # Task fire-and-forget di _emit: vanno tenuti referenziati finché
        # non completano, altrimenti l'event loop ne tiene solo una weak
        # reference e possono essere garbage-collected a metà broadcast.
        self._bg_tasks: set = set()

    def set_terminal_bridge(self, bridge) -> None:
        """app.py registra qui il TerminalBridge per la modalità terminale."""
        self._terminal_bridge = bridge

    @property
    def mode(self) -> str:
        return self._mode

    async def set_mode(self, mode: str) -> bool:
        """
        Cambia modalità globale ('chat' | 'terminal'). Broadcast WS.

        In modalità terminale la voce viene forzata OFF (shell con
        comportamento prevedibile): se l'assistente sta parlando, la
        riproduzione viene tagliata all'istante. Tornando in chat lo stato
        precedente del TTS viene ripristinato, così l'utente non deve
        riattivare la voce manualmente ogni volta.
        """
        if mode not in ("chat", "terminal"):
            return False
        if mode == self._mode:
            return True
        self._mode = mode

        if mode == "terminal":
            # Ricorda lo stato voce per ripristinarlo al ritorno in chat e
            # zittisci subito (set_tts_enabled taglia anche l'audio in corso).
            self._tts_before_terminal = self._tts_enabled
            if self._tts_enabled:
                self.set_tts_enabled(False)
        else:  # chat
            prev = getattr(self, "_tts_before_terminal", None)
            if prev:
                self.set_tts_enabled(True)
            self._tts_before_terminal = None

        # Notifica subito la UI così cambia interfaccia
        self._emit({"type": "mode", "mode": mode})
        logger.info("ui.bridge | mode → {}", mode)
        return True

    def set_persist_callback(self, cb) -> None:
        """app.py registra qui la funzione che salva _sessions su disco."""
        self._persist_cb = cb

    def _persist(self) -> None:
        if self._persist_cb is not None:
            try:
                self._persist_cb(self._sessions)
            except Exception as e:
                logger.warning("ui.bridge | persist fallito: {}", e)

    # ── Session management ───────────────────────────────────────────────
    def _register_session(self, sid: str, name: str) -> None:
        import time as _t
        self._sessions[sid] = {
            "id": sid, "name": name,
            "created": _t.time(), "last_active": _t.time(),
            "messages": [],
            "rag_files": [],
        }

    def list_sessions(self) -> list[dict]:
        out = []
        for sid, s in self._sessions.items():
            msgs = s["messages"]
            last = msgs[-1]["text"] if msgs else ""
            out.append({
                "id": sid, "name": s["name"],
                "active": sid == self._session_id,
                "created": s["created"], "last_active": s["last_active"],
                "preview": (last[:50] + "…") if len(last) > 50 else last,
                "count": len(msgs),
            })
        # Most recently active first
        out.sort(key=lambda x: x["last_active"], reverse=True)
        return out

    def new_session(self, name: str | None = None) -> str:
        import uuid, time as _t
        sid = str(uuid.uuid4())[:8]
        n = name or f"Chat {len(self._sessions) + 1}"
        self._register_session(sid, n)
        self._session_id = sid
        self._emit({"type": "sessions", "sessions": self.list_sessions()})
        logger.info("ui.bridge | nuova sessione '{}' ({})", n, sid)
        return sid

    def switch_session(self, sid: str) -> bool:
        if sid not in self._sessions:
            logger.warning("ui.bridge | sessione non trovata: {}", sid)
            return False
        self._session_id = sid
        self._sessions[sid]["last_active"] = __import__("time").time()
        self._emit({
            "type": "session_switch",
            "id": sid,
            "messages": self._sessions[sid]["messages"],
            "sessions": self.list_sessions(),
        })
        logger.info("ui.bridge | switch sessione → {}", sid)
        return True

    def delete_session(self, sid: str) -> bool:
        if sid not in self._sessions or len(self._sessions) <= 1:
            return False
        del self._sessions[sid]
        try:
            self._orch.clear_session(sid)
        except Exception:
            pass
        if self._session_id == sid:
            # Switch to the most recently active remaining session
            self._session_id = max(
                self._sessions.items(),
                key=lambda kv: kv[1].get("last_active", 0),
            )[0]
        self._persist()
        self._emit({"type": "sessions", "sessions": self.list_sessions()})
        return True

    def rename_session(self, sid: str, name: str) -> bool:
        if sid not in self._sessions:
            return False
        self._sessions[sid]["name"] = name.strip()[:40] or self._sessions[sid]["name"]
        self._emit({"type": "sessions", "sessions": self.list_sessions()})
        return True

    def restore_histories_from_sessions(self) -> int:
        """
        Ricostruisce la finestra conversazionale dell'orchestratore
        (_session_histories) a partire dai messaggi salvati su disco.

        Senza questo, dopo un riavvio la UI mostra la chat completa ma
        l'LLM non ha memoria verbatim dei turni precedenti: il primo
        messaggio ripartirebbe con cronologia vuota.

        Va chiamato DOPO load() (orchestratore pronto). Ritorna il numero
        di sessioni ripristinate. Errori non fatali.
        """
        from core.orchestrator import _trim_history
        restored = 0
        try:
            window = self._orch._context_window
        except Exception:
            window = 20
        for sid, s in self._sessions.items():
            msgs = s.get("messages") or []
            history: list[dict[str, str]] = []
            for m in msgs:
                role = "assistant" if m.get("role") == "asst" else m.get("role", "user")
                content = m.get("text", "")
                if content:
                    history.append({"role": role, "content": content})
            if history:
                try:
                    self._orch._session_histories[sid] = _trim_history(history, window)
                    restored += 1
                except Exception as exc:
                    logger.warning("ui.bridge | restore history {}: {}", sid, exc)
        logger.info("ui.bridge | history orchestratore ripristinata per {} sessioni", restored)
        return restored

    def restore_rag_files_from_sessions(self) -> int:
        """
        Reinietta nell'orchestrator i file RAG salvati per sessione, così dopo
        un riavvio il RAG dei file caricati torna disponibile senza ricaricarli.
        Gemello di restore_histories_from_sessions; va chiamato DOPO load().
        Errori non fatali. Ritorna il numero di sessioni ripristinate.
        """
        restored = 0
        for sid, s in self._sessions.items():
            rag = s.get("rag_files") or []
            if not rag:
                continue
            try:
                self._orch._session_rag_files[sid] = list(rag)
                restored += 1
            except Exception as exc:
                logger.warning("ui.bridge | restore rag_files {}: {}", sid, exc)
        logger.info("ui.bridge | rag_files ripristinati per {} sessioni", restored)
        return restored

    def _record_message(self, role: str, text: str) -> None:
        import time as _t
        s = self._sessions.get(self._session_id)
        if s is not None:
            s["messages"].append({"role": role, "text": text, "ts": _t.time()})
            s["last_active"] = _t.time()
            self._persist()   # salva su disco dopo OGNI messaggio

    # ── Personality ──────────────────────────────────────────────────
    def list_personalities(self) -> list[dict]:
        try:
            pm = self._orch._personality
            return [{"name":p.name,"display_name":p.display_name,"description":p.description}
                    for p in [pm.get(n) for n in pm.list_profiles()]]
        except Exception as e:
            logger.warning("ui.bridge | list_personalities: {}", e); return []

    @property
    def active_personality(self) -> str:
        try: return self._orch._personality.active.name
        except: return "default"

    def switch_personality(self, name: str) -> bool:
        try:
            self._orch.switch_personality(name)
            p = self._orch._personality.active
            self._emit({"type":"personality","name":p.name,"display_name":p.display_name})
            return True
        except KeyError:
            logger.warning("ui.bridge | personality non trovata: {}", name); return False

    # ── Model ────────────────────────────────────────────────────────
    async def list_models(self) -> list[str]:
        try: return await self._orch._llm.list_models()
        except Exception as e:
            logger.warning("ui.bridge | list_models: {}", e); return []

    @property
    def active_model(self) -> str:
        from core.context import ModelRole
        try:
            return self._orch.active_model(ModelRole.CHAT)
        except Exception:
            from config.settings import settings
            return settings.ollama.chat_model

    def switch_model(self, name: str) -> bool:
        try:
            from core.context import ModelRole
            # Stato isolato sull'orchestratore: non muta settings.ollama.*
            self._orch.set_model(name, ModelRole.CHAT)
            self._emit({"type":"model","name":name})
            logger.info("ui.bridge | modello → '{}'", name)
            return True
        except Exception as e:
            logger.error("ui.bridge | switch_model: {}", e); return False

    # ── Warmup scheduling (fire-and-forget) ──────────────────────────
    # Lo switch di modello/modalità non deve bloccare la UI: l'override del
    # modello vale comunque dal turno successivo. Qui lanciamo il warmup
    # mirato in background, con segnalino "warmup", e teniamo il task
    # referenziato (come _emit) per evitare il GC a metà esecuzione.
    def _spawn(self, coro) -> None:
        try:
            t = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            # Nessun event loop in esecuzione: niente da schedulare.
            try:
                coro.close()
            except Exception:
                pass
            return
        self._bg_tasks.add(t)
        t.add_done_callback(self._bg_tasks.discard)

    def schedule_warmup(self, model: str, *, include_tts: bool = False) -> None:
        """Warmup mirato del modello indicato, in background. No-op se vuoto."""
        if not model:
            return
        self._spawn(self.warmup_switch(model, include_tts=include_tts))

    def schedule_mode_warmup(self) -> None:
        """
        Warmup del modello rilevante per la modalità CORRENTE (da chiamare
        dopo set_mode): in 'terminal' è il modello del TerminalBridge, in
        'chat' è il modello chat attivo dell'orchestratore. Il TTS NON viene
        riscaldato qui (il suo warmup è una preoccupazione d'avvio): in
        terminale la voce è forzata OFF, in chat il modello TTS è già
        residente dal boot.
        """
        if self._mode == "terminal":
            model = getattr(self._terminal_bridge, "current_model", None) \
                if self._terminal_bridge is not None else None
        else:
            model = self.active_model
        self.schedule_warmup(model)

    # ── Voice ───────────────────────────────────────────────────────────
    async def list_voices(self) -> list[str]:
        # Durante l'avvio lazy il TTS può non essere ancora pronto:
        # ritorna [] senza loggare (non è un errore, è transitorio).
        if self._tts is None:
            return []
        try: return await self._tts.available_profiles()
        except Exception as e:
            logger.warning("ui.bridge | list_voices: {}", e); return []

    def _voice_fallback(self) -> str:
        """
        Voce di ripiego quando il server TTS non è ancora pronto.
        Ordine: voce salvata in ui-settings.json → settings.tts.voice → "".
        Non solleva mai eccezioni.
        """
        try:
            from ui.server import _load_ui_settings
            saved = (_load_ui_settings() or {}).get("voice")
            if saved:
                return saved
        except Exception:
            pass
        try:
            from config.settings import settings
            return settings.tts.voice
        except Exception:
            return ""

    async def active_voice_from_server(self) -> str:
        """
        Legge la voce attiva direttamente dal server TTS.
        Se il TTS non è ancora caricato (avvio lazy), ritorna un
        fallback sensato senza crashare — errore non fatale.
        """
        if self._tts is None:
            return self._voice_fallback()
        try:
            r = await self._tts._http.get("/health")
            r.raise_for_status()
            return r.json().get("profile", self._tts._profile)
        except Exception:
            try:
                return self._tts._profile
            except Exception:
                return self._voice_fallback()

    @property
    def active_voice(self) -> str:
        if self._tts is None:
            return self._voice_fallback()
        try:
            return self._tts._profile
        except Exception:
            return self._voice_fallback()

    async def switch_voice(self, name: str) -> bool:
        try:
            r = await self._tts._http.post(f"/switch/{name}")
            r.raise_for_status()
            data = r.json()
            # Server confirms the new active profile
            confirmed = data.get("profile", name)
            self._tts._profile = confirmed
            self._emit({"type": "voice", "name": confirmed})
            logger.info("ui.bridge | voce → '{}' (confermata: '{}')", name, confirmed)
            return True
        except Exception as e:
            logger.error("ui.bridge | switch_voice '{}': {}", name, e)
            return False

    # ── TTS on/off ───────────────────────────────────────────────────
    def set_tts_enabled(self, enabled: bool) -> bool:
        """
        Attiva/disattiva la sintesi vocale.
        La chat testuale con l'LLM continua a funzionare in entrambi i casi.

        Se viene disattivata MENTRE l'assistente sta parlando, la voce viene
        zittita all'istante (riproduzione corrente tagliata + audio in coda
        scartato), senza interrompere lo streaming del testo — come ci si
        aspetta da un pulsante "muta".

        Notifica la UI via WebSocket. Ritorna sempre True (operazione locale).
        """
        was_enabled = self._tts_enabled
        self._tts_enabled = bool(enabled)
        # Mute durante il parlato: taglia subito l'audio in corso/in coda.
        if was_enabled and not self._tts_enabled:
            self.interrupt_tts()
        self._emit({"type": "tts", "enabled": self._tts_enabled})
        logger.info("ui.bridge | TTS {}", "attivo" if self._tts_enabled else "disattivato")
        return True

    # ── PTT ──────────────────────────────────────────────────────────
    async def send_text(self, text: str) -> None:
        text = text.strip()
        if not text: return
        try: self._turn_queue.put_nowait(_TextMessage(text=text))
        except asyncio.QueueFull: logger.warning("ui.bridge | queue piena")

    def set_ptt_key(self, key_str: str) -> bool:
        r = _resolve_key(key_str)
        if r is None: return False
        self._ptt_key_str = key_str; self._ptt_key_obj = r
        self._emit({"type":"ptt_key","key":key_str})
        return True

    @property
    def ptt_key(self): return self._ptt_key_str

    # Invocati dall'handler WebSocket (stesso event loop): segnalano che il
    # tasto PTT è premuto/rilasciato. _run_ptt aspetta questo Event.
    def ptt_down(self) -> None:
        self._ptt_held.set()

    def ptt_up(self) -> None:
        self._ptt_held.clear()

    async def broadcast_init(self) -> None:
        """
        Invia a TUTTI i client WS connessi il payload 'init' completo.

        Serve a chiudere la race d'avvio: la webview può connettersi
        mentre il loop non è ancora pronto (state['loop'] None nel server),
        ricevendo uno stato vuoto. Una volta caricati orchestratore/TTS e
        ripristinate le impostazioni, ribroadcastiamo init così i client
        già connessi si popolano senza dover riconnettere. Il client
        gestisce 'init' in modo idempotente.
        """
        try:
            payload = {
                "type": "init",
                "state": self.state,
                "ptt_key": self.ptt_key,
                "session": self.session_id,
                "sessions": self.list_sessions(),
                "active_messages": self._sessions.get(self.session_id, {}).get("messages", []),
                "personality": self.active_personality,
                "personalities": self.list_personalities(),
                "model": self.active_model,
                "voice": await self.active_voice_from_server(),
                "voices": await self.list_voices(),
                "tts_enabled": self.tts_enabled,
                "stats": self.stats.to_log_dict(),
            }
            await self._ws.broadcast(payload)
            logger.debug("ui.bridge | init ribroadcastato a {} client", self._ws.n_clients)
        except Exception as exc:
            logger.warning("ui.bridge | broadcast_init: {}", exc)

    # ── WS overrides ─────────────────────────────────────────────────
    def _emit(self, payload: dict) -> None:
        try:
            t = asyncio.get_running_loop().create_task(self._ws.broadcast(payload))
        except RuntimeError:
            return
        self._bg_tasks.add(t)
        t.add_done_callback(self._bg_tasks.discard)

    def _set_state(self, state: str) -> None:
        super()._set_state(state)
        self._emit({"type":"state","value":state})

    async def _process_turn(self, stt_result: Any) -> None:
        """
        Override completo di VoiceLoop._process_turn.
        Identico al base ma aggiunge broadcast di testo utente,
        statistiche e latenze dopo ogni turno.
        """
        from core.context import AssistantContext, InputMode, OutputMode
        from core.voice_loop import _PTT_ECHO_GRACE_S

        user_text = stt_result.text.strip()
        if not user_text:
            return

        # ── Deviazione modalità terminale ──────────────────────────────
        # In "terminal" la trascrizione NON entra nel turn chat: viene
        # passata al TerminalBridge che la mostra nel box di input.
        # L'utente premerà Invio per confermare → POST /api/terminal/propose.
        if self._mode == "terminal" and self._terminal_bridge is not None:
            try:
                await self._terminal_bridge.handle_voice_input(user_text)
            except Exception as exc:
                logger.warning("ui.bridge | terminal handle_voice_input: {}", exc)
            return

        # Notifica UI del testo utente + registra nella sessione
        self._record_message("user", user_text)
        await self._ws.broadcast({
            "type": "user", "text": user_text, "session": self._session_id
        })

        # ── Replica esatta di VoiceLoop._process_turn ──────────────────
        self._is_speaking = True
        self._set_state("thinking")
        self._stats.turns += 1
        self._stats.total_words_in += len(user_text.split())

        print(f"\n{chr(8212)*50}")
        print(f"\U0001f3a4  Tu:  {user_text}")
        print(f"\U0001f916  Assistente: ", end="", flush=True)

        ctx = AssistantContext(
            user_text   = user_text,
            session_id  = self._session_id,
            input_mode  = InputMode.TEXT,
            output_mode = OutputMode.TEXT,
            model_role  = self._model_role,
        )

        try:
            await self._stream_and_speak(ctx)
        except Exception as exc:
            logger.error("ui.bridge | _process_turn fallito: {}", exc)
            print(f"\n  ⚠ Errore: {exc}")
        finally:
            self._is_speaking = False
            if self._tts_last_play_end > 0:
                self._echo_block_until = self._tts_last_play_end + _PTT_ECHO_GRACE_S
            self._tts_last_play_end  = 0.0
            self._tts_speaking_start = 0.0
            self._set_state("listening")
            self._stats.total_words_out += len(ctx.assistant_text.split())
            # Salva i file RAG della sessione in _sessions[sid]: il _persist()
            # di _record_message li scrive su disco e così sopravvivono al
            # riavvio (l'orchestrator li tiene solo in RAM in _session_rag_files).
            try:
                rag = self._orch._session_rag_files.get(self._session_id)
                if rag:
                    self._sessions[self._session_id]["rag_files"] = list(rag)
            except Exception as exc:
                logger.warning("ui.bridge | persist rag_files: {}", exc)
            if ctx.assistant_text.strip():
                self._record_message("asst", ctx.assistant_text.strip())
            print()
            logger.info("ui.bridge | turno {} | {}", self._stats.turns, ctx.to_log_dict())

            # ── Broadcast statistiche + latenze ───────────────────────────
            llm_ms = self._last_llm_ms or None
            stt_ms = self._last_stt_ms or None
            tts_ms = self._last_tts_ms or None
            total  = round(sum(v for v in [stt_ms, llm_ms, tts_ms] if v), 1)
            await self._ws.broadcast({
                "type":      "stats",
                "turns":     self._stats.turns,
                "words_in":  self._stats.total_words_in,
                "words_out": self._stats.total_words_out,
                "stt_errors":self._stats.stt_errors,
                "tts_errors":self._stats.tts_errors,
                "latency": {
                    "stt_ms":   stt_ms,
                    "llm_ms":   llm_ms,
                    "tts_ms":   tts_ms,
                    "total_ms": total,
                },
            })

    def _on_llm_chunk(self, chunk: str) -> None:
        # Hook del loop base: inoltra ogni chunk LLM alla UI via WebSocket.
        # Niente monkey-patching: le latenze sono misurate dal base loop.
        self._emit({"type": "chunk", "text": chunk})

    async def _run_ptt(self) -> None:
        import numpy as np, sounddevice as sd
        # Niente listener globale pynput (backend X11/Xlib → muto su Wayland):
        # il tasto premuto/rilasciato arriva dal browser via WS e pilota
        # _ptt_held (ptt_down/ptt_up). Stesso event loop, nessun thread.
        held = self._ptt_held
        held.clear()
        logger.info("ui.bridge | PTT attivo (WS) — tasto: '{}'", self._ptt_key_str)
        try:
            while not self._stop_event.is_set():
                try: await asyncio.wait_for(held.wait(), timeout=0.5)
                except asyncio.TimeoutError: continue
                if self._stop_event.is_set(): break
                if self._is_speaking or time.monotonic() < self._echo_block_until:
                    while held.is_set(): await asyncio.sleep(0.02)
                    continue
                self._set_state(LoopState.RECORDING)
                frames, t0 = [], time.monotonic()
                stream = sd.InputStream(samplerate=_PTT_SAMPLE_RATE, channels=1, dtype="int16", blocksize=_PTT_BLOCK_SIZE)
                stream.start()
                try:
                    while held.is_set() and not self._stop_event.is_set():
                        data, _ = stream.read(_PTT_BLOCK_SIZE)
                        frames.append(data.copy()); await asyncio.sleep(0.005)
                finally: stream.stop(); stream.close()
                dur = time.monotonic() - t0
                self._set_state(LoopState.LISTENING)
                if dur < _PTT_MIN_DURATION_S or not frames: continue
                try:
                    _t0_stt = time.monotonic()
                    r = await self._stt.transcribe(np.concatenate(frames).tobytes())
                    self._last_stt_ms = round((time.monotonic() - _t0_stt) * 1000, 1)
                    if not r.is_empty():
                        try: self._turn_queue.put_nowait(r)
                        except asyncio.QueueFull: pass
                except Exception as e:
                    self._stats.stt_errors += 1; logger.error("ui.bridge | STT: {}", e)
        finally: held.clear()
