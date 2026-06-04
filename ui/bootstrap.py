"""
ui/bootstrap.py
Sequenza di avvio condivisa per la UI web di local-assistant.

Sia l'app nativa Qt (ui/app.py) sia il launcher headless (scripts/run_ui.py)
avviano lo stesso stack: server uvicorn + UIBridge + TerminalBridge, con
ripristino di sessioni/impostazioni da disco, ricostruzione della cronologia
dell'LLM, warmup dei modelli e cleanup degli upload.

Prima questa sequenza era duplicata: la versione completa in ui/app.py e una
copia degradata in scripts/run_ui.py (senza persistenza, terminale, restore
impostazioni né cleanup). Tenerla qui come unica fonte di verità evita che i
due entry-point divergano di nuovo.

I due chiamanti differiscono solo per cosa fanno quando il server è in
ascolto: l'app Qt emette il segnale `ready` (carica la webview), il launcher
headless apre il browser. Questo è l'unico punto di variazione, esposto come
hook `on_server_ready`.
"""
from __future__ import annotations

import asyncio
from typing import Callable, Optional

import uvicorn

from config.settings import settings
from core.logger import logger, setup_logging
from core.uploads_cleanup import run_cleanup_at_boot
from ui.bridge import UIBridge, WSManager
from ui.server import (
    _load_sessions_disk,
    _load_ui_settings,
    _save_sessions_disk,
    create_app,
)
from ui.terminal_bridge import TerminalBridge


# ---------------------------------------------------------------------------
# Helper interni
# ---------------------------------------------------------------------------

def _spawn_bg(coro, *, label: str, registry: set) -> "asyncio.Task":
    """
    Lancia un task best-effort tenendone il riferimento finché non completa
    (altrimenti l'event loop ne tiene solo una weak reference e il GC può
    ucciderlo a metà esecuzione). Logga eventuali eccezioni senza propagarle.
    """
    task = asyncio.create_task(coro)
    registry.add(task)

    def _done(t: "asyncio.Task") -> None:
        registry.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.warning("ui.bootstrap | task '{}': {}", label, exc)

    task.add_done_callback(_done)
    return task


def _restore_sessions_from_disk(ui_loop: UIBridge) -> None:
    """
    Sostituisce le sessioni di default con quelle salvate su disco (con i loro
    messaggi) e punta alla sessione usata più di recente. Errori non fatali.
    """
    try:
        saved = _load_sessions_disk()
        if saved and isinstance(saved, dict) and len(saved) > 0:
            ui_loop._sessions = saved
            most_recent = max(
                saved.items(),
                key=lambda kv: kv[1].get("last_active", 0),
            )[0]
            ui_loop._session_id = most_recent
            n_msgs = sum(len(s.get("messages", [])) for s in saved.values())
            logger.info(
                "ui.bootstrap | {} sessioni ripristinate ({} messaggi)",
                len(saved), n_msgs,
            )
    except Exception as exc:
        logger.warning("ui.bootstrap | restore sessions: {}", exc)


async def _restore_ui_settings(
    ui_loop: UIBridge,
    terminal_bridge: TerminalBridge,
) -> None:
    """
    Ripristina modello/voce/personalità/modalità da ui-settings.json.

    Eseguita DOPO ui_loop.load() (quindi con _orch e _tts pronti), altrimenti
    switch_personality/switch_voice fallirebbero perché i moduli sottostanti
    non sono ancora inizializzati. Errori non fatali: logga e prosegue col
    default.
    """
    try:
        s = _load_ui_settings() or {}
    except Exception as exc:
        logger.warning("ui.bootstrap | restore ui-settings (load): {}", exc)
        return

    saved_personality = s.get("personality")
    if saved_personality:
        try:
            ok = ui_loop.switch_personality(saved_personality)
        except Exception as exc:
            ok = False
            logger.warning(
                "ui.bootstrap | switch_personality '{}': {}",
                saved_personality, exc,
            )
        logger.info(
            "ui.bootstrap | personalità ripristinata → '{}'"
            if ok else
            "ui.bootstrap | personalità salvata non valida: '{}' — uso default",
            saved_personality,
        )

    saved_model = s.get("model")
    if saved_model:
        try:
            ok = ui_loop.switch_model(saved_model)
        except Exception as exc:
            ok = False
            logger.warning("ui.bootstrap | switch_model '{}': {}", saved_model, exc)
        logger.info(
            "ui.bootstrap | modello ripristinato → '{}'"
            if ok else
            "ui.bootstrap | modello salvato non valido: '{}' — uso default",
            saved_model,
        )

    saved_voice = s.get("voice")
    if saved_voice:
        try:
            ok = await ui_loop.switch_voice(saved_voice)
        except Exception as exc:
            ok = False
            logger.warning("ui.bootstrap | switch_voice '{}': {}", saved_voice, exc)
        logger.info(
            "ui.bootstrap | voce ripristinata → '{}'"
            if ok else
            "ui.bootstrap | voce salvata non valida: '{}' — uso default",
            saved_voice,
        )

    saved_tts = s.get("tts_enabled")
    if saved_tts is not None:
        try:
            ui_loop.set_tts_enabled(bool(saved_tts))
            logger.info(
                "ui.bootstrap | TTS ripristinato → {}",
                "attivo" if saved_tts else "disattivato",
            )
        except Exception as exc:
            logger.warning("ui.bootstrap | restore tts_enabled: {}", exc)

    # ── Terminale: hidden_models + modello + modalità ────────────────────
    # Applichiamo SEMPRE i nascosti (anche restando in chat), così quando
    # l'utente passerà a terminal la lista è già corretta.
    try:
        hidden = s.get("terminal_hidden_models") or []
        if isinstance(hidden, list):
            terminal_bridge.set_hidden_models(hidden)
    except Exception as exc:
        logger.warning("ui.bootstrap | restore terminal_hidden_models: {}", exc)

    # Carichiamo SEMPRE il TerminalBridge (non lazy al primo switch): lo switch
    # chat→terminal dal frontend è una semplice POST /api/mode; se il bridge non
    # è caricato a quel punto, gli endpoint terminal/* tornano 500.
    try:
        await terminal_bridge.load()
        saved_term_model = s.get("terminal_model")
        if saved_term_model:
            ok = await terminal_bridge.switch_model(saved_term_model)
            if not ok:
                logger.info(
                    "ui.bootstrap | modello terminale salvato non più "
                    "disponibile: '{}'", saved_term_model,
                )
    except Exception as exc:
        logger.warning(
            "ui.bootstrap | load terminal_bridge fallito: {} — "
            "la modalità terminale non sarà disponibile", exc,
        )

    saved_mode = s.get("mode")
    if saved_mode == "terminal":
        try:
            await ui_loop.set_mode("terminal")
            logger.info("ui.bootstrap | modalità ripristinata → terminal")
        except Exception as exc:
            logger.warning(
                "ui.bootstrap | restore terminal mode fallito: {} — "
                "resto in chat", exc,
            )


# ---------------------------------------------------------------------------
# Entry point condiviso
# ---------------------------------------------------------------------------

async def serve(
    *,
    personality: Optional[str] = None,
    ptt_key: str = "space",
    on_server_ready: Optional[Callable[[], None]] = None,
) -> None:
    """
    Avvia l'intero stack UI e blocca finché server e loop sono in esecuzione.

    Sequenza:
        1. uvicorn in ascolto (host/port da settings.api) → i client possono
           connettersi subito.
        2. on_server_ready() (hook del chiamante: emit segnale Qt / apri
           browser). Best-effort.
        3. UIBridge + TerminalBridge, persistenza, restore sessioni da disco.
        4. async with ui_loop: restore impostazioni, ricostruzione cronologia
           LLM, broadcast_init, warmup + cleanup in background.
        5. gather(loop PTT, server) fino allo stop.

    Args:
        personality:     profilo da attivare (None = default da settings).
        ptt_key:         tasto push-to-talk.
        on_server_ready: callback invocata quando il server è in ascolto, prima
                         del caricamento dell'assistente. Non deve sollevare.
    """
    # Attiva i sink su file (assistant.log / errors.log in PROJECT_ROOT/logs).
    # Qui, nell'entry point condiviso, così sia run_ui.py sia l'app Qt scrivono
    # i log nello stesso posto a prescindere da dove vengono lanciati.
    setup_logging()

    ws_mgr = WSManager()
    app, state = create_app(ws_mgr)

    host = settings.api.host
    port = settings.api.port

    # ── 1. Server subito ─────────────────────────────────────────────────
    server = uvicorn.Server(uvicorn.Config(
        app, host=host, port=port, log_level="warning", loop="none",
    ))
    server_task = asyncio.create_task(server.serve())
    # Breve pausa per garantire che uvicorn sia in ascolto prima del callback.
    await asyncio.sleep(0.4)

    # ── 2. Hook "server pronto" ──────────────────────────────────────────
    if on_server_ready is not None:
        try:
            on_server_ready()
        except Exception as exc:
            logger.warning("ui.bootstrap | on_server_ready: {}", exc)
    logger.info("ui.bootstrap | server avviato → http://{}:{}", host, port)

    # Notifica i client già connessi che il caricamento è in corso.
    await ws_mgr.broadcast({"type": "state", "value": "loading"})

    # ── 3. Bridge + persistenza + restore sessioni ───────────────────────
    ui_loop = UIBridge(ws_manager=ws_mgr, ptt_key=ptt_key, personality=personality)

    terminal_bridge = TerminalBridge(ws_manager=ws_mgr)
    ui_loop.set_terminal_bridge(terminal_bridge)
    state["terminal"] = terminal_bridge

    ui_loop.set_persist_callback(_save_sessions_disk)
    _restore_sessions_from_disk(ui_loop)

    state["loop"] = ui_loop

    bg_tasks: set = set()
    try:
        async with ui_loop:
            # ui_loop.load() eseguito da __aenter__: _orch e _tts pronti.
            await _restore_ui_settings(ui_loop, terminal_bridge)

            # Ricostruisce la finestra conversazionale dell'LLM dai messaggi
            # salvati: dopo un riavvio l'assistente "ricorda" i turni
            # precedenti (non solo la UI).
            try:
                ui_loop.restore_histories_from_sessions()
            except Exception as exc:
                logger.warning("ui.bootstrap | restore histories: {}", exc)

            # Ripristina i file RAG per sessione: dopo un riavvio le chat con
            # file caricati tornano interrogabili senza ricaricare il file.
            try:
                ui_loop.restore_rag_files_from_sessions()
            except Exception as exc:
                logger.warning("ui.bootstrap | restore rag_files: {}", exc)

            # Chiude la race d'avvio: i client connessi prima che il loop fosse
            # pronto ricevono ora l'init completo senza dover riconnettere.
            try:
                await ui_loop.broadcast_init()
            except Exception as exc:
                logger.warning("ui.bootstrap | broadcast_init: {}", exc)

            # Warmup modelli (chat + embed) e cleanup uploads in background:
            # non bloccano la UI. Best-effort, riferimenti tenuti per il GC.
            _spawn_bg(ui_loop.warmup(), label="warmup", registry=bg_tasks)
            _spawn_bg(run_cleanup_at_boot(), label="cleanup", registry=bg_tasks)

            # ── Loop PTT + server in parallelo ───────────────────────────
            try:
                await asyncio.gather(ui_loop.run(), server_task)
            except asyncio.CancelledError:
                pass
    finally:
        # Chiudi il TerminalBridge se caricato (idempotente) e ferma il server.
        try:
            await terminal_bridge.aclose()
        except Exception as exc:
            logger.debug("ui.bootstrap | aclose terminal_bridge: {}", exc)
        server.should_exit = True
