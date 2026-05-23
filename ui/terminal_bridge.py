"""
ui/terminal_bridge.py
TerminalBridge — bridge tra il TerminalAgent e la UI web.

Responsabilità:
- Mantiene un'istanza di TerminalAgent caricata.
- Gestisce un piccolo registro di proposte "in attesa di conferma"
  (proposal_id → CommandProposal), così la UI può confermare/annullare
  in modo asincrono.
- Gestisce la lista dei modelli "thinking" mostrati nel selettore
  della modalità terminale (filtro per family + overrides utente).
- Espone metodi che ritornano dict JSON-ready per gli endpoint REST,
  e fa broadcast WS dei cambi di stato (proposta, risultato, cwd, modello).

Pattern: mirror semantico di UIBridge ma senza l'eredità da VoiceLoop:
il terminale non usa TTS, non ha personalità né sessioni, non ha PTT
proprio (la trascrizione STT arriva dal UIBridge che la devia qui
quando _mode == "terminal").

Lifecycle:
    bridge = TerminalBridge(ws_manager=...)
    await bridge.load()
    ...
    await bridge.aclose()

In ui/app.py viene creato accanto a UIBridge e collegato:
    ui_loop.set_terminal_bridge(terminal_bridge)
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from core.logger import logger
from modules.terminal_agent import (
    AgentTurn,
    CommandProposal,
    CommandResult,
    RiskLevel,
    TerminalAgent,
)


# ---------------------------------------------------------------------------
# Helpers di serializzazione (dataclass del modulo → dict JSON-ready)
# ---------------------------------------------------------------------------

def _proposal_to_dict(p: CommandProposal) -> dict[str, Any]:
    return {
        "proposal_id":        p.proposal_id,
        "command":            p.command,
        "rationale":          p.rationale,
        "risk_level":         p.risk_level.value,
        "needs_confirmation": p.needs_confirmation,
        "cwd":                p.cwd,
        "search_used":        p.search_used,
        "sources":            list(p.sources),
    }


def _result_to_dict(r: CommandResult) -> dict[str, Any]:
    return {
        "proposal_id": r.proposal_id,
        "command":     r.command,
        "exit_code":   r.exit_code,
        "stdout":      r.stdout,
        "stderr":      r.stderr,
        "duration_ms": r.duration_ms,
        "cwd_before":  r.cwd_before,
        "cwd_after":   r.cwd_after,
        "truncated":   r.truncated,
        "success":     r.success,
    }


def _turn_to_dict(t: AgentTurn) -> dict[str, Any]:
    return {
        "user_request": t.user_request,
        "proposal":     _proposal_to_dict(t.proposal) if t.proposal else None,
        "result":       _result_to_dict(t.result)     if t.result   else None,
        "analysis":     t.analysis,
        "skipped":      t.skipped,
        "error":        t.error,
    }


# ---------------------------------------------------------------------------
# TerminalBridge
# ---------------------------------------------------------------------------

class TerminalBridge:
    """
    Bridge tra TerminalAgent e UI. Gestisce il ciclo propose → confirm/cancel
    in modo asincrono via WebSocket.
    """

    # ── lifecycle ────────────────────────────────────────────────────────

    def __init__(self, ws_manager: Any) -> None:
        self._ws = ws_manager
        self._agent: Optional[TerminalAgent] = None
        self._loaded: bool = False

        # proposte in attesa di conferma: proposal_id → CommandProposal
        self._pending: dict[str, CommandProposal] = {}

        # history dei turni completati (memoria di sessione, no disco)
        self._history: list[AgentTurn] = []

        # modello attivo per il terminale. None finché load() non sceglie
        # il primo modello visibile (deciso runtime da /api/tags + filter).
        # NON usiamo settings.ollama.chat_model come fallback perché
        # quello potrebbe essere un non-thinking (es. gemma3) che fa
        # crashare il TerminalAgent con 400 su think:true.
        self._current_model: Optional[str] = None

        # whitelist family per i modelli "thinking"
        self._thinking_families: list[str] = list(
            getattr(settings.terminal_agent, "thinking_families",
                    ["qwen35", "qwen35moe", "qwen3", "qwen3moe"])
        )

        # overrides utente: {"hidden": ["model:tag", ...]}
        # popolato da ui/server.py da ui-settings.json
        self._hidden_models: set[str] = set()

    async def load(self) -> None:
        if self._loaded:
            return

        # crea il TerminalAgent con il modello corrente
        self._agent = TerminalAgent(
            model_role=None,  # default da settings, override via switch_model
        )
        await self._agent.load()
        self._loaded = True

        # ── Localizzazione: percorsi XDG ─────────────────────────────────
        # Detect dei percorsi utente nella lingua/configurazione corrente.
        # Su Ubuntu IT: Scrivania/Scaricati/Documenti/Immagini/Video/Musica.
        # I path vengono passati all'agent che li include nel system prompt,
        # così il modello non inventa più Desktop/Downloads in inglese.
        try:
            xdg = _detect_xdg_user_paths()
            if xdg:
                self._agent.set_xdg_paths(xdg)
        except Exception as exc:
            logger.debug("ui.terminal_bridge | detect xdg: {}", exc)

        # Locale corrente
        try:
            import locale as _locale
            current_locale = _locale.getlocale()[0] or os.environ.get("LANG", "C")
            self._agent.set_locale(current_locale)
        except Exception as exc:
            logger.debug("ui.terminal_bridge | detect locale: {}", exc)

        # Scegli il modello iniziale: il primo visibile della lista
        # filtrata, oppure settings.ollama.chat_model se compatibile.
        # Se non c'è nessun thinking-model installato, _current_model
        # resta None e l'agent userà il default di settings (rischioso,
        # ma è il meglio che possiamo fare).
        try:
            visible = await self.list_visible_models()
            if visible:
                # Preferisci settings.ollama.chat_model se è tra i visibili
                preferred = settings.ollama.chat_model
                if preferred in visible:
                    self._current_model = preferred
                else:
                    self._current_model = visible[0]
                self._agent.set_model(self._current_model)
        except Exception as exc:
            logger.warning(
                "ui.terminal_bridge | impossibile scegliere modello iniziale: {}",
                exc,
            )

        logger.info(
            "ui.terminal_bridge | inizializzato | cwd={} model={}",
            self._agent.cwd, self._current_model,
        )

    async def aclose(self) -> None:
        if self._agent is not None:
            try:
                await self._agent.aclose()
            except Exception as exc:
                logger.debug("ui.terminal_bridge | aclose agent: {}", exc)
            self._agent = None
        self._loaded = False

    # ── property / introspezione ─────────────────────────────────────────

    @property
    def cwd(self) -> str:
        return self._agent.cwd if self._agent else ""

    @property
    def current_model(self) -> str:
        return self._current_model

    @property
    def history_count(self) -> int:
        return len(self._history)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    # ── modelli visibili ────────────────────────────────────────────────

    def set_hidden_models(self, hidden: list[str]) -> None:
        """Sovrascrive la lista dei modelli nascosti dall'utente."""
        self._hidden_models = set(hidden or [])

    def is_thinking_family(self, family: str) -> bool:
        return family in self._thinking_families

    async def list_available_models(self) -> list[dict[str, Any]]:
        """
        Ritorna TUTTI i modelli installati su Ollama. Ogni elemento contiene:
            {name, family, parameter_size, hidden: bool, current: bool}

        Nota: prima filtravamo per `family ∈ thinking_families` come difesa
        contro modelli non-thinking che crashavano con 400 sul flag think:true.
        Ora il filtro è rimosso per dare massima libertà all'utente. Conseguenza:
        selezionare un modello non-thinking può generare errore 400 alla prima
        chiamata. L'utente decide.

        Non solleva: se Ollama non risponde, ritorna [].
        """
        try:
            import httpx
            base = settings.ollama.base_url.rstrip("/")
            async with httpx.AsyncClient(timeout=5.0) as cli:
                r = await cli.get(f"{base}/api/tags")
                r.raise_for_status()
                data = r.json()
        except Exception as exc:
            logger.warning("ui.terminal_bridge | list_available_models: {}", exc)
            return []

        out: list[dict[str, Any]] = []
        for m in data.get("models", []):
            name    = m.get("name", "")
            details = m.get("details", {})
            family  = details.get("family", "")
            psize   = details.get("parameter_size", "")
            if not name:
                continue
            out.append({
                "name":           name,
                "family":         family,
                "parameter_size": psize,
                "hidden":         name in self._hidden_models,
                "current":        name == self._current_model,
            })
        # ordine: visibili prima, poi alfabetico
        out.sort(key=lambda x: (x["hidden"], x["name"]))
        return out

    async def list_visible_models(self) -> list[str]:
        """Solo i nomi dei modelli mostrati nel selettore (non nascosti)."""
        all_models = await self.list_available_models()
        return [m["name"] for m in all_models if not m["hidden"]]

    async def switch_model(self, name: str) -> bool:
        """
        Cambia il modello per le prossime chiamate. Verifica che sia
        nella lista visibile. Ritorna True se ok.
        """
        visible = await self.list_visible_models()
        if name not in visible:
            logger.warning(
                "ui.terminal_bridge | switch_model rifiutato: '{}' non visibile",
                name,
            )
            return False
        self._current_model = name
        # Setta il modello sull'agent SENZA toccare i settings globali
        # (importante: settings.ollama.chat_model è usato dalla modalità chat
        # e non vogliamo sporcarlo).
        if self._agent is not None:
            self._agent.set_model(name)
        await self._broadcast({"type": "terminal.model", "name": name})
        logger.info("ui.terminal_bridge | modello → {}", name)
        return True

    # ── flusso propose / confirm / cancel ────────────────────────────────

    async def propose(self, user_request: str) -> dict[str, Any]:
        """
        Chiede una proposta all'agente. Se è auto-eseguibile (safe),
        la esegue subito ed esegue anche analyze; altrimenti la mette
        in pending e ritorna solo la proposta.

        Ritorna sempre un dict serializzabile:
            {
              "turn": <AgentTurn>,            # con proposal+result+analysis se safe
              "needs_confirmation": bool,
            }
        oppure
            {"error": "..."}
        """
        self._require_loaded()
        user_request = (user_request or "").strip()
        if not user_request:
            return {"error": "richiesta vuota"}

        # Genera la proposta
        try:
            proposal = await self._agent.propose(user_request)
        except Exception as exc:
            logger.warning("ui.terminal_bridge | propose fallito: {}", exc)
            await self._broadcast({"type": "terminal.error", "message": str(exc)})
            return {"error": str(exc)}

        # Broadcast della proposta SEMPRE (così la UI la vede appena pronta)
        await self._broadcast({
            "type":     "terminal.proposal",
            "proposal": _proposal_to_dict(proposal),
        })

        # Se richiede conferma → in pending, l'utente deciderà
        if proposal.needs_confirmation:
            self._pending[proposal.proposal_id] = proposal
            turn = AgentTurn(
                user_request=user_request,
                proposal=proposal,
                skipped=True,
            )
            return {
                "turn":               _turn_to_dict(turn),
                "needs_confirmation": True,
            }

        # SAFE → auto-execute + analyze immediatamente
        turn = await self._execute_and_analyze(proposal, user_request)
        return {
            "turn":               _turn_to_dict(turn),
            "needs_confirmation": False,
        }

    async def confirm(self, proposal_id: str) -> dict[str, Any]:
        """
        Conferma una proposta in attesa: esegue + analizza. Ritorna il turno.
        """
        self._require_loaded()
        proposal = self._pending.pop(proposal_id, None)
        if proposal is None:
            return {"error": f"proposta non trovata: {proposal_id}"}

        # Usa la richiesta originale se l'abbiamo (non la salviamo nel
        # CommandProposal, quindi qui ripieghiamo su una stringa generica)
        turn = await self._execute_and_analyze(proposal, user_request="(conferma)")
        return {
            "turn":               _turn_to_dict(turn),
            "needs_confirmation": False,
        }

    async def cancel(self, proposal_id: str) -> dict[str, Any]:
        """Annulla una proposta in attesa, niente esecuzione."""
        proposal = self._pending.pop(proposal_id, None)
        if proposal is None:
            return {"ok": False, "error": "proposta non in attesa"}
        await self._broadcast({
            "type":        "terminal.cancelled",
            "proposal_id": proposal_id,
        })
        logger.info("ui.terminal_bridge | proposta '{}' annullata", proposal_id)
        return {"ok": True}

    async def _execute_and_analyze(
        self,
        proposal: CommandProposal,
        user_request: str,
    ) -> AgentTurn:
        """Helper interno: esegue + analizza, broadcast, history. Non solleva."""
        turn = AgentTurn(user_request=user_request, proposal=proposal)

        cwd_before = self._agent.cwd
        try:
            result = await self._agent.execute(proposal, confirmed=True)
            turn.result = result
        except Exception as exc:
            turn.error = f"execute: {exc}"
            logger.warning("ui.terminal_bridge | execute fallito: {}", exc)
            self._history.append(turn)
            # Propago anche all'agent così "correggi il comando precedente"
            # nei turni successivi ha contesto del fallimento.
            self._agent.add_turn_to_history(turn)
            await self._broadcast({
                "type":  "terminal.error",
                "message": turn.error,
            })
            return turn

        # broadcast cwd se cambiata
        if self._agent.cwd != cwd_before:
            await self._broadcast({
                "type": "terminal.cwd",
                "cwd":  self._agent.cwd,
            })

        # analyze in coda
        try:
            turn.analysis = await self._agent.analyze(
                result, user_request=user_request,
            )
        except Exception as exc:
            turn.error = f"analyze: {exc}"
            logger.warning("ui.terminal_bridge | analyze fallito: {}", exc)

        self._history.append(turn)
        # Propago anche all'agent così "correggi il comando precedente"
        # nei turni successivi ha contesto su comando/result/analisi precedenti.
        # IMPORTANTE: senza questo, agent._turn_history resta vuoto e il modello
        # risponde "non ci sono comandi precedenti" quando l'utente chiede di
        # correggere/modificare.
        self._agent.add_turn_to_history(turn)

        await self._broadcast({
            "type":     "terminal.result",
            "result":   _result_to_dict(result),
            "analysis": turn.analysis,
        })
        return turn

    # ── input vocale (Q3 = B) ────────────────────────────────────────────

    async def handle_voice_input(self, text: str) -> None:
        """
        Riceve una trascrizione STT deviata dal UIBridge. In modalità
        terminale NON eseguiamo direttamente: emettiamo un messaggio
        WS che la UI userà per popolare il box di input. L'utente preme
        Invio per confermare e parte una POST /api/terminal/propose.
        """
        text = (text or "").strip()
        if not text:
            return
        await self._broadcast({
            "type": "terminal.transcription",
            "text": text,
        })
        logger.info("ui.terminal_bridge | transcription deviata: {}", text[:80])

    # ── reset ────────────────────────────────────────────────────────────

    async def reset(self) -> None:
        """Resetta cwd, history, pending. Mantiene il modello corrente."""
        self._require_loaded()
        self._agent.reset()
        self._pending.clear()
        self._history.clear()
        await self._broadcast({
            "type": "terminal.reset",
            "cwd":  self._agent.cwd,
        })
        logger.info("ui.terminal_bridge | reset")

    # ── stato per ws/api ─────────────────────────────────────────────────

    def state_payload(self) -> dict[str, Any]:
        """
        Snapshot completo dello stato del bridge — usato in /api/terminal/state
        e nel payload init del WS.
        """
        return {
            "cwd":           self.cwd,
            "current_model": self._current_model,
            "history_count": len(self._history),
            "pending_count": len(self._pending),
            "pending":       [_proposal_to_dict(p) for p in self._pending.values()],
            "history":       [_turn_to_dict(t)     for t in self._history[-20:]],
        }

    # ── interno ──────────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if not self._loaded or self._agent is None:
            raise RuntimeError("TerminalBridge non caricato — chiama load()")

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        try:
            await self._ws.broadcast(payload)
        except Exception as exc:
            logger.debug("ui.terminal_bridge | broadcast fallito: {}", exc)

    def __repr__(self) -> str:
        if self._loaded:
            return (
                f"<TerminalBridge cwd='{self.cwd}' model='{self._current_model}' "
                f"history={len(self._history)} pending={len(self._pending)}>"
            )
        return "<TerminalBridge [non caricato]>"


# ---------------------------------------------------------------------------
# Detect dei percorsi utente localizzati (XDG user-dirs)
# ---------------------------------------------------------------------------

# Mapping chiave standard XDG → label leggibile.
# `xdg-user-dir KEY` ritorna il path; se la cartella non esiste o la chiave
# non è configurata, ritorna $HOME (lo trattiamo come "non disponibile").
_XDG_KEYS: list[tuple[str, str]] = [
    ("DESKTOP",     "Scrivania"),
    ("DOWNLOAD",    "Scaricati"),
    ("DOCUMENTS",   "Documenti"),
    ("MUSIC",       "Musica"),
    ("PICTURES",    "Immagini"),
    ("VIDEOS",      "Video"),
    ("TEMPLATES",   "Modelli"),
    ("PUBLICSHARE", "Pubblici"),
]


def _detect_xdg_user_paths() -> dict[str, str]:
    """
    Detect dei percorsi utente XDG usando il binario `xdg-user-dir` standard
    su Ubuntu/freedesktop. Ritorna un dict label → path assoluto.

    Su locale italiano restituisce ad esempio:
        {"Scrivania": "/home/mauro/Scrivania",
         "Scaricati": "/home/mauro/Scaricati", ...}

    Se `xdg-user-dir` non è disponibile o un path non è configurato (ritorna
    $HOME), quella chiave viene omessa.

    Non solleva eccezioni: in caso di errore ritorna {} e logga in debug.
    """
    if shutil.which("xdg-user-dir") is None:
        logger.debug("ui.terminal_bridge | xdg-user-dir non trovato")
        return {}

    home = os.path.expanduser("~")
    out: dict[str, str] = {}
    for key, label in _XDG_KEYS:
        try:
            r = subprocess.run(
                ["xdg-user-dir", key],
                capture_output=True, text=True, timeout=2,
            )
            path = r.stdout.strip()
            # xdg-user-dir ritorna $HOME se la chiave non è configurata
            # oppure se la cartella è disabilitata.
            if path and path != home and os.path.isdir(path):
                out[label] = path
        except Exception as exc:
            logger.debug("ui.terminal_bridge | xdg-user-dir {}: {}", key, exc)
    return out
