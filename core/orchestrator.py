"""
core/orchestrator.py
Orchestratore del loop conversazionale di local-assistant.

Coordina ogni turno:
    INPUT (voice|text)
        → STT (se InputMode.VOICE)
        → MemoryManager.populate_context()
        → PersonalityManager.apply_to_context()
        → Web search (se trigger keyword + tool consentito)
        → costruzione messages (system + memory + web + history + user)
        → OllamaClient.stream() o .chat()
        → salvataggio memoria (turno utente + risposta)
        → TTS (se OutputMode.VOICE|BOTH)
    OUTPUT (text e/o audio)

API pubblica:
    async with Orchestrator() as orch:
        async for chunk in orch.turn(ctx):        # streaming
            print(chunk, end="", flush=True)

        ctx = await orch.turn_sync(ctx)           # risposta completa

        result = await orch.add_memory(text, {})  # aggiungi memoria manuale

Errori non fatali:
    - TTS down       → continua in testo  (ctx.error rimane None)
    - memoria down   → continua senza RAG (ctx.error rimane None)
    - web search down→ continua senza ricerca (ctx.error rimane None)
    - STT fallisce   → ctx.error settato, turn si interrompe
    - LLM fallisce   → ctx.error settato, turn si interrompe
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

from config.settings import settings
from core.context import (
    AssistantContext,
    InputMode,
    MemoryChunk,
    ModelRole,
    OutputMode,
)
from core.logger import logger

# Import leggeri — le classi usano httpx/piccoli wrapper
from modules.llm import OllamaClient
from modules.llm.base_llm import Message, Role
from modules.memory import MemoryManager
from modules.memory.base_memory import SaveResult
from modules.personality import PersonalityManager
from modules.web_search import SearXNGClient


# ---------------------------------------------------------------------------
# Dataclass pubblica
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorStatus:
    """Stato corrente dell'orchestratore (per diagnostica e health-check)."""
    llm_ok:         bool = False
    memory_ok:      bool = False
    personality_ok: bool = False
    stt_ok:         bool = False
    tts_ok:         bool = False
    web_search_ok:  bool = False
    active_sessions: int = 0

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "llm":          self.llm_ok,
            "memory":       self.memory_ok,
            "personality":  self.personality_ok,
            "stt":          self.stt_ok,
            "tts":          self.tts_ok,
            "web_search":   self.web_search_ok,
            "sessions":     self.active_sessions,
        }


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

_MEMORY_SOURCE_USER = "conversation:user"
_MEMORY_SOURCE_ASST = "conversation:assistant"

# Prefisso iniettato nel system prompt per le memorie recuperate
_MEMORY_HEADER = "\n\n---\nCONTESTO DALLA MEMORIA SEMANTICA (usa queste info se pertinenti):\n"
_MEMORY_FOOTER = "\n---\n"

# Nome del tool (deve coincidere con allowed_tools nei profili YAML)
_WEB_SEARCH_TOOL = "web_search"

# Prefisso iniettato nel system prompt per i risultati di ricerca web
_WEBSEARCH_HEADER = (
    "\n\n---\nRISULTATI RICERCA WEB (informazioni aggiornate dal web, "
    "cita le fonti pertinenti nella risposta):\n"
)
_WEBSEARCH_FOOTER = "\n---\n"

# Parole chiave che attivano la ricerca web (lowercase, match su sottostringa)
_WEBSEARCH_TRIGGERS: tuple[str, ...] = (
    "cerca online",
    "cerca su internet",
    "cerca sul web",
    "cerca in rete",
    "fai una ricerca",
    "ricerca online",
    "ricerca sul web",
    "cerca su google",
    "guarda online",
    "guarda su internet",
    "ultime notizie",
    "ultimissime",
    "notizie di oggi",
    "cosa dicono",
    "cosa si dice",
    "novità su",
    "aggiornamenti su",
    "search online",
    "search the web",
)


# ---------------------------------------------------------------------------
# Helper — costruzione prompt
# ---------------------------------------------------------------------------

def _should_search(text: str) -> bool:
    """
    Euristica leggera: True se il testo utente contiene una delle
    keyword-trigger per la ricerca web.
    """
    if not text:
        return False
    low = text.lower()
    return any(trigger in low for trigger in _WEBSEARCH_TRIGGERS)


def _clean_query(text: str) -> str:
    """
    Rimuove le frasi-trigger dalla query per passare a SearXNG
    solo il contenuto utile. Best-effort: se non resta nulla,
    restituisce il testo originale.
    """
    low = text.lower()
    cleaned = text
    for trigger in _WEBSEARCH_TRIGGERS:
        idx = low.find(trigger)
        if idx != -1:
            # rimuove la sottostringa trigger mantenendo il resto
            cleaned = (cleaned[:idx] + cleaned[idx + len(trigger):])
            low = cleaned.lower()
    # normalizza spazi multipli e punteggiatura residua ai bordi
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.strip(" ,.:;!?\"'-")
    return cleaned if cleaned else text.strip()

def _format_memory_block(chunks: list[MemoryChunk]) -> str:
    """Formatta i MemoryChunk in un blocco testuale da iniettare nel prompt."""
    if not chunks:
        return ""
    lines = [_MEMORY_HEADER]
    for i, chunk in enumerate(chunks, 1):
        score = f"{chunk.relevance_score:.2f}"
        lines.append(f"[{i}] (source={chunk.source}, score={score})\n{chunk.content}")
    lines.append(_MEMORY_FOOTER)
    return "\n".join(lines)


def _format_search_block(results: list[Any]) -> str:
    """
    Formatta i SearchResult in un blocco testuale da iniettare nel prompt.
    Gemello di _format_memory_block ma per la ricerca web.
    """
    if not results:
        return ""
    lines = [_WEBSEARCH_HEADER]
    for i, r in enumerate(results, 1):
        snippet = (r.snippet or "").strip()
        lines.append(f"[{i}] {r.title}\n    URL: {r.url}\n    {snippet}")
    lines.append(_WEBSEARCH_FOOTER)
    return "\n".join(lines)


def _build_messages(ctx: AssistantContext) -> list[Message]:
    """
    Costruisce la lista di Message da inviare a OllamaClient.

    Ordine:
        1. System prompt (+ memoria iniettata)
        2. Conversation history (finestra scorrevole)
        3. Messaggio utente corrente
    """
    # --- system prompt con eventuale blocco di memoria ---
    system_text = ctx.system_prompt
    if ctx.retrieved_memories:
        system_text += _format_memory_block(ctx.retrieved_memories)

    messages: list[Message] = []

    # System come primo messaggio
    if system_text:
        messages.append(Message(role=Role.SYSTEM, content=system_text))

    # Cronologia (già trimmata dalla finestra scorrevole)
    for entry in ctx.conversation_history:
        role_str = entry.get("role", "user")
        try:
            role = Role(role_str)
        except ValueError:
            role = Role.USER
        messages.append(Message(role=role, content=entry.get("content", "")))

    # Messaggio utente corrente
    if ctx.user_text:
        messages.append(Message(role=Role.USER, content=ctx.user_text))

    return messages


def _trim_history(
    history: list[dict[str, str]],
    max_messages: int,
) -> list[dict[str, str]]:
    """
    Mantiene la finestra scorrevole: taglia i messaggi più vecchi.
    max_messages deve essere pari per mantenere coppie user/assistant intere.
    """
    if max_messages <= 0:
        return history
    if len(history) <= max_messages:
        return history
    # Taglia da sinistra (messaggi più vecchi)
    return history[-max_messages:]


# ---------------------------------------------------------------------------
# Orchestratore
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Coordina il loop completo per ogni turno di conversazione.

    Args:
        personality:    Nome del profilo da attivare all'avvio
                        (default: settings.personality.default_personality).
        enable_tts:     Se True, tenta di avviare il TTS. Se il server
                        non è disponibile, continua silenziosamente.
        enable_stt:     Se True, carica WhisperSTT per i turni voice.
        enable_web_search: Se True, abilita la ricerca web SearXNG
                        (soggetta anche a allowed_tools del profilo).
        context_window: Dimensione max della finestra di storia (n. messaggi).
                        0 = usa settings.memory.context_window_messages.

    Esempio:
        async with Orchestrator() as orch:
            ctx = AssistantContext(user_text="Ciao!", session_id="s1")
            async for chunk in orch.turn(ctx):
                print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        personality:       Optional[str] = None,
        enable_tts:        bool          = True,
        enable_stt:        bool          = True,
        enable_web_search: bool          = True,
        context_window:    int           = 0,
    ) -> None:
        self._personality_name  = personality or settings.personality.default_personality
        self._enable_tts        = enable_tts
        self._enable_stt        = enable_stt
        self._enable_web_search = enable_web_search
        self._context_window    = context_window or settings.memory.context_window_messages

        # Moduli — inizializzati in load()
        self._llm:         Optional[OllamaClient]      = None
        self._memory:      Optional[MemoryManager]     = None
        self._personality: Optional[PersonalityManager] = None
        self._stt:         Optional[Any]               = None  # WhisperSTT (import lazy)
        self._tts:         Optional[Any]               = None  # Qwen3TTS   (import lazy)
        self._web_search:  Optional[SearXNGClient]     = None

        self._loaded: bool = False

        # Storico separato per sessione: session_id → list[dict]
        # (permette sessioni parallele isolate)
        self._session_histories: dict[str, list[dict[str, str]]] = defaultdict(list)

        # Override del modello per ruolo, impostati a runtime (es. dalla UI).
        # Evita di mutare il singleton globale settings.ollama.*: lo stato
        # del modello attivo vive qui, isolato per istanza di orchestratore.
        self._model_overrides: dict[ModelRole, str] = {}

    # -----------------------------------------------------------------------
    # Context manager
    # -----------------------------------------------------------------------

    async def __aenter__(self) -> "Orchestrator":
        await self.load()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Chiude tutte le risorse aperte."""
        tasks = []
        if self._llm:
            tasks.append(self._llm.aclose())
        if self._memory:
            tasks.append(self._memory.aclose())
        # Qwen3TTS espone __aexit__ (chiude http + termina sottoprocesso), non aclose()
        if self._tts:
            tasks.append(self._tts.__aexit__(None, None, None))
        # WhisperSTT espone __aexit__ (azzera riferimenti ai modelli), non aclose()
        if self._stt:
            tasks.append(self._stt.__aexit__(None, None, None))
        # SearXNGClient espone aclose() (chiude httpx), come OllamaClient
        if self._web_search:
            tasks.append(self._web_search.aclose())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._llm = self._memory = self._personality = None
        self._stt = self._tts = self._web_search = None
        self._loaded = False
        logger.info("orchestrator | chiuso")

    # -----------------------------------------------------------------------
    # Inizializzazione
    # -----------------------------------------------------------------------

    async def load(self) -> None:
        """
        Inizializza tutti i moduli in parallelo dove possibile.
        I moduli non critici (TTS, STT) vengono avviati con gestione
        degli errori non fatale.
        """
        logger.info("orchestrator | avvio inizializzazione")
        t0 = time.monotonic()

        # --- LLM (httpx, leggero) ---
        self._llm = OllamaClient()

        # --- Personality (YAML, veloce) ---
        self._personality = PersonalityManager(default_name=self._personality_name)
        await self._personality.load()

        # --- Memory (ChromaDB + Ollama embed) ---
        try:
            self._memory = MemoryManager()
            await self._memory.load()
        except Exception as exc:
            logger.warning("orchestrator | memory non disponibile: {} — continuo senza RAG", exc)
            self._memory = None

        # --- STT (import pesante — solo se abilitato) ---
        if self._enable_stt:
            try:
                await self._load_stt()
            except Exception as exc:
                logger.warning("orchestrator | STT non disponibile: {}", exc)
                self._stt = None

        # --- TTS (sottoprocesso FastAPI — solo se abilitato) ---
        if self._enable_tts:
            try:
                await self._load_tts()
            except Exception as exc:
                logger.warning("orchestrator | TTS non disponibile: {} — continuo in testo", exc)
                self._tts = None

        # --- Web search (httpx, leggero — come OllamaClient) ---
        if self._enable_web_search:
            try:
                self._web_search = SearXNGClient()
                logger.debug("orchestrator | web_search inizializzato")
            except Exception as exc:
                logger.warning(
                    "orchestrator | web_search non disponibile: {} — continuo senza", exc
                )
                self._web_search = None

        self._loaded = True
        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "orchestrator | pronto in {:.0f}ms | llm=✓ memory={} stt={} tts={} web={}",
            elapsed,
            "✓" if self._memory else "✗",
            "✓" if self._stt else "✗",
            "✓" if self._tts else "✗",
            "✓" if self._web_search else "✗",
        )

    async def _load_stt(self) -> None:
        """Import lazy di WhisperSTT per non bloccare l'avvio."""
        from modules.stt import WhisperSTT  # noqa: F401

        stt = WhisperSTT()
        # _load_models() carica Whisper + Silero VAD in run_in_executor internamente
        await stt._load_models()
        self._stt = stt
        logger.debug("orchestrator | STT caricato")

    async def _load_tts(self) -> None:
        """Import lazy di Qwen3TTS — avvia il sottoprocesso server."""
        from modules.tts import Qwen3TTS  # import leggero (solo httpx)

        tts = Qwen3TTS()
        # __aenter__ chiama _ensure_server() e crea self._http
        # senza questo, synthesize() lancia RuntimeError("Qwen3TTS non inizializzato")
        await tts.__aenter__()
        self._tts = tts
        logger.debug("orchestrator | TTS caricato")

    # -----------------------------------------------------------------------
    # API pubblica
    # -----------------------------------------------------------------------

    async def turn(
        self,
        ctx: AssistantContext,
    ) -> AsyncGenerator[str, None]:
        """
        Esegue un turno completo in modalità streaming.

        Yields:
            Chunk di testo dell'assistente man mano che arrivano da Ollama.
            Dopo l'ultimo chunk, ctx è completamente popolato (audio, memorie, timings).

        Il metodo gestisce internamente errori non fatali (TTS, memoria).
        In caso di errore fatale (LLM, STT) setta ctx.error e interrompe.
        """
        self._require_loaded()
        logger.info(
            "orchestrator.turn | id={} session={} mode={}",
            ctx.turn_id,
            ctx.session_id,
            ctx.input_mode,
        )

        # 1. STT — trascrizione audio (se voice input)
        if ctx.input_mode == InputMode.VOICE:
            ok = await self._run_stt(ctx)
            if not ok:
                return

        if not ctx.user_text.strip():
            logger.warning("orchestrator.turn | user_text vuoto — skip")
            return

        # 2. Sincronizza history dal session store → ctx
        ctx.conversation_history = list(
            self._session_histories[ctx.session_id]
        )

        # 3. Memory RAG
        await self._run_memory(ctx)

        # 4. Personality
        self._personality.apply_to_context(ctx)

        # 4.5 Web search (se trigger keyword + tool consentito dal profilo)
        await self._run_web_search(ctx)

        # 5. Costruzione messaggi
        messages = _build_messages(ctx)

        # 6. LLM streaming
        full_text = ""
        try:
            t0 = time.monotonic()
            model_name = self.active_model(ctx.model_role)
            ctx.model_name = model_name
            async for chunk in self._llm.stream(messages, ctx.model_role, model=model_name):
                full_text += chunk
                yield chunk
            ctx.set_timing("llm", (time.monotonic() - t0) * 1000)
        except Exception as exc:
            ctx.error = f"LLM error: {exc}"
            logger.error("orchestrator.turn | LLM fallito: {}", exc)
            return

        ctx.assistant_text = full_text

        # 7. Aggiorna conversation history (finestra scorrevole)
        self._update_history(ctx)

        # 8. Salvataggio in memoria
        await self._save_turn_to_memory(ctx)

        # 9. TTS
        if ctx.output_mode in (OutputMode.VOICE, OutputMode.BOTH):
            await self._run_tts(ctx)

        logger.info(
            "orchestrator.turn | completato | {}",
            ctx.to_log_dict(),
        )

    async def turn_sync(self, ctx: AssistantContext) -> AssistantContext:
        """
        Esegue un turno completo raccogliendo l'intera risposta prima
        di restituire il contesto.

        Returns:
            Lo stesso ctx popolato con assistant_text, audio_chunks, timings.
        """
        chunks: list[str] = []
        async for chunk in self.turn(ctx):
            chunks.append(chunk)
        if not ctx.assistant_text and chunks:
            ctx.assistant_text = "".join(chunks)
        return ctx

    async def add_memory(
        self,
        text:     str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[SaveResult]:
        """
        Aggiunge manualmente un chunk di testo alla memoria semantica.

        Returns:
            SaveResult se la memoria è disponibile, None altrimenti.
        """
        if not self._memory:
            logger.warning("orchestrator.add_memory | memoria non disponibile")
            return None
        return await self._memory.save(text, metadata or {})

    @property
    def status(self) -> OrchestratorStatus:
        """Snapshot dello stato corrente per diagnostica."""
        return OrchestratorStatus(
            llm_ok         = self._llm is not None,
            memory_ok      = self._memory is not None,
            personality_ok = self._personality is not None and self._personality._loaded,
            stt_ok         = self._stt is not None,
            tts_ok         = self._tts is not None,
            web_search_ok  = self._web_search is not None,
            active_sessions= len(self._session_histories),
        )

    def switch_personality(self, name: str) -> None:
        """Cambia personalità attiva per i turni successivi."""
        self._require_loaded()
        self._personality.switch(name)
        logger.info("orchestrator | personalità → '{}'", name)

    def _default_model(self, role: ModelRole) -> str:
        """Modello da settings per il ruolo dato (nessun override applicato)."""
        return {
            ModelRole.CHAT:   settings.ollama.chat_model,
            ModelRole.CODE:   settings.ollama.code_model,
            ModelRole.VISION: settings.ollama.vision_model,
        }[role]

    def active_model(self, role: ModelRole = ModelRole.CHAT) -> str:
        """Modello attualmente attivo per il ruolo (override runtime o default)."""
        return self._model_overrides.get(role) or self._default_model(role)

    def set_model(self, name: str, role: ModelRole = ModelRole.CHAT) -> None:
        """
        Imposta il modello per i turni successivi del ruolo dato, senza
        mutare il singleton globale settings.ollama.* (che resta il default).
        """
        self._model_overrides[role] = name
        logger.info("orchestrator | modello[{}] → '{}'", role.value, name)

    def clear_session(self, session_id: str) -> None:
        """Azzera la cronologia di una sessione."""
        self._session_histories.pop(session_id, None)
        logger.info("orchestrator | sessione '{}' azzerata", session_id)

    # -----------------------------------------------------------------------
    # Pipeline interna
    # -----------------------------------------------------------------------

    async def _run_stt(self, ctx: AssistantContext) -> bool:
        """
        Trascrive ctx.raw_audio → ctx.user_text.
        Ritorna False (con ctx.error settato) se la trascrizione fallisce.
        """
        if not ctx.raw_audio:
            ctx.error = "STT: raw_audio mancante"
            logger.error("orchestrator._run_stt | raw_audio assente")
            return False

        if not self._stt:
            ctx.error = "STT non disponibile"
            logger.error("orchestrator._run_stt | STT non caricato")
            return False

        try:
            t0     = time.monotonic()
            result = await self._stt.transcribe(ctx.raw_audio)
            ctx.user_text = result.text
            ctx.set_timing("stt", (time.monotonic() - t0) * 1000)
            logger.debug(
                "orchestrator._run_stt | '{}' [{:.0f}ms]",
                result.text[:60],
                ctx.timings["stt"],
            )
            return True
        except Exception as exc:
            ctx.error = f"STT error: {exc}"
            logger.error("orchestrator._run_stt | fallito: {}", exc)
            return False

    async def _run_web_search(self, ctx: AssistantContext) -> None:
        """
        Se il testo utente contiene una keyword-trigger e il profilo attivo
        consente il tool 'web_search', esegue la ricerca su SearXNG e inietta
        i risultati nel system prompt (stesso meccanismo del blocco memoria).

        Registra l'invocazione in ctx.tool_calls / ctx.tool_results.
        Errori non fatali: continua senza ricerca.
        """
        if not self._web_search:
            return

        # Trigger: keyword nel testo utente
        if not _should_search(ctx.user_text):
            return

        # Permesso: il profilo attivo deve abilitare il tool
        try:
            allowed = self._personality.active.allows_tool(_WEB_SEARCH_TOOL)
        except Exception:
            allowed = False
        if not allowed:
            logger.debug(
                "orchestrator._run_web_search | tool '{}' non consentito dal profilo '{}'",
                _WEB_SEARCH_TOOL,
                ctx.personality_name,
            )
            return

        query = _clean_query(ctx.user_text)
        logger.info("orchestrator._run_web_search | query='{}'", query[:80])

        try:
            t0   = time.monotonic()
            resp = await self._web_search.search(query)
            ctx.set_timing("web_search", (time.monotonic() - t0) * 1000)
        except Exception as exc:
            logger.warning(
                "orchestrator._run_web_search | ricerca fallita: {} — continuo", exc
            )
            return

        if resp.error:
            logger.warning(
                "orchestrator._run_web_search | SearXNG error: {} — continuo", resp.error
            )
            return

        if resp.is_empty():
            logger.debug("orchestrator._run_web_search | nessun risultato per '{}'", query[:60])
            return

        # Inietta i risultati nel system prompt (dopo eventuale blocco memoria)
        block = _format_search_block(resp.results)
        ctx.system_prompt = (ctx.system_prompt or "") + block

        # Traccia l'uso del tool nel contesto
        ctx.add_tool_call(
            tool=_WEB_SEARCH_TOOL,
            args={"query": query, "max_results": len(resp.results)},
            result=[r.to_log_dict() for r in resp.results],
        )
        logger.info(
            "orchestrator._run_web_search | {} risultati iniettati | {}",
            len(resp.results),
            resp.to_log_dict(),
        )

    async def _run_memory(self, ctx: AssistantContext) -> None:
        """
        Popola ctx.retrieved_memories con RAG.
        Errori non fatali: continua senza memoria.
        """
        if not self._memory:
            return
        try:
            await self._memory.populate_context(ctx)
        except Exception as exc:
            logger.warning("orchestrator._run_memory | errore RAG: {} — continuo", exc)
            ctx.retrieved_memories = []

    async def _run_tts(self, ctx: AssistantContext) -> None:
        """
        Sintetizza ctx.assistant_text → ctx.audio_chunks.
        Errori non fatali: continua senza audio.
        """
        if not self._tts or not ctx.assistant_text:
            return
        try:
            t0     = time.monotonic()
            result = await self._tts.synthesize(ctx.assistant_text)
            ctx.audio_chunks = [result.audio_bytes]
            ctx.set_timing("tts", (time.monotonic() - t0) * 1000)
            logger.debug(
                "orchestrator._run_tts | {:.0f}ms",
                ctx.timings["tts"],
            )
        except Exception as exc:
            logger.warning(
                "orchestrator._run_tts | TTS fallito: {} — continuo in testo", exc
            )

    async def _save_turn_to_memory(self, ctx: AssistantContext) -> None:
        """
        Salva in ChromaDB il testo utente e la risposta dell'assistente.
        Errori non fatali.
        """
        if not self._memory:
            return

        base_meta = {
            "session_id": ctx.session_id,
            "turn_id":    ctx.turn_id,
            "timestamp":  ctx.timestamp.isoformat(),
        }

        try:
            await self._memory.save(
                ctx.user_text,
                {**base_meta, "source": _MEMORY_SOURCE_USER, "role": "user"},
            )
        except Exception as exc:
            logger.warning("orchestrator._save_turn | salvataggio user fallito: {}", exc)

        if ctx.assistant_text:
            try:
                await self._memory.save(
                    ctx.assistant_text,
                    {**base_meta, "source": _MEMORY_SOURCE_ASST, "role": "assistant"},
                )
            except Exception as exc:
                logger.warning(
                    "orchestrator._save_turn | salvataggio assistant fallito: {}", exc
                )

    def _update_history(self, ctx: AssistantContext) -> None:
        """
        Aggiunge user_text e assistant_text alla cronologia della sessione
        e applica la finestra scorrevole.
        """
        history = self._session_histories[ctx.session_id]

        if ctx.user_text:
            history.append({"role": "user", "content": ctx.user_text})
        if ctx.assistant_text:
            history.append({"role": "assistant", "content": ctx.assistant_text})

        # Applica finestra scorrevole
        self._session_histories[ctx.session_id] = _trim_history(
            history, self._context_window
        )

        # Propaga al contesto per ispezione
        ctx.conversation_history = list(self._session_histories[ctx.session_id])

    # -----------------------------------------------------------------------
    # Interno
    # -----------------------------------------------------------------------

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "Orchestrator non inizializzato — "
                "usa 'async with Orchestrator()' oppure chiama 'await orch.load()'"
            )

    def __repr__(self) -> str:
        if self._loaded:
            return (
                f"<Orchestrator personality='{self._personality_name}' "
                f"memory={'✓' if self._memory else '✗'} "
                f"tts={'✓' if self._tts else '✗'} "
                f"stt={'✓' if self._stt else '✗'}>"
            )
        return "<Orchestrator [non caricato]>"
