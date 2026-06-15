"""
core/orchestrator.py
Orchestratore del loop conversazionale di local-assistant.

Coordina ogni turno:
    INPUT (voice|text)
        → STT (se InputMode.VOICE)
        → MemoryManager.populate_context()
        → PersonalityManager.apply_to_context()
        → File analysis (se path rilevato + tool consentito)
        → Web search (se trigger keyword + tool consentito)
        → costruzione messages (system + memory + file + web + history + user)
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
    - TTS down        → continua in testo  (ctx.error rimane None)
    - memoria down    → continua senza RAG (ctx.error rimane None)
    - web search down → continua senza ricerca (ctx.error rimane None)
    - file_analysis   → file falliti vengono saltati, gli altri proseguono
    - STT fallisce    → ctx.error settato, turn si interrompe
    - LLM fallisce    → ctx.error settato, turn si interrompe
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
from modules.file_analysis import AnalysisResult, FileAnalyzer
from modules.file_rag import FileRAG
from modules.intent import Intent, IntentClassifier
from modules.map_reduce import MapReduceEngine


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
    file_analysis_ok: bool = False
    active_sessions: int = 0

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "llm":          self.llm_ok,
            "memory":       self.memory_ok,
            "personality":  self.personality_ok,
            "stt":          self.stt_ok,
            "tts":          self.tts_ok,
            "web_search":   self.web_search_ok,
            "file_analysis": self.file_analysis_ok,
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


# Nome del tool file_analysis (deve coincidere con allowed_tools nei profili)
_FILE_ANALYSIS_TOOL = "file_analysis"

# Prefisso iniettato nel system prompt per i file analizzati
_FILE_ANALYSIS_HEADER = (
    "\n\n---\nCONTENUTO FILE ANALIZZATI (usa queste informazioni per "
    "rispondere; cita il path del file quando ti riferisci ai contenuti):\n"
)
_FILE_ANALYSIS_FOOTER = "\n---\n"

# Prefisso iniettato per i passaggi recuperati dal RAG sui file grandi.
_FILE_RAG_HEADER = (
    "\n\n---\nPASSAGGI RILEVANTI DAL FILE (recuperati semanticamente per la "
    "domanda corrente). Basa la risposta PRINCIPALMENTE su questi passaggi, "
    "non sull'estratto introduttivo qui sopra; cita SEMPRE file e pagina "
    '(es. "a pagina 157") quando riporti un contenuto:\n'
)
_FILE_RAG_FOOTER = "\n---\n"

# Regex per riconoscere path file con estensione supportata nel testo utente.
# Cattura: path assoluti (/...), home-relativi (~/...) e path relativi
# espliciti (./..., ../...). I path nudi senza directory ("foo.pdf") NON
# vengono catturati: troppo ambigui, sarebbero falsi positivi (es. titoli
# di articoli). Le 19 estensioni gestite sono identiche a quelle del
# FileAnalyzer per non avere riconoscimenti orfani.
import re as _re_file_analysis  # alias locale per non collidere altrove

_FILE_PATH_RE = _re_file_analysis.compile(
    r"""
    (?<![\w./~])                # non preceduto da char di path (evita match parziali)
    (?P<path>
        (?:~|\.{1,2})?           # opzionale: ~, ., ..
        (?:/[\w\-.+@]+)+         # almeno un segmento di percorso (NO spazi)
        \.
        (?:pdf|docx|txt|md|log|rst|yaml|yml|html|htm|
           json|csv|tsv|xml|wav|mp3|ogg|flac|m4a|opus)
    )
    """,
    flags=_re_file_analysis.IGNORECASE | _re_file_analysis.VERBOSE,
)


def _extract_file_paths(text: str, max_paths: int = 3) -> list[str]:
    """
    Estrae i path con estensione supportata dal testo utente.

    Tollerante ai delimitatori: rimuove virgolette/apici/backtick e
    punteggiatura finale (`.`, `,`, `;`, `?`, `)`). I path duplicati
    vengono compattati mantenendo l'ordine di prima occorrenza.

    Args:
        text:      testo utente
        max_paths: numero massimo di path restituiti (cap di sicurezza)

    Returns:
        Lista di stringhe path (non-resolved). Il FileAnalyzer si occupa
        di espandere ~ e validare l'esistenza.
    """
    if not text:
        return []

    found: list[str] = []
    seen:  set[str]  = set()

    for match in _FILE_PATH_RE.finditer(text):
        raw = match.group("path")
        # Cleanup bordi:
        # - lstrip SOLO caratteri di apertura (quote/backtick/parentesi) per
        #   NON rimuovere il '.' iniziale di './' o '../' che fanno parte
        #   del path.
        # - rstrip caratteri di chiusura + punteggiatura tipica di fine
        #   frase ('.', ',', ';', ':', '?', '!').
        cleaned = raw.lstrip("\"'`<({[").rstrip("\"'`>)}],.;:?!")
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        found.append(cleaned)
        if len(found) >= max_paths:
            break

    return found


def _format_file_analysis_block(
    results: list[AnalysisResult],
    *,
    rag_preview_chars: int | None = None,
) -> str:
    """
    Formatta i risultati di FileAnalyzer in un blocco testuale da iniettare
    nel system prompt. Gemello di _format_search_block ma per i file.
    Salta i risultati con contenuto vuoto, mostra path + contenuto.

    Se rag_preview_chars è valorizzato, i file destinati al RAG (full_content
    presente) vengono iniettati inline solo come breve estratto introduttivo:
    il dettaglio arriva dai PASSAGGI RILEVANTI recuperati semanticamente, così
    non si satura il context iniettando due volte lo stesso contenuto.
    """
    usable = [r for r in results if r.error is None and r.content.strip()]
    if not usable:
        return ""
    lines = [_FILE_ANALYSIS_HEADER]
    for i, r in enumerate(usable, 1):
        # Header: path + tipo + metadati strutturali (es. page_count) + troncamento.
        # Il page_count va mostrato in chiaro cosi' l'LLM non lo deduce dal testo
        # troncato (causa del bug "il libro ha 5 pagine" quando ne ha centinaia).
        parts = [f"tipo: {r.file_type}"]
        page_count = r.metadata.get("page_count")
        if isinstance(page_count, int) and page_count > 0:
            parts.append(f"{page_count} {'pagina' if page_count == 1 else 'pagine'}")
        # File grande destinato al RAG: inline solo un estratto introduttivo,
        # i dettagli arrivano dai chunk recuperati semanticamente.
        in_rag = bool(getattr(r, "full_content", None))
        body = r.content
        if rag_preview_chars and in_rag and len(body) > rag_preview_chars:
            body = (
                body[:rag_preview_chars]
                + "\n\n[...estratto introduttivo. Il contenuto completo è "
                "indicizzato: usa i PASSAGGI RILEVANTI qui sotto per i dettagli.]"
            )
            parts.append("estratto introduttivo — dettagli nei passaggi RAG")
        elif r.truncated:
            parts.append(f"testo troncato a {r.char_count}/{r.original_char_count} char")
        lines.append(f"[{i}] {r.path}  ({', '.join(parts)})")
        lines.append(body)
    lines.append(_FILE_ANALYSIS_FOOTER)
    return "\n".join(lines)


def _format_file_rag_block(chunks: list[MemoryChunk]) -> str:
    """
    Formatta i chunk recuperati dal RAG sui file grandi. Mostra file e pagina
    di provenienza così l'LLM può citarli ("a pagina 142 del libro...").
    """
    if not chunks:
        return ""
    lines = [_FILE_RAG_HEADER]
    for i, c in enumerate(chunks, 1):
        meta = c.metadata or {}
        source = meta.get("source", "file")
        ps, pe = meta.get("page_start", 0), meta.get("page_end", 0)
        if ps and pe and ps != pe:
            loc = f"{source}, pagine {ps}-{pe}"
        elif ps:
            loc = f"{source}, pagina {ps}"
        else:
            loc = source
        lines.append(f"[{i}] ({loc})\n{c.content}")
    lines.append(_FILE_RAG_FOOTER)
    return "\n".join(lines)


_MAPREDUCE_INTENTS = {
    Intent.FILE_GLOBAL, Intent.FILE_STRUCTURAL, Intent.FILE_POSITIONAL,
}
_MAPREDUCE_BLOCK_CHARS = 40000      # ~10K token per blocco sul context 16K
_POSITIONAL_BLOCKS = 2              # primi N + ultimi N per le domande posizionali

_MAP_REDUCE_HEADER = (
    "\n\n---\nSINTESI DAL DOCUMENTO COMPLETO (ottenuta scandendo l'intero file; "
    "basati su questa per rispondere e cita le pagine indicate):\n"
)


def _format_map_reduce_block(content: str) -> str:
    """Formatta la sintesi map-reduce come blocco di contesto da iniettare."""
    if not content or not content.strip():
        return ""
    return _MAP_REDUCE_HEADER + content.strip() + "\n"


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
        enable_file_analysis: Se True, abilita l'analisi di file locali
                        citati nel testo utente (soggetta anche a
                        allowed_tools del profilo). Riusa lo STT
                        eventualmente caricato per i file audio.
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
        personality:         Optional[str] = None,
        enable_tts:          bool          = True,
        enable_stt:          bool          = True,
        enable_web_search:   bool          = True,
        enable_file_analysis: bool         = True,
        context_window:      int           = 0,
    ) -> None:
        self._personality_name    = personality or settings.personality.default_personality
        self._enable_tts          = enable_tts
        self._enable_stt          = enable_stt
        self._enable_web_search   = enable_web_search
        self._enable_file_analysis = enable_file_analysis
        self._context_window      = context_window or settings.memory.context_window_messages

        # Moduli — inizializzati in load()
        self._llm:           Optional[OllamaClient]      = None
        self._memory:        Optional[MemoryManager]     = None
        self._personality:   Optional[PersonalityManager] = None
        self._stt:           Optional[Any]               = None  # WhisperSTT (import lazy)
        self._tts:           Optional[Any]               = None  # Qwen3TTS   (import lazy)
        self._web_search:    Optional[SearXNGClient]     = None
        self._file_analyzer: Optional[FileAnalyzer]      = None
        self._file_rag:      Optional[FileRAG]           = None

        self._loaded: bool = False

        # Storico separato per sessione: session_id → list[dict]
        # (permette sessioni parallele isolate)
        self._session_histories: dict[str, list[dict[str, str]]] = defaultdict(list)

        # File grandi indicizzati nel RAG, per sessione: session_id → list[dict]
        # ({file_id, source}). Stesso ciclo di vita della history: sync-in a inizio
        # turn(), write-back nel finally, pulizia in clear_session().
        self._session_rag_files: dict[str, list[dict]] = defaultdict(list)

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
        # FileAnalyzer.aclose() è no-op ma manteniamo il pattern simmetrico
        if self._file_analyzer:
            tasks.append(self._file_analyzer.aclose())
        if self._file_rag:
            tasks.append(self._file_rag.aclose())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._llm = self._memory = self._personality = None
        self._stt = self._tts = self._web_search = None
        self._file_analyzer = None
        self._file_rag = None
        self._intent_classifier = None
        self._map_reduce = None
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

        # --- Classificatore di intenti (gira sul modello chat, gia' in VRAM) ---
        self._intent_classifier = IntentClassifier(self._llm)

        # --- Motore map-reduce (scansione globale dei documenti) ---
        self._map_reduce = MapReduceEngine(self._llm)

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

        # --- File analysis (estrattore puro, costo zero in caricamento) ---
        # Riceve l'istanza STT eventualmente già caricata (per l'estrazione
        # audio); se STT non è disponibile, gli estrattori non-audio
        # funzionano comunque e l'audio degrada gentilmente.
        if self._enable_file_analysis:
            try:
                cfg = settings.file_analysis
                self._file_analyzer = FileAnalyzer(
                    stt              = self._stt,
                    max_chars_inline = cfg.max_chars_per_file,
                    max_file_bytes   = cfg.max_file_bytes,
                    safe_dirs        = cfg.safe_dirs,
                )
                logger.debug("orchestrator | file_analysis inizializzato")
            except Exception as exc:
                logger.warning(
                    "orchestrator | file_analysis non disponibile: {} — continuo senza", exc
                )
                self._file_analyzer = None

        # --- File RAG (indicizzazione file grandi; riusa ChromaDB+embed) ---
        # Gated su rag_enabled e sulla disponibilità del file_analyzer (senza
        # estrazione non c'è nulla da indicizzare). Best-effort.
        if (
            self._enable_file_analysis
            and self._file_analyzer is not None
            and getattr(settings.file_analysis, "rag_enabled", True)
        ):
            try:
                self._file_rag = FileRAG()
                await self._file_rag.load()
                logger.debug("orchestrator | file_rag inizializzato")
            except Exception as exc:
                logger.warning(
                    "orchestrator | file_rag non disponibile: {} — continuo senza", exc
                )
                self._file_rag = None

        self._loaded = True
        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "orchestrator | pronto in {:.0f}ms | llm=✓ memory={} stt={} tts={} web={} file={}",
            elapsed,
            "✓" if self._memory else "✗",
            "✓" if self._stt else "✗",
            "✓" if self._tts else "✗",
            "✓" if self._web_search else "✗",
            "✓" if self._file_analyzer else "✗",
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

    async def warmup(self, *, keep_alive: Optional[str] = None) -> None:
        """
        Pre-carica i modelli Ollama per azzerare il cold-start del primo turno:
          1. modello chat attivo (caricamento pesi in VRAM/RAM);
          2. modello di embedding (usato dal RAG memoria PRIMA dell'LLM a ogni
             turno) — solo se la memoria è disponibile.

        Pensato per girare in background all'avvio: ogni passo è best-effort e
        non solleva mai, così non può rompere lo startup.
        """
        if not self._loaded or self._llm is None:
            return
        ka = keep_alive if keep_alive is not None else getattr(
            settings.ollama, "warmup_keep_alive", None
        )

        # 1. Modello chat attivo
        t0 = time.monotonic()
        model = self.active_model(ModelRole.CHAT)
        try:
            ok = await self._llm.warmup(model=model, keep_alive=ka)
        except Exception as exc:
            ok = False
            logger.warning("orchestrator.warmup | chat '{}' eccezione: {}", model, exc)
        logger.info(
            "orchestrator.warmup | chat '{}' {} ({:.0f}ms)",
            model, "✓" if ok else "✗", (time.monotonic() - t0) * 1000,
        )

        # 2. Embedding (solo se il RAG memoria è attivo)
        if self._memory is not None:
            t1 = time.monotonic()
            try:
                await self._llm.embed(["warmup"])
                logger.info(
                    "orchestrator.warmup | embed '{}' ✓ ({:.0f}ms)",
                    settings.ollama.embed_model, (time.monotonic() - t1) * 1000,
                )
            except Exception as exc:
                logger.warning("orchestrator.warmup | embed fallito: {}", exc)

    async def warmup_model(
        self,
        model: str,
        *,
        keep_alive: Optional[str] = None,
    ) -> bool:
        """
        Warmup mirato di UN modello specifico per nome (non l'embedding).

        Usato dagli switch a runtime (cambio modello dalla tendina chat o
        terminale, cambio modalità chat↔terminale): il warmup di Ollama è
        globale al daemon e indicizzato sul nome del modello, quindi questo
        metodo scalda qualunque modello — incluso quello del terminale —
        riusando l'unico OllamaClient.

        Best-effort: non solleva mai. Ritorna True solo se la richiesta è
        andata a buon fine.
        """
        if not self._loaded or self._llm is None or not model:
            return False
        ka = keep_alive if keep_alive is not None else getattr(
            settings.ollama, "warmup_keep_alive", None
        )
        t0 = time.monotonic()
        try:
            ok = await self._llm.warmup(model=model, keep_alive=ka)
        except Exception as exc:
            ok = False
            logger.warning("orchestrator.warmup_model | '{}' eccezione: {}", model, exc)
        logger.info(
            "orchestrator.warmup_model | '{}' {} ({:.0f}ms)",
            model, "✓" if ok else "✗", (time.monotonic() - t0) * 1000,
        )
        return ok

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
        # 2b. Sincronizza i file RAG della sessione → ctx (sync-in).
        self._sync_session_rag(ctx)

        # 3. Memory RAG
        await self._run_memory(ctx)

        # 4. Personality
        self._personality.apply_to_context(ctx)

        # 4.4 File analysis (se il testo utente cita path con estensione
        # supportata + tool consentito dal profilo)
        await self._run_file_analysis(ctx)

        # 4.42 Classificazione intenti del turno: instrada il web (e, dagli stadi
        # successivi, i percorsi sui file). Popola ctx.metadata["intents"].
        await self._classify_intents(ctx)

        # 4.45 File RAG: recupera passaggi rilevanti dai file grandi indicizzati
        # (di questo turno o di turni precedenti della sessione)
        await self._run_file_rag(ctx)

        # 4.47 Map-reduce on-demand: domande globali/strutturali/posizionali sul
        # documento (instradate dal classificatore). Inietta la sintesi.
        await self._run_map_reduce(ctx)

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
        finally:
            # Eseguito su ogni uscita dal blocco, inclusa la cancellazione:
            #  - fine naturale dello stream;
            #  - GeneratorExit (consumer che chiude il generatore tra un chunk
            #    e l'altro);
            #  - CancelledError (chunk annullato mentre si attendeva il token).
            # In tutti i casi il parziale entra nella finestra conversazionale,
            # così il turno successivo resta coerente anche dopo uno stop —
            # esattamente come fanno Claude/Gemini. _update_history è sincrono,
            # quindi è sicuro anche durante l'unwinding di GeneratorExit.
            # Memoria e TTS restano fuori dal try: su cancel vengono saltati.
            ctx.assistant_text = full_text
            if full_text.strip():
                try:
                    self._update_history(ctx)
                except Exception as exc:
                    logger.warning("orchestrator.turn | update_history: {}", exc)
            # Persiste i file RAG della sessione (write-back). Nel finally di
            # proposito: l'indicizzazione su ChromaDB avviene prima dell'LLM,
            # quindi anche su cancellazione dello stream il file_id non va perso.
            self._persist_session_rag(ctx)

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
            llm_ok          = self._llm is not None,
            memory_ok       = self._memory is not None,
            personality_ok  = self._personality is not None and self._personality._loaded,
            stt_ok          = self._stt is not None,
            tts_ok          = self._tts is not None,
            web_search_ok   = self._web_search is not None,
            file_analysis_ok= self._file_analyzer is not None,
            active_sessions = len(self._session_histories),
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

    def _sync_session_rag(self, ctx: AssistantContext) -> None:
        """Copia i file RAG della sessione nel ctx del turno (sync-in).

        Copia difensiva con list(): _index_large_files fa append, non vogliamo
        mutare lo store di sessione finché il turno non è andato a buon fine.
        """
        ctx.metadata["rag_files"] = list(self._session_rag_files[ctx.session_id])

    def _persist_session_rag(self, ctx: AssistantContext) -> None:
        """Salva i file RAG accumulati nel turno nello store di sessione (write-back).

        Chiamato nel finally di turn(): l'indicizzazione su ChromaDB avviene
        prima dell'LLM, quindi il file_id va persistito anche su cancellazione.
        """
        rag_files = ctx.metadata.get("rag_files")
        if rag_files:
            self._session_rag_files[ctx.session_id] = rag_files

    def clear_session(self, session_id: str) -> None:
        """Azzera la cronologia e i file RAG di una sessione."""
        self._session_histories.pop(session_id, None)
        self._session_rag_files.pop(session_id, None)
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

    async def _run_file_analysis(self, ctx: AssistantContext) -> None:
        """
        Se il testo utente contiene path con estensione supportata e il
        profilo attivo consente il tool 'file_analysis', analizza fino a
        max_files_per_turn file e inietta i contenuti nel system prompt.

        Pattern identico a _run_web_search:
            1. Skip se il modulo non è inizializzato.
            2. Skip se nessun path è riconosciuto nel testo.
            3. Skip se il profilo non consente il tool.
            4. Analizza i file in parallelo.
            5. Applica fair-share sui caratteri totali (cap globale).
            6. Inietta il blocco nel system prompt e registra il tool call.

        Errori non fatali: file singoli falliti vengono saltati (con log)
        e gli altri proseguono. Tutto il modulo failing è errore non fatale.
        """
        if not self._file_analyzer:
            return

        cfg = settings.file_analysis

        # 1) Riconosci path nel testo utente
        paths = _extract_file_paths(ctx.user_text, max_paths=cfg.max_files_per_turn)
        if not paths:
            return

        # 2) Permesso dal profilo
        try:
            allowed = self._personality.active.allows_tool(_FILE_ANALYSIS_TOOL)
        except Exception:
            allowed = False
        if not allowed:
            logger.debug(
                "orchestrator._run_file_analysis | tool '{}' non consentito dal profilo '{}'",
                _FILE_ANALYSIS_TOOL,
                ctx.personality_name,
            )
            return

        logger.info(
            "orchestrator._run_file_analysis | {} path rilevati: {}",
            len(paths), [p[-60:] for p in paths],
        )

        # 3) Analisi in parallelo. asyncio.gather con return_exceptions=True
        # protegge contro estrattori che potrebbero sollevare (FileAnalyzer
        # normalmente non solleva, ma siamo difensivi sul livello superiore).
        t0 = time.monotonic()
        try:
            raw_results = await asyncio.gather(
                *(self._file_analyzer.analyze(p) for p in paths),
                return_exceptions=True,
            )
        except Exception as exc:
            logger.warning(
                "orchestrator._run_file_analysis | analisi fallita: {} — continuo", exc
            )
            return
        ctx.set_timing("file_analysis", (time.monotonic() - t0) * 1000)

        # 4) Filtra/normalizza i risultati
        results: list[AnalysisResult] = []
        for path, item in zip(paths, raw_results):
            if isinstance(item, Exception):
                logger.warning(
                    "orchestrator._run_file_analysis | eccezione su '{}': {}",
                    path[-60:], item,
                )
                continue
            results.append(item)

        if not results:
            return

        # 5) Fair-share sui caratteri totali. Se i contenuti utili insieme
        # superano max_total_chars, ridistribuiamo il budget in parti uguali
        # tra i file con contenuto valido, troncando ciascuno.
        usable = [r for r in results if r.error is None and r.content.strip()]
        if usable:
            total_chars = sum(r.char_count for r in usable)
            if total_chars > cfg.max_total_chars:
                budget_per_file = max(500, cfg.max_total_chars // len(usable))
                logger.debug(
                    "orchestrator._run_file_analysis | fair-share: {} file, "
                    "budget/file={}", len(usable), budget_per_file,
                )
                for r in usable:
                    if r.char_count > budget_per_file:
                        # Tronchiamo "in place" il content; original_char_count
                        # già rispecchia la dimensione pre-truncation iniziale,
                        # qui aggiorniamo solo il visibile e segniamo truncated.
                        notice = (
                            f"\n\n[...contenuto troncato dal cap globale: "
                            f"mostrati {budget_per_file} di {r.char_count} char]"
                        )
                        r.content = r.content[:budget_per_file] + notice
                        r.char_count = len(r.content)
                        r.truncated = True

        # 6) Inietta nel system prompt. Per i file destinati al RAG l'estratto
        # inline è ridotto a rag_inline_preview_chars: il dettaglio lo portano
        # i PASSAGGI RILEVANTI, evitando di saturare il context.
        block = _format_file_analysis_block(
            results, rag_preview_chars=cfg.rag_inline_preview_chars
        )
        if block:
            ctx.system_prompt = (ctx.system_prompt or "") + block

        # 6b) RAG: indicizza i file il cui testo COMPLETO supera la soglia
        # (AnalysisResult.full_content valorizzato da FileAnalyzer). Registra
        # i file_id nella sessione per il retrieval di questo e dei prossimi
        # turni. Indicizzazione sincrona qui (best-effort); il segnalino
        # "indexing" in background arriva nel commit successivo.
        await self._index_large_files(ctx, results)

        # 7) Registra il tool call (anche se tutti i file sono falliti, per
        # diagnostica)
        ctx.add_tool_call(
            tool=_FILE_ANALYSIS_TOOL,
            args={"paths": paths, "max_files_per_turn": cfg.max_files_per_turn},
            result=[r.to_log_dict() for r in results],
        )

        n_ok = sum(1 for r in results if r.error is None and r.content.strip())
        n_err = sum(1 for r in results if r.error is not None)
        logger.info(
            "orchestrator._run_file_analysis | analizzati={} ok={} err={} chars_iniettati={}",
            len(results), n_ok, n_err, len(block),
        )

    async def _index_large_files(
        self, ctx: AssistantContext, results: list[AnalysisResult],
    ) -> None:
        """
        Indicizza nel RAG i file con full_content valorizzato (testo > soglia).
        Registra i file_id in ctx.metadata["rag_files"] (lista di dict
        {file_id, source}) per il retrieval. Best-effort: non solleva.
        """
        if self._file_rag is None:
            return
        large = [r for r in results if getattr(r, "full_content", None)]
        if not large:
            return

        rag_files: list[dict] = ctx.metadata.get("rag_files", [])
        known_ids = {f["file_id"] for f in rag_files}

        t0 = time.monotonic()
        for r in large:
            source = r.path.rsplit("/", 1)[-1]  # nome file leggibile
            try:
                res = await self._file_rag.index_file(r.full_content, source=source)
            except Exception as exc:
                logger.warning(
                    "orchestrator._index_large_files | '{}' fallito: {}", source, exc
                )
                continue
            if res.file_id not in known_ids:
                rag_files.append({"file_id": res.file_id, "source": source})
                known_ids.add(res.file_id)

        ctx.metadata["rag_files"] = rag_files
        ctx.set_timing("file_rag_index", (time.monotonic() - t0) * 1000)
        logger.info(
            "orchestrator._index_large_files | file_grandi={} sessione_rag_files={}",
            len(large), len(rag_files),
        )

    async def _run_file_rag(self, ctx: AssistantContext) -> None:
        """
        Se la sessione ha file indicizzati, recupera i passaggi rilevanti per
        la domanda corrente e li inietta nel system prompt. Fuso col resto del
        contesto (memoria, file inline). Best-effort: non solleva.

        I file_id vivono in ctx.metadata["rag_files"], sincronizzati a ogni
        turno da/verso self._session_rag_files[session_id] (vedi turn()): la
        persistenza tra i turni è interna all'orchestrator, il chiamante non
        deve fare nulla. Il metodo è robusto anche se la lista è vuota.
        """
        if self._file_rag is None:
            return
        # Si fa da parte solo se il classificatore ha instradato al map-reduce
        # (file globale/strutturale/posizionale) SENZA intento locale: in quel
        # caso risponde _run_map_reduce. Negli altri casi (locale, None, vuoto)
        # il RAG semantico gira come oggi — nessuna rete tolta.
        intents = ctx.metadata.get("intents")
        if isinstance(intents, set) and (intents & _MAPREDUCE_INTENTS) and (
            Intent.FILE_LOCAL not in intents
        ):
            return
        rag_files = ctx.metadata.get("rag_files") or []
        if not rag_files:
            return
        query = ctx.user_text
        if not query.strip():
            return

        top_k = getattr(settings.file_analysis, "rag_top_k", 5)
        t0 = time.monotonic()
        all_chunks: list = []
        for f in rag_files:
            try:
                chunks = await self._file_rag.search_file(f["file_id"], query, top_k=top_k)
            except Exception as exc:
                logger.warning("orchestrator._run_file_rag | search '{}': {}", f.get("source"), exc)
                continue
            all_chunks.extend(chunks)

        ctx.set_timing("file_rag", (time.monotonic() - t0) * 1000)
        if not all_chunks:
            return

        # Ordina per rilevanza decrescente e tieni i migliori top_k globali.
        all_chunks.sort(key=lambda c: c.relevance_score, reverse=True)
        all_chunks = all_chunks[:top_k]

        block = _format_file_rag_block(all_chunks)
        if block:
            ctx.system_prompt = (ctx.system_prompt or "") + block
        logger.info(
            "orchestrator._run_file_rag | chunk_iniettati={} files={}",
            len(all_chunks), len(rag_files),
        )

    async def _run_map_reduce(self, ctx: AssistantContext) -> None:
        """
        Per gli intenti FILE_GLOBAL/STRUCTURAL/POSITIONAL: scandisce il documento
        a blocchi (on-demand) e inietta la sintesi nel system prompt; poi il
        modello principale streamma la risposta finale. Best-effort: non solleva.
        Su intents None (classificatore fallito) NON parte: copre il RAG semantico.
        """
        if self._file_rag is None or self._map_reduce is None:
            return
        intents = ctx.metadata.get("intents")
        if not isinstance(intents, set):
            return
        mr = intents & _MAPREDUCE_INTENTS
        if not mr:
            return
        rag_files = ctx.metadata.get("rag_files") or []
        if not rag_files:
            return
        query = ctx.user_text
        if not query.strip():
            return

        positional_only = mr == {Intent.FILE_POSITIONAL}
        t0 = time.monotonic()
        blocks: list[dict] = []
        for f in rag_files:
            try:
                chunks = await self._file_rag.get_ordered_chunks(f["file_id"])
            except Exception as exc:
                logger.warning(
                    "orchestrator._run_map_reduce | chunks '{}': {}", f.get("source"), exc
                )
                continue
            fb = list(FileRAG.iter_blocks(chunks, _MAPREDUCE_BLOCK_CHARS))
            if positional_only and len(fb) > 2 * _POSITIONAL_BLOCKS:
                fb = fb[:_POSITIONAL_BLOCKS] + fb[-_POSITIONAL_BLOCKS:]
            blocks.extend(fb)

        if not blocks:
            return
        try:
            result = await self._map_reduce.run(question=query, blocks=blocks)
        except Exception as exc:
            logger.warning("orchestrator._run_map_reduce | motore: {}", exc)
            return
        ctx.set_timing("map_reduce", (time.monotonic() - t0) * 1000)

        block = _format_map_reduce_block(result.content)
        if block:
            ctx.system_prompt = (ctx.system_prompt or "") + block
        logger.info("orchestrator._run_map_reduce | {}", result.to_log_dict())

    async def _classify_intents(self, ctx: AssistantContext) -> None:
        """
        Classifica la query del turno e parcheggia il risultato in
        ctx.metadata["intents"]: set[Intent] (riuscita, anche vuoto) oppure None
        (fallita -> i consumatori usano il fallback). Va chiamato DOPO
        _run_file_analysis, cosi' has_file vede anche il file appena caricato.
        Non fatale: senza classificatore parcheggia None.
        """
        if not self._intent_classifier:
            ctx.metadata["intents"] = None
            return
        # I file vivono in ctx.metadata["rag_files"] (stessa fonte di
        # _run_file_rag/_run_map_reduce), gia' popolata da _run_file_analysis in
        # questo turno: cosi' has_file e' corretto anche al PRIMO turno con un
        # file appena caricato (self._session_rag_files si aggiorna solo a fine
        # turno, sarebbe in ritardo).
        has_file = bool(ctx.metadata.get("rag_files"))
        ctx.metadata["intents"] = await self._intent_classifier.classify(
            ctx.user_text, has_file=has_file,
        )

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

        # Trigger: intento WEB_SEARCH dal classificatore; se la classificazione
        # e' fallita (None), fallback all'euristica keyword.
        intents = ctx.metadata.get("intents")
        if intents is None:
            triggered = _should_search(ctx.user_text)
        else:
            triggered = Intent.WEB_SEARCH in intents
        if not triggered:
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

        Eccezione: nei turni con un file in gioco (analisi inline o RAG sulla
        sessione) la risposta dell'assistente NON viene memorizzata. Il suo
        contenuto è derivato dal file e vive già nel file RAG, effimero e isolato
        per sessione; la memoria invece è globale (search senza filtro di
        sessione), quindi salvarlo lo farebbe riemergere in altre chat. Il testo
        utente si continua a salvare (preserva fatti personali detti nel turno).
        """
        if not self._memory:
            return

        file_in_turn = (
            bool(ctx.metadata.get("rag_files"))
            or any(t.get("tool") == _FILE_ANALYSIS_TOOL for t in ctx.tool_calls)
        )

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

        if ctx.assistant_text and not file_in_turn:
            try:
                await self._memory.save(
                    ctx.assistant_text,
                    {**base_meta, "source": _MEMORY_SOURCE_ASST, "role": "assistant"},
                )
            except Exception as exc:
                logger.warning(
                    "orchestrator._save_turn | salvataggio assistant fallito: {}", exc
                )
        elif ctx.assistant_text and file_in_turn:
            logger.debug(
                "orchestrator._save_turn | risposta file-grounded non memorizzata"
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
                f"stt={'✓' if self._stt else '✗'} "
                f"file={'✓' if self._file_analyzer else '✗'}>"
            )
        return "<Orchestrator [non caricato]>"
