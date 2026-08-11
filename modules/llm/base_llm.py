"""
modules/llm/base.py
OllamaClient — unico punto di accesso a tutti i modelli Ollama del progetto.

Modelli gestiti (configurati in config/settings.py → OllamaSettings,
nomi esatti definiti lì o sovrascritti via .env):
    CHAT   → settings.ollama.chat_model
    CODE   → settings.ollama.code_model
    VISION → settings.ollama.vision_model
    embed  → settings.ollama.embed_model  (nomic-embed-text)

API pubblica:
    client.chat(messages, role, **kw)       → LLMResponse          (async)
    client.stream(messages, role, **kw)     → AsyncGenerator[str]  (async)
    client.embed(texts)                     → list[list[float]]    (async)
    client.vision(prompt, images, **kw)     → LLMResponse          (async)
    client.is_available()                   → bool                 (async)
    client.list_models()                    → list[str]            (async)
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Iterable, Optional, Union

import httpx

from config.settings import settings
from core.context import ModelRole  # unica definizione nel progetto
from core.logger import logger


# ---------------------------------------------------------------------------
# Enums & Dataclasses
# ---------------------------------------------------------------------------

class Role(str, Enum):
    """Ruolo mittente in una conversazione."""
    SYSTEM    = "system"
    USER      = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Un singolo messaggio nella conversazione."""
    role:    Role
    content: str
    images:  list[str] = field(default_factory=list)
    """Immagini base64-encoded (solo per ModelRole.VISION)."""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.images:
            d["images"] = self.images
        return d


@dataclass
class LLMResponse:
    """Risposta completa da Ollama."""
    content:           str
    model:             str
    role:              ModelRole
    prompt_tokens:     int  = 0
    completion_tokens: int  = 0
    total_tokens:      int  = 0
    done:              bool = True

    @property
    def tokens(self) -> dict[str, int]:
        return {
            "prompt":     self.prompt_tokens,
            "completion": self.completion_tokens,
            "total":      self.total_tokens,
        }


# ---------------------------------------------------------------------------
# Helpers interni
# ---------------------------------------------------------------------------

def _model_for_role(role: ModelRole) -> str:
    """Risolve ModelRole → nome modello Ollama da settings."""
    return {
        ModelRole.CHAT:   settings.ollama.chat_model,
        ModelRole.CODE:   settings.ollama.code_model,
        ModelRole.VISION: settings.ollama.vision_model,
    }[role]


def _encode_image(source: Union[str, Path, bytes]) -> str:
    """Accetta path file, bytes o stringa già base64 → restituisce base64 str."""
    if isinstance(source, bytes):
        return base64.b64encode(source).decode()
    p = Path(source)
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return str(source)  # già base64


def _token_counts(raw: dict) -> tuple[int, int, int]:
    p = raw.get("prompt_eval_count", 0) or 0
    c = raw.get("eval_count", 0) or 0
    return p, c, p + c


# ---------------------------------------------------------------------------
# Filtro tag <think>...</think>
# ---------------------------------------------------------------------------
# Difesa post-Ollama: alcuni modelli (anche con `think:false` sul payload)
# possono comunque emettere blocchi di ragionamento dentro `message.content`,
# inframmezzati come <think>...</think> o <thinking>...</thinking>. Quando
# accade, l'utente li vede comparire nello stream come se fossero parte della
# risposta. Questo filtro li rimuove in modo robusto allo streaming:
#   - gestisce tag spezzati su più chunk;
#   - non emette mai un suffisso che potrebbe essere il prefisso di un tag,
#     finché il chunk successivo non chiarisce di cosa si tratti;
#   - è un no-op sui modelli che non emettono questi tag.
# Volutamente non tocca il campo `message.thinking`, che Ollama già separa
# correttamente quando think=true: lì il filtro non serve.

_THINK_OPEN  = ("<think>", "<thinking>")
_THINK_CLOSE = ("</think>", "</thinking>")


def _suffix_overlap(s: str, tag: str) -> int:
    """Lunghezza massima k tale che s termini con tag[:k] (0 se nessun match)."""
    m = min(len(s), len(tag))
    for k in range(m, 0, -1):
        if s.endswith(tag[:k]):
            return k
    return 0


class _StripThink:
    """
    Filtro stateful che rimuove blocchi <think>...</think> e
    <thinking>...</thinking> da uno stream di chunk di testo.

    Uso:
        s = _StripThink()
        for chunk in chunks:
            safe = s.feed(chunk); ... # emetti `safe` (potrebbe essere "")
        tail = s.flush()              # emetti `tail` a fine stream
    """

    def __init__(self) -> None:
        self._buf: str = ""
        self._in_think: bool = False

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        out: list[str] = []
        while True:
            if self._in_think:
                pos, used = -1, ""
                for tag in _THINK_CLOSE:
                    i = self._buf.find(tag)
                    if i >= 0 and (pos < 0 or i < pos):
                        pos, used = i, tag
                if pos < 0:
                    # chiusura non ancora arrivata: scarta tutto tranne un
                    # possibile prefisso di tag di chiusura in coda al buffer.
                    keep = max(_suffix_overlap(self._buf, t) for t in _THINK_CLOSE)
                    self._buf = self._buf[-keep:] if keep else ""
                    break
                self._buf = self._buf[pos + len(used):]
                self._in_think = False
                continue
            # fuori da un blocco think
            pos, used = -1, ""
            for tag in _THINK_OPEN:
                i = self._buf.find(tag)
                if i >= 0 and (pos < 0 or i < pos):
                    pos, used = i, tag
            if pos < 0:
                # nessun tag completo: emetti tutto tranne un eventuale
                # prefisso di tag di apertura sospeso a fine buffer.
                keep = max(_suffix_overlap(self._buf, t) for t in _THINK_OPEN)
                if keep:
                    out.append(self._buf[:-keep])
                    self._buf = self._buf[-keep:]
                else:
                    out.append(self._buf)
                    self._buf = ""
                break
            out.append(self._buf[:pos])
            self._buf = self._buf[pos + len(used):]
            self._in_think = True
            continue
        return "".join(out)

    def flush(self) -> str:
        """A fine stream: emetti il residuo se siamo fuori da un blocco think."""
        if self._in_think:
            self._buf = ""
            return ""
        tail, self._buf = self._buf, ""
        return tail


def _strip_think_tags(text: str) -> str:
    """Versione single-shot per testo non in streaming (es. chat() non-stream)."""
    s = _StripThink()
    return s.feed(text) + s.flush()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Client async per Ollama. Usa un singolo httpx.AsyncClient per sessione.
    Preferire come async context manager.

    Esempio:
        async with OllamaClient() as llm:
            resp = await llm.chat([Message(Role.USER, "Ciao!")], ModelRole.CHAT)
            print(resp.content)
    """

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=settings.ollama.base_url,
            timeout=httpx.Timeout(settings.ollama.timeout),
        )
        # Modelli per cui ABBIAMO fatto richieste: solo questi vengono
        # scaricati dalla VRAM alla chiusura (vedi unload_models).
        self._touched: set[str] = set()

    def _resolve(self, model: Optional[str], role: Optional[ModelRole]) -> str:
        """Nome modello per la richiesta, registrandolo come "nostro"."""
        name = model or (_model_for_role(role) if role else settings.ollama.embed_model)
        self._touched.add(name)
        return name

    # -- context manager -------------------------------------------------------

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.unload_models()
        await self._http.aclose()

    async def unload_models(self) -> int:
        """
        Scarica dalla VRAM i modelli che abbiamo tenuto residenti.

        Ogni richiesta viaggia con `keep_alive` (30m di default) perché tra
        un turno e l'altro il modello NON debba ricaricarsi. Alla chiusura
        però quella residenza diventa spreco: senza questo passo l'assistente
        continua a occupare ~8 GB di VRAM per mezz'ora dopo l'uscita.
        `keep_alive: 0` dice a Ollama di liberarlo subito.

        Best-effort e mirato: scarica solo i modelli per cui QUESTA istanza
        ha fatto richieste (`_touched`, che include gli override scelti a
        runtime dalla UI), mai quelli caricati da altre applicazioni. Non
        solleva mai.

        Returns:
            Numero di modelli per cui lo scarico è stato richiesto.
        """
        if not self._touched:
            return 0
        ours = set(self._touched)
        # Ollama riporta i nomi con tag esplicito (":latest"): normalizziamo
        # entrambi i lati, altrimenti "modello" non combacia con "modello:latest".
        norm = lambda n: n if ":" in n else f"{n}:latest"
        ours = {norm(n) for n in ours}
        try:
            r = await self._http.get("/api/ps")
            r.raise_for_status()
            loaded = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as exc:
            logger.debug("llm.unload | lista modelli non disponibile: {}", exc)
            return 0

        done = 0
        for name in loaded:
            if name not in ours:
                continue
            try:
                # generate con prompt vuoto: nessuna inferenza, solo lo
                # scarico immediato (endpoint documentato da Ollama).
                r = await self._http.post(
                    "/api/generate",
                    json={"model": name, "keep_alive": 0},
                )
                r.raise_for_status()
                done += 1
                logger.info("llm.unload | '{}' scaricato dalla VRAM", name)
            except Exception as exc:
                logger.warning("llm.unload | '{}' fallito: {}", name, exc)
        return done

    # -- utility ---------------------------------------------------------------

    async def is_available(self) -> bool:
        """Verifica che Ollama sia raggiungibile."""
        try:
            r = await self._http.get("/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """Elenco modelli installati su Ollama."""
        r = await self._http.get("/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    # -- chat ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[Message],
        role:     ModelRole = ModelRole.CHAT,
        *,
        model:   Optional[str]  = None,
        options: Optional[dict] = None,
        system:  Optional[str]  = None,
    ) -> LLMResponse:
        """
        Chiamata chat — raccoglie l'intera risposta prima di restituirla.

        Args:
            messages: Cronologia conversazione.
            role:     Seleziona il modello (CHAT/CODE/VISION). Override con `model`.
            options:  Parametri Ollama (temperature, num_ctx, …).
            system:   System prompt (aggiunto in testa se non già presente).
        """
        resolved = self._resolve(model, role)
        payload  = self._build_payload(resolved, messages, options, system, stream=False)

        logger.debug("llm.chat | model={} msgs={}", resolved, len(messages))
        r = await self._http.post("/api/chat", json=payload)
        r.raise_for_status()

        data = r.json()
        p, c, t = _token_counts(data)
        # Filtro <think>: il modello, anche con think:false sul payload, a volte
        # emette il ragionamento dentro `content`. Lo rimuoviamo qui in modo
        # trasparente per ogni chiamante (intent, map_reduce, terminal_agent…).
        # Sui modelli puliti è un no-op.
        return LLMResponse(
            content=_strip_think_tags(data.get("message", {}).get("content", "")),
            model=resolved,
            role=role,
            prompt_tokens=p,
            completion_tokens=c,
            total_tokens=t,
            done=data.get("done", True),
        )

    # -- stream ----------------------------------------------------------------

    async def stream(
        self,
        messages: list[Message],
        role:     ModelRole = ModelRole.CHAT,
        *,
        model:   Optional[str]  = None,
        options: Optional[dict] = None,
        system:  Optional[str]  = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming: yield ogni chunk di testo non appena disponibile.
        Il TTS (Qwen3) si aggancia qui per iniziare a parlare al primo chunk.

        Uso:
            async for chunk in client.stream(messages, ModelRole.CHAT):
                print(chunk, end="", flush=True)
        """
        resolved = self._resolve(model, role)
        payload  = self._build_payload(resolved, messages, options, system, stream=True)

        logger.debug("llm.stream | model={}", resolved)
        # Filtro <think> streaming: vedi _StripThink. Gestisce tag spezzati su
        # più chunk; sui modelli che non emettono <think> in content è inerte.
        stripper = _StripThink()
        async with self._http.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    safe = stripper.feed(chunk)
                    if safe:
                        yield safe
                if data.get("done"):
                    tail = stripper.flush()
                    if tail:
                        yield tail
                    break

    # -- warmup ----------------------------------------------------------------

    async def warmup(
        self,
        *,
        model:      Optional[str]   = None,
        role:       ModelRole       = ModelRole.CHAT,
        keep_alive: Optional[str]   = None,
    ) -> bool:
        """
        Forza Ollama a caricare il modello in memoria, così il primo turno
        reale non paga il cold-start (caricamento pesi → VRAM/RAM).

        Esegue una generazione minima da 1 token; `keep_alive` (es. "30m",
        "-1" per residenza indefinita) estende quanto a lungo Ollama tiene
        il modello caricato dopo il warmup. Best-effort: non solleva, ritorna
        True solo se la richiesta è andata a buon fine.
        """
        resolved = self._resolve(model, role)
        # num_ctx coerente coi settings: se il warmup carica il modello con
        # context 4K (default Ollama) e poi la prima chat reale arriva con
        # 8K, Ollama deve ricaricare e il warmup non scalda nulla. Allineare
        # i due rende il warmup effettivo.
        warmup_opts: dict[str, Any] = {"num_predict": 1}
        num_ctx = getattr(settings.ollama, "num_ctx", None)
        if num_ctx:
            warmup_opts["num_ctx"] = num_ctx
        payload: dict[str, Any] = {
            "model":    resolved,
            "messages": [{"role": "user", "content": "ok"}],
            "stream":   False,
            "options":  warmup_opts,
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        try:
            logger.debug("llm.warmup | model={} keep_alive={}", resolved, keep_alive)
            r = await self._http.post("/api/chat", json=payload)
            r.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("llm.warmup | '{}' fallito: {}", resolved, exc)
            return False

    # -- vision ----------------------------------------------------------------

    async def vision(
        self,
        prompt:  str,
        images:  list[Union[str, Path, bytes]],
        *,
        system:  Optional[str]  = None,
        options: Optional[dict] = None,
    ) -> LLMResponse:
        """
        Shortcut per qwen3-vl:8b.
        images: lista di path, bytes o stringhe base64.
        """
        encoded = [_encode_image(img) for img in images]
        msg = Message(role=Role.USER, content=prompt, images=encoded)
        return await self.chat([msg], role=ModelRole.VISION, system=system, options=options)

    # -- embed -----------------------------------------------------------------

    async def embed(
        self,
        texts: Union[str, list[str]],
        *,
        model: Optional[str] = None,
    ) -> list[list[float]]:
        """
        Genera embeddings con nomic-embed-text.
        Usato da modules/memory per popolare ChromaDB.

        Returns:
            Lista di vettori float — uno per ogni testo.
        """
        resolved = self._resolve(model, None)
        if isinstance(texts, str):
            texts = [texts]

        logger.debug("llm.embed | model={} n={}", resolved, len(texts))

        # /api/embed — Ollama >= 0.3 (accetta lista)
        # keep_alive: tiene residente anche il modello di embedding (usato dal
        # RAG memoria PRIMA dell'LLM a ogni turno), evitando il suo cold-start.
        embed_payload: dict[str, Any] = {"model": resolved, "input": texts}
        ka = self._keep_alive()
        if ka is not None:
            embed_payload["keep_alive"] = ka
        r = await self._http.post("/api/embed", json=embed_payload)

        if r.status_code == 404:
            # fallback Ollama < 0.3 — /api/embeddings, un testo alla volta
            results = []
            for text in texts:
                r2 = await self._http.post(
                    "/api/embeddings", json={"model": resolved, "prompt": text}
                )
                r2.raise_for_status()
                results.append(r2.json()["embedding"])
            return results

        r.raise_for_status()
        data = r.json()
        return data.get("embeddings") or [data["embedding"]]

    # -- interno ---------------------------------------------------------------

    @staticmethod
    def _keep_alive() -> Optional[str]:
        """
        Durata `keep_alive` da applicare a OGNI richiesta di
        generazione/embedding, così Ollama tiene il modello residente tra un
        turno e l'altro.

        Senza questo, dopo ogni risposta Ollama riparte dal default (5 min):
        basta una pausa perché il modello venga scaricato e il messaggio
        successivo paghi di nuovo il cold-start (caricamento pesi in VRAM/RAM)
        — la causa principale delle risposte "a volte lente". Riusa la stessa
        durata del warmup d'avvio per un comportamento coerente; gated sul
        flag `warmup` così resta disattivabile. Ritorna None se non applicabile.
        """
        if not getattr(settings.ollama, "warmup", True):
            return None
        ka = getattr(settings.ollama, "warmup_keep_alive", None)
        return ka or None

    def _build_payload(
        self,
        model:    str,
        messages: list[Message],
        options:  Optional[dict],
        system:   Optional[str],
        stream:   bool,
    ) -> dict[str, Any]:
        # think va al top level del payload, NON dentro options
        # (limitazione Ollama — issue #14809)
        think = False
        opts: dict[str, Any] = {}
        for k, v in (options or {}).items():
            if k == "think":
                think = v
            else:
                opts[k] = v

        # num_ctx: default dai settings, ma il chiamante può sovrascriverlo
        # esplicitamente passandolo in `options` (es. il map-reduce del
        # riassunto può chiedere context più grande per chunk pesanti).
        # Senza questo, Ollama userebbe il suo default interno di 4096 token,
        # che è troppo basso per file analysis e conversazioni lunghe.
        if "num_ctx" not in opts:
            num_ctx = getattr(settings.ollama, "num_ctx", None)
            if num_ctx:
                opts["num_ctx"] = num_ctx

        msg_list: list[dict] = []
        if system and not any(m.role == Role.SYSTEM for m in messages):
            msg_list.append({"role": "system", "content": system})
        msg_list.extend(m.to_dict() for m in messages)

        payload: dict[str, Any] = {
            "model":    model,
            "messages": msg_list,
            "stream":   stream,
            "think":    think,   # top level — unico modo funzionante
            "options":  opts,
        }
        ka = self._keep_alive()
        if ka is not None:
            payload["keep_alive"] = ka
        return payload
