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
from typing import Any, AsyncGenerator, Optional, Union

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

    # -- context manager -------------------------------------------------------

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

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
        resolved = model or _model_for_role(role)
        payload  = self._build_payload(resolved, messages, options, system, stream=False)

        logger.debug("llm.chat | model={} msgs={}", resolved, len(messages))
        r = await self._http.post("/api/chat", json=payload)
        r.raise_for_status()

        data = r.json()
        p, c, t = _token_counts(data)
        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
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
        resolved = model or _model_for_role(role)
        payload  = self._build_payload(resolved, messages, options, system, stream=True)

        logger.debug("llm.stream | model={}", resolved)
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
                    yield chunk
                if data.get("done"):
                    break

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
        resolved = model or settings.ollama.embed_model
        if isinstance(texts, str):
            texts = [texts]

        logger.debug("llm.embed | model={} n={}", resolved, len(texts))

        # /api/embed — Ollama >= 0.3 (accetta lista)
        r = await self._http.post("/api/embed", json={"model": resolved, "input": texts})

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

        msg_list: list[dict] = []
        if system and not any(m.role == Role.SYSTEM for m in messages):
            msg_list.append({"role": "system", "content": system})
        msg_list.extend(m.to_dict() for m in messages)

        return {
            "model":    model,
            "messages": msg_list,
            "stream":   stream,
            "think":    think,   # top level — unico modo funzionante
            "options":  opts,
        }
