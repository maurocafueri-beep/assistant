"""
modules/personality/base_personality.py
PersonalityManager — carica e gestisce i profili di personalità YAML.

I profili vivono in config/personalities/<nome>.yaml (già presenti: default, dev).
Struttura YAML attesa:

    name:          default
    display_name:  "Assistente"
    description:   "Assistente generico"
    language:      it
    tts_voice:     af_sarah
    lora_adapter:  null          # path relativo o null
    allowed_tools: []            # lista di tool abilitati
    system_prompt: |
      Sei un assistente AI locale...

API pubblica:
    manager.active                          → PersonalityProfile  (property)
    manager.get(name)                       → PersonalityProfile
    manager.switch(name)                    → PersonalityProfile
    manager.list_profiles()                 → list[str]
    manager.reload()                        → dict[str, PersonalityProfile]
    manager.apply_to_context(ctx)           → None  (popola AssistantContext)

Uso rapido:
    async with PersonalityManager() as pm:
        profile = pm.active
        print(profile.system_prompt)
        pm.switch("dev")

Uso senza context manager (i profili sono caricati in modo sincrono):
    pm = PersonalityManager()
    await pm.load()
    profile = pm.get("default")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from core.context import AssistantContext
from core.logger import logger


# ---------------------------------------------------------------------------
# Dataclass pubblica
# ---------------------------------------------------------------------------

@dataclass
class PersonalityProfile:
    """
    Rappresentazione in memoria di un profilo YAML.

    Tutti i campi sono già validati e normalizzati all'atto del caricamento.
    """
    name:          str
    display_name:  str
    description:   str
    language:      str
    tts_voice:     str
    lora_adapter:  Optional[str]
    allowed_tools: list[str]
    system_prompt: str

    # --- utili a runtime ---

    @property
    def is_multilingual(self) -> bool:
        """True se il profilo non forza una singola lingua (language == "auto")."""
        return self.language.lower() == "auto"

    def allows_tool(self, tool_name: str) -> bool:
        """Verifica se un tool è abilitato per questo profilo."""
        return not self.allowed_tools or tool_name in self.allowed_tools

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "name":          self.name,
            "display_name":  self.display_name,
            "language":      self.language,
            "tts_voice":     self.tts_voice,
            "lora_adapter":  self.lora_adapter,
            "allowed_tools": self.allowed_tools,
            "prompt_len":    len(self.system_prompt),
        }

    def __str__(self) -> str:
        return f"<PersonalityProfile '{self.name}' ({self.display_name})>"


# ---------------------------------------------------------------------------
# Parsing YAML
# ---------------------------------------------------------------------------

def _parse_yaml(path: Path) -> PersonalityProfile:
    """
    Legge e valida un file YAML → PersonalityProfile.
    Import di yaml solo qui (non è pesante, ma manteniamo il pattern).
    """
    import yaml  # già in PyYAML, non richiederebbe executor ma coerente

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # --- normalizzazione con valori di default sicuri ---
    name = raw.get("name") or path.stem
    return PersonalityProfile(
        name=name,
        display_name=str(raw.get("display_name", name)),
        description=str(raw.get("description", "")),
        language=str(raw.get("language", "it")).lower(),
        tts_voice=str(raw.get("tts_voice", settings.tts.voice)),
        lora_adapter=raw.get("lora_adapter") or None,
        allowed_tools=list(raw.get("allowed_tools") or []),
        system_prompt=str(raw.get("system_prompt", "")).strip(),
    )


def _load_all_from_dir(config_dir: Path) -> dict[str, PersonalityProfile]:
    """Legge tutti i file .yaml/.yml da config_dir → dict nome→profilo."""
    profiles: dict[str, PersonalityProfile] = {}
    if not config_dir.exists():
        logger.warning("personality | directory non trovata: {}", config_dir)
        return profiles

    for path in sorted(config_dir.glob("*.yaml")):
        try:
            p = _parse_yaml(path)
            profiles[p.name] = p
            logger.debug("personality | caricato '{}' da {}", p.name, path.name)
        except Exception as exc:
            logger.error("personality | errore caricamento '{}': {}", path.name, exc)

    # supporta anche estensione .yml
    for path in sorted(config_dir.glob("*.yml")):
        if path.stem not in profiles:
            try:
                p = _parse_yaml(path)
                profiles[p.name] = p
                logger.debug("personality | caricato '{}' da {}", p.name, path.name)
            except Exception as exc:
                logger.error("personality | errore caricamento '{}': {}", path.name, exc)

    return profiles


# ---------------------------------------------------------------------------
# Manager principale
# ---------------------------------------------------------------------------

class PersonalityManager:
    """
    Gestisce il ciclo di vita dei profili di personalità.

    Carica tutti i YAML all'avvio, permette il cambio di personalità a runtime
    e l'iniezione del profilo attivo nel AssistantContext.

    Args:
        config_dir:       Directory dei YAML (default: settings.personality.config_dir).
        default_name:     Nome del profilo attivo all'avvio
                          (default: settings.personality.default_personality).

    Esempio — context manager:
        async with PersonalityManager() as pm:
            pm.switch("dev")
            ctx.personality_name = pm.active.name
            ctx.system_prompt    = pm.active.system_prompt

    Esempio — standalone:
        pm = PersonalityManager()
        await pm.load()
        print(pm.list_profiles())
    """

    def __init__(
        self,
        config_dir:   Optional[Path] = None,
        default_name: Optional[str]  = None,
    ) -> None:
        self._config_dir   = config_dir   or settings.personality.config_dir
        self._default_name = default_name or settings.personality.default_personality
        self._profiles:    dict[str, PersonalityProfile] = {}
        self._active_name: str = self._default_name
        self._loaded:      bool = False

    # -- context manager -------------------------------------------------------

    async def __aenter__(self) -> "PersonalityManager":
        await self.load()
        return self

    async def __aexit__(self, *_: Any) -> None:
        # Nessuna risorsa da liberare; azzeriamo per sicurezza
        self._loaded = False

    # -- caricamento -----------------------------------------------------------

    async def load(self) -> dict[str, PersonalityProfile]:
        """
        Carica (o ricarica) tutti i profili YAML in un executor.
        Ritorna il dizionario nome→profilo.
        """
        loop = asyncio.get_running_loop()
        profiles = await loop.run_in_executor(
            None, lambda: _load_all_from_dir(self._config_dir)
        )
        self._profiles = profiles
        self._loaded   = True

        if not profiles:
            logger.warning(
                "personality | nessun profilo trovato in {}",
                self._config_dir,
            )
        else:
            logger.info(
                "personality | {} profili caricati: {}",
                len(profiles),
                list(profiles.keys()),
            )

        # Verifica che il profilo default esista
        if self._default_name not in self._profiles:
            fallback = next(iter(self._profiles), None)
            if fallback:
                logger.warning(
                    "personality | profilo default '{}' non trovato — "
                    "uso '{}' come fallback",
                    self._default_name,
                    fallback,
                )
                self._active_name = fallback
            else:
                logger.error("personality | nessun profilo disponibile")
        else:
            self._active_name = self._default_name

        return self._profiles

    async def reload(self) -> dict[str, PersonalityProfile]:
        """
        Ricarica i profili dal disco senza riavviare il manager.
        Utile per raccogliere nuovi YAML aggiunti a runtime.
        """
        logger.info("personality | reload da {}", self._config_dir)
        return await self.load()

    # -- accesso profili -------------------------------------------------------

    @property
    def active(self) -> PersonalityProfile:
        """Profilo attivo corrente. Lancia RuntimeError se non ancora caricato."""
        self._require_loaded()
        return self._profiles[self._active_name]

    def get(self, name: str) -> PersonalityProfile:
        """
        Restituisce un profilo per nome.
        Lancia KeyError se non esiste.
        """
        self._require_loaded()
        if name not in self._profiles:
            available = list(self._profiles.keys())
            raise KeyError(
                f"Profilo '{name}' non trovato. Disponibili: {available}"
            )
        return self._profiles[name]

    def switch(self, name: str) -> PersonalityProfile:
        """
        Cambia il profilo attivo.
        Lancia KeyError se il profilo non esiste.
        Ritorna il nuovo profilo attivo.
        """
        profile = self.get(name)          # lancia KeyError se assente
        old_name = self._active_name
        self._active_name = name
        if old_name != name:
            logger.info(
                "personality | cambio profilo: '{}' → '{}'",
                old_name,
                name,
            )
        return profile

    def list_profiles(self) -> list[str]:
        """Elenco dei nomi di tutti i profili caricati, in ordine alfabetico."""
        self._require_loaded()
        return sorted(self._profiles.keys())

    def has_profile(self, name: str) -> bool:
        """Verifica l'esistenza di un profilo senza lanciare eccezioni."""
        return name in self._profiles

    # -- integrazione con AssistantContext -------------------------------------

    def apply_to_context(self, ctx: AssistantContext) -> None:
        """
        Popola i campi rilevanti di AssistantContext con il profilo attivo.

        Campi scritti:
            ctx.personality_name  → profile.name
            ctx.system_prompt     → profile.system_prompt
            ctx.model_name        → invariato (il profilo non seleziona il modello)

        Non sovrascrive system_prompt se è già stato impostato dall'esterno
        (es. l'orchestratore ha aggiunto contesto aggiuntivo).
        """
        profile = self.active
        ctx.personality_name = profile.name
        if not ctx.system_prompt:
            ctx.system_prompt = profile.system_prompt
        logger.debug(
            "personality | applicato '{}' al contesto {}",
            profile.name,
            ctx.turn_id,
        )

    # -- interno ---------------------------------------------------------------

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "PersonalityManager non inizializzato — "
                "usa 'async with PersonalityManager()' oppure chiama 'await pm.load()'"
            )

    # -- rappresentazione ------------------------------------------------------

    def __repr__(self) -> str:
        if self._loaded:
            return (
                f"<PersonalityManager active='{self._active_name}' "
                f"profiles={list(self._profiles.keys())}>"
            )
        return "<PersonalityManager [non caricato]>"
