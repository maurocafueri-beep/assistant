"""
config/settings.py
Unica fonte di verità per tutta la configurazione.
Uso: from config.settings import settings

NOTA su env_file e subclass: pydantic-settings 2.x NON eredita
`env_file` dalle BaseSettings annidate via `default_factory`. Per
questo motivo ogni subclass deve dichiarare esplicitamente un suo
`SettingsConfigDict(env_file=PROJECT_ROOT/".env", ...)`. Senza questa
ripetizione, le variabili nel `.env` verrebbero ignorate per quella
subclass anche se i nomi/prefissi sono corretti.

Convenzioni per env_prefix:
- OllamaSettings        → OLLAMA_*
- STTSettings           → STT_*           (alias storici: VAD_THRESHOLD, VAD_SILENCE_DURATION)
- TTSSettings           → TTS_*
- SpeakerSettings       → SPEAKER_*
- MemorySettings        → MEMORY_*
- WebSearchSettings     → WEB_SEARCH_*    (alias storici: SEARXNG_URL, WEB_SCRAPE_TIMEOUT)
- PCControlSettings     → PC_CONTROL_*
- TerminalAgentSettings → TERMINAL_AGENT_*
- FileAnalysisSettings  → FILE_ANALYSIS_*
- PersonalitySettings   → PERSONALITY_*   (alias storico: DEFAULT_PERSONALITY)
- APISettings           → API_*

Quando un campo ha un alias esplicito, l'alias ha precedenza sul
prefisso: SEARXNG_URL viene letto al posto di WEB_SEARCH_SEARXNG_URL.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent

import os
os.environ.setdefault("HF_HOME",    str(PROJECT_ROOT / "data" / "hf-cache"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "data" / "torch-cache"))

class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    base_url: str = "http://localhost:11434"
    # Default = tag Ollama realmente installati. Override via .env
    # (OLLAMA_CHAT_MODEL, OLLAMA_CODE_MODEL, ...). Devono esistere su
    # `ollama list`, altrimenti ogni turno LLM fallisce.
    chat_model: str = "qwen3:14b-q8_0"
    code_model: str = "qwen3:30b-a3b-q4_K_M"
    vision_model: str = "qwen3-vl:8b"
    embed_model: str = "nomic-embed-text"
    timeout: int = 120

class STTSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="STT_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    model: str = "large-v3-turbo"
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    compute_type: Literal["int8", "float16", "float32"] = "int8"
    language: str = "it"
    vad_threshold: float = Field(0.5, alias="VAD_THRESHOLD")
    vad_silence_duration: float = Field(0.8, alias="VAD_SILENCE_DURATION")

class TTSSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TTS_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    engine: Literal["kokoro", "qwen3"] = "qwen3"
    voice: str = "mercoledì"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    sample_rate: int = 24000  # Qwen3-TTS output
    cuda_device_index: int = Field(1, alias="TTS_CUDA_DEVICE")
    model_size: Literal["0.6B", "1.7B"] = Field("1.7B", alias="TTS_MODEL_SIZE")
    model_type: Literal["Base", "CustomVoice", "VoiceDesign"] = Field("Base", alias="TTS_MODEL_TYPE")

class SpeakerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SPEAKER_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    id_threshold: float = Field(0.75, alias="SPEAKER_ID_THRESHOLD")
    enroll_seconds: int = Field(30, alias="SPEAKER_ENROLL_SECONDS")

class MemorySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEMORY_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    chroma_persist_dir: Path = PROJECT_ROOT / "data" / "embeddings"
    # NB: l'embedding è generato da Ollama (settings.ollama.embed_model,
    # nomic-embed-text). Non esiste un embedding sentence-transformers locale.
    rag_top_k: int = 5
    context_window_messages: int = 20

class WebSearchSettings(BaseSettings):
    # NOTA: env_prefix è "WEB_SEARCH_", ma alcuni campi mantengono alias
    # storici (SEARXNG_URL, WEB_SCRAPE_TIMEOUT) per non rompere i .env
    # esistenti. Quando un campo ha alias, pydantic-settings ignora il
    # prefix per quel campo (l'alias ha precedenza assoluta).
    model_config = SettingsConfigDict(
        env_prefix="WEB_SEARCH_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    # alias mantenuto: SEARXNG_URL (no prefisso)
    searxng_url: str = Field("http://localhost:8080", alias="SEARXNG_URL")
    # senza alias: viene letta come WEB_SEARCH_MAX_RESULTS (era già così)
    max_results: int = 5
    # alias mantenuto: WEB_SCRAPE_TIMEOUT (storico, senza prefisso WEB_SEARCH_)
    scrape_timeout: int = Field(10, alias="WEB_SCRAPE_TIMEOUT")

class PCControlSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PC_CONTROL_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    allow_delete: bool = False
    allow_sudo: bool = False
    safe_dirs: list[str] = ["/home", "/tmp", "/media"]

    @field_validator("safe_dirs", mode="before")
    @classmethod
    def parse_dirs(cls, v):
        if isinstance(v, str):
            return [d.strip() for d in v.split(",")]
        return v

class TerminalAgentSettings(BaseSettings):
    """Configurazione del terminale agentico (modules/terminal_agent)."""
    # NOTA: env_file esplicito perché pydantic-settings 2.x non eredita
    # questa config dalle BaseSettings annidate via default_factory.
    # Senza, TERMINAL_AGENT_* nel .env verrebbero ignorate.
    model_config = SettingsConfigDict(
        env_prefix="TERMINAL_AGENT_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Modello LLM: "chat" -> settings.ollama.chat_model,
    #              "code" -> settings.ollama.code_model
    model_role: Literal["chat", "code"] = "chat"
    use_thinking_propose: bool = True
    use_thinking_analyze: bool = False

    # Family Ollama considerate "thinking" — solo i modelli di queste family
    # appaiono nel selettore della modalità terminale (ulteriormente filtrati
    # dalla blacklist utente in ui-settings.json). Override via env:
    #     TERMINAL_AGENT_THINKING_FAMILIES=qwen35,qwen3,deepseek
    thinking_families: list[str] = ["qwen35", "qwen35moe", "qwen3", "qwen3moe"]

    @field_validator("thinking_families", mode="before")
    @classmethod
    def parse_families(cls, v):
        if isinstance(v, str):
            return [f.strip() for f in v.split(",") if f.strip()]
        return v

    # Esecuzione comando
    command_timeout: int = Field(30, ge=1, le=600)
    max_output_bytes: int = Field(64 * 1024, ge=1024)

    # Loop agentico (ReAct)
    max_iterations: int = Field(6, ge=1, le=10)
    enable_web_search: bool = True
    max_search_results: int = Field(5, ge=1, le=20)

    # Sicurezza: None = eredita da pc_control
    allow_sudo: bool | None = None
    allow_delete: bool | None = None
    safe_dirs: list[str] | None = None
    initial_cwd: str = ""

    @field_validator("safe_dirs", mode="before")
    @classmethod
    def parse_dirs(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return [d.strip() for d in v.split(",") if d.strip()]
        return v

class FileAnalysisSettings(BaseSettings):
    """
    Configurazione del modulo file_analysis (analisi di file locali iniettati
    nel system prompt dell'LLM).

    NOTA: env_file esplicito perché pydantic-settings 2.x non eredita
    questa config dalle BaseSettings annidate via default_factory. Stesso
    pattern di TerminalAgentSettings.
    """
    model_config = SettingsConfigDict(
        env_prefix="FILE_ANALYSIS_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Soglia hard-cut per il testo iniettato nel system prompt (per singolo
    # file). Caratteri, non token: 8000 char ≈ 2000 token per la maggior
    # parte dei modelli, abbondante anche con history e RAG accanto.
    max_chars_per_file: int = Field(8_000, ge=500, le=200_000)

    # Numero massimo di file analizzati nello stesso turno. Oltre, vengono
    # menzionati ma non analizzati. Cap necessario per non gonfiare il
    # contesto se l'utente incolla 10 path.
    max_files_per_turn: int = Field(3, ge=1, le=10)

    # Tetto globale sui caratteri iniettati dai file_analysis nel turno.
    # Quando N file insieme superano questa soglia, si divide il budget in
    # parti uguali (fair share) e si tronca ogni file di conseguenza.
    max_total_chars: int = Field(12_000, ge=1_000, le=500_000)

    # Limite di sicurezza sulla dimensione del file su disco. Evita di
    # tentare l'estrazione di file enormi (PDF da centinaia di MB, etc.).
    max_file_bytes: int = Field(50 * 1024 * 1024, ge=1024)

    # Sicurezza: None = eredita da pc_control.safe_dirs (path consentiti).
    # Override esplicito via env: FILE_ANALYSIS_SAFE_DIRS="/home,/tmp"
    safe_dirs: list[str] | None = None

    @field_validator("safe_dirs", mode="before")
    @classmethod
    def parse_dirs(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return [d.strip() for d in v.split(",") if d.strip()]
        return v


class PersonalitySettings(BaseSettings):
    # env_prefix "PERSONALITY_", ma il campo default_personality mantiene
    # alias storico "DEFAULT_PERSONALITY" (senza prefisso) per backcompat.
    model_config = SettingsConfigDict(
        env_prefix="PERSONALITY_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    default_personality: str = Field("default", alias="DEFAULT_PERSONALITY")
    config_dir: Path = PROJECT_ROOT / "config" / "personalities"

class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    data_dir: Path = PROJECT_ROOT / "data"
    log_dir: Path = PROJECT_ROOT / "logs"
    hf_token: str = Field("", alias="HF_TOKEN")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    speaker: SpeakerSettings = Field(default_factory=SpeakerSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    pc_control: PCControlSettings = Field(default_factory=PCControlSettings)
    terminal_agent: TerminalAgentSettings = Field(default_factory=TerminalAgentSettings)
    file_analysis: FileAnalysisSettings = Field(default_factory=FileAnalysisSettings)
    personality: PersonalitySettings = Field(default_factory=PersonalitySettings)
    api: APISettings = Field(default_factory=APISettings)

    def ensure_dirs(self) -> None:
        for d in [
            self.data_dir, self.log_dir,
            self.data_dir / "embeddings", self.data_dir / "speaker_profiles",
            self.data_dir / "Conversations", self.data_dir / "models" / "lora",
            self.personality.config_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

settings = Settings()
