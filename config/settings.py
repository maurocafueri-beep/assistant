"""
config/settings.py
Unica fonte di verità per tutta la configurazione.
Uso: from config.settings import settings
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
    model_config = SettingsConfigDict(env_prefix="OLLAMA_")
    base_url: str = "http://localhost:11434"
    chat_model: str = "qwen3.5:9b-q8_0"
    code_model: str = "qwen3.6:35b-a3b"
    vision_model: str = "qwen3-vl:8b"
    embed_model: str = "nomic-embed-text"
    timeout: int = 120

class STTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STT_")
    model: str = "large-v3-turbo"
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    compute_type: Literal["int8", "float16", "float32"] = "int8"
    language: str = "it"
    vad_threshold: float = Field(0.5, alias="VAD_THRESHOLD")
    vad_silence_duration: float = Field(0.8, alias="VAD_SILENCE_DURATION")

class TTSSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TTS_")
    engine: Literal["kokoro", "qwen3"] = "qwen3"
    voice: str = "mercoledì"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    sample_rate: int = 24000  # Qwen3-TTS output
    cuda_device_index: int = Field(1, alias="TTS_CUDA_DEVICE")
    model_size: Literal["0.6B", "1.7B"] = Field("1.7B", alias="TTS_MODEL_SIZE")
    model_type: Literal["Base", "CustomVoice", "VoiceDesign"] = Field("Base", alias="TTS_MODEL_TYPE")

class SpeakerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SPEAKER_")
    id_threshold: float = Field(0.75, alias="SPEAKER_ID_THRESHOLD")
    enroll_seconds: int = Field(30, alias="SPEAKER_ENROLL_SECONDS")

class MemorySettings(BaseSettings):
    chroma_persist_dir: Path = PROJECT_ROOT / "data" / "embeddings"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    rag_top_k: int = 5
    context_window_messages: int = 20

class WebSearchSettings(BaseSettings):
    searxng_url: str = Field("http://localhost:8080", alias="SEARXNG_URL")
    max_results: int = Field(5, alias="WEB_SEARCH_MAX_RESULTS")
    scrape_timeout: int = Field(10, alias="WEB_SCRAPE_TIMEOUT")

class PCControlSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PC_CONTROL_")
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
    model_config = SettingsConfigDict(env_prefix="TERMINAL_AGENT_")

    # Modello LLM: "chat" -> qwen3.5:9b-q8_0, "code" -> qwen3.6:35b-a3b
    model_role: Literal["chat", "code"] = "chat"
    use_thinking: bool = True

    # Esecuzione comando
    command_timeout: int = Field(30, ge=1, le=600)
    max_output_bytes: int = Field(64 * 1024, ge=1024)

    # Loop agentico (ReAct)
    max_iterations: int = Field(4, ge=1, le=10)
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

class PersonalitySettings(BaseSettings):
    default_personality: str = Field("default", alias="DEFAULT_PERSONALITY")
    config_dir: Path = PROJECT_ROOT / "config" / "personalities"

class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="API_")
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
