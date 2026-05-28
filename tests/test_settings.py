"""
tests/test_settings.py
Test suite per config/settings.py — verifica che ogni subclass annidata
legga effettivamente le variabili d'ambiente dal .env / monkeypatch.

Contesto: pydantic-settings 2.x non eredita `env_file` dalle BaseSettings
annidate via `default_factory`. Questi test sono la "rete di sicurezza"
che impedisce a un futuro contributore di aggiungere una nuova subclass
dimenticando di passare env_file esplicito (con conseguente bug silente).

Strategia per ogni subclass:
    1. test che istanzia la subclass direttamente, applica monkeypatch.setenv()
       a un campo, e verifica che il valore sia letto.
    2. test che ENV_FILE attribute sia popolato in model_config (metatest).

asyncio_mode = "auto" → non serve @pytest.mark.asyncio.

Esecuzione:
    venv-runtime/bin/pytest tests/test_settings.py -v
"""

from __future__ import annotations

import pytest

from config.settings import (
    APISettings,
    FileAnalysisSettings,
    MemorySettings,
    OllamaSettings,
    PCControlSettings,
    PersonalitySettings,
    Settings,
    SpeakerSettings,
    STTSettings,
    TerminalAgentSettings,
    TTSSettings,
    WebSearchSettings,
)

# Tutte le subclass attive (esclusa Settings root e ev. future)
_ALL_SUBCLASSES = [
    OllamaSettings,
    STTSettings,
    TTSSettings,
    SpeakerSettings,
    MemorySettings,
    WebSearchSettings,
    PCControlSettings,
    TerminalAgentSettings,
    FileAnalysisSettings,
    PersonalitySettings,
    APISettings,
]


# ===========================================================================
# Meta-test — env_file deve essere popolato su OGNI subclass
# ===========================================================================

class TestEnvFileMetaCheck:
    """
    Rete di sicurezza: garantisce che nessuna subclass perda env_file in
    futuro. Se questo test rompe, qualcuno ha aggiunto una BaseSettings
    senza il pattern stabilito.
    """

    @pytest.mark.parametrize("cls", _ALL_SUBCLASSES, ids=lambda c: c.__name__)
    def test_subclass_has_env_file_in_model_config(self, cls):
        cfg = getattr(cls, "model_config", None)
        assert cfg is not None, f"{cls.__name__}: model_config mancante"
        env_file = cfg.get("env_file")
        assert env_file is not None, (
            f"{cls.__name__}: env_file mancante in model_config. "
            "Aggiungi 'env_file=PROJECT_ROOT/\".env\"' al SettingsConfigDict."
        )

    @pytest.mark.parametrize("cls", _ALL_SUBCLASSES, ids=lambda c: c.__name__)
    def test_subclass_has_extra_ignore(self, cls):
        cfg = getattr(cls, "model_config", None)
        assert cfg is not None
        extra = cfg.get("extra")
        assert extra == "ignore", (
            f"{cls.__name__}: extra deve essere 'ignore' (è {extra!r})"
        )


# ===========================================================================
# OllamaSettings — env_prefix=OLLAMA_
# ===========================================================================

class TestOllamaSettings:
    def test_default_values(self):
        s = OllamaSettings()
        assert s.base_url.startswith("http")
        assert s.chat_model
        assert s.code_model
        assert s.embed_model

    def test_env_override_chat_model(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_CHAT_MODEL", "test:tag")
        s = OllamaSettings()
        assert s.chat_model == "test:tag"

    def test_env_override_base_url(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://altro:9999")
        s = OllamaSettings()
        assert s.base_url == "http://altro:9999"

    def test_num_ctx_default(self):
        s = OllamaSettings()
        # Default 8192 (raddoppio sicuro del default Ollama di 4096).
        assert s.num_ctx == 8192

    def test_num_ctx_env_override(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_NUM_CTX", "16384")
        s = OllamaSettings()
        assert s.num_ctx == 16384


# ===========================================================================
# STTSettings — env_prefix=STT_, alias VAD_THRESHOLD / VAD_SILENCE_DURATION
# ===========================================================================

class TestSTTSettings:
    def test_env_override_model(self, monkeypatch):
        monkeypatch.setenv("STT_MODEL", "tiny")
        s = STTSettings()
        assert s.model == "tiny"

    def test_env_override_language(self, monkeypatch):
        monkeypatch.setenv("STT_LANGUAGE", "en")
        s = STTSettings()
        assert s.language == "en"

    def test_vad_threshold_alias_no_prefix(self, monkeypatch):
        """L'alias storico VAD_THRESHOLD (senza prefisso STT_) deve funzionare."""
        monkeypatch.setenv("VAD_THRESHOLD", "0.75")
        s = STTSettings()
        assert s.vad_threshold == 0.75

    def test_vad_silence_duration_alias_no_prefix(self, monkeypatch):
        monkeypatch.setenv("VAD_SILENCE_DURATION", "1.5")
        s = STTSettings()
        assert s.vad_silence_duration == 1.5


# ===========================================================================
# TTSSettings — env_prefix=TTS_
# ===========================================================================

class TestTTSSettings:
    def test_env_override_voice(self, monkeypatch):
        monkeypatch.setenv("TTS_VOICE", "mauro")
        s = TTSSettings()
        assert s.voice == "mauro"

    def test_env_override_speed(self, monkeypatch):
        monkeypatch.setenv("TTS_SPEED", "1.5")
        s = TTSSettings()
        assert s.speed == 1.5

    def test_cuda_device_alias(self, monkeypatch):
        """L'alias TTS_CUDA_DEVICE è già allineato al prefisso e va letto."""
        monkeypatch.setenv("TTS_CUDA_DEVICE", "2")
        s = TTSSettings()
        assert s.cuda_device_index == 2


# ===========================================================================
# SpeakerSettings — env_prefix=SPEAKER_, alias SPEAKER_ID_THRESHOLD / SPEAKER_ENROLL_SECONDS
# ===========================================================================

class TestSpeakerSettings:
    def test_id_threshold_alias(self, monkeypatch):
        monkeypatch.setenv("SPEAKER_ID_THRESHOLD", "0.9")
        s = SpeakerSettings()
        assert s.id_threshold == 0.9

    def test_enroll_seconds_alias(self, monkeypatch):
        monkeypatch.setenv("SPEAKER_ENROLL_SECONDS", "60")
        s = SpeakerSettings()
        assert s.enroll_seconds == 60


# ===========================================================================
# MemorySettings — env_prefix=MEMORY_ (NUOVO)
# ===========================================================================

class TestMemorySettings:
    def test_default_values(self):
        s = MemorySettings()
        assert s.rag_top_k > 0
        assert s.context_window_messages > 0

    def test_env_override_rag_top_k(self, monkeypatch):
        monkeypatch.setenv("MEMORY_RAG_TOP_K", "10")
        s = MemorySettings()
        assert s.rag_top_k == 10

    def test_env_override_context_window(self, monkeypatch):
        monkeypatch.setenv("MEMORY_CONTEXT_WINDOW_MESSAGES", "50")
        s = MemorySettings()
        assert s.context_window_messages == 50


# ===========================================================================
# WebSearchSettings — env_prefix=WEB_SEARCH_, alias SEARXNG_URL / WEB_SCRAPE_TIMEOUT
# ===========================================================================

class TestWebSearchSettings:
    def test_searxng_url_alias_backcompat(self, monkeypatch):
        """L'alias storico SEARXNG_URL deve continuare a funzionare."""
        monkeypatch.setenv("SEARXNG_URL", "http://nuovo:9090")
        s = WebSearchSettings()
        assert s.searxng_url == "http://nuovo:9090"

    def test_max_results_via_prefix(self, monkeypatch):
        """max_results non ha alias → letta via env_prefix WEB_SEARCH_*."""
        monkeypatch.setenv("WEB_SEARCH_MAX_RESULTS", "20")
        s = WebSearchSettings()
        assert s.max_results == 20

    def test_scrape_timeout_alias_backcompat(self, monkeypatch):
        """L'alias storico WEB_SCRAPE_TIMEOUT (senza prefix) deve funzionare."""
        monkeypatch.setenv("WEB_SCRAPE_TIMEOUT", "30")
        s = WebSearchSettings()
        assert s.scrape_timeout == 30

    def test_alias_takes_precedence_over_prefix(self, monkeypatch):
        """
        Se entrambi sono settati (alias + nome prefisso), l'alias vince.
        Garanzia di pydantic-settings che il fix non rompe l'utente.
        """
        monkeypatch.setenv("SEARXNG_URL", "http://alias-wins")
        monkeypatch.setenv("WEB_SEARCH_SEARXNG_URL", "http://prefix-loses")
        s = WebSearchSettings()
        assert s.searxng_url == "http://alias-wins"


# ===========================================================================
# PCControlSettings — env_prefix=PC_CONTROL_
# ===========================================================================

class TestPCControlSettings:
    def test_env_override_allow_sudo(self, monkeypatch):
        monkeypatch.setenv("PC_CONTROL_ALLOW_SUDO", "true")
        s = PCControlSettings()
        assert s.allow_sudo is True

    def test_env_override_safe_dirs_csv(self, monkeypatch):
        """Il @field_validator accetta CSV per safe_dirs via env."""
        # pydantic-settings parsa list come JSON di default; il validator
        # supporta anche stringhe semplici. Passiamo programmaticamente
        # per esercitare il validator.
        s = PCControlSettings(safe_dirs="/x,/y,/z")
        assert s.safe_dirs == ["/x", "/y", "/z"]


# ===========================================================================
# FileAnalysisSettings — già coperta dal commit precedente; smoke check
# ===========================================================================

class TestFileAnalysisSettings:
    def test_env_override_max_chars(self, monkeypatch):
        monkeypatch.setenv("FILE_ANALYSIS_MAX_CHARS_PER_FILE", "5000")
        s = FileAnalysisSettings()
        assert s.max_chars_per_file == 5000


# ===========================================================================
# TerminalAgentSettings — già fixata; verifichiamo che il pattern regga
# ===========================================================================

class TestTerminalAgentSettings:
    def test_env_override_command_timeout(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_AGENT_COMMAND_TIMEOUT", "60")
        s = TerminalAgentSettings()
        assert s.command_timeout == 60


# ===========================================================================
# PersonalitySettings — env_prefix=PERSONALITY_ (NUOVO), alias DEFAULT_PERSONALITY
# ===========================================================================

class TestPersonalitySettings:
    def test_default_personality_alias(self, monkeypatch):
        """L'alias storico DEFAULT_PERSONALITY (senza prefix) deve funzionare."""
        monkeypatch.setenv("DEFAULT_PERSONALITY", "Gwen")
        s = PersonalitySettings()
        assert s.default_personality == "Gwen"

    def test_default_personality_default_value(self):
        s = PersonalitySettings()
        assert s.default_personality == "default"


# ===========================================================================
# APISettings — env_prefix=API_
# ===========================================================================

class TestAPISettings:
    def test_env_override_port(self, monkeypatch):
        monkeypatch.setenv("API_PORT", "9000")
        s = APISettings()
        assert s.port == 9000

    def test_env_override_host(self, monkeypatch):
        monkeypatch.setenv("API_HOST", "0.0.0.0")
        s = APISettings()
        assert s.host == "0.0.0.0"


# ===========================================================================
# Settings root — leggibile e contiene tutte le subclass
# ===========================================================================

class TestSettingsRoot:
    def test_has_all_subclasses(self):
        s = Settings()
        assert isinstance(s.ollama,         OllamaSettings)
        assert isinstance(s.stt,            STTSettings)
        assert isinstance(s.tts,            TTSSettings)
        assert isinstance(s.speaker,        SpeakerSettings)
        assert isinstance(s.memory,         MemorySettings)
        assert isinstance(s.web_search,     WebSearchSettings)
        assert isinstance(s.pc_control,     PCControlSettings)
        assert isinstance(s.terminal_agent, TerminalAgentSettings)
        assert isinstance(s.file_analysis,  FileAnalysisSettings)
        assert isinstance(s.personality,    PersonalitySettings)
        assert isinstance(s.api,            APISettings)

    def test_hf_token_and_log_level(self, monkeypatch):
        """Top-level settings (HF_TOKEN, LOG_LEVEL) restano alias diretti."""
        monkeypatch.setenv("HF_TOKEN",  "hf_abc")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.hf_token  == "hf_abc"
        assert s.log_level == "DEBUG"
