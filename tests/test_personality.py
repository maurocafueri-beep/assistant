"""
tests/test_personality.py
Test suite per modules/personality.

I test veloci usano profili YAML temporanei creati in tmp_path — zero
dipendenze da filesystem del progetto o modelli AI.
I test @pytest.mark.slow testano il caricamento dei profili reali
da config/personalities/ ed il ciclo di vita completo.

Esecuzione:
    make test
    SKIP_SLOW=1 venv-runtime/bin/pytest tests/test_personality.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

import pytest

from modules.personality.base_personality import (
    PersonalityManager,
    PersonalityProfile,
    _parse_yaml,
    _load_all_from_dir,
)

SKIP_SLOW = os.getenv("SKIP_SLOW", "0") == "1"
slow = pytest.mark.skipif(SKIP_SLOW, reason="SKIP_SLOW=1")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

YAML_DEFAULT = dedent("""\
    name: default
    display_name: "Assistente"
    description: "Assistente generico"
    language: it
    tts_voice: af_sarah
    lora_adapter: null
    allowed_tools: []
    system_prompt: |
      Sei un assistente AI locale.
      Parli in italiano.
""")

YAML_DEV = dedent("""\
    name: dev
    display_name: "Dev"
    description: "Senior software engineer"
    language: it
    tts_voice: am_adam
    lora_adapter: null
    allowed_tools:
      - code_execution
      - file_read
      - web_search
    system_prompt: |
      Sei un senior software engineer. Sei diretto e tecnico.
""")

YAML_MULTILINGUAL = dedent("""\
    name: multi
    display_name: "Multi"
    description: "Profilo multilingua"
    language: auto
    tts_voice: af_sarah
    allowed_tools:
      - web_search
    system_prompt: You are a helpful multilingual assistant.
""")


@pytest.fixture
def profiles_dir(tmp_path: Path) -> Path:
    """Directory temporanea con tre profili YAML validi."""
    d = tmp_path / "personalities"
    d.mkdir()
    (d / "default.yaml").write_text(YAML_DEFAULT, encoding="utf-8")
    (d / "dev.yaml").write_text(YAML_DEV, encoding="utf-8")
    (d / "multi.yaml").write_text(YAML_MULTILINGUAL, encoding="utf-8")
    return d


@pytest.fixture
def single_profile_dir(tmp_path: Path) -> Path:
    """Directory con un solo profilo (default)."""
    d = tmp_path / "personalities"
    d.mkdir()
    (d / "default.yaml").write_text(YAML_DEFAULT, encoding="utf-8")
    return d


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    d = tmp_path / "empty"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# PersonalityProfile — unit test
# ---------------------------------------------------------------------------

class TestPersonalityProfile:
    def _make(self, **kwargs) -> PersonalityProfile:
        defaults = dict(
            name="test",
            display_name="Test",
            description="Desc",
            language="it",
            tts_voice="af_sarah",
            lora_adapter=None,
            allowed_tools=[],
            system_prompt="System prompt.",
        )
        defaults.update(kwargs)
        return PersonalityProfile(**defaults)

    def test_to_log_dict_keys(self):
        assert self._make().to_log_dict().keys() == {
            "name", "display_name", "language",
            "tts_voice", "lora_adapter", "allowed_tools", "prompt_len",
        }

    def test_to_log_dict_prompt_len(self):
        p = self._make(system_prompt="ciao mondo")
        assert p.to_log_dict()["prompt_len"] == len("ciao mondo")

    def test_is_multilingual_false(self):
        assert not self._make(language="it").is_multilingual

    def test_is_multilingual_true(self):
        assert self._make(language="auto").is_multilingual

    def test_is_multilingual_case_insensitive(self):
        assert self._make(language="AUTO").is_multilingual

    def test_allows_tool_empty_list_means_all(self):
        p = self._make(allowed_tools=[])
        assert p.allows_tool("any_tool") is True

    def test_allows_tool_in_list(self):
        p = self._make(allowed_tools=["web_search", "code_execution"])
        assert p.allows_tool("web_search") is True

    def test_allows_tool_not_in_list(self):
        p = self._make(allowed_tools=["web_search"])
        assert p.allows_tool("file_write") is False

    def test_str_repr(self):
        p = self._make(name="dev", display_name="Dev")
        assert "dev" in str(p)
        assert "Dev" in str(p)


# ---------------------------------------------------------------------------
# _parse_yaml
# ---------------------------------------------------------------------------

class TestParseYaml:
    def test_full_yaml(self, tmp_path):
        f = tmp_path / "default.yaml"
        f.write_text(YAML_DEFAULT, encoding="utf-8")
        p = _parse_yaml(f)
        assert p.name          == "default"
        assert p.display_name  == "Assistente"
        assert p.language      == "it"
        assert p.tts_voice     == "af_sarah"
        assert p.lora_adapter  is None
        assert p.allowed_tools == []
        assert "assistente ai" in p.system_prompt.lower()

    def test_name_fallback_to_stem(self, tmp_path):
        f = tmp_path / "my_profile.yaml"
        f.write_text("display_name: Test\nsystem_prompt: ok", encoding="utf-8")
        p = _parse_yaml(f)
        assert p.name == "my_profile"

    def test_allowed_tools_list(self, tmp_path):
        f = tmp_path / "dev.yaml"
        f.write_text(YAML_DEV, encoding="utf-8")
        p = _parse_yaml(f)
        assert "code_execution" in p.allowed_tools
        assert "web_search"     in p.allowed_tools

    def test_empty_yaml_uses_defaults(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("", encoding="utf-8")
        p = _parse_yaml(f)
        assert p.name     == "empty"
        assert p.language == "it"
        assert p.system_prompt == ""

    def test_missing_optional_fields(self, tmp_path):
        f = tmp_path / "minimal.yaml"
        f.write_text("name: minimal\nsystem_prompt: Ciao.", encoding="utf-8")
        p = _parse_yaml(f)
        assert p.name == "minimal"
        assert p.lora_adapter is None
        assert p.allowed_tools == []

    def test_language_normalised_to_lowercase(self, tmp_path):
        f = tmp_path / "p.yaml"
        f.write_text("name: p\nlanguage: IT\nsystem_prompt: x", encoding="utf-8")
        p = _parse_yaml(f)
        assert p.language == "it"


# ---------------------------------------------------------------------------
# _load_all_from_dir
# ---------------------------------------------------------------------------

class TestLoadAllFromDir:
    def test_loads_all_yaml(self, profiles_dir):
        profiles = _load_all_from_dir(profiles_dir)
        assert set(profiles.keys()) == {"default", "dev", "multi"}

    def test_returns_empty_for_nonexistent_dir(self, tmp_path):
        profiles = _load_all_from_dir(tmp_path / "nonexistent")
        assert profiles == {}

    def test_returns_empty_for_empty_dir(self, empty_dir):
        profiles = _load_all_from_dir(empty_dir)
        assert profiles == {}

    def test_skips_invalid_yaml_gracefully(self, tmp_path):
        d = tmp_path / "p"
        d.mkdir()
        (d / "good.yaml").write_text(YAML_DEFAULT, encoding="utf-8")
        (d / "bad.yaml").write_text("{ invalid: yaml: content:", encoding="utf-8")
        profiles = _load_all_from_dir(d)
        assert "default" in profiles          # il buono è caricato
        # "bad" potrebbe essere skippato o avere valori default — nessun crash

    def test_profile_values_correct(self, profiles_dir):
        profiles = _load_all_from_dir(profiles_dir)
        assert profiles["dev"].tts_voice == "am_adam"
        assert "web_search" in profiles["dev"].allowed_tools


# ---------------------------------------------------------------------------
# PersonalityManager — init e load
# ---------------------------------------------------------------------------

class TestPersonalityManagerLoad:
    async def test_context_manager_loads(self, profiles_dir):
        async with PersonalityManager(config_dir=profiles_dir) as pm:
            assert pm.list_profiles() == ["default", "dev", "multi"]

    async def test_standalone_load(self, profiles_dir):
        pm = PersonalityManager(config_dir=profiles_dir)
        await pm.load()
        assert pm.has_profile("default")

    async def test_not_loaded_raises(self):
        pm = PersonalityManager.__new__(PersonalityManager)
        pm._loaded = False
        pm._profiles = {}
        pm._active_name = "default"
        with pytest.raises(RuntimeError, match="non inizializzato"):
            _ = pm.active

    async def test_empty_dir_no_crash(self, empty_dir):
        async with PersonalityManager(config_dir=empty_dir) as pm:
            assert pm.list_profiles() == []

    async def test_reload_picks_up_new_file(self, profiles_dir):
        async with PersonalityManager(config_dir=profiles_dir) as pm:
            assert len(pm.list_profiles()) == 3
            # aggiungiamo un nuovo profilo a runtime
            (profiles_dir / "new.yaml").write_text(
                "name: new\ndisplay_name: New\nsystem_prompt: Hello.",
                encoding="utf-8",
            )
            await pm.reload()
            assert "new" in pm.list_profiles()

    async def test_default_fallback_when_missing(self, tmp_path):
        d = tmp_path / "p"
        d.mkdir()
        (d / "other.yaml").write_text(
            "name: other\ndisplay_name: Other\nsystem_prompt: Hi.",
            encoding="utf-8",
        )
        pm = PersonalityManager(config_dir=d, default_name="nonexistent")
        await pm.load()
        assert pm.active.name == "other"   # primo profilo disponibile


# ---------------------------------------------------------------------------
# PersonalityManager — active, get, switch
# ---------------------------------------------------------------------------

class TestPersonalityManagerActive:
    async def test_active_is_default(self, profiles_dir):
        async with PersonalityManager(
            config_dir=profiles_dir, default_name="default"
        ) as pm:
            assert pm.active.name == "default"

    async def test_get_existing(self, profiles_dir):
        async with PersonalityManager(config_dir=profiles_dir) as pm:
            dev = pm.get("dev")
            assert dev.name == "dev"

    async def test_get_missing_raises_key_error(self, profiles_dir):
        async with PersonalityManager(config_dir=profiles_dir) as pm:
            with pytest.raises(KeyError, match="nonexistent"):
                pm.get("nonexistent")

    async def test_switch_changes_active(self, profiles_dir):
        async with PersonalityManager(
            config_dir=profiles_dir, default_name="default"
        ) as pm:
            pm.switch("dev")
            assert pm.active.name == "dev"

    async def test_switch_returns_profile(self, profiles_dir):
        async with PersonalityManager(config_dir=profiles_dir) as pm:
            returned = pm.switch("dev")
            assert isinstance(returned, PersonalityProfile)
            assert returned.name == "dev"

    async def test_switch_invalid_raises(self, profiles_dir):
        async with PersonalityManager(config_dir=profiles_dir) as pm:
            with pytest.raises(KeyError):
                pm.switch("inesistente")

    async def test_switch_same_profile_no_error(self, profiles_dir):
        async with PersonalityManager(
            config_dir=profiles_dir, default_name="default"
        ) as pm:
            pm.switch("default")
            assert pm.active.name == "default"

    async def test_has_profile_true(self, profiles_dir):
        async with PersonalityManager(config_dir=profiles_dir) as pm:
            assert pm.has_profile("dev") is True

    async def test_has_profile_false(self, profiles_dir):
        async with PersonalityManager(config_dir=profiles_dir) as pm:
            assert pm.has_profile("xyz") is False

    async def test_list_profiles_sorted(self, profiles_dir):
        async with PersonalityManager(config_dir=profiles_dir) as pm:
            names = pm.list_profiles()
            assert names == sorted(names)

    async def test_repr_loaded(self, profiles_dir):
        async with PersonalityManager(config_dir=profiles_dir) as pm:
            r = repr(pm)
            assert "active=" in r

    async def test_repr_not_loaded(self):
        pm = PersonalityManager()
        assert "non caricato" in repr(pm)


# ---------------------------------------------------------------------------
# PersonalityManager — apply_to_context
# ---------------------------------------------------------------------------

class TestPersonalityManagerApplyToContext:
    async def test_sets_personality_name(self, profiles_dir):
        from core.context import AssistantContext
        ctx = AssistantContext()
        async with PersonalityManager(
            config_dir=profiles_dir, default_name="default"
        ) as pm:
            pm.apply_to_context(ctx)
        assert ctx.personality_name == "default"

    async def test_sets_system_prompt(self, profiles_dir):
        from core.context import AssistantContext
        ctx = AssistantContext()
        async with PersonalityManager(
            config_dir=profiles_dir, default_name="default"
        ) as pm:
            pm.apply_to_context(ctx)
        assert len(ctx.system_prompt) > 0

    async def test_does_not_overwrite_existing_prompt(self, profiles_dir):
        from core.context import AssistantContext
        ctx = AssistantContext()
        ctx.system_prompt = "Prompt personalizzato."
        async with PersonalityManager(config_dir=profiles_dir) as pm:
            pm.apply_to_context(ctx)
        # non deve sovrascrivere
        assert ctx.system_prompt == "Prompt personalizzato."

    async def test_apply_dev_profile(self, profiles_dir):
        from core.context import AssistantContext
        ctx = AssistantContext()
        async with PersonalityManager(config_dir=profiles_dir) as pm:
            pm.switch("dev")
            pm.apply_to_context(ctx)
        assert ctx.personality_name == "dev"
        assert "engineer" in ctx.system_prompt.lower()


# ---------------------------------------------------------------------------
# Test slow — usa i profili reali del progetto
# ---------------------------------------------------------------------------

@slow
class TestPersonalityManagerReal:
    async def test_loads_default_and_dev(self):
        async with PersonalityManager() as pm:
            assert pm.has_profile("default")
            assert pm.has_profile("dev")

    async def test_default_active(self):
        async with PersonalityManager() as pm:
            assert pm.active.name == "default"

    async def test_switch_to_dev(self):
        async with PersonalityManager() as pm:
            pm.switch("dev")
            assert pm.active.name == "dev"
            assert "code_execution" in pm.active.allowed_tools

    async def test_apply_to_real_context(self):
        from core.context import AssistantContext
        ctx = AssistantContext()
        async with PersonalityManager() as pm:
            pm.apply_to_context(ctx)
        assert ctx.personality_name in pm.list_profiles()
        assert ctx.system_prompt

    async def test_dev_system_prompt_not_empty(self):
        async with PersonalityManager() as pm:
            dev = pm.get("dev")
        assert len(dev.system_prompt) > 20
