"""
tests/test_pc_control.py
HyprlandPCControl (subprocess mockato: niente hyprctl/grim reali) e il
percorso visivo dell'orchestratore (_run_screen_look).
"""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import core.orchestrator as O
from core.context import AssistantContext, InputMode, ModelRole, OutputMode
from core.orchestrator import Orchestrator, _format_screen_block, _should_look_at_screen
from modules.pc_control import BasePCControl, HyprlandPCControl


def _cp(returncode=0, stdout=b"", stderr=b"") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


def _pc(run_result=None) -> HyprlandPCControl:
    pc = HyprlandPCControl()
    pc._run = MagicMock(return_value=run_result or _cp())  # type: ignore[method-assign]
    return pc


# ---------------------------------------------------------------------------
# HyprlandPCControl
# ---------------------------------------------------------------------------

class TestOpenApplication:
    def test_lancia_app_valida(self):
        pc = _pc(_cp(stdout=b"ok"))
        assert pc.open_application("firefox") is True
        pc._run.assert_called_once_with(["hyprctl", "dispatch", "exec", "firefox"])

    def test_rifiuta_shell_injection(self):
        pc = _pc()
        assert pc.open_application("firefox; rm -rf ~") is False
        assert pc.open_application("app && evil") is False
        assert pc.open_application("app arg") is False       # niente argomenti
        assert pc.open_application("") is False
        pc._run.assert_not_called()

    def test_nomi_con_punto_e_trattino_ok(self):
        pc = _pc(_cp(stdout=b"ok"))
        assert pc.open_application("org.gnome.Calculator") is True
        pc = _pc(_cp(stdout=b"ok"))
        assert pc.open_application("visual-studio-code") is True

    def test_fallimento_hyprctl(self):
        pc = _pc(_cp(returncode=1, stdout=b""))
        assert pc.open_application("firefox") is False


class TestOpenFile:
    def test_apre_file_in_safe_dir(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_text("x")
        pc = HyprlandPCControl()
        with patch("modules.pc_control.base_pc_control.subprocess.Popen") as popen:
            assert pc.open_file(f) is True
            assert popen.call_args.args[0][0] == "xdg-open"

    def test_rifiuta_path_fuori_safe_dirs(self):
        pc = HyprlandPCControl()
        with patch("modules.pc_control.base_pc_control.subprocess.Popen") as popen:
            assert pc.open_file(Path("/etc/passwd")) is False
            popen.assert_not_called()

    def test_rifiuta_file_inesistente(self, tmp_path):
        pc = HyprlandPCControl()
        with patch("modules.pc_control.base_pc_control.subprocess.Popen") as popen:
            assert pc.open_file(tmp_path / "manca.txt") is False
            popen.assert_not_called()


class TestScreenshot:
    def test_ritorna_png_bytes(self):
        pc = _pc(_cp(stdout=b"\x89PNG..."))
        assert pc.take_screenshot() == b"\x89PNG..."
        pc._run.assert_called_once_with(["grim", "-"])

    def test_fallimento_solleva(self):
        pc = _pc(_cp(returncode=1, stderr=b"no output"))
        with pytest.raises(RuntimeError, match="grim"):
            pc.take_screenshot()


class TestWindowsAndClipboard:
    def test_list_open_windows(self):
        clients = [{"title": "Editor", "class": "code"},
                   {"title": "", "class": "kitty"},
                   {"title": "", "class": ""}]
        pc = _pc(_cp(stdout=json.dumps(clients).encode()))
        assert pc.list_open_windows() == ["Editor [code]", " [kitty]"]

    def test_list_windows_json_rotto(self):
        pc = _pc(_cp(stdout=b"non-json"))
        assert pc.list_open_windows() == []

    def test_read_clipboard(self):
        pc = _pc(_cp(stdout="testo copiato".encode()))
        assert pc.read_clipboard() == "testo copiato"

    def test_read_clipboard_vuota(self):
        pc = _pc(_cp(returncode=1))          # wl-paste esce 1 se vuota
        assert pc.read_clipboard() == ""

    def test_write_clipboard(self):
        pc = _pc(_cp())
        pc.write_clipboard("ciao")
        assert pc._run.call_args.kwargs["input_bytes"] == b"ciao"

    def test_write_clipboard_fallita_solleva(self):
        pc = _pc(_cp(returncode=1))
        with pytest.raises(RuntimeError, match="wl-copy"):
            pc.write_clipboard("x")


class TestAvailable:
    def test_tutti_i_binari_presenti(self, monkeypatch):
        monkeypatch.setattr("modules.pc_control.base_pc_control.shutil.which",
                            lambda b: f"/usr/bin/{b}")
        assert HyprlandPCControl.available() is True

    def test_binario_mancante(self, monkeypatch):
        monkeypatch.setattr("modules.pc_control.base_pc_control.shutil.which",
                            lambda b: None if b == "grim" else f"/usr/bin/{b}")
        assert HyprlandPCControl.available() is False

    def test_implementa_interfaccia(self):
        assert issubclass(HyprlandPCControl, BasePCControl)


# ---------------------------------------------------------------------------
# Percorso visivo dell'orchestratore
# ---------------------------------------------------------------------------

def make_ctx(**kwargs) -> AssistantContext:
    defaults = dict(
        user_text="guarda lo schermo e dimmi che errore c'è",
        session_id="s1",
        input_mode=InputMode.TEXT,
        output_mode=OutputMode.TEXT,
        model_role=ModelRole.CHAT,
        system_prompt="Sei un assistente.",
        personality_name="dev",
    )
    defaults.update(kwargs)
    return AssistantContext(**defaults)


def _orch_screen(allows=True, description="Vedo un traceback Python."):
    orch = Orchestrator.__new__(Orchestrator)
    orch._pc_control = MagicMock()
    orch._pc_control.take_screenshot = MagicMock(return_value=b"\x89PNGxxx")
    orch._llm = MagicMock()
    orch._llm.vision = AsyncMock(return_value=SimpleNamespace(content=description))
    pers = MagicMock()
    pers.active.allows_tool = MagicMock(return_value=allows)
    orch._personality = pers
    return orch


class TestShouldLookAtScreen:
    def test_trigger(self):
        assert _should_look_at_screen("Guarda lo schermo e aiutami")
        assert _should_look_at_screen("cosa vedi adesso?")
        assert _should_look_at_screen("leggi lo schermo per favore")

    def test_no_trigger(self):
        assert not _should_look_at_screen("Spiegami i decoratori Python")
        assert not _should_look_at_screen("")


class TestRunScreenLook:
    async def test_inietta_analisi_e_traccia_tool(self):
        orch = _orch_screen()
        ctx = make_ctx()
        await orch._run_screen_look(ctx)
        assert "traceback Python" in ctx.system_prompt
        assert ctx.tool_calls and ctx.tool_calls[0]["tool"] == "pc_control"
        # la domanda dell'utente arriva nel prompt del modello vision
        assert "che errore" in orch._llm.vision.call_args.kwargs["prompt"]

    async def test_skip_senza_trigger(self):
        orch = _orch_screen()
        ctx = make_ctx(user_text="Che ore sono?")
        await orch._run_screen_look(ctx)
        orch._pc_control.take_screenshot.assert_not_called()

    async def test_skip_tool_non_consentito(self):
        orch = _orch_screen(allows=False)
        ctx = make_ctx()
        await orch._run_screen_look(ctx)
        orch._pc_control.take_screenshot.assert_not_called()

    async def test_skip_senza_pc_control(self):
        orch = _orch_screen()
        orch._pc_control = None
        ctx = make_ctx()
        before = ctx.system_prompt
        await orch._run_screen_look(ctx)
        assert ctx.system_prompt == before

    async def test_screenshot_fallito_non_solleva(self):
        orch = _orch_screen()
        orch._pc_control.take_screenshot = MagicMock(side_effect=RuntimeError("grim"))
        ctx = make_ctx()
        before = ctx.system_prompt
        await orch._run_screen_look(ctx)
        assert ctx.system_prompt == before

    async def test_vision_fallita_non_solleva(self):
        orch = _orch_screen()
        orch._llm.vision = AsyncMock(side_effect=RuntimeError("ollama"))
        ctx = make_ctx()
        before = ctx.system_prompt
        await orch._run_screen_look(ctx)
        assert ctx.system_prompt == before


class TestFormatScreenBlock:
    def test_vuoto(self):
        assert _format_screen_block("") == ""
        assert _format_screen_block("   ") == ""

    def test_contiene_descrizione(self):
        block = _format_screen_block("Un terminale con un errore.")
        assert "ANALISI DELLO SCHERMO" in block
        assert "Un terminale con un errore." in block
