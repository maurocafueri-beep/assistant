"""
tests/test_terminal_bridge.py
Test suite per ui/terminal_bridge.py.

Mockiamo:
- TerminalAgent (per non chiamare LLM né subprocess)
- WSManager (per catturare i broadcast)
- httpx.AsyncClient (per simulare /api/tags di Ollama)

asyncio_mode = "auto" → niente @pytest.mark.asyncio.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.terminal_agent import (
    AgentTurn,
    CommandProposal,
    CommandResult,
    RiskLevel,
)
from ui.terminal_bridge import (
    TerminalBridge,
    _proposal_to_dict,
    _result_to_dict,
    _turn_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_proposal(
    cmd: str = "ls -la",
    risk: RiskLevel = RiskLevel.SAFE,
    needs_conf: bool = False,
    proposal_id: str = "abc12345",
    cwd: str = "/tmp/test",
) -> CommandProposal:
    return CommandProposal(
        command=cmd, rationale="test", risk_level=risk,
        needs_confirmation=needs_conf, cwd=cwd,
        proposal_id=proposal_id,
    )


def _make_result(
    cmd: str = "ls -la",
    exit_code: int = 0,
    stdout: str = "file1\nfile2",
    stderr: str = "",
    proposal_id: str = "abc12345",
    cwd: str = "/tmp/test",
) -> CommandResult:
    return CommandResult(
        command=cmd, exit_code=exit_code, stdout=stdout, stderr=stderr,
        duration_ms=10.0, cwd_before=cwd, cwd_after=cwd,
        proposal_id=proposal_id,
    )


@pytest.fixture
def mock_ws():
    """WSManager mockato — cattura tutti i broadcast in .broadcast_calls."""
    ws = MagicMock()
    ws.broadcast = AsyncMock()
    return ws


@pytest.fixture
def mock_agent():
    """TerminalAgent mockato. Stubbe tutti i metodi async."""
    ag = MagicMock()
    ag.cwd = "/tmp/test"
    ag.load    = AsyncMock()
    ag.aclose  = AsyncMock()
    ag.propose = AsyncMock()
    ag.execute = AsyncMock()
    ag.analyze = AsyncMock(return_value="analisi mock")
    ag.reset   = MagicMock()
    return ag


@pytest.fixture
async def bridge(mock_ws, mock_agent):
    """TerminalBridge pre-caricato col TerminalAgent mockato."""
    b = TerminalBridge(ws_manager=mock_ws)
    # Inietto direttamente il mock invece di passare dal load() reale
    b._agent = mock_agent
    b._loaded = True
    yield b
    # cleanup: niente da fare


# ---------------------------------------------------------------------------
# Serializzatori
# ---------------------------------------------------------------------------

class TestSerializers:
    def test_proposal_to_dict_keys(self):
        p = _make_proposal()
        d = _proposal_to_dict(p)
        assert set(d.keys()) == {
            "proposal_id", "command", "rationale", "risk_level",
            "needs_confirmation", "cwd", "search_used", "sources",
        }
        assert d["risk_level"] == "safe"

    def test_result_to_dict_keys(self):
        r = _make_result()
        d = _result_to_dict(r)
        assert "success" in d
        assert d["success"] is True
        assert "stdout" in d

    def test_turn_to_dict_handles_none_proposal(self):
        t = AgentTurn(user_request="x", proposal=None, result=None)
        d = _turn_to_dict(t)
        assert d["proposal"] is None
        assert d["result"] is None
        assert d["user_request"] == "x"


# ---------------------------------------------------------------------------
# Lifecycle / introspezione
# ---------------------------------------------------------------------------

class TestLifecycle:
    async def test_repr_not_loaded(self, mock_ws):
        b = TerminalBridge(ws_manager=mock_ws)
        assert "non caricato" in repr(b)

    async def test_repr_loaded(self, bridge):
        r = repr(bridge)
        assert "TerminalBridge" in r
        assert "/tmp/test" in r

    async def test_require_loaded_raises_on_propose(self, mock_ws):
        b = TerminalBridge(ws_manager=mock_ws)
        with pytest.raises(RuntimeError, match="non caricato"):
            await b.propose("ciao")

    async def test_aclose_idempotent(self, bridge, mock_agent):
        await bridge.aclose()
        await bridge.aclose()
        assert mock_agent.aclose.await_count == 1


# ---------------------------------------------------------------------------
# Filtraggio modelli (thinking families + hidden)
# ---------------------------------------------------------------------------

class TestModelFilter:
    """
    Per testare list_available_models() patchiamo httpx.AsyncClient
    nel modulo httpx (l'import in terminal_bridge è lazy dentro la funzione,
    quindi patch("ui.terminal_bridge.httpx") non funzionerebbe).
    """

    @staticmethod
    def _patch_ollama_tags(fake_response: dict):
        """
        Helper: ritorna un context manager che patcha httpx.AsyncClient
        per ritornare fake_response a /api/tags.
        """
        cli = MagicMock()
        cli.get = AsyncMock(return_value=MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value=fake_response),
        ))
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=cli)
        ctx.__aexit__  = AsyncMock(return_value=None)
        return patch("httpx.AsyncClient", return_value=ctx)

    async def test_list_available_models_filters_by_family(self, bridge):
        fake_tags = {
            "models": [
                {"name": "qwen3.5:9b", "details": {"family": "qwen35", "parameter_size": "9.7B"}},
                {"name": "gemma3:12b", "details": {"family": "gemma3", "parameter_size": "12B"}},
                {"name": "qwen3:14b",  "details": {"family": "qwen3",  "parameter_size": "14B"}},
                {"name": "llama3:8b",  "details": {"family": "llama3", "parameter_size": "8B"}},
            ]
        }
        with self._patch_ollama_tags(fake_tags):
            out = await bridge.list_available_models()

        names = [m["name"] for m in out]
        assert "qwen3.5:9b" in names
        assert "qwen3:14b"  in names
        assert "gemma3:12b" not in names
        assert "llama3:8b"  not in names

    async def test_hidden_models_marked(self, bridge):
        fake_tags = {
            "models": [
                {"name": "qwen3.5:9b",  "details": {"family": "qwen35", "parameter_size": "9.7B"}},
                {"name": "qwen3:14b",   "details": {"family": "qwen3",  "parameter_size": "14B"}},
            ]
        }
        bridge.set_hidden_models(["qwen3:14b"])
        with self._patch_ollama_tags(fake_tags):
            out = await bridge.list_available_models()

        hidden_flags = {m["name"]: m["hidden"] for m in out}
        assert hidden_flags["qwen3:14b"]  is True
        assert hidden_flags["qwen3.5:9b"] is False

    async def test_visible_models_excludes_hidden(self, bridge):
        fake_tags = {
            "models": [
                {"name": "qwen3.5:9b", "details": {"family": "qwen35", "parameter_size": "9.7B"}},
                {"name": "qwen3:14b",  "details": {"family": "qwen3",  "parameter_size": "14B"}},
            ]
        }
        bridge.set_hidden_models(["qwen3:14b"])
        with self._patch_ollama_tags(fake_tags):
            visible = await bridge.list_visible_models()

        assert visible == ["qwen3.5:9b"]

    async def test_list_available_models_handles_ollama_down(self, bridge):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(side_effect=Exception("connection refused"))
        ctx.__aexit__  = AsyncMock(return_value=None)
        with patch("httpx.AsyncClient", return_value=ctx):
            out = await bridge.list_available_models()
        assert out == []

    async def test_switch_model_rejects_unknown(self, bridge):
        with patch.object(bridge, "list_visible_models",
                          AsyncMock(return_value=["qwen3.5:9b"])):
            ok = await bridge.switch_model("unknown:tag")
        assert ok is False
        # broadcast non deve essere stato fatto
        bridge._ws.broadcast.assert_not_called()

    async def test_switch_model_accepts_visible(self, bridge):
        with patch.object(bridge, "list_visible_models",
                          AsyncMock(return_value=["qwen3.5:9b"])):
            ok = await bridge.switch_model("qwen3.5:9b")
        assert ok is True
        assert bridge.current_model == "qwen3.5:9b"
        # Broadcast del cambio modello
        bridge._ws.broadcast.assert_awaited_once()
        payload = bridge._ws.broadcast.await_args.args[0]
        assert payload["type"] == "terminal.model"
        assert payload["name"] == "qwen3.5:9b"


# ---------------------------------------------------------------------------
# propose() — safe vs needs_confirmation
# ---------------------------------------------------------------------------

class TestProposeFlow:
    async def test_safe_proposal_runs_automatically(self, bridge, mock_agent):
        """SAFE → execute + analyze automatici, ritorna turno completo."""
        proposal = _make_proposal(risk=RiskLevel.SAFE, needs_conf=False)
        result   = _make_result()
        mock_agent.propose.return_value = proposal
        mock_agent.execute.return_value = result
        mock_agent.analyze.return_value = "tutto ok"

        out = await bridge.propose("mostrami i file")

        # turn ritornato
        assert out["needs_confirmation"] is False
        assert out["turn"]["proposal"]["command"] == "ls -la"
        assert out["turn"]["result"]["exit_code"] == 0
        assert out["turn"]["analysis"] == "tutto ok"

        # ha eseguito + analizzato
        mock_agent.execute.assert_awaited_once()
        mock_agent.analyze.assert_awaited_once()

        # history aggiornata
        assert bridge.history_count == 1
        assert bridge.pending_count == 0

    async def test_moderate_proposal_goes_to_pending(self, bridge, mock_agent):
        """MODERATE → finisce in _pending, niente execute."""
        proposal = _make_proposal(risk=RiskLevel.MODERATE, needs_conf=True)
        mock_agent.propose.return_value = proposal

        out = await bridge.propose("crea una cartella")

        assert out["needs_confirmation"] is True
        assert out["turn"]["skipped"] is True
        assert out["turn"]["result"] is None

        # execute NON deve essere stato chiamato
        mock_agent.execute.assert_not_awaited()
        mock_agent.analyze.assert_not_awaited()

        # pending registrato
        assert bridge.pending_count == 1
        assert proposal.proposal_id in bridge._pending

    async def test_empty_request_returns_error(self, bridge):
        out = await bridge.propose("   ")
        assert "error" in out

    async def test_propose_broadcasts_proposal(self, bridge, mock_agent, mock_ws):
        mock_agent.propose.return_value = _make_proposal(risk=RiskLevel.MODERATE, needs_conf=True)

        await bridge.propose("test")

        # almeno un broadcast con type=terminal.proposal
        types = [c.args[0]["type"] for c in mock_ws.broadcast.await_args_list]
        assert "terminal.proposal" in types

    async def test_propose_agent_failure_returns_error(self, bridge, mock_agent):
        mock_agent.propose.side_effect = RuntimeError("ollama down")
        out = await bridge.propose("test")
        assert "error" in out
        assert "ollama down" in out["error"]


# ---------------------------------------------------------------------------
# confirm() / cancel()
# ---------------------------------------------------------------------------

class TestConfirmCancel:
    async def test_confirm_executes_pending(self, bridge, mock_agent):
        proposal = _make_proposal(risk=RiskLevel.MODERATE, needs_conf=True, proposal_id="ABC")
        bridge._pending["ABC"] = proposal
        mock_agent.execute.return_value = _make_result(exit_code=0)
        mock_agent.analyze.return_value = "ok"

        out = await bridge.confirm("ABC")

        assert out["turn"]["result"]["exit_code"] == 0
        assert out["turn"]["analysis"] == "ok"
        mock_agent.execute.assert_awaited_once()
        # pending svuotato
        assert "ABC" not in bridge._pending

    async def test_confirm_unknown_proposal_returns_error(self, bridge):
        out = await bridge.confirm("nonexistent")
        assert "error" in out

    async def test_cancel_removes_pending(self, bridge, mock_ws):
        proposal = _make_proposal(proposal_id="ABC")
        bridge._pending["ABC"] = proposal

        out = await bridge.cancel("ABC")
        assert out["ok"] is True
        assert "ABC" not in bridge._pending

        # broadcast con type=terminal.cancelled
        types = [c.args[0]["type"] for c in mock_ws.broadcast.await_args_list]
        assert "terminal.cancelled" in types

    async def test_cancel_unknown_returns_ok_false(self, bridge):
        out = await bridge.cancel("nonexistent")
        assert out["ok"] is False


# ---------------------------------------------------------------------------
# Esecuzione: errori, cwd change, broadcast
# ---------------------------------------------------------------------------

class TestExecuteAndAnalyze:
    async def test_execute_failure_logs_and_returns_turn_with_error(self, bridge, mock_agent):
        mock_agent.propose.return_value = _make_proposal(risk=RiskLevel.SAFE, needs_conf=False)
        mock_agent.execute.side_effect = ValueError("comando bloccato")

        out = await bridge.propose("test")

        # Il turno è in history, con error settato
        assert bridge.history_count == 1
        turn = bridge._history[-1]
        assert turn.error is not None
        assert "comando bloccato" in turn.error
        # E un broadcast di errore è partito
        types = [c.args[0]["type"] for c in bridge._ws.broadcast.await_args_list]
        assert "terminal.error" in types

    async def test_cwd_change_broadcast(self, bridge, mock_agent, mock_ws):
        proposal = _make_proposal(risk=RiskLevel.SAFE, needs_conf=False, cwd="/tmp/test")
        mock_agent.propose.return_value = proposal
        mock_agent.execute.return_value = _make_result()

        # simulo cwd cambiata dopo execute
        def _side(*a, **kw):
            mock_agent.cwd = "/tmp/test/sub"
            return _make_result(cwd="/tmp/test/sub")
        mock_agent.execute.side_effect = _side

        await bridge.propose("cd sub")

        types = [c.args[0]["type"] for c in mock_ws.broadcast.await_args_list]
        assert "terminal.cwd" in types
        # E il payload deve avere la nuova cwd
        cwd_payload = next(c.args[0] for c in mock_ws.broadcast.await_args_list
                           if c.args[0]["type"] == "terminal.cwd")
        assert cwd_payload["cwd"] == "/tmp/test/sub"


# ---------------------------------------------------------------------------
# Voice input deviato
# ---------------------------------------------------------------------------

class TestVoiceInput:
    async def test_handle_voice_input_broadcasts_transcription(self, bridge, mock_ws):
        await bridge.handle_voice_input("trovami i file grossi")

        # deve aver broadcastato terminal.transcription
        assert mock_ws.broadcast.await_count == 1
        payload = mock_ws.broadcast.await_args.args[0]
        assert payload["type"] == "terminal.transcription"
        assert payload["text"] == "trovami i file grossi"

    async def test_handle_voice_input_ignores_empty(self, bridge, mock_ws):
        await bridge.handle_voice_input("   ")
        mock_ws.broadcast.assert_not_called()

    async def test_handle_voice_input_does_not_call_agent(self, bridge, mock_agent):
        await bridge.handle_voice_input("trovami i file grossi")
        # NON deve aver chiamato propose direttamente — solo broadcast
        mock_agent.propose.assert_not_awaited()


# ---------------------------------------------------------------------------
# reset() e state
# ---------------------------------------------------------------------------

class TestReset:
    async def test_reset_clears_history_and_pending(self, bridge, mock_agent, mock_ws):
        bridge._pending["X"] = _make_proposal(proposal_id="X")
        bridge._history.append(AgentTurn(user_request="prev"))

        await bridge.reset()

        assert bridge.history_count == 0
        assert bridge.pending_count == 0
        mock_agent.reset.assert_called_once()
        types = [c.args[0]["type"] for c in mock_ws.broadcast.await_args_list]
        assert "terminal.reset" in types


class TestStatePayload:
    async def test_payload_has_required_keys(self, bridge):
        d = bridge.state_payload()
        assert set(d.keys()) == {
            "cwd", "current_model", "history_count", "pending_count",
            "pending", "history",
        }

    async def test_payload_truncates_history_to_20(self, bridge):
        for i in range(30):
            bridge._history.append(AgentTurn(user_request=f"req {i}"))
        d = bridge.state_payload()
        assert len(d["history"]) == 20
        assert d["history_count"] == 30


# ---------------------------------------------------------------------------
# Settings — thinking_families
# ---------------------------------------------------------------------------

class TestSettingsThinkingFamilies:
    def test_default_thinking_families_present(self):
        from config.settings import settings
        cfg = settings.terminal_agent
        assert hasattr(cfg, "thinking_families")
        assert isinstance(cfg.thinking_families, list)
        assert "qwen35" in cfg.thinking_families
        assert "qwen3"  in cfg.thinking_families

    def test_env_override_thinking_families_json(self, monkeypatch):
        """
        pydantic-settings parsa l'env come JSON quando il tipo è list[str].
        Quindi per override via env serve un JSON array.
        Il @field_validator del modulo accetta anche stringa CSV ma viene
        chiamato DOPO il parsing JSON, quindi serve per override programmatico
        (Settings(thinking_families="a,b,c")), non per env var.
        """
        monkeypatch.setenv("TERMINAL_AGENT_THINKING_FAMILIES",
                           '["qwen35", "deepseek", "llama"]')
        from config.settings import TerminalAgentSettings
        cfg = TerminalAgentSettings()
        assert cfg.thinking_families == ["qwen35", "deepseek", "llama"]

    def test_field_validator_accepts_csv_string(self):
        """
        Override programmatico con CSV: invocando direttamente la classe
        passando una stringa, il @field_validator converte CSV → list.
        """
        from config.settings import TerminalAgentSettings
        cfg = TerminalAgentSettings(thinking_families="qwen35, deepseek , llama")
        assert cfg.thinking_families == ["qwen35", "deepseek", "llama"]
