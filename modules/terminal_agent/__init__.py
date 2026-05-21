"""
modules/terminal_agent
Terminale agentico: l'LLM propone comandi shell, può consultare il web,
li esegue (con conferma per i comandi non read-only), legge l'output
e aiuta nel debug.

Esporta:
    TerminalAgent        — la classe principale
    CommandProposal      — output di propose()
    CommandResult        — output di execute()
    AgentTurn            — turno completo (propose + execute + analyze)
    RiskLevel            — enum del livello di rischio
"""
from .base_terminal_agent import (
    AgentTurn,
    CommandProposal,
    CommandResult,
    RiskLevel,
    TerminalAgent,
)

__all__ = [
    "TerminalAgent",
    "CommandProposal",
    "CommandResult",
    "AgentTurn",
    "RiskLevel",
]
