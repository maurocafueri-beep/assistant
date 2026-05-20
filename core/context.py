"""
core/context.py
Oggetto contesto condiviso tra tutti i moduli per ogni turno di conversazione.
"""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class InputMode(str, Enum):
    VOICE = "voice"
    TEXT = "text"

class OutputMode(str, Enum):
    VOICE = "voice"
    TEXT = "text"
    BOTH = "both"

class ModelRole(str, Enum):
    CHAT = "chat"
    CODE = "code"
    VISION = "vision"

@dataclass
class SpeakerInfo:
    id: str
    display_name: str
    confidence: float
    is_known: bool = True

@dataclass
class MemoryChunk:
    content: str
    source: str
    relevance_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class AssistantContext:
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = ""
    input_mode: InputMode = InputMode.TEXT
    raw_audio: bytes | None = None
    user_text: str = ""
    speaker: SpeakerInfo | None = None
    personality_name: str = "default"
    system_prompt: str = ""
    model_role: ModelRole = ModelRole.CHAT
    model_name: str = ""
    retrieved_memories: list[MemoryChunk] = field(default_factory=list)
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    output_mode: OutputMode = OutputMode.BOTH
    assistant_text: str = ""
    audio_chunks: list[bytes] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def set_timing(self, module: str, elapsed_ms: float) -> None:
        self.timings[module] = round(elapsed_ms, 1)

    def add_tool_call(self, tool: str, args: dict, result: Any) -> None:
        self.tool_calls.append({"tool": tool, "args": args})
        self.tool_results.append({"tool": tool, "result": result})

    def has_error(self) -> bool:
        return self.error is not None

    def speaker_label(self) -> str:
        if self.speaker and self.speaker.is_known:
            return self.speaker.display_name
        return "Utente"

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "speaker": self.speaker_label(),
            "personality": self.personality_name,
            "model": self.model_name,
            "input_len": len(self.user_text),
            "output_len": len(self.assistant_text),
            "tools_used": [t["tool"] for t in self.tool_calls],
            "timings_ms": self.timings,
            "error": self.error,
        }
