"""modules/gaming/base.py — Interfaccia astratta Gaming Agent"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class GameState:
    screenshot: bytes
    description: str
    metadata: dict

class BaseGamingAgent(ABC):
    @abstractmethod
    def start_game(self, rom_path: str, core: str) -> None: ...
    @abstractmethod
    def get_state(self) -> GameState: ...
    @abstractmethod
    def send_action(self, action: str) -> None: ...
    @abstractmethod
    def stop_game(self) -> None: ...
