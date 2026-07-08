"""
modules/pc_control/base_pc_control.py
PC Control — interfaccia astratta + implementazione Hyprland/Wayland.

HyprlandPCControl usa solo binari di sistema, senza dipendenze Python extra:
    hyprctl   → lancio applicazioni, elenco finestre
    grim      → screenshot (PNG bytes, base del percorso visivo con qwen3-vl)
    wl-copy / wl-paste → clipboard
    xdg-open  → apertura file col programma predefinito
    pynput    → digitazione testo (via Xwayland, già dipendenza del PTT)

Sicurezza:
    - open_application accetta SOLO un nome di programma (token singolo,
      [A-Za-z0-9._-]): niente shell injection via `hyprctl dispatch exec`.
    - open_file valida il path contro settings.pc_control.safe_dirs.

Tutte le chiamate sono sincrone e veloci (subprocess locali); i chiamanti
async le eseguono con asyncio.to_thread.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from config.settings import settings
from core.logger import logger

_APP_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SUBPROCESS_TIMEOUT_S = 10.0


class BasePCControl(ABC):
    @abstractmethod
    def open_application(self, app_name: str) -> bool: ...
    @abstractmethod
    def open_file(self, path: Path) -> bool: ...
    @abstractmethod
    def type_text(self, text: str) -> None: ...
    @abstractmethod
    def take_screenshot(self) -> bytes: ...
    @abstractmethod
    def list_open_windows(self) -> list[str]: ...
    @abstractmethod
    def read_clipboard(self) -> str: ...
    @abstractmethod
    def write_clipboard(self, text: str) -> None: ...


class HyprlandPCControl(BasePCControl):
    """Controllo del desktop su Hyprland (Wayland)."""

    #: binari richiesti da available(); pynput è opzionale (solo type_text).
    REQUIRED_BINARIES = ("hyprctl", "grim", "wl-copy", "wl-paste", "xdg-open")

    @classmethod
    def available(cls) -> bool:
        """True se tutti i binari richiesti sono nel PATH."""
        return all(shutil.which(b) for b in cls.REQUIRED_BINARIES)

    @staticmethod
    def _run(cmd: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd, input=input_bytes, capture_output=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
        )

    # -- azioni ----------------------------------------------------------------

    def open_application(self, app_name: str) -> bool:
        """
        Lancia un'applicazione per nome. Solo token singoli ([A-Za-z0-9._-]):
        `hyprctl dispatch exec` passa da una shell, qualunque altro carattere
        è rifiutato per costruzione (niente argomenti, niente injection).
        """
        app_name = app_name.strip()
        if not _APP_NAME_RE.match(app_name):
            logger.warning("pc_control | nome applicazione rifiutato: {!r}", app_name)
            return False
        r = self._run(["hyprctl", "dispatch", "exec", app_name])
        ok = r.returncode == 0 and b"ok" in r.stdout.lower()
        logger.info("pc_control | open_application '{}' → {}", app_name, ok)
        return ok

    def open_file(self, path: Path) -> bool:
        """Apre un file col programma predefinito, se dentro le safe_dirs."""
        p = Path(path).expanduser().resolve()
        safe = [Path(d).resolve() for d in settings.pc_control.safe_dirs]
        if not any(p.is_relative_to(d) for d in safe):
            logger.warning("pc_control | path fuori dalle safe_dirs: {}", p)
            return False
        if not p.exists():
            logger.warning("pc_control | file inesistente: {}", p)
            return False
        # Detached: xdg-open può bloccarsi finché l'app non chiude.
        subprocess.Popen(
            ["xdg-open", str(p)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        logger.info("pc_control | open_file {}", p)
        return True

    def type_text(self, text: str) -> None:
        """Digita testo nella finestra attiva (pynput via Xwayland)."""
        from pynput.keyboard import Controller
        Controller().type(text)

    def take_screenshot(self) -> bytes:
        """Screenshot dell'intero output attivo, PNG bytes (via grim)."""
        r = self._run(["grim", "-"])
        if r.returncode != 0:
            raise RuntimeError(
                f"grim fallito (rc={r.returncode}): {r.stderr.decode(errors='replace')[:120]}"
            )
        return r.stdout

    def list_open_windows(self) -> list[str]:
        """Finestre aperte come 'titolo [classe]' (via hyprctl clients)."""
        r = self._run(["hyprctl", "clients", "-j"])
        if r.returncode != 0:
            return []
        try:
            clients = json.loads(r.stdout.decode())
        except Exception:
            return []
        out = []
        for c in clients:
            title = (c.get("title") or "").strip()
            klass = (c.get("class") or "").strip()
            if title or klass:
                out.append(f"{title} [{klass}]" if klass else title)
        return out

    def read_clipboard(self) -> str:
        r = self._run(["wl-paste", "--no-newline"])
        # wl-paste esce 1 a clipboard vuota: non è un errore.
        return r.stdout.decode(errors="replace") if r.returncode == 0 else ""

    def write_clipboard(self, text: str) -> None:
        r = self._run(["wl-copy"], input_bytes=text.encode())
        if r.returncode != 0:
            raise RuntimeError(f"wl-copy fallito (rc={r.returncode})")
