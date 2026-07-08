"""
modules/pc_control
==================
Controllo del desktop: apri app/file, digita testo, screenshot, clipboard.

    from modules.pc_control import HyprlandPCControl

HyprlandPCControl è l'implementazione per Hyprland/Wayland (hyprctl, grim,
wl-clipboard, xdg-open); HyprlandPCControl.available() dice se i binari ci
sono. Lo screenshot alimenta il percorso visivo dell'orchestratore
("guarda lo schermo" → qwen3-vl). Sicurezza: open_file rispetta
settings.pc_control.safe_dirs, open_application accetta solo nomi-token.
"""

from .base_pc_control import BasePCControl, HyprlandPCControl

__all__ = ["BasePCControl", "HyprlandPCControl"]
