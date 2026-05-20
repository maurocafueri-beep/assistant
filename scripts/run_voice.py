"""
scripts/run_voice.py
Entry point CLI per l'assistente vocale locale.

Avvia il loop completo: microfono → STT → LLM → TTS → speaker.

Requisiti:
    - Ollama attivo con il modello chat configurato
    - Server Qwen3-TTS (avviato automaticamente in venv-tts)
    - Microfono e altoparlanti configurati nel sistema

Uso:
    venv-runtime/bin/python scripts/run_voice.py
    venv-runtime/bin/python scripts/run_voice.py --personality dev
    venv-runtime/bin/python scripts/run_voice.py --tts-profile squib
    venv-runtime/bin/python scripts/run_voice.py --no-memory
    venv-runtime/bin/python scripts/run_voice.py --session mia-sessione-1
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import logger
from core.voice_loop import VoiceLoop


async def main(args: argparse.Namespace) -> None:
    try:
        async with VoiceLoop(
            personality   = args.personality,
            tts_profile   = args.tts_profile,
            session_id    = args.session,
            enable_memory = not args.no_memory,
        ) as loop:
            await loop.run()

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.error("run_voice | errore fatale: {}", exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assistente vocale locale — local-assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  %(prog)s                              # avvio con impostazioni di default
  %(prog)s --personality dev            # profilo sviluppatore
  %(prog)s --tts-profile squib          # voce alternativa
  %(prog)s --no-memory                  # senza RAG (più veloce)
  %(prog)s --session sess-lavoro        # sessione nominata (riprendibile)
        """,
    )
    parser.add_argument(
        "--personality",
        default=None,
        metavar="NOME",
        help="Profilo personalità (default: da settings)",
    )
    parser.add_argument(
        "--tts-profile",
        default=None,
        metavar="PROFILO",
        help="Profilo voce TTS (default: da settings.tts.voice)",
    )
    parser.add_argument(
        "--session",
        default=None,
        metavar="ID",
        help="ID sessione per storia conversazione (default: generato)",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disabilita memoria semantica RAG",
    )

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n\nArrivederci.")
        sys.exit(0)
