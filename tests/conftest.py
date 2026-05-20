"""tests/conftest.py — fixture condivise"""
import struct
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
def sample_audio() -> bytes:
    num_samples = 16000
    raw = struct.pack(f"<{num_samples}h", *([0] * num_samples))
    return raw

@pytest.fixture(scope="session")
def sample_text() -> str:
    return "Ciao, come posso aiutarti oggi?"
