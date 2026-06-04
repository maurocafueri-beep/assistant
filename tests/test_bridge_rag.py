"""
tests/test_bridge_rag.py
Persistenza cross-restart dei file RAG: ripristino lato UIBridge.
"""
from types import SimpleNamespace

from ui.bridge import UIBridge


def _bridge(sessions: dict) -> UIBridge:
    # __new__ salta __init__ (niente load()/orchestrator reale): impostiamo a
    # mano solo gli attributi usati dal metodo, come nei test dell'orchestrator.
    b = UIBridge.__new__(UIBridge)
    b._sessions = sessions
    b._orch = SimpleNamespace(_session_rag_files={})
    return b


class TestRestoreRagFilesFromSessions:
    def test_reinjects_saved_rag_files(self):
        b = _bridge({
            "s1": {"rag_files": [{"file_id": "abc", "source": "libro.pdf"}]},
            "s2": {"rag_files": []},     # sessione senza file
            "s3": {},                    # chiave assente
        })
        n = b.restore_rag_files_from_sessions()
        assert n == 1
        assert b._orch._session_rag_files["s1"] == [{"file_id": "abc", "source": "libro.pdf"}]
        assert "s2" not in b._orch._session_rag_files
        assert "s3" not in b._orch._session_rag_files

    def test_copia_difensiva(self):
        # La lista reiniettata è una copia: mutarla nell'orchestrator non tocca
        # lo store di sessione su cui si baserà il prossimo persist.
        sessions = {"s1": {"rag_files": [{"file_id": "a", "source": "x.pdf"}]}}
        b = _bridge(sessions)
        b.restore_rag_files_from_sessions()
        b._orch._session_rag_files["s1"].append({"file_id": "b", "source": "y.pdf"})
        assert len(sessions["s1"]["rag_files"]) == 1

    def test_no_sessions_no_error(self):
        b = _bridge({})
        assert b.restore_rag_files_from_sessions() == 0
