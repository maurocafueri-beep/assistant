"""
modules/file_analysis/__init__.py
Punto di ingresso pubblico del modulo file_analysis.

Uso:
    from modules.file_analysis import FileAnalyzer, AnalysisResult, FileAnalysisError

Esempio rapido:
    async with FileAnalyzer(stt=orch._stt) as fa:
        result = await fa.analyze("~/Scaricati/contratto.pdf")
        if result.error is None:
            print(result.content)
"""

from modules.file_analysis.base_file_analysis import (
    AnalysisResult,
    FileAnalysisError,
    FileAnalyzer,
)

__all__ = [
    "FileAnalyzer",
    "AnalysisResult",
    "FileAnalysisError",
]
