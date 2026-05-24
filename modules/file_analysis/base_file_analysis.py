"""
modules/file_analysis/base_file_analysis.py
FileAnalyzer — estrattore di contenuto testuale da file locali.

Supporta documenti (PDF, DOCX), testo (TXT, MD, LOG, RST, YAML), web
(HTML), dati strutturati (JSON, CSV/TSV, XML) e file audio
(WAV, MP3, OGG, FLAC, M4A, OPUS) — questi ultimi via WhisperSTT.

API pubblica:
    analyzer.analyze(path)              → AnalysisResult   (async)
    analyzer.is_supported(path)         → bool
    analyzer.supported_extensions()     → set[str]

Uso tipico (dall'orchestratore):
    async with FileAnalyzer(stt=stt) as fa:
        result = await fa.analyze("~/Scaricati/contratto.pdf")
        if result.error is None:
            print(result.content)

Filosofia:
    - L'LLM riceve TESTO. La nostra unica responsabilità è estrarlo bene.
    - Hard cut a `max_chars_inline` (default 8000 ≈ 2000 token). Strategie
      di summarization a chunk arrivano in un'iterazione successiva.
    - Errori non fatali: sollevati come FileAnalysisError con `code`
      categorico, il chiamante decide se mostrare il messaggio all'utente.
    - File audio: gratis grazie a WhisperSTT.transcribe_file() già caricato
      dall'orchestratore. Se STT non è disponibile, l'estrattore audio
      degrada con un FileAnalysisError("stt_unavailable").

Errori (FileAnalysisError.code):
    not_found            → percorso inesistente
    out_of_safe_dirs     → percorso fuori da settings.pc_control.safe_dirs
    too_large            → file più grande di max_file_bytes
    unsupported          → estensione non riconosciuta
    stt_unavailable      → file audio richiesto ma STT non iniettato
    extraction_failed    → l'estrattore ha sollevato un'eccezione
"""

from __future__ import annotations

import asyncio
import csv as _csv_stdlib
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from core.logger import logger


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

# Estensioni → categoria. Tutte lowercase, sempre confrontate con
# Path.suffix.lower() per non avere sorprese con "FILE.PDF".
_TEXT_EXTS:  frozenset[str] = frozenset({".txt", ".md", ".log", ".rst", ".yaml", ".yml"})
_PDF_EXTS:   frozenset[str] = frozenset({".pdf"})
_DOCX_EXTS:  frozenset[str] = frozenset({".docx"})
_HTML_EXTS:  frozenset[str] = frozenset({".html", ".htm"})
_JSON_EXTS:  frozenset[str] = frozenset({".json"})
_CSV_EXTS:   frozenset[str] = frozenset({".csv", ".tsv"})
_XML_EXTS:   frozenset[str] = frozenset({".xml"})
_AUDIO_EXTS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".opus"}
)

# Soglie operative — valori sensati per gemma4:26B con 8 GB di contesto e
# anche tutta la memoria/personality/web_search iniettata accanto.
_DEFAULT_MAX_CHARS_INLINE: int = 8_000
_DEFAULT_MAX_FILE_BYTES:   int = 50 * 1024 * 1024   # 50 MB

# Marcatore di troncamento aggiunto in coda al contenuto quando viene tagliato.
# Formato pensato per essere riconoscibile dall'LLM nel system prompt.
_TRUNCATION_NOTICE = (
    "\n\n[...contenuto troncato: mostrati {shown} di {total} caratteri totali]"
)

# Anteprima massima per il box CSV: dopo queste righe ci fermiamo, anche se
# non abbiamo raggiunto max_chars_inline. Tabelle enormi sono inutili da
# leggere intere per un LLM: meglio prima 50 righe ben formattate.
_CSV_MAX_PREVIEW_ROWS: int = 50


# ---------------------------------------------------------------------------
# Exception class
# ---------------------------------------------------------------------------

class FileAnalysisError(Exception):
    """
    Errore raised dagli estrattori. Il campo `code` permette al chiamante
    (orchestratore, UI) di produrre messaggi user-friendly senza fare
    pattern-matching su stringhe.
    """

    def __init__(self, code: str, message: str, *, path: Optional[str] = None) -> None:
        super().__init__(message)
        self.code:    str           = code
        self.message: str           = message
        self.path:    Optional[str] = path

    def __str__(self) -> str:
        if self.path:
            return f"[{self.code}] {self.message} ({self.path})"
        return f"[{self.code}] {self.message}"


# ---------------------------------------------------------------------------
# Dataclass pubblica
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Risultato di una chiamata FileAnalyzer.analyze()."""
    path:               str
    file_type:          str                            # "pdf"|"docx"|"text"|"html"|"json"|"csv"|"xml"|"audio"|"unknown"
    content:            str                            # testo estratto, già troncato se necessario
    char_count:         int                            # len(content) dopo troncamento
    original_char_count: int                           # caratteri prima del troncamento
    truncated:          bool                           # True se è stato applicato il hard cut
    elapsed_ms:         float                          # tempo totale (I/O + parsing/STT)
    extractor:          str                            # nome del metodo che ha estratto (per log/debug)
    error:              Optional[str] = None           # popolato se l'estrazione è fallita (con .code:msg)
    metadata:           dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.content.strip()

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "path":            self.path[-80:],        # solo gli ultimi 80 char (privacy + lunghezza)
            "file_type":       self.file_type,
            "char_count":      self.char_count,
            "orig_char_count": self.original_char_count,
            "truncated":       self.truncated,
            "elapsed_ms":      round(self.elapsed_ms, 1),
            "extractor":       self.extractor,
            "error":           self.error,
        }


# ---------------------------------------------------------------------------
# Helper puri — path e classificazione
# ---------------------------------------------------------------------------

def _resolve_path(raw: str | Path) -> Path:
    """
    Espande `~`, variabili d'ambiente e risolve link simbolici.
    Non verifica che il percorso esista (sarà compito di analyze()).
    """
    p = Path(os.path.expandvars(os.path.expanduser(str(raw))))
    # resolve(strict=False) non solleva se il file non esiste ma normalizza i `..`.
    return p.resolve(strict=False)


def _is_safe_path(path: Path, safe_dirs: list[str]) -> bool:
    """
    True se `path` è dentro almeno una delle directory permesse.
    safe_dirs vuoto → consentito tutto (use case: test con tmp_path).
    """
    if not safe_dirs:
        return True
    try:
        path_resolved = path.resolve(strict=False)
    except Exception:
        return False
    for d in safe_dirs:
        try:
            base = Path(os.path.expanduser(d)).resolve(strict=False)
        except Exception:
            continue
        try:
            path_resolved.relative_to(base)
            return True
        except ValueError:
            continue
    return False


def _classify_path(path: Path) -> str:
    """Restituisce la categoria del file in base all'estensione."""
    ext = path.suffix.lower()
    if ext in _PDF_EXTS:   return "pdf"
    if ext in _DOCX_EXTS:  return "docx"
    if ext in _TEXT_EXTS:  return "text"
    if ext in _HTML_EXTS:  return "html"
    if ext in _JSON_EXTS:  return "json"
    if ext in _CSV_EXTS:   return "csv"
    if ext in _XML_EXTS:   return "xml"
    if ext in _AUDIO_EXTS: return "audio"
    return "unknown"


def _truncate(content: str, max_chars: int) -> tuple[str, int, bool]:
    """
    Tronca `content` a `max_chars` caratteri aggiungendo un avviso.
    Ritorna (testo_finale, char_count_originale, truncated_flag).
    """
    original_len = len(content)
    if original_len <= max_chars:
        return content, original_len, False

    notice = _TRUNCATION_NOTICE.format(shown=max_chars, total=original_len)
    return content[:max_chars] + notice, original_len, True


# ---------------------------------------------------------------------------
# Estrattori sincroni (lanciati in run_in_executor)
# ---------------------------------------------------------------------------

def _extract_text(path: Path) -> str:
    """TXT, MD, LOG, RST, YAML, YML — read_text con encoding tollerante."""
    # `errors="replace"` evita crash su file con BOM strani o byte non-UTF8.
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_json(path: Path) -> str:
    """JSON — parse + pretty-print. Se non parsa, ritorna il testo grezzo."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(raw)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        # JSON malformato → mostra il testo originale (più utile di un errore).
        return raw


def _extract_csv(path: Path) -> str:
    """
    CSV/TSV — anteprima tabellare ASCII delle prime _CSV_MAX_PREVIEW_ROWS
    righe, con colonne allineate dal massimo dei valori in colonna.
    Delimiter auto-detection (sniffer stdlib).
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return ""

    # Sniff: prova a indovinare separatore (,|;|tab) sui primi 4KB.
    try:
        dialect = _csv_stdlib.Sniffer().sniff(raw[:4096], delimiters=",;\t|")
    except Exception:
        # Fallback: TSV se .tsv, CSV altrimenti.
        dialect = _csv_stdlib.excel_tab if path.suffix.lower() == ".tsv" else _csv_stdlib.excel

    reader = _csv_stdlib.reader(io.StringIO(raw), dialect)
    rows: list[list[str]] = []
    total_rows = 0
    for i, row in enumerate(reader):
        total_rows += 1
        if i < _CSV_MAX_PREVIEW_ROWS:
            rows.append([str(c) for c in row])

    if not rows:
        return ""

    # Larghezza colonna = max len su tutte le righe selezionate.
    n_cols  = max(len(r) for r in rows)
    widths  = [0] * n_cols
    for r in rows:
        for j, cell in enumerate(r):
            widths[j] = max(widths[j], len(cell))

    def fmt_row(r: list[str]) -> str:
        cells = [(r[j] if j < len(r) else "").ljust(widths[j]) for j in range(n_cols)]
        return " | ".join(cells).rstrip()

    lines = [fmt_row(rows[0])]
    if len(rows) > 1:
        lines.append("-+-".join("-" * w for w in widths))
        lines.extend(fmt_row(r) for r in rows[1:])

    if total_rows > _CSV_MAX_PREVIEW_ROWS:
        lines.append(
            f"\n[anteprima: prime {_CSV_MAX_PREVIEW_ROWS} righe di {total_rows} totali]"
        )
    return "\n".join(lines)


def _extract_xml(path: Path) -> str:
    """XML — restituisce il testo grezzo. La tassonomia tag-by-tag non aggiunge valore."""
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_html(path: Path) -> str:
    """HTML — testo visibile via BeautifulSoup, con normalizzazione di spazi."""
    from bs4 import BeautifulSoup  # lazy: ~30ms import

    raw  = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")

    # Rimuovi script/style: rumore puro nel contesto LLM.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Compatta più newline consecutivi (HTML genera molti vuoti).
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def _extract_pdf(path: Path) -> str:
    """PDF — pypdf, una stringa per pagina separata da form-feed."""
    from pypdf import PdfReader  # lazy: ~80ms import

    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception as exc:
            logger.debug("file_analysis._extract_pdf | pagina {} fallita: {}", i + 1, exc)
            txt = ""
        if txt.strip():
            pages.append(f"--- pagina {i + 1} ---\n{txt.strip()}")
    return "\n\n".join(pages)


def _extract_docx(path: Path) -> str:
    """DOCX — paragrafi + tabelle (celle separate da tab)."""
    from docx import Document  # python-docx, lazy

    doc   = Document(str(path))
    parts: list[str] = []

    # Paragrafi
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text.strip())

    # Tabelle: rappresenta ogni riga con celle separate da \t.
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append("\t".join(cells))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# FileAnalyzer
# ---------------------------------------------------------------------------

class FileAnalyzer:
    """
    Estrattore di contenuto da file locali. Async-friendly.

    Args:
        stt:               Istanza WhisperSTT già caricata. Se None,
                           gli estrattori audio falliscono con FileAnalysisError(
                           "stt_unavailable").
        max_chars_inline:  Soglia hard-cut sul testo estratto. Default
                           _DEFAULT_MAX_CHARS_INLINE (8000).
        max_file_bytes:    Limite di sicurezza sulla dimensione su disco
                           (evita di leggere PDF da 500 MB). Default 50 MB.
        safe_dirs:         Lista di directory permesse. None = eredita da
                           settings.pc_control.safe_dirs.

    Esempio:
        async with FileAnalyzer(stt=orch._stt) as fa:
            result = await fa.analyze("~/Scaricati/note.pdf")
            print(result.content[:500])

    Note di design:
        - Niente stato persistente: ogni analyze() è autonoma.
        - aclose() è no-op per ora (nessuna risorsa aperta lungo termine).
        - Gli estrattori sincroni sono eseguiti in run_in_executor per non
          bloccare l'event loop su PDF di centinaia di pagine.
    """

    def __init__(
        self,
        stt:              Optional[Any] = None,
        max_chars_inline: Optional[int] = None,
        max_file_bytes:   Optional[int] = None,
        safe_dirs:        Optional[list[str]] = None,
    ) -> None:
        self._stt             = stt
        self._max_chars       = max_chars_inline if max_chars_inline is not None else _DEFAULT_MAX_CHARS_INLINE
        self._max_file_bytes  = max_file_bytes   if max_file_bytes   is not None else _DEFAULT_MAX_FILE_BYTES

        if safe_dirs is not None:
            self._safe_dirs: list[str] = list(safe_dirs)
        else:
            # Eredita pigramente dai settings: i test possono passare safe_dirs=[]
            # per disattivare il controllo.
            self._safe_dirs = list(settings.pc_control.safe_dirs)

    # -- context manager -------------------------------------------------------

    async def __aenter__(self) -> "FileAnalyzer":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """No-op: nessuna risorsa attiva. Manteniamo l'API per coerenza."""
        return None

    # -- utility pubbliche -----------------------------------------------------

    def supported_extensions(self) -> set[str]:
        """Tutte le estensioni gestite, con il punto iniziale."""
        return set(
            _TEXT_EXTS | _PDF_EXTS | _DOCX_EXTS | _HTML_EXTS
            | _JSON_EXTS | _CSV_EXTS | _XML_EXTS | _AUDIO_EXTS
        )

    def is_supported(self, path: str | Path) -> bool:
        """True se l'estensione del file è gestita dal modulo."""
        return _classify_path(Path(str(path))) != "unknown"

    # -- API principale --------------------------------------------------------

    async def analyze(self, path: str | Path) -> AnalysisResult:
        """
        Estrae il contenuto testuale di `path`. Mai solleva:
        in caso d'errore restituisce AnalysisResult con `error` valorizzato
        e `content` vuoto.

        I codici d'errore (in `error`, prefissati con "[code]") sono:
            not_found, out_of_safe_dirs, too_large, unsupported,
            stt_unavailable, extraction_failed.
        """
        t0      = time.perf_counter()
        resolved = _resolve_path(path)
        path_str = str(resolved)

        # 1) Esistenza
        if not resolved.exists() or not resolved.is_file():
            return self._error_result(
                path_str, "unknown", "not_found",
                f"File non trovato: {resolved}",
                t0,
            )

        # 2) Safety: dentro safe_dirs?
        if not _is_safe_path(resolved, self._safe_dirs):
            return self._error_result(
                path_str, "unknown", "out_of_safe_dirs",
                f"Percorso fuori dalle directory permesse: {resolved}",
                t0,
            )

        # 3) Dimensione su disco
        try:
            file_size = resolved.stat().st_size
        except OSError as exc:
            return self._error_result(
                path_str, "unknown", "extraction_failed",
                f"Stat fallita: {exc}", t0,
            )

        if file_size > self._max_file_bytes:
            return self._error_result(
                path_str, _classify_path(resolved), "too_large",
                f"File troppo grande ({file_size} byte > {self._max_file_bytes}). "
                f"Soglia regolabile in FileAnalyzer(max_file_bytes=...).",
                t0,
            )

        # 4) Classificazione
        file_type = _classify_path(resolved)
        if file_type == "unknown":
            return self._error_result(
                path_str, "unknown", "unsupported",
                f"Estensione non supportata: '{resolved.suffix}'. "
                f"Supportate: {', '.join(sorted(self.supported_extensions()))}.",
                t0,
            )

        logger.debug(
            "file_analysis.analyze | path='{}' type={} size={}B",
            path_str[-80:], file_type, file_size,
        )

        # 5) Dispatch estrattore
        try:
            content, extractor_name = await self._dispatch(resolved, file_type)
        except FileAnalysisError as exc:
            return self._error_result(
                path_str, file_type, exc.code, exc.message, t0,
            )
        except Exception as exc:
            logger.warning(
                "file_analysis.analyze | estrattore '{}' fallito: {}",
                file_type, exc,
            )
            return self._error_result(
                path_str, file_type, "extraction_failed",
                f"Estrazione fallita ({type(exc).__name__}): {exc}",
                t0,
            )

        # 6) Truncation
        final_text, original_len, was_truncated = _truncate(content, self._max_chars)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = AnalysisResult(
            path                = path_str,
            file_type           = file_type,
            content             = final_text,
            char_count          = len(final_text),
            original_char_count = original_len,
            truncated           = was_truncated,
            elapsed_ms          = elapsed_ms,
            extractor           = extractor_name,
            metadata            = {"file_size_bytes": file_size},
        )
        logger.info("file_analysis.analyze | OK | {}", result.to_log_dict())
        return result

    # -- dispatch interno ------------------------------------------------------

    async def _dispatch(self, path: Path, file_type: str) -> tuple[str, str]:
        """
        Smista al giusto estrattore. Ritorna (testo, nome_extractor).
        Solleva FileAnalysisError per audio senza STT; per il resto, le
        eccezioni risalgono al chiamante che le maperà a extraction_failed.
        """
        if file_type == "audio":
            return await self._extract_audio(path), "audio_stt"

        # Tutti gli altri estrattori sono sincroni → run_in_executor.
        loop = asyncio.get_running_loop()
        if file_type == "pdf":
            txt = await loop.run_in_executor(None, _extract_pdf, path)
            return txt, "pdf_pypdf"
        if file_type == "docx":
            txt = await loop.run_in_executor(None, _extract_docx, path)
            return txt, "docx_python-docx"
        if file_type == "html":
            txt = await loop.run_in_executor(None, _extract_html, path)
            return txt, "html_bs4"
        if file_type == "json":
            txt = await loop.run_in_executor(None, _extract_json, path)
            return txt, "json_stdlib"
        if file_type == "csv":
            txt = await loop.run_in_executor(None, _extract_csv, path)
            return txt, "csv_stdlib"
        if file_type == "xml":
            txt = await loop.run_in_executor(None, _extract_xml, path)
            return txt, "xml_raw"
        if file_type == "text":
            txt = await loop.run_in_executor(None, _extract_text, path)
            return txt, "text_raw"

        # Difensivo: non dovrebbe accadere (filtrato in analyze()).
        raise FileAnalysisError(
            "unsupported", f"Estrattore mancante per categoria '{file_type}'",
        )

    async def _extract_audio(self, path: Path) -> str:
        """
        Audio → trascrizione via WhisperSTT.transcribe_file().
        Solleva FileAnalysisError("stt_unavailable") se self._stt è None.
        """
        if self._stt is None:
            raise FileAnalysisError(
                "stt_unavailable",
                "Modulo STT non disponibile: l'analisi audio richiede Whisper. "
                "Riavvia l'assistente abilitando lo STT.",
                path=str(path),
            )

        try:
            result = await self._stt.transcribe_file(path)
        except Exception as exc:
            raise FileAnalysisError(
                "extraction_failed",
                f"Trascrizione fallita: {exc}",
                path=str(path),
            ) from exc

        text = (result.text or "").strip()
        if not text:
            # Trascrizione vuota — file silenzioso o non riconosciuto come
            # parlato. Restituiamo stringa vuota, non un errore: l'utente
            # vedrà "contenuto vuoto" e capirà.
            logger.debug(
                "file_analysis._extract_audio | trascrizione vuota per '{}'",
                str(path)[-80:],
            )
        return text

    # -- helper interno --------------------------------------------------------

    def _error_result(
        self,
        path:     str,
        ftype:    str,
        code:     str,
        message:  str,
        t0:       float,
    ) -> AnalysisResult:
        """Costruisce un AnalysisResult d'errore con timing già misurato."""
        elapsed_ms = (time.perf_counter() - t0) * 1000
        err_str    = f"[{code}] {message}"
        logger.warning("file_analysis.analyze | KO | {}", err_str)
        return AnalysisResult(
            path                = path,
            file_type           = ftype,
            content             = "",
            char_count          = 0,
            original_char_count = 0,
            truncated           = False,
            elapsed_ms          = elapsed_ms,
            extractor           = "n/a",
            error               = err_str,
        )
