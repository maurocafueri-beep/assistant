"""
modules/terminal_agent/base_terminal_agent.py
TerminalAgent — terminale agentico Linux orchestrato da LLM.

Pipeline di un turno:
    propose(user_request)
        ├── loop ReAct (max N iterazioni):
        │     LLM → JSON {"action": "search"|"propose"}
        │       ├── search → web_search.search() → risultati al prompt
        │       └── propose → esce dal loop
        └── classificazione deterministica del comando → CommandProposal

    execute(proposal, confirmed=...)
        ├── verifica needs_confirmation
        ├── rifiuto deterministico se BLOCKED
        ├── subprocess con cwd persistente + sentinel per tracking cwd
        └── output troncato → CommandResult

    analyze(result)
        └── LLM legge stdout/stderr e produce diagnosi/suggerimenti

API pubblica:
    async with TerminalAgent() as ta:
        proposal = await ta.propose("trova i file più grandi nella mia home")
        result   = await ta.execute(proposal, confirmed=True)
        analysis = await ta.analyze(result)

    # oppure tutto insieme:
        turn = await ta.run("...", auto_confirm=False)

    # gestione stato:
        ta.cwd               # stringa
        ta.reset()           # resetta cwd, history, env override

Errori:
    - Comando bloccato (BLOCKED)             → ValueError sollevato in execute()
    - Comando non confermato (needs_conf.)   → ValueError sollevato in execute()
    - Timeout subprocess                     → CommandResult con exit_code=-1
    - LLM non parsa JSON (max retry)         → RuntimeError sollevato in propose()
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from core.context import ModelRole
from core.logger import logger
from modules.llm import OllamaClient
from modules.llm.base_llm import Message, Role


# ---------------------------------------------------------------------------
# Enums & Dataclasses pubbliche
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    SAFE      = "safe"        # read-only, auto-run
    MODERATE  = "moderate"    # modifica file/sistema reversibile
    DANGEROUS = "dangerous"   # distruttivo o sudo
    BLOCKED   = "blocked"     # mai eseguito


@dataclass
class CommandProposal:
    """Proposta di comando da parte dell'LLM, dopo classificazione di rischio."""
    command:            str
    rationale:          str
    risk_level:         RiskLevel
    needs_confirmation: bool
    cwd:                str
    search_used:        bool        = False
    sources:            list[str]   = field(default_factory=list)
    proposal_id:        str         = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "proposal_id":        self.proposal_id,
            "command":            self.command[:120],
            "risk_level":         self.risk_level.value,
            "needs_confirmation": self.needs_confirmation,
            "cwd":                self.cwd,
            "search_used":        self.search_used,
            "n_sources":          len(self.sources),
        }


@dataclass
class CommandResult:
    """Esito dell'esecuzione di un comando."""
    command:     str
    exit_code:   int
    stdout:      str
    stderr:      str
    duration_ms: float
    cwd_before:  str
    cwd_after:   str
    truncated:   bool = False
    proposal_id: str  = ""

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "exit_code":   self.exit_code,
            "duration_ms": round(self.duration_ms, 1),
            "stdout_len":  len(self.stdout),
            "stderr_len":  len(self.stderr),
            "truncated":   self.truncated,
            "cwd_after":   self.cwd_after,
        }


@dataclass
class AgentTurn:
    """Turno completo: proposta + (eventuale) esecuzione + analisi."""
    user_request: str
    proposal:     Optional[CommandProposal] = None
    result:       Optional[CommandResult]   = None
    analysis:     str                       = ""
    skipped:      bool                      = False   # comando non confermato dall'utente
    error:        Optional[str]             = None

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "request_len": len(self.user_request),
            "command":     (self.proposal.command[:80] if self.proposal else None),
            "exit_code":   (self.result.exit_code      if self.result   else None),
            "skipped":     self.skipped,
            "error":       self.error,
        }


# ---------------------------------------------------------------------------
# Classificazione del rischio
# ---------------------------------------------------------------------------

# Prefissi di comando read-only — auto-run senza conferma.
# La verifica è sulla prima "word" del comando (dopo eventuali env-var inline).
SAFE_COMMANDS: frozenset[str] = frozenset({
    # listing / cwd
    "ls", "ll", "la", "dir", "tree", "pwd", "cd",
    # lettura file
    "cat", "head", "tail", "less", "more", "bat",
    # info sistema
    "whoami", "id", "hostname", "uname", "uptime", "date",
    "df", "du", "free", "lscpu", "lsblk", "lsmem", "lsusb", "lspci",
    "env", "printenv",
    # lookup binari
    "which", "whereis", "type",
    # processi (read-only)
    "ps", "pgrep", "pidof", "jobs",
    # file metadata
    "stat", "file", "readlink", "realpath", "basename", "dirname",
    # testo / parsing
    "echo", "printf",
    "wc", "sort", "uniq", "cut", "tr", "rev", "tac", "nl",
    "grep", "egrep", "fgrep", "rg", "ag", "ack",
    "awk", "gawk",   # awk è read-only se non usa >, >>, system()
    # ricerca file (cautela su find — controllata sotto)
    "find", "locate", "fd",
    # rete (read-only)
    "ping", "dig", "host", "nslookup", "traceroute", "tracepath",
    "ip", "ifconfig", "netstat", "ss",
    # git read-only
    # NOTE: "git" da solo è ambiguo; la safety è controllata su prima+seconda word
    # package managers read-only
    # NOTE: "pip", "apt", "dpkg" hanno sub-cmd safe/unsafe; controllati sotto
    # shell builtins read-only
    "history", "alias", "help",
})

# Sub-comandi safe per "git" (prima word == "git", seconda word in questo set).
SAFE_GIT_SUBCOMMANDS: frozenset[str] = frozenset({
    "status", "log", "diff", "branch", "show", "remote", "config",
    "tag", "describe", "blame", "reflog", "stash",  # stash list/show
    "rev-parse", "ls-files", "ls-tree", "shortlog",
})

# Sub-comandi safe per package manager.
SAFE_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "git":  SAFE_GIT_SUBCOMMANDS,
    "pip":  frozenset({"list", "show", "freeze", "check", "config", "--version"}),
    "pip3": frozenset({"list", "show", "freeze", "check", "config", "--version"}),
    "apt":  frozenset({"list", "search", "show", "policy", "--version"}),
    "dpkg": frozenset({"-l", "--list", "-L", "-s", "--status", "-S", "--search"}),
    "npm":  frozenset({"list", "ls", "view", "info", "outdated", "config"}),
    "docker": frozenset({"ps", "images", "logs", "inspect", "version", "info",
                         "history", "port", "stats", "top"}),
    "systemctl": frozenset({"status", "is-active", "is-enabled", "list-units",
                            "list-unit-files", "show", "cat"}),
}

# Pattern che NON sono mai eseguiti, neanche con conferma esplicita.
BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+(-[a-zA-Z]*[rRf][a-zA-Z]*\s+)+/(\s|$)"),       # rm -rf /
    re.compile(r"\brm\s+(-[a-zA-Z]*[rRf][a-zA-Z]*\s+)+/\s*\*"),        # rm -rf /*
    re.compile(r"\brm\s+(-[a-zA-Z]*[rRf][a-zA-Z]*\s+)+~\s*/?\s*$"),    # rm -rf ~
    re.compile(r"\bdd\b[^|]*\bof=/dev/(sd|nvme|hd|mmcblk)"),           # dd to disk
    re.compile(r"\bmkfs\.[a-z0-9]+\b"),                                # format
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}"),                   # fork bomb
    re.compile(r">\s*/dev/(sd[a-z]|nvme|hd|mmcblk)"),                  # write to raw disk
    re.compile(r"\bchmod\s+(-R\s+)?(0?[0-9]{3,4})\s+/(\s|$)"),         # chmod XXX /
    re.compile(r"\bchown\s+-R\s+\S+\s+/(\s|$)"),                       # chown -R /
    re.compile(r"\b(shutdown|reboot|halt|poweroff|init\s+0|init\s+6)\b"),
]

# Pattern che richiedono conferma (oltre alla classificazione "non-safe").
# Usati per marcare un comando come "dangerous" anche se per prefisso sarebbe "moderate".
DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\b"),                       # qualunque rm
    re.compile(r"\bdd\b"),                       # qualunque dd
    re.compile(r"\bsudo\b"),
    re.compile(r"\bpkexec\b"),
    re.compile(r"\bsu\s+-"),
    re.compile(r"\bchmod\s+(-R\s+)?0?7{2,3}\b"), # chmod 777
    re.compile(r"\bkill\s+-9\b"),
    re.compile(r"\bkillall\b"),
    re.compile(r">\s*/etc/"),
    re.compile(r"\bcurl\b[^|;]*\|\s*(sh|bash|zsh)\b"),  # curl | sh
    re.compile(r"\bwget\b[^|;]*\|\s*(sh|bash|zsh)\b"),  # wget | sh
]

# Pattern che invalidano la "safety" di un comando altrimenti safe.
# Es: `find ... -delete` non è più read-only.
UNSAFE_FIND_FLAGS = re.compile(r"\bfind\b[^;]*(\s-(delete|exec\s+rm|execdir\s+rm))")


def _strip_quoted_strings(cmd: str) -> str:
    """
    Rimuove il contenuto delle stringhe quoted (single + double) dal comando,
    rispettando gli escape. Lascia le virgolette esterne ma svuota il contenuto.

    Serve a evitare falsi positivi nelle regex di safety. Esempio:
        echo "sudo apt update"  →  echo ""
    Senza questo, `\\bsudo\\b` matcha la `sudo` dentro la stringa e classifica
    `echo` come pericoloso.

    Funziona riga per riga sui caratteri (no shlex perché vogliamo preservare
    la struttura per le regex successive — separatori, redirezioni, ecc.).
    """
    out: list[str] = []
    i = 0
    while i < len(cmd):
        c = cmd[i]
        if c == '"' or c == "'":
            quote = c
            out.append(quote)
            i += 1
            # avanza fino alla chiusura, rispettando \" o \' (solo per ")
            while i < len(cmd):
                if quote == '"' and cmd[i] == "\\" and i + 1 < len(cmd):
                    # in double-quoted, \ può fare escape
                    i += 2
                    continue
                if cmd[i] == quote:
                    out.append(quote)
                    i += 1
                    break
                i += 1
            # nota: se la stringa non chiude mai, terminiamo il loop senza
            # aggiungere niente alla output — comportamento accettabile
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _first_word(cmd: str) -> str:
    """Estrae il nome del comando (skipping env-var inline tipo FOO=bar cmd)."""
    parts = shlex.split(cmd, posix=True) if cmd.strip() else []
    for p in parts:
        if "=" in p and not p.startswith(("-", "/")) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", p):
            continue  # env var inline, skippa
        return p
    return ""


def _second_word(cmd: str) -> str:
    """Estrae la seconda 'word' utile (per git/pip/apt sub-commands)."""
    try:
        parts = shlex.split(cmd, posix=True)
    except ValueError:
        return ""
    skipped_env = False
    found_first = False
    for p in parts:
        if not skipped_env and "=" in p and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", p):
            continue
        skipped_env = True
        if not found_first:
            found_first = True
            continue
        return p
    return ""


def _classify(cmd: str, *, allow_sudo: bool) -> RiskLevel:
    """
    Classifica un comando shell. Logica deterministica, indipendente dall'LLM.
    """
    stripped = cmd.strip()
    if not stripped:
        return RiskLevel.BLOCKED

    # Per le regex di safety usiamo il comando con le stringhe quoted SVUOTATE.
    # Così `echo "sudo apt"` non viene classificato come pericoloso a causa
    # del contenuto della stringa. La classificazione del prefisso (_first_word)
    # continua a usare `stripped` originale perché non guarda dentro le stringhe.
    safety_view = _strip_quoted_strings(stripped)

    # 1) BLOCKED patterns hard
    for pat in BLOCKED_PATTERNS:
        if pat.search(safety_view):
            return RiskLevel.BLOCKED

    # 2) sudo / pkexec / su senza permesso
    if re.search(r"\bsudo\b|\bpkexec\b|\bsu\s+-", safety_view) and not allow_sudo:
        return RiskLevel.BLOCKED

    # 3) DANGEROUS patterns
    for pat in DANGEROUS_PATTERNS:
        if pat.search(safety_view):
            return RiskLevel.DANGEROUS

    # 4) find con -delete / -exec rm → non più safe
    if UNSAFE_FIND_FLAGS.search(safety_view):
        return RiskLevel.MODERATE

    # 5) redirezioni a file / pipe in scrittura → moderate
    #    (gestiamo solo le redirezioni "ovvie": >, >>, &>, > tee)
    if re.search(r"(^|\s)>>?\s*[^&\s]", safety_view) or re.search(r"\|\s*tee\b", safety_view):
        return RiskLevel.MODERATE

    # 6) compound commands (&&, ||, ;, |) → controlla ogni "leg"
    if re.search(r"(\|\||\&\&|;)", safety_view):
        # split grossolano sui separatori top-level
        legs = re.split(r"\|\||\&\&|;", stripped)
        worst = RiskLevel.SAFE
        order = [RiskLevel.SAFE, RiskLevel.MODERATE, RiskLevel.DANGEROUS, RiskLevel.BLOCKED]
        for leg in legs:
            leg = leg.strip()
            if not leg:
                continue
            sub = _classify(leg, allow_sudo=allow_sudo)
            if order.index(sub) > order.index(worst):
                worst = sub
        return worst

    # 7) prima word check
    first = _first_word(stripped)
    if not first:
        return RiskLevel.MODERATE

    # binario via path assoluto? bidirezionale (es. /usr/bin/ls è safe)
    first_name = Path(first).name if "/" in first else first

    if first_name in SAFE_COMMANDS:
        return RiskLevel.SAFE

    if first_name in SAFE_SUBCOMMANDS:
        second = _second_word(stripped)
        # gestisci anche flag come "dpkg -l"
        if second in SAFE_SUBCOMMANDS[first_name]:
            return RiskLevel.SAFE
        return RiskLevel.MODERATE

    # 8) default: moderate (richiede conferma)
    return RiskLevel.MODERATE


# ---------------------------------------------------------------------------
# Helpers — output, parsing JSON, gestione cwd
# ---------------------------------------------------------------------------

_CWD_SENTINEL_TAG = "__TERMINAL_AGENT_CWD__"
_CWD_SENTINEL_RE  = re.compile(rf"{_CWD_SENTINEL_TAG}=(.+?)(?:\n|$)")


def _wrap_command_with_cwd_tracking(cmd: str) -> str:
    """
    Avvolge il comando in un wrapper bash che stampa la cwd finale.
    Si usa con `bash -c <wrapper>`.
    Preserva l'exit code originale.
    """
    # Eseguito sotto bash -c, niente di che da escapare nel wrapper:
    # `cmd` viene passato come stringa dentro un blocco { ... }.
    return f"{{ {cmd}; }}; __ec=$?; printf '\\n{_CWD_SENTINEL_TAG}=%s\\n' \"$PWD\"; exit $__ec"


def _extract_cwd(stdout: str, fallback: str) -> tuple[str, str]:
    """
    Estrae la cwd dal sentinel in coda allo stdout.
    Restituisce (stdout_pulito, cwd_finale).
    """
    matches = list(_CWD_SENTINEL_RE.finditer(stdout))
    if not matches:
        return stdout, fallback
    last = matches[-1]
    new_cwd = last.group(1).strip()
    # rimuovi il sentinel e l'eventuale newline che lo precede
    cleaned = stdout[:last.start()].rstrip("\n")
    return cleaned, new_cwd


def _truncate(text: str, max_bytes: int) -> tuple[str, bool]:
    """Tronca a max_bytes (in bytes UTF-8). Restituisce (testo, truncated)."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text, False
    cut = encoded[:max_bytes].decode("utf-8", errors="replace")
    return cut + f"\n... [output troncato a {max_bytes} bytes]", True


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """
    Estrae il primo oggetto JSON da una stringa, anche se circondato da prosa.
    Usa parentesi-matching robusto (rispetta stringhe e escape).
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc    = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _is_inside(path: str, allowed: list[str]) -> bool:
    """Verifica se `path` è dentro almeno una delle directory `allowed`."""
    try:
        target = Path(path).resolve()
    except (OSError, RuntimeError):
        return False
    for a in allowed:
        try:
            base = Path(a).resolve()
        except (OSError, RuntimeError):
            continue
        try:
            target.relative_to(base)
            return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """Sei un assistente terminale agentico per Linux. \
L'utente ti chiederà di fare qualcosa nel suo computer e tu devi proporre il \
comando shell (bash) appropriato.

CONTESTO CORRENTE:
- working directory: {cwd}
- shell: bash
- OS: Linux
- locale: {locale}
- safe_dirs (directory in cui l'utente ti permette di operare): {safe_dirs}
- privilegi root permessi: {allow_sudo}
- IMPORTANTE: l'app gira come processo GUI senza terminale interattivo,
  quindi `sudo` NON funziona (non può chiedere la password). Per comandi
  che richiedono root usa SEMPRE `pkexec` — apre un dialog grafico
  polkit che chiede la password all'utente. Esempio:
    invece di: sudo apt update
    usa:       pkexec apt update
  Se devi concatenare più comandi con privilegi, racchiudili in `bash -c`:
    pkexec bash -c "apt update && apt upgrade -y"

{xdg_paths_block}REGOLE FONDAMENTALI:
1. Rispondi SEMPRE e SOLO in JSON valido, in una di queste due forme:

   {{"action": "search", "query": "stringa di ricerca"}}

   oppure

   {{"action": "propose", "command": "il_comando_shell", \
"rationale": "spiegazione breve", "sources": ["url1", "url2"]}}

   Qualunque altra risposta è un errore. NON spiegare a parole, NON dire \
"non posso fare X": se non puoi fare quello che chiede l'utente, proponi \
comunque un comando `echo "..."` che spiega il problema, e classificalo \
come safe.

2. Per comandi non banali (qualsiasi cosa oltre `ls`, `cd`, `pwd`, `cat`, \
`echo`, `whoami`, `date`), devi PRIMA fare almeno una ricerca web per \
verificare sintassi e flag aggiornati della distro Linux corrente. Usa \
action "search" finché non sei sicuro della sintassi.

3. Puoi fare più ricerche in sequenza. Quando hai abbastanza informazioni, \
emetti l'azione "propose".

4. Nel campo `command`:
   - usa bash, non sh
   - se serve cambiare directory, includi `cd` nel comando
   - una sola riga (puoi usare && o ; per concatenare)
   - niente comandi distruttivi (rm -rf /, dd, mkfs, ecc.): saranno rifiutati a monte
   - se l'utente NON ti ha dato il permesso per i privilegi root (vedi sopra),
     NON usare né `sudo` né `pkexec`. Se l'utente li ha permessi, usa SEMPRE
     `pkexec` (mai `sudo`) per via del contesto GUI.
   - usa i percorsi XDG sopra elencati per le cartelle utente: NON inventare
     `/home/X/Desktop` o `/home/X/Downloads` se la lista XDG dice `Scrivania`
     o `Scaricati`

5. Nel campo `rationale` spiega in italiano in 1-2 frasi cosa fa il comando.

6. Nel campo `sources` metti gli URL delle ricerche che hai usato per \
costruire il comando (vuoto se non hai cercato).

7. META-ISTRUZIONI DELL'UTENTE: se l'utente dice "correggi/modifica/riprova/\
cambia il comando precedente" o simili, NON cercare un binario chiamato \
"correggi": guarda i turni precedenti nella conversazione, identifica l'ultimo \
comando che ha avuto un problema (exit_code != 0 o errore in stderr) o quello \
che l'utente vuole cambiare, e produci una versione corretta nel campo \
`command`. Il `rationale` deve spiegare COSA hai cambiato rispetto al \
precedente. Esempi di trigger: "correggi", "modifica", "riprova con", \
"cambia in", "usa X invece di Y".

Niente testo fuori dal JSON. Nessun commento, nessuna spiegazione esterna.
"""


_ANALYSIS_PROMPT_TEMPLATE = """Sei un analista di terminale. Stile: asciutto, tecnico, telegrafico.
NIENTE elogi del comando, NIENTE consigli ovvi (es. "puoi verificare con ls"),
NIENTE chiusure decorative ("ottimo!", "perfetto!"). Vai dritto al punto.

L'utente ti aveva chiesto: "{request}"

Comando eseguito:
    $ {command}
    (cwd: {cwd_before})

Esito:
- exit code: {exit_code}
- duration:  {duration_ms:.0f}ms
- cwd dopo:  {cwd_after}

STDOUT{stdout_truncated_note}:
{stdout}

STDERR:
{stderr}

REGOLE DI BREVITÀ (VINCOLANTI):
- exit_code=0 e output < 200 byte → 1-2 righe. Dici cosa ha fatto e basta.
- exit_code=0 e output più lungo → 2-4 righe. Riassumi il contenuto, non lo descrivi.
- exit_code!=0 oppure stderr non vuoto → 3-6 righe. Causa concreta + come correggere
  (con il comando giusto, se applicabile). Niente prosa generica.

NON ripetere il comando, NON elencare le opzioni usate, NON spiegare l'ovvio,
NON suggerire `ls` o `cd ~` come "passo successivo" se non è pertinente.

Esempi del tono atteso:
- "Sei in /tmp/foo. Niente di insolito."
- "Trovati 12 file .py, nessuno sotto node_modules."
- "SyntaxError: parentesi mancante. Usa: python3 -c 'print(\\"hello\\")'."

Rispondi in italiano, in prosa, NON in JSON.
"""


# ---------------------------------------------------------------------------
# TerminalAgent
# ---------------------------------------------------------------------------

class TerminalAgent:
    """
    Terminale agentico: orchestratore tra LLM, web_search e subprocess.

    Args:
        llm:               Client Ollama (uno nuovo viene creato se None).
        web_searcher:      Istanza già caricata di SearXNGClient. Se None e
                           settings.terminal_agent.enable_web_search è True,
                           viene creata al primo `propose()` che ne ha bisogno.
        model_role:        Override del ModelRole (default: da settings).
        initial_cwd:       Override della cwd iniziale (default: settings → $HOME).
        safe_dirs:         Override delle directory permesse (default: settings
                           → pc_control.safe_dirs).
        allow_sudo:        Override per il permesso sudo (default: settings →
                           pc_control.allow_sudo).

    Priorità per safe_dirs e allow_sudo:
        parametro costruttore > settings.terminal_agent.* > settings.pc_control.*

    Esempio:
        async with TerminalAgent() as ta:
            turn = await ta.run("trovami i file più grandi nella mia home",
                                auto_confirm=False)
            print(turn.proposal.command)
            print(turn.result.stdout)
            print(turn.analysis)
    """

    def __init__(
        self,
        llm:          Optional[OllamaClient] = None,
        web_searcher: Optional[Any]          = None,
        model_role:   Optional[ModelRole]    = None,
        initial_cwd:  Optional[str]          = None,
        safe_dirs:    Optional[list[str]]    = None,
        allow_sudo:   Optional[bool]         = None,
    ) -> None:
        self._cfg = settings.terminal_agent

        self._llm:          Optional[OllamaClient] = llm
        self._owned_llm:    bool                   = llm is None
        self._web_searcher: Optional[Any]          = web_searcher
        self._owned_web:    bool                   = web_searcher is None

        # ModelRole: stringa "chat"/"code" → enum
        if model_role is not None:
            self._model_role = model_role
        else:
            self._model_role = (
                ModelRole.CODE if self._cfg.model_role == "code" else ModelRole.CHAT
            )

        # Safety: priorità parametro → settings.terminal_agent → settings.pc_control
        if allow_sudo is not None:
            self._allow_sudo: bool = allow_sudo
        elif self._cfg.allow_sudo is not None:
            self._allow_sudo = self._cfg.allow_sudo
        else:
            self._allow_sudo = settings.pc_control.allow_sudo

        self._allow_delete: bool = (
            self._cfg.allow_delete
            if self._cfg.allow_delete is not None
            else settings.pc_control.allow_delete
        )

        if safe_dirs is not None:
            self._safe_dirs: list[str] = list(safe_dirs)
        elif self._cfg.safe_dirs is not None:
            self._safe_dirs = list(self._cfg.safe_dirs)
        else:
            self._safe_dirs = list(settings.pc_control.safe_dirs)

        # cwd iniziale
        cwd_choice = initial_cwd or self._cfg.initial_cwd or os.path.expanduser("~")
        self._cwd: str = str(Path(cwd_choice).resolve())

        # Stato sessione: storia degli ultimi turni per dare contesto all'LLM
        self._turn_history: list[AgentTurn] = []

        # Ultima proposta emessa (per conferma asincrona dalla UI)
        self._last_proposal: Optional[CommandProposal] = None

        # Override del modello Ollama. Se settato, ha precedenza su
        # settings.ollama.chat_model/code_model: è il modo per usare un
        # modello specifico (es. uno "thinking") senza toccare le settings
        # globali (che potrebbero essere un modello chat che non supporta
        # think:true, tipo gemma3).
        self._model_override: Optional[str] = None

        # Locale corrente (es. "it_IT.UTF-8"). Determinato all'avvio.
        # Passato nel system prompt per dare contesto al modello.
        self._locale: str = os.environ.get("LANG", "C")

        # Percorsi XDG dell'utente, popolati esternamente via set_xdg_paths().
        # Esempio: {"Scrivania": "/home/mauro/Scrivania",
        #           "Scaricati": "/home/mauro/Scaricati", ...}.
        # Se vuoto, il system prompt non include la sezione "PERCORSI UTENTE".
        self._xdg_paths: dict[str, str] = {}

        # Cache dei modelli che NON supportano think:true (popolata runtime al
        # primo 400 di Ollama). Vedi _chat_with_think_fallback.
        self._models_without_thinking: set[str] = set()

        self._loaded: bool = False

    # -----------------------------------------------------------------------
    # Context manager
    # -----------------------------------------------------------------------

    async def __aenter__(self) -> "TerminalAgent":
        await self.load()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def load(self) -> None:
        if self._llm is None:
            self._llm = OllamaClient()
        self._loaded = True
        logger.info(
            "terminal_agent | inizializzato | model_role={} cwd={} safe_dirs={} sudo={}",
            self._model_role.value,
            self._cwd,
            self._safe_dirs,
            self._allow_sudo,
        )

    async def aclose(self) -> None:
        if self._owned_llm and self._llm is not None:
            await self._llm.aclose()
            self._llm = None
        if self._owned_web and self._web_searcher is not None:
            close = getattr(self._web_searcher, "aclose", None)
            if close:
                try:
                    await close()
                except Exception as exc:
                    logger.debug("terminal_agent | aclose web_searcher: {}", exc)
            self._web_searcher = None
        self._loaded = False

    # -----------------------------------------------------------------------
    # Proprietà / utility
    # -----------------------------------------------------------------------

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def last_proposal(self) -> Optional[CommandProposal]:
        return self._last_proposal

    @property
    def model(self) -> Optional[str]:
        """Modello Ollama corrente (None = usa settings per il ModelRole)."""
        return self._model_override

    def set_model(self, name: Optional[str]) -> None:
        """
        Imposta il modello Ollama da usare per propose() e analyze().
        Passa None per tornare a usare settings.ollama (in base al ModelRole).
        Non valida la disponibilità del modello — il chiamante (es. il
        TerminalBridge) deve averlo già verificato contro /api/tags.
        """
        self._model_override = name or None
        logger.info("terminal_agent | model_override → {}", self._model_override)

    def set_xdg_paths(self, paths: dict[str, str]) -> None:
        """
        Imposta i percorsi XDG dell'utente (dict label → path assoluto).
        Esempio: {"Scrivania": "/home/mauro/Scrivania", ...}.
        Vengono inclusi nel system prompt per evitare che il modello inventi
        percorsi inglesi su locale non-EN.
        """
        self._xdg_paths = dict(paths or {})
        logger.info("terminal_agent | xdg_paths → {} chiavi", len(self._xdg_paths))

    def set_locale(self, locale: str) -> None:
        """Imposta il locale corrente (es. 'it_IT.UTF-8') per il system prompt."""
        self._locale = locale or "C"

    def add_turn_to_history(self, turn: AgentTurn) -> None:
        """
        Aggiunge un turno alla history interna dell'agent.

        Quando si usano propose()/execute()/analyze() separatamente (es. dalla UI
        con conferma asincrona), il chiamante DEVE chiamare questo metodo dopo
        ogni turno completato. Altrimenti la history resta vuota e il modello
        non ha contesto per richieste "correggi il comando precedente".

        TerminalAgent.run() lo fa già internamente — questo metodo serve solo
        per i flussi che bypassano run().
        """
        self._turn_history.append(turn)

    async def _chat_with_think_fallback(
        self,
        messages: list[Message],
        *,
        think: bool,
        system: Optional[str] = None,
    ):
        """
        Wrapper di self._llm.chat() che gestisce il caso di modelli che NON
        supportano l'opzione `think`. Strategia:

        1. Se sappiamo già che il modello corrente non supporta think,
           chiamiamo senza il flag.
        2. Altrimenti proviamo con think=requested. Se Ollama risponde 400,
           lo segniamo nella cache e ritentiamo senza think.
        3. Errori non-400 (timeout, connection, ecc.) li ripropaghiamo.

        Risolve il caso 'gemma3' / 'gemma4' / qualunque modello non-thinking
        che l'utente seleziona dalla UI dopo che abbiamo rimosso il filtro
        per family.
        """
        # Quale modello stiamo davvero usando?
        current = self._model_override  # se None, OllamaClient userà il default
        # Se è già marcato come non-thinking, niente think dall'inizio
        if current and current in self._models_without_thinking:
            return await self._llm.chat(
                messages,
                role=self._model_role,
                model=self._model_override,
                system=system,
                options={"think": False},
            )

        # Primo tentativo: con think come richiesto
        try:
            return await self._llm.chat(
                messages,
                role=self._model_role,
                model=self._model_override,
                system=system,
                options={"think": think},
            )
        except Exception as exc:
            # Tentiamo il fallback senza think se l'errore SEMBRA legato al
            # mancato supporto di think. Cause possibili:
            #   - 400 Bad Request (alcuni modelli)
            #   - 500 Internal Server Error con body '"... does not support thinking"'
            #     (es. gemma3 su Ollama 0.23.x: status 500 + JSON error)
            # Su 500 generici (es. OOM, runner crash) il retry senza think
            # quasi certamente fallirà uguale, ma è safe ritentare comunque:
            # se è davvero OOM, il secondo tentativo ridarà lo stesso errore
            # e lo propaghiamo all'utente.
            msg = str(exc).lower()
            looks_like_think_issue = (
                "400" in msg
                or "500" in msg
                or "bad request" in msg
                or "internal server error" in msg
                or "does not support thinking" in msg
                or "thinking is not supported" in msg
            )
            if not (think and looks_like_think_issue):
                raise  # qualunque altro errore (timeout, network, ecc.) va propagato

            logger.info(
                "terminal_agent | chat fallita con think=true ({}), "
                "ritento senza think | modello={}",
                type(exc).__name__, current or "(default)",
            )
            # Ritenta senza il flag think
            try:
                result = await self._llm.chat(
                    messages,
                    role=self._model_role,
                    model=self._model_override,
                    system=system,
                    options={"think": False},
                )
            except Exception as exc2:
                # Anche il retry è fallito: il problema NON era think.
                # Propagiamo l'errore originale (più informativo).
                logger.warning(
                    "terminal_agent | anche il retry senza think è fallito: {} "
                    "— propago l'errore originale", exc2,
                )
                raise exc
            # Retry riuscito: cache il modello come non-thinking per evitare
            # di rifare il primo tentativo inutilmente in futuro.
            if current:
                self._models_without_thinking.add(current)
                logger.info(
                    "terminal_agent | '{}' marcato come non-thinking",
                    current,
                )
            return result

    def reset(self) -> None:
        """Resetta cwd, history dei turni e ultima proposta."""
        self._cwd = str(Path(self._cfg.initial_cwd or os.path.expanduser("~")).resolve())
        self._turn_history.clear()
        self._last_proposal = None
        logger.info("terminal_agent | reset")

    # -----------------------------------------------------------------------
    # API: propose / execute / analyze / run
    # -----------------------------------------------------------------------

    async def propose(self, user_request: str) -> CommandProposal:
        """
        Loop ReAct con l'LLM. Ritorna una CommandProposal classificata.

        Raises:
            RuntimeError: se l'LLM non produce JSON parsabile entro max_iterations.
        """
        self._require_loaded()
        if not user_request.strip():
            raise ValueError("user_request vuoto")

        t0 = time.monotonic()

        # Costruisci il blocco "PERCORSI UTENTE" se abbiamo i path XDG.
        # Lo includiamo nel system prompt come istruzione esplicita —
        # vincere il bias del modello che propone sempre "Desktop"/"Downloads"
        # anche su locale italiano.
        if self._xdg_paths:
            xdg_lines = ["PERCORSI UTENTE (usa SEMPRE questi, non i nomi inglesi):"]
            for k, v in self._xdg_paths.items():
                if v:
                    xdg_lines.append(f"- {k}: {v}")
            xdg_block = "\n".join(xdg_lines) + "\n\n"
        else:
            xdg_block = ""

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            cwd=self._cwd,
            locale=self._locale,
            xdg_paths_block=xdg_block,
            safe_dirs=", ".join(self._safe_dirs),
            allow_sudo=("sì" if self._allow_sudo else "no"),
        )

        messages: list[Message] = []

        # Storia: ultimi 2 turni completati, come contesto utente→assistant.
        # Passiamo un summary RICCO: comando, exit_code, stdout E stderr troncati,
        # analisi. Serve perché l'utente può dire "correggi il comando precedente"
        # e il modello deve avere abbastanza contesto per capire cosa correggere.
        for past in self._turn_history[-2:]:
            messages.append(Message(role=Role.USER, content=past.user_request))
            if past.proposal and past.result:
                parts = [
                    f"[esecuzione precedente — exit={past.result.exit_code}]",
                    f"$ {past.proposal.command}",
                ]
                if past.result.stdout:
                    parts.append(f"stdout: {past.result.stdout[:400]}")
                if past.result.stderr:
                    parts.append(f"stderr: {past.result.stderr[:400]}")
                if past.analysis:
                    parts.append(f"analisi: {past.analysis[:300]}")
                summary = "\n".join(parts)
                messages.append(Message(role=Role.ASSISTANT, content=summary))
            elif past.proposal and past.skipped:
                messages.append(Message(
                    role=Role.ASSISTANT,
                    content=f"[proposta non eseguita — annullata o in attesa]\n"
                            f"$ {past.proposal.command}",
                ))

        # Turno corrente
        messages.append(Message(role=Role.USER, content=user_request))

        sources: list[str] = []
        search_used = False

        for iteration in range(self._cfg.max_iterations):
            response = await self._chat_with_think_fallback(
                messages,
                think=self._cfg.use_thinking_propose,
                system=system_prompt,
            )
            raw = response.content
            parsed = _extract_json(raw)

            if parsed is None:
                logger.warning(
                    "terminal_agent.propose | iter={} JSON non parsabile, retry. raw={!r}",
                    iteration, raw[:200],
                )
                # feedback al modello per il retry
                messages.append(Message(role=Role.ASSISTANT, content=raw))
                messages.append(Message(
                    role=Role.USER,
                    content='Non hai risposto in JSON valido. Riprova con SOLO un oggetto JSON '
                            'della forma {"action": "search"|"propose", ...}. '
                            'Niente testo fuori dalle parentesi graffe.',
                ))
                continue

            action = parsed.get("action")

            if action == "search":
                if not self._cfg.enable_web_search:
                    # disattivato → forziamo la proposta
                    messages.append(Message(role=Role.ASSISTANT, content=raw))
                    messages.append(Message(
                        role=Role.USER,
                        content="La ricerca web è disabilitata. Proponi il comando "
                                "con le tue conoscenze, action='propose'.",
                    ))
                    continue

                query = (parsed.get("query") or "").strip()
                if not query:
                    messages.append(Message(role=Role.ASSISTANT, content=raw))
                    messages.append(Message(
                        role=Role.USER,
                        content='Hai chiesto una ricerca ma il campo "query" è vuoto. '
                                'Riprova con una query non vuota oppure passa a action="propose".',
                    ))
                    continue

                search_used = True
                results_block, urls = await self._do_search(query)
                sources.extend(urls)

                messages.append(Message(role=Role.ASSISTANT, content=raw))
                messages.append(Message(
                    role=Role.USER,
                    content=f"Risultati di ricerca per '{query}':\n{results_block}\n\n"
                            f"Ora puoi fare un'altra ricerca oppure proporre il comando.",
                ))
                continue

            if action == "propose":
                command   = (parsed.get("command")   or "").strip()
                rationale = (parsed.get("rationale") or "").strip()
                llm_srcs  = parsed.get("sources") or []
                if isinstance(llm_srcs, list):
                    for s in llm_srcs:
                        if isinstance(s, str) and s and s not in sources:
                            sources.append(s)

                if not command:
                    messages.append(Message(role=Role.ASSISTANT, content=raw))
                    messages.append(Message(
                        role=Role.USER,
                        content='Il campo "command" è vuoto. Riprova.',
                    ))
                    continue

                # classifica deterministicamente
                risk = _classify(command, allow_sudo=self._allow_sudo)
                needs_conf = risk in (RiskLevel.MODERATE, RiskLevel.DANGEROUS)

                proposal = CommandProposal(
                    command=command,
                    rationale=rationale,
                    risk_level=risk,
                    needs_confirmation=needs_conf,
                    cwd=self._cwd,
                    search_used=search_used,
                    sources=sources,
                )
                self._last_proposal = proposal
                logger.info(
                    "terminal_agent.propose | {} | elapsed={:.0f}ms",
                    proposal.to_log_dict(),
                    (time.monotonic() - t0) * 1000,
                )
                return proposal

            # action sconosciuta
            messages.append(Message(role=Role.ASSISTANT, content=raw))
            messages.append(Message(
                role=Role.USER,
                content=f'Action "{action}" non valida. Usa "search" o "propose".',
            ))

        raise RuntimeError(
            f"terminal_agent: LLM non ha prodotto una proposta valida in "
            f"{self._cfg.max_iterations} iterazioni"
        )

    async def execute(
        self,
        proposal: CommandProposal,
        *,
        confirmed: bool = False,
    ) -> CommandResult:
        """
        Esegue il comando proposto. Gestisce confirmation, blocking,
        timeout, troncamento output, tracking della cwd.

        Args:
            proposal:  La proposta da `propose()`.
            confirmed: True se l'utente ha approvato (richiesto per
                       proposal.needs_confirmation=True).

        Raises:
            ValueError: se il comando è BLOCKED o richiede conferma e
                        confirmed=False.
        """
        self._require_loaded()

        if proposal.risk_level == RiskLevel.BLOCKED:
            raise ValueError(
                f"Comando rifiutato (BLOCKED): {proposal.command!r}. "
                f"Pattern pericoloso o sudo non permesso."
            )

        if proposal.needs_confirmation and not confirmed:
            raise ValueError(
                f"Comando '{proposal.risk_level.value}' richiede conferma esplicita. "
                f"Passa confirmed=True per eseguirlo."
            )

        # safety: cwd deve essere dentro safe_dirs
        if not _is_inside(self._cwd, self._safe_dirs):
            raise ValueError(
                f"cwd corrente ({self._cwd}) non è dentro safe_dirs ({self._safe_dirs}). "
                f"Usa reset() o cambia configurazione."
            )

        cwd_before = self._cwd
        wrapped    = _wrap_command_with_cwd_tracking(proposal.command)

        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", wrapped,
                cwd=cwd_before,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )
        except FileNotFoundError as exc:
            return CommandResult(
                command=proposal.command, exit_code=-1,
                stdout="", stderr=f"bash non trovato: {exc}",
                duration_ms=(time.monotonic() - t0) * 1000,
                cwd_before=cwd_before, cwd_after=cwd_before,
                proposal_id=proposal.proposal_id,
            )

        timed_out = False
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=self._cfg.command_timeout,
            )
        except asyncio.TimeoutError:
            timed_out = True
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            stdout_b, stderr_b = b"", b""

        duration_ms = (time.monotonic() - t0) * 1000

        stdout_raw = stdout_b.decode("utf-8", errors="replace")
        stderr_raw = stderr_b.decode("utf-8", errors="replace")

        # estrai cwd finale dal sentinel
        stdout_clean, new_cwd = _extract_cwd(stdout_raw, fallback=cwd_before)

        # verifica che la nuova cwd sia ancora dentro safe_dirs
        if new_cwd != cwd_before:
            if _is_inside(new_cwd, self._safe_dirs):
                self._cwd = new_cwd
            else:
                logger.warning(
                    "terminal_agent.execute | cwd '{}' fuori safe_dirs, mantengo '{}'",
                    new_cwd, cwd_before,
                )
                new_cwd = cwd_before

        # tronca output
        stdout_out, t_out = _truncate(stdout_clean, self._cfg.max_output_bytes)
        stderr_out, t_err = _truncate(stderr_raw,   self._cfg.max_output_bytes)

        exit_code = -1 if timed_out else (proc.returncode or 0)
        if timed_out:
            stderr_out = (
                stderr_out + f"\n[terminal_agent] timeout dopo "
                f"{self._cfg.command_timeout}s"
            ).strip()

        result = CommandResult(
            command=proposal.command,
            exit_code=exit_code,
            stdout=stdout_out,
            stderr=stderr_out,
            duration_ms=duration_ms,
            cwd_before=cwd_before,
            cwd_after=new_cwd,
            truncated=(t_out or t_err),
            proposal_id=proposal.proposal_id,
        )
        logger.info("terminal_agent.execute | {}", result.to_log_dict())
        return result

    async def analyze(
        self,
        result: CommandResult,
        *,
        user_request: str = "",
    ) -> str:
        """
        Fa leggere stdout/stderr all'LLM e produce diagnosi + suggerimenti.
        Non solleva mai: in caso di errore restituisce stringa con messaggio.
        """
        self._require_loaded()
        request = user_request or "(richiesta originale non disponibile)"

        prompt = _ANALYSIS_PROMPT_TEMPLATE.format(
            request=request,
            command=result.command,
            cwd_before=result.cwd_before,
            cwd_after=result.cwd_after,
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
            stdout=(result.stdout or "(vuoto)"),
            stderr=(result.stderr or "(vuoto)"),
            stdout_truncated_note=(" [TRONCATO]" if result.truncated else ""),
        )

        try:
            response = await self._chat_with_think_fallback(
                [Message(role=Role.USER, content=prompt)],
                think=self._cfg.use_thinking_analyze,
            )
            return response.content.strip()
        except Exception as exc:
            logger.warning("terminal_agent.analyze | LLM fallito: {}", exc)
            return f"[analisi non disponibile: {exc}]"

    async def run(
        self,
        user_request: str,
        *,
        auto_confirm: bool = False,
    ) -> AgentTurn:
        """
        Esegue un turno completo: propose → (execute) → analyze.

        Se la proposta richiede conferma e auto_confirm=False, ritorna un
        AgentTurn con `skipped=True` e nessun result/analysis (l'utente
        deve confermare manualmente chiamando execute()+analyze()).
        """
        self._require_loaded()
        turn = AgentTurn(user_request=user_request)
        try:
            proposal = await self.propose(user_request)
            turn.proposal = proposal
        except Exception as exc:
            turn.error = f"propose: {exc}"
            logger.warning("terminal_agent.run | propose fallito: {}", exc)
            return turn

        if proposal.needs_confirmation and not auto_confirm:
            turn.skipped = True
            self._turn_history.append(turn)
            return turn

        try:
            result = await self.execute(proposal, confirmed=auto_confirm)
            turn.result = result
        except Exception as exc:
            turn.error = f"execute: {exc}"
            logger.warning("terminal_agent.run | execute fallito: {}", exc)
            self._turn_history.append(turn)
            return turn

        try:
            turn.analysis = await self.analyze(result, user_request=user_request)
        except Exception as exc:
            turn.error = f"analyze: {exc}"
            logger.warning("terminal_agent.run | analyze fallito: {}", exc)

        self._turn_history.append(turn)
        logger.info("terminal_agent.run | {}", turn.to_log_dict())
        return turn

    # -----------------------------------------------------------------------
    # Interno
    # -----------------------------------------------------------------------

    async def _do_search(self, query: str) -> tuple[str, list[str]]:
        """
        Esegue una ricerca web e ritorna (blocco_testuale, lista_url).
        Lazy-load del client web al primo uso.
        """
        if self._web_searcher is None:
            try:
                # import lazy: web_search ha dipendenze pesanti (httpx)
                from modules.web_search import SearXNGClient  # type: ignore
                self._web_searcher = SearXNGClient()
            except Exception as exc:
                logger.warning(
                    "terminal_agent | impossibile creare SearXNGClient: {} — "
                    "ricerca skippata", exc,
                )
                return "[ricerca web non disponibile]", []

        try:
            response = await self._web_searcher.search(
                query, max_results=self._cfg.max_search_results,
            )
        except Exception as exc:
            logger.warning("terminal_agent | search fallita: {}", exc)
            return f"[ricerca fallita: {exc}]", []

        # response è SearchResponse: .results è lista di SearchResult
        # (con .title, .url, .snippet). Se mai venisse passato un mock che
        # ritorna lista direttamente, _get() qui sotto funziona comunque.
        results = _get(response, "results", response)
        err     = _get(response, "error",   None)

        if err:
            logger.debug("terminal_agent | search ritorna errore: {}", err)
            return f"[ricerca fallita: {err}]", []

        lines: list[str] = []
        urls:  list[str] = []
        for i, r in enumerate(results, 1):
            title   = _get(r, "title",   "(senza titolo)")
            url     = _get(r, "url",     "")
            snippet = _get(r, "snippet", "") or _get(r, "content", "")
            snippet = (snippet or "")[:400]
            lines.append(f"[{i}] {title}\n    {url}\n    {snippet}")
            if url:
                urls.append(url)

        return ("\n".join(lines) if lines else "[nessun risultato]"), urls

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "TerminalAgent non inizializzato — usa 'async with TerminalAgent()' "
                "oppure 'await ta.load()'"
            )

    def __repr__(self) -> str:
        if self._loaded:
            return (
                f"<TerminalAgent cwd='{self._cwd}' role={self._model_role.value} "
                f"sudo={'on' if self._allow_sudo else 'off'} "
                f"history={len(self._turn_history)}>"
            )
        return "<TerminalAgent [non caricato]>"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Accesso uniforme a key su dict o attr su dataclass."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
