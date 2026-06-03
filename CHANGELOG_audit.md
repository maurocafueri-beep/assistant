# Audit integrità strutturale — pass 2 (2026-05-31)

Secondo giro di analisi su ridondanze ed errori di progettazione. La base
resta solida (zero import circolari, 936 test verdi); i problemi trovati erano
concentrati su packaging, tooling e file di supporto.

## Bug

- **`check_health.py` non rilevava mai i fallimenti.** Il riepilogo finale
  spacchettava le tuple a 5 campi con `for *_, s, _ in results`, assegnando a
  `s` il campo *detail* invece di *status*. Risultato: `ok_n`/`fail_n`/`warn_n`
  sempre a 0 e `sys.exit(1)` mai raggiunto — `make check` usciva 0 anche con
  dipendenze mancanti. Corretto in `for _c, _n, s, _d, _col in results`.
- **`pyproject.toml` build-backend inesistente.** Era
  `setuptools.backends.legacy:build` (modulo che non esiste →
  `ModuleNotFoundError` su qualunque build PEP 517). Allineato a
  `setuptools.build_meta`.

## Riferimenti morti

- **`pyproject.toml` packages.find:** rimosso `api*` (la cartella `api/` era
  già stata eliminata nel pass 1) e aggiunto `ui*` (package reale con
  `__init__.py`, prima escluso dal packaging).
- **`make train`** puntava a `training/lora_trainer.py`, inesistente — stessa
  classe del target morto `enroll` chiuso nel pass 1. Reso robusto: ora
  verifica la presenza dello script e altrimenti stampa una nota di roadmap.
- **Import morto** `compute_file_id` da `modules.file_rag` in
  `core/orchestrator.py` (mai usato): rimosso.
- **`check_health.py`** suggeriva `make setup-docker` (target inesistente) per
  SearXNG/Open WebUI: corretto in `cd docker && docker compose up -d`.
- **`struttura.txt`** non elencava `config/personalities/Aurora.yml`: aggiunto,
  data e conteggio file aggiornati.

## Refactor — sequenza di avvio condivisa

- **Logica di avvio duplicata e divergente** tra `ui/app.py` (`_AsyncWorker`)
  e `scripts/run_ui.py`: entrambi avviavano uvicorn + `UIBridge`, ma
  `run_ui.py` era una copia degradata che *non* faceva ripristino sessioni da
  disco, restore di `ui-settings.json`, setup del `TerminalBridge`,
  `broadcast_init()` né cleanup uploads. Estratta l'intera sequenza in
  `ui/bootstrap.py::serve(*, personality, ptt_key, on_server_ready)`, unica
  fonte di verità. I due entry-point ora vi delegano e differiscono solo per
  l'hook `on_server_ready`: l'app Qt emette il segnale `ready` (carica la
  webview), il launcher headless apre il browser. Rimosso anche l'import morto
  `httpx` da `ui/app.py`.

## Verifica

936 test verdi (31 slow skippati), `py_compile` pulito su
`ui/app.py`/`scripts/run_ui.py`/`ui/bootstrap.py`, build-backend,
`core.orchestrator` e `ui.bootstrap` importabili.

- **`tts_studio.sh` — `LD_LIBRARY_PATH` ridondante rimosso.** Lo script
  esportava le lib CUDA di `venv-tts` e poi lanciava `venv-runtime/bin/python`,
  dando l'impressione di un mismatch di venv. Verificato che non lo è:
  `tools/tts_studio.py` è un orchestratore web leggero (fastapi/httpx/uvicorn)
  che gira correttamente in `venv-runtime` e non carica CUDA; il vero server
  TTS è lanciato come sottoprocesso con `venv-tts/bin/python`, e quel codice
  (`_ensure_tts_server`) imposta già `LD_LIBRARY_PATH` sulle lib CUDA di
  `venv-tts` nell'`env` del figlio, in modo robusto (`if nvidia.exists()`).
  L'export nello script era quindi morto: rimosso e sostituito con un commento
  esplicativo.

---

# Audit integrità strutturale — 8 fix

**Commit:** `0821834` — "Audit integrità: 8 fix strutturali e di progettazione"
**Data:** 2026-05-24

Analisi completa del progetto (74 file Python) e correzione di 8 problemi di
progettazione/coerenza. La struttura di base è risultata solida: nessun import
circolare, cleanup delle risorse corretto, degradazione non fatale ben gestita
(TTS/memoria/web search down → l'app continua senza crashare).

## Alta priorità

- **History desync al riavvio.** L'orchestratore non ricostruiva la finestra
  conversazionale (`_session_histories`) dai messaggi salvati su disco, quindi
  dopo un riavvio l'LLM "dimenticava" i turni precedenti pur mostrandoli in UI.
  Aggiunto `UIBridge.restore_histories_from_sessions()`, invocato in `app.py`
  dopo il load dell'orchestratore.
- **Default modelli fittizi** in `settings.py` (`qwen3.5:9b-q8_0` /
  `qwen3.6:35b-a3b`, inesistenti come tag Ollama): un clone senza `.env`
  falliva ogni turno LLM. Allineati ai tag reali (`qwen3:14b-q8_0`,
  `qwen3:30b-a3b-q4_K_M`); commenti corretti in `base_llm.py`.

## Media priorità

- **Monkey-patching rimosso.** `orch.turn` e `tts.synthesize` non vengono più
  sostituiti a runtime per la strumentazione dei tempi: le latenze sono misurate
  nel loop base e i chunk passano per il nuovo hook `_on_llm_chunk()`.
- **`_emit` task-safe.** I task fire-and-forget di broadcast vengono ora
  referenziati (`_bg_tasks`) per evitare il garbage-collection prematuro.
- **Config morta rimossa.** `MemorySettings.embedding_model` non era mai usato
  (l'embedding è `nomic-embed-text` via Ollama).
- **`switch_model` senza mutazione globale.** Il modello attivo vive in
  `Orchestrator._model_overrides` e viene passato a `stream(model=...)` invece
  di mutare il singleton `settings.ollama.chat_model`; `ctx.model_name` ora
  viene popolato.

## Bassa priorità

- Rimosso lo scaffolding vuoto `api/` e corretti i target morti del **Makefile**
  (`start` → `run_app.py`, nuovi `start-ui` / `start-voice`, `stop` con i
  pattern di processo reali, rimosso `enroll` che puntava a uno script
  inesistente). `struttura.txt` rigenerato e datato.
- Chiusa la **race d'avvio Qt**: nuovo `UIBridge.broadcast_init()` invocato a
  loop pronto, così i client connessi mentre il backend non era ancora
  inizializzato ricevono l'`init` completo senza dover riconnettere.

## Verifica

238 test verdi (125 core + 80 personality + 33 terminal_bridge), compilazione
pulita di tutti i moduli, import sani. Zero regressioni.

## File toccati

`Makefile`, `config/settings.py`, `core/orchestrator.py`, `core/voice_loop.py`,
`modules/llm/base_llm.py`, `struttura.txt`, `ui/app.py`, `ui/bridge.py`;
rimossa la cartella `api/`.
