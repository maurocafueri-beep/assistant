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
