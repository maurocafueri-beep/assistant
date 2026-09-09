# local-assistant

Assistente AI personale che gira **interamente in locale**: nessuna chiamata a servizi cloud, nessuna chiave API, nessun dato che esce dalla macchina. Parla e ascolta, ricorda le conversazioni, legge i documenti che gli passi, cerca sul web e — se glielo permetti — esegue comandi sul sistema.

Nasce come progetto personale per capire fino a che punto si può portare un assistente vocale senza appoggiarsi a servizi esterni.

> **Stato del progetto:** funzionante e in uso quotidiano, ma pensato per la mia macchina. Sviluppato prima su Ubuntu, oggi gira su **Arch Linux**. Non è stato provato su macOS né su Windows.

---

## Cosa fa

- **Chat testuale e vocale** — parlato riconosciuto con Whisper (`faster-whisper`), risposta sintetizzata in voce, attivazione con wake word oppure con un tasto push-to-talk.
- **Memoria delle conversazioni** — le sessioni sono persistenti e l'assistente recupera il contesto passato quando serve.
- **Analisi dei documenti (RAG)** — carichi un file e puoi farci domande sopra: il contenuto viene indicizzato in un database vettoriale e recuperato per somiglianza semantica. I documenti lunghi passano per un riassunto map-reduce.
- **Ricerca web** — tramite un'istanza locale di SearXNG, quindi anche le ricerche restano in casa.
- **Riconoscimento del parlante** — distingue chi sta parlando tra le voci registrate.
- **Personalità configurabili** — profili diversi (tono, stile, istruzioni di sistema) selezionabili a caldo.
- **Controllo del PC e agente da terminale** — esecuzione di comandi con conferma esplicita e vincoli configurabili (cartelle consentite, divieto di `sudo` e di cancellazione).

---

## Architettura

Il cuore è un **orchestratore asincrono** (`core/orchestrator.py`) che compone ogni turno di conversazione chiamando i moduli che servono, e solo quelli:

```
        INPUT (voce | testo)
              │
              ▼
        STT  (se l'input è vocale)
              │
              ▼
        Memoria ──► recupero del contesto rilevante
              │
              ▼
        Personalità ──► istruzioni di sistema
              │
              ├──► Analisi file / RAG   (se è stato rilevato un documento)
              ├──► Ricerca web          (se la richiesta la richiede)
              └──► Vision               (se serve guardare lo schermo)
              │
              ▼
        Costruzione del prompt ──► LLM via Ollama (streaming)
              │
              ├──► salvataggio in memoria
              └──► TTS  (se l'output è vocale)
              │
              ▼
        OUTPUT (testo e/o audio)
```

Ogni funzionalità vive in un modulo indipendente sotto `modules/`, con una classe base che ne definisce l'interfaccia. L'orchestratore conosce le interfacce, non le implementazioni: si aggiunge o si sostituisce un modulo senza toccare il resto.

```
core/          orchestratore, contesto del turno, loop vocale, logging
modules/       llm · stt · tts · wake_word · speaker_id · memory · file_rag
               file_analysis · map_reduce · web_search · intent · personality
               pc_control · terminal_agent · gaming
ui/            server FastAPI (REST + WebSocket), app nativa Qt, bridge
config/        impostazioni tipizzate e profili di personalità
tests/         suite pytest, un modulo per componente
scripts/       avvio, health check, setup di sistema
docker/        SearXNG per la ricerca web
```

Il backend è **FastAPI**: oltre 30 endpoint REST per sessioni, upload, modelli, voci e impostazioni, più un canale **WebSocket** su cui la risposta arriva in streaming token per token. L'interfaccia è un'app nativa Qt Quick che parla con lo stesso backend, quindi la UI è sostituibile senza toccare la logica.

---

## Requisiti

| | |
|---|---|
| Sistema | Linux (sviluppato su Ubuntu, oggi su Arch). Non testato su macOS/Windows |
| Python | 3.11+ |
| [Ollama](https://ollama.com) | **da installare a parte** — è il motore che esegue i modelli |
| GPU | consigliata NVIDIA con CUDA; senza GPU funziona ma le risposte sono lente |
| Docker | solo se vuoi la ricerca web (SearXNG) |
| Audio | PortAudio, ffmpeg, sox, libsndfile |

---

## Installazione

**Su Arch Linux** c'è uno script che fa tutto — pacchetti di sistema, Ollama, modelli, ambienti virtuali, gruppi utente, Docker — ed è idempotente, quindi si può rilanciare senza danni:

```bash
git clone https://github.com/maurocafueri-beep/assistant.git
cd assistant
bash scripts/setup_arch.sh                    # interattivo
NVIDIA_INSTALL=1 bash scripts/setup_arch.sh   # installa anche driver e CUDA
```

**Su altre distribuzioni**, a mano:

```bash
# 1. Ollama (vedi ollama.com per la tua distro), poi i modelli
ollama pull nomic-embed-text
ollama pull qwen3-vl:8b

# 2. Ambiente Python
python -m venv venv-runtime
venv-runtime/bin/pip install -r requirements-runtime.txt

# 3. Configurazione
cp .env.example .env      # e adatta i modelli alle tue risorse
```

I modelli indicati in `.env.example` sono tarati sulla mia macchina: se hai meno VRAM, sostituiscili con varianti più piccole prima del primo avvio.

---

## Avvio

Tutto passa dal `Makefile`:

```bash
make start          # app nativa Qt (modalità normale)
make start-ui       # solo backend + UI web, utile in sviluppo
make start-voice    # loop vocale da terminale, senza interfaccia
make check          # health check: Ollama, modelli, audio, dipendenze
make status         # stato dei servizi
make logs           # log in tempo reale
make stop           # ferma tutto
```

Se qualcosa non parte, `make check` è il primo posto da guardare.

---

## Test

```bash
make test                          # tutta la suite
venv-runtime/bin/pytest tests/test_orchestrator.py -v   # un singolo modulo
```

I test che hanno bisogno di servizi reali (Ollama attivo, server TTS, SearXNG) sono marcati e si **auto-escludono** se il servizio non risponde: la suite gira anche su una macchina spoglia senza fallire per motivi ambientali.

```bash
SKIP_SLOW=1 make test              # salta anche i test lenti
```

---

## Configurazione

Tutto sta in `.env` (parti da `.env.example`) ed è validato all'avvio con Pydantic Settings, così un valore sbagliato viene segnalato subito invece di esplodere a metà esecuzione. Le voci principali:

| Variabile | A cosa serve |
|---|---|
| `OLLAMA_CHAT_MODEL` | modello usato per la conversazione |
| `OLLAMA_EMBED_MODEL` | modello per gli embedding del RAG |
| `STT_MODEL`, `STT_LANGUAGE` | riconoscimento vocale |
| `TTS_ENGINE`, `TTS_VOICE` | sintesi vocale |
| `WAKE_WORD` | parola di attivazione |
| `PC_CONTROL_ALLOW_SUDO`, `PC_CONTROL_SAFE_DIRS` | limiti del controllo di sistema |
| `API_HOST`, `API_PORT` | indirizzo del backend |

Il file `.env` non è tracciato da git e non deve esserlo: usa `.env.example` come riferimento.

---

## Note

Il progetto è nato per uso personale e ne porta i segni: la configurazione è tarata sul mio hardware e alcune scelte (Qt, Arch, modelli specifici) riflettono la mia macchina più che una ricerca di portabilità. Parte del codice è stata scritta con l'aiuto di strumenti di AI, con progettazione, integrazione e test a mio carico.

Idee, segnalazioni e critiche sono benvenute: apri pure una issue.
