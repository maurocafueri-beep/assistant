SHELL := /bin/bash
RUNTIME_VENV := venv-runtime
TRAINING_VENV := venv-training
PYTHON := $(RUNTIME_VENV)/bin/python
PIP    := $(RUNTIME_VENV)/bin/pip
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo ""
	@echo "  Assistente AI Locale"
	@echo "  ──────────────────────────────────────────"
	@echo "  make start          Avvia l'app nativa (Qt)"
	@echo "  make start-ui       Solo UI web + server (dev, no finestra)"
	@echo "  make start-voice    Loop vocale CLI (no UI)"
	@echo "  make stop           Ferma tutto"
	@echo "  make check          Verifica sistema"
	@echo "  make status         Stato servizi"
	@echo "  make logs           Log in tempo reale"
	@echo "  make pull-models    Scarica modelli Ollama"
	@echo "  make train          Avvia fine-tuning"
	@echo "  make test           Esegui test"
	@echo "  make clean          Rimuovi cache"
	@echo ""

.PHONY: start
start:
	$(PYTHON) scripts/run_app.py

.PHONY: start-ui
start-ui:
	$(PYTHON) scripts/run_ui.py

.PHONY: start-voice
start-voice:
	$(PYTHON) scripts/run_voice.py

.PHONY: stop
stop:
	@pkill -f "scripts/run_app.py" 2>/dev/null || true
	@pkill -f "scripts/run_ui.py" 2>/dev/null || true
	@pkill -f "scripts/run_voice.py" 2>/dev/null || true
	@cd docker && docker compose stop

.PHONY: check
check:
	$(PYTHON) scripts/check_health.py

.PHONY: status
status:
	@systemctl is-active ollama && ollama list || echo "⚠ Ollama non attivo"
	@cd docker && docker compose ps

.PHONY: logs
logs:
	@tail -f logs/assistant.log

.PHONY: pull-models
pull-models:
	ollama pull qwen3:14b-q8_0
	ollama pull qwen3:30b-a3b-q4_K_M
	ollama pull qwen3-vl:8b
	ollama pull nomic-embed-text

.PHONY: train
train:
	$(TRAINING_VENV)/bin/python training/lora_trainer.py

.PHONY: test
test:
	$(RUNTIME_VENV)/bin/pytest tests/ -v --tb=short

.PHONY: clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	@echo "✓ Pulizia completata"
