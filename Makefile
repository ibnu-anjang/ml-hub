SHELL := /bin/bash
VENV := faceid/venv312
PY   := $(VENV)/bin/python

.PHONY: web cloudlens-train cloudlens-app help

help:
	@echo "Targets:"
	@echo "  make web              - jalankan ML Hub web (http://localhost:8000)"
	@echo "  make cloudlens-train  - training cloudlens model"
	@echo "  make cloudlens-app    - cloudlens CLI inference"

web:
	@echo "ML Hub → http://localhost:8000"
	$(VENV)/bin/uvicorn web.main:app --reload --host 0.0.0.0 --port 8000

cloudlens-train:
	cd cloudlens && make train

cloudlens-app:
	cd cloudlens && $(PY) app.py $(ARGS)
