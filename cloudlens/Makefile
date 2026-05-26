SHELL := /bin/bash
PY := venv312/bin/python
PIP := venv312/bin/pip

.PHONY: install dataset train run web clean

install:
	/home/iben/.pyenv/versions/3.12.13/bin/python -m venv venv312
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

dataset:
	$(PY) prepare_data.py

train:
	@echo "Running with low priority (nice 15) + 4 thread cap"
	nice -n 15 $(PY) train.py

run:
	$(PY) app.py $(ARGS)

web:
	venv312/bin/uvicorn web.main:app --reload --host 0.0.0.0 --port 8000

clean:
	rm -rf __pycache__ */__pycache__
