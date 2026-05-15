SHELL := /bin/bash
.PHONY: run train install

run:
	TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=2 venv312/bin/python app.py

train:
	TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=2 venv312/bin/python train.py

install:
	/home/iben/.pyenv/versions/3.12.13/bin/python -m venv venv312 && venv312/bin/pip install -r requirements.txt
