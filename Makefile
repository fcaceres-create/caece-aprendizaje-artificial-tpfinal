# Atajos para macOS/Linux. En Windows usar directamente los comandos `python -m ...`.
PY ?= .venv/bin/python

.PHONY: all setup data eda features train evaluate report api ui test

all:
	$(PY) -m bank_captacion.pipeline

setup:
	python3 -m venv .venv && $(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

data:
	$(PY) -m bank_captacion.pipeline --stage data

eda:
	$(PY) -m bank_captacion.pipeline --stage eda

features:
	$(PY) -m bank_captacion.pipeline --stage features

train:
	$(PY) -m bank_captacion.pipeline --stage train

evaluate:
	$(PY) -m bank_captacion.pipeline --stage evaluate

report:
	$(PY) -m bank_captacion.pipeline --stage report

api:
	$(PY) -m uvicorn app.api.main:app --port 8000

ui:
	$(PY) -m streamlit run app/ui/streamlit_app.py

test:
	$(PY) -m pytest -q
