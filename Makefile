.PHONY: install install-dev data synth prepare vae gan snn calibrate evaluate benchmark pipeline smoke test api dashboard docker lint clean

CONFIG ?= configs/default.yaml
NIDS = nids --config $(CONFIG)

install:
	pip install -e .

install-dev:
	pip install -e ".[serve,track,dev]"

# --- full pipeline on real CICIDS2017 (put the CSVs in data/raw/cicids2017 first)
prepare:
	$(NIDS) prepare
vae:
	$(NIDS) train-vae
gan:
	$(NIDS) train-gan
snn:
	$(NIDS) train-snn
calibrate:
	$(NIDS) calibrate
evaluate:
	$(NIDS) evaluate
benchmark:
	$(NIDS) benchmark
pipeline: prepare vae gan snn calibrate evaluate

# --- 2-minute CPU smoke test on synthetic data (metrics are NOT a benchmark)
smoke:
	nids --config configs/smoke.yaml run-all --synthetic

test:
	pytest

api:
	uvicorn nids.serving.api:app --reload --port 8000
dashboard:
	streamlit run src/nids/dashboard/app.py

docker:
	docker compose up --build

lint:
	ruff check src tests

clean:
	rm -rf artifacts_smoke reports_smoke data/synthetic .pytest_cache
