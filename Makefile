PYTHON ?= python
CONFIG ?= configs/m0_smoke.json

.PHONY: env lint test smoke-data import-mobile-actions smoke-model manifest-smoke train-donor train-recipient-baselines cache-activations rank-layers eval-layer-candidate eval-main fit-same-size-transplant param-budget summarize-baselines donor-gap-gate

env:
	$(PYTHON) scripts/task.py env

lint:
	$(PYTHON) scripts/task.py lint

test:
	$(PYTHON) scripts/task.py test

smoke-data:
	$(PYTHON) scripts/task.py smoke-data --config $(CONFIG)

import-mobile-actions:
	$(PYTHON) scripts/task.py import-mobile-actions --config $(CONFIG)

smoke-model:
	$(PYTHON) scripts/task.py smoke-model --config $(CONFIG)

manifest-smoke:
	$(PYTHON) scripts/task.py manifest-smoke --config $(CONFIG)

train-donor:
	$(PYTHON) scripts/task.py train-donor --config $(CONFIG)

train-recipient-baselines:
	$(PYTHON) scripts/task.py train-recipient-baselines --config $(CONFIG)

cache-activations:
	$(PYTHON) scripts/task.py cache-activations --config $(CONFIG)

rank-layers:
	$(PYTHON) scripts/task.py rank-layers --config $(CONFIG)

eval-layer-candidate:
	$(PYTHON) scripts/task.py eval-layer-candidate --config $(CONFIG)

eval-main:
	$(PYTHON) scripts/task.py eval-main --config $(CONFIG)

fit-same-size-transplant:
	$(PYTHON) scripts/task.py fit-same-size-transplant --config $(CONFIG)

param-budget:
	$(PYTHON) scripts/task.py param-budget --config $(CONFIG)

summarize-baselines:
	$(PYTHON) scripts/task.py summarize-baselines --config $(CONFIG)

donor-gap-gate:
	$(PYTHON) scripts/task.py donor-gap-gate --config $(CONFIG)
