PYTHON ?= python

.PHONY: test prepare baseline family-report dry-train

test:
	$(PYTHON) -m pytest -q

prepare:
	$(PYTHON) -m src.competition.parser --config configs/data.yaml

baseline:
	$(PYTHON) -m src.experiments.run_baseline --config configs/eval.yaml

family-report:
	$(PYTHON) -m src.experiments.run_rule_analysis --config configs/data.yaml

dry-train:
	$(PYTHON) -m src.student.lora_train --config configs/train_lora.yaml --dry-run
