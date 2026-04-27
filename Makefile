PYTHON ?= python

.PHONY: test prepare baseline family-report dry-train probe-submission-size probe-submission-size-trained final-acceptance no-gpu-readiness research-registry research-rescue-data lb-correlation-log

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

probe-submission-size:
	$(PYTHON) scripts/probe_adapter_submission_size.py \
	  --config configs/train_stage2_selected_trace.yaml \
	  --output artifacts/adapter_submission_probe.json \
	  --tiny-mode

probe-submission-size-trained:
	@if [ -z "$(ADAPTER_DIR)" ]; then \
		echo "ERROR: ADAPTER_DIR=path/to/adapter required"; exit 2; \
	fi
	$(PYTHON) scripts/probe_adapter_submission_size.py \
	  --config configs/train_stage2_selected_trace.yaml \
	  --output artifacts/adapter_submission_probe.json \
	  --adapter-dir $(ADAPTER_DIR)

final-acceptance:
	$(PYTHON) scripts/run_local_final_acceptance.py

no-gpu-readiness:
	$(PYTHON) scripts/run_no_gpu_readiness_gate.py --mode full

research-registry:
	$(PYTHON) scripts/build_research_candidate_registry.py --check --output configs/research_breakout_candidates.json

research-rescue-data:
	$(PYTHON) scripts/build_research_rescue_data.py

lb-correlation-log:
	@if [ -z "$(CANDIDATE)" ] || [ -z "$(PUBLIC_SCORE)" ]; then \
		echo "ERROR: CANDIDATE=name PUBLIC_SCORE=0.xx required"; exit 2; \
	fi
	$(PYTHON) scripts/update_lb_correlation_log.py \
	  --candidate $(CANDIDATE) \
	  --public-score $(PUBLIC_SCORE)
