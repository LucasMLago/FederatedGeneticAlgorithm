.PHONY: help install smoke dry-run run tables figures clean

PYTHON   ?= uv run --project federatedgeneticalgorithm python
CONFIG   ?= configs/smoke_cifar.yaml
SEED     ?= 0
FED      ?= local-simulation-gpu

help:
	@echo "FederatedGeneticAlgorithm: common tasks"
	@echo ""
	@echo "  make install        Install dependencies via uv sync"
	@echo "  make smoke          2-round CPU smoke run (~6 min) to verify the harness"
	@echo "  make dry-run CONFIG=configs/<scenario>.yaml SEED=42"
	@echo "                      Resolve a YAML + seed and print the JSON; no launch"
	@echo "  make run CONFIG=configs/<scenario>.yaml SEED=42 FED=local-simulation-gpu"
	@echo "                      Run an experiment via the runner CLI"
	@echo "  make tables         Regenerate the results tables + bias report from run telemetry"
	@echo "  make figures        Regenerate the paper figures into analysis/figures/"
	@echo "  make clean          Remove runner snapshot tempfiles in /tmp"

install:
	uv sync --project federatedgeneticalgorithm

smoke:
	$(PYTHON) -m federatedgeneticalgorithm.runner \
		--config configs/smoke_cifar.yaml \
		--federation local-simulation \
		--tag make-smoke

dry-run:
	$(PYTHON) -m federatedgeneticalgorithm.runner \
		--config $(CONFIG) \
		--seed $(SEED) \
		--dry-run

run:
	$(PYTHON) -m federatedgeneticalgorithm.runner \
		--config $(CONFIG) \
		--seed $(SEED) \
		--federation $(FED) \
		--tag make-run

tables:
	$(PYTHON) analysis/results_tables.py --markdown analysis/results_tables.md
	$(PYTHON) analysis/fitness_bias.py --markdown analysis/fitness_bias_report.md

figures:
	$(PYTHON) analysis/paper_figures.py

clean:
	rm -f /tmp/fga_config_*.json
