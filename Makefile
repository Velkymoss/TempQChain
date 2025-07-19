.PHONY: install baseline-yn primal-dual-yn primal-dual-qchain-yn baseline-fr primal-dual-fr primal-dual-qchain-fr tests

PY_ARGS=--epoch 8 --train_size 1000000 --test_size 1000000 --cuda 0 --lr 8e-6 --batch_size 8
MODEL ?= bert

install:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "uv not found. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	uv sync

baseline-yn:
	uv run python src/main.py $(PY_ARGS) --data_path $(DATA_PATH) --results_path $(RES_PATH) --use_chains $(CHAINS) --train_file ORIGIN --test_file ORIGIN --model $(MODEL)

primal-dual-yn:
	uv run python src/main.py $(PY_ARGS) --data_path $(DATA_PATH) --results_path $(RES_PATH) --use_chains $(CHAINS) --train_file ORIGIN --test_file ORIGIN --pmd T --constraints T --model $(MODEL)

primal-dual-qchain-yn:
	uv run python src/main.py $(PY_ARGS) --data_path $(DATA_PATH) --results_path $(RES_PATH) --use_chains $(CHAINS) --train_file SPARTUN --test_file SPARTUN --pmd T --constraints T --save T --save_file Q_chain_T5 --model $(MODEL)

baseline-fr:
	uv run python src/main_rel.py $(PY_ARGS) --data_path $(DATA_PATH) --results_path $(RES_PATH) --use_chains $(CHAINS) --train_file ORIGIN --test_file ORIGIN --model $(MODEL)

primal-dual-fr:
	uv run python src/main_rel.py $(PY_ARGS) --data_path $(DATA_PATH) --results_path $(RES_PATH) --use_chains $(CHAINS) --train_file ORIGIN --test_file ORIGIN --pmd T --constraints T --model $(MODEL)

primal-dual-qchain-fr:
	uv run python src/main_rel.py $(PY_ARGS) --data_path $(DATA_PATH) --results_path $(RES_PATH) --use_chains $(CHAINS) --train_file SPARTUN --test_file SPARTUN --pmd T --constraints T --save T --save_file Q_chain_T5 --model $(MODEL)

tests:
	uv run pytest tests --cov=src --cov-report=term-missing