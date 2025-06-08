.PHONY: install baseline-yn primal-dual-yn primal-dual-qchain-yn baseline-fr primal-dual-fr primal-dual-qchain-fr tests

PY_ARGS=--epoch 8 --train_size 1000000 --test_size 1000000 --cuda 0 --lr 8e-6 --batch_size 8
MODEL ?= bert

install:
	@if ! command -v poetry >/dev/null 2>&1; then \
		echo "Poetry not found. Installing..."; \
		pip install --user poetry; \
	fi
	poetry install

baseline-yn:
	python src/main.py $(PY_ARGS) --train_file ORIGIN --test_file ORIGIN --model $(MODEL)

primal-dual-yn:
	python src/main.py $(PY_ARGS) --train_file ORIGIN --test_file ORIGIN --pmd T --constraints T --model $(MODEL)

primal-dual-qchain-yn:
	python src/main.py $(PY_ARGS) --train_file SPARTUN --test_file SPARTUN --pmd T --constraints T --save T --save_file Q_chain_T5 --model $(MODEL)

baseline-fr:
	python src/main_rel.py $(PY_ARGS) --train_file ORIGIN --test_file ORIGIN --model $(MODEL)

primal-dual-fr:
	python src/main_rel.py $(PY_ARGS) --train_file ORIGIN --test_file ORIGIN --pmd T --constraints T --model $(MODEL)

primal-dual-qchain-fr:
	python src/main_rel.py $(PY_ARGS) --train_file SPARTUN --test_file SPARTUN --pmd T --constraints T --save T --save_file Q_chain_T5 --model $(MODEL)

tests:
	poetry run pytest tests --cov=src --cov-report=term-missing