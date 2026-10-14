.PHONY: install doctor lint test demo figures notebook clean

install:
	pip install -r requirements-dev.txt

doctor:
	python scripts/doctor.py

lint:
	ruff check .
	ruff format --check .

test:
	python -m pytest

# Whole pipeline on synthetic data.
demo:
	python main.py --mode all --demo --num_leaves 15 --min_data_in_leaf 20 --num_boost_round 500 --catboost_iterations 500

# Figures for a finished run: make figures RUN=runs/run_a DATA=data/processed
figures:
	python scripts/make_figures.py --run_dir $(RUN) --data_dir $(DATA) --export_dir docs/figures

notebook:
	python scripts/build_notebook.py

clean:
	rm -rf .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
