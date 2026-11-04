.PHONY: install test run fedprox compare sweep plots all
install:   ; pip install -e ".[dev]"
test:      ; python -m pytest -q
run:       ; ECOMFL_RESULTS=$(PWD)/results python scripts/run_simulation.py --tag fedavg
fedprox:   ; ECOMFL_RESULTS=$(PWD)/results python scripts/run_simulation.py --tag fedprox --mu 0.1
compare:   ; python scripts/compare.py --fl fedavg fedprox
plots:     ; python scripts/plot_training.py fedavg fedprox
sweep:     ; python scripts/run_sweep.py --het 0.0 0.5 1.0 --seeds 1 2 --rounds 10
all: test run fedprox compare plots
