"""Run the federation in-process with Flower's Python simulation API.

Equivalent to `flwr run .` but uses your *current* environment (no per-run dependency install),
which is handy for notebooks, CI, and quick experiments / sweeps.

    python scripts/run_simulation.py --tag fedavg
    python scripts/run_simulation.py --tag fedprox --mu 0.1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
from pathlib import Path

import json
import tomllib
from flwr.simulation import run_simulation

from ecomfl.client_app import app as client_app
from ecomfl.server_app import app as server_app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fedavg")
    ap.add_argument("--mu", type=float, default=0.0, help="FedProx mu (0 = FedAvg)")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--stores", type=int, default=None)
    ap.add_argument("--het", type=float, default=None)
    ap.add_argument("--local-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="data seed")
    a = ap.parse_args()

    cfg = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
    cfg.update({"run-tag": a.tag, "proximal-mu": a.mu})
    if a.rounds is not None: cfg["num-server-rounds"] = a.rounds
    if a.stores is not None: cfg["num-stores"] = a.stores
    if a.het is not None: cfg["heterogeneity"] = a.het
    if a.local_epochs is not None: cfg["local-epochs"] = a.local_epochs
    if a.seed is not None: cfg["data-seed"] = a.seed

    os.environ.setdefault("ECOMFL_RESULTS", str(Path("results").resolve()))
    os.environ["ECOMFL_OVERRIDES"] = json.dumps(cfg)  # read by ecomfl.config.get_cfg in every worker
    backend_config = {
        "client_resources": {"num_cpus": 1, "num_gpus": 0.0},
        "init_args": {"num_cpus": max(2, os.cpu_count() or 2), "include_dashboard": False},
    }
    run_simulation(server_app=server_app, client_app=client_app,
                   num_supernodes=int(cfg["num-stores"]), backend_config=backend_config)


if __name__ == "__main__":
    main()
