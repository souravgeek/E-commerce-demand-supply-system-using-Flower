"""Run-config resolution.

Priority (low -> high): DEFAULTS < env var ECOMFL_OVERRIDES (JSON) < Flower run_config.
`flwr run .` always supplies the full run_config from pyproject.toml / --run-config, while the
Python simulation helper (scripts/run_simulation.py) uses the env var for overrides.
Keep DEFAULTS in sync with [tool.flwr.app.config] in pyproject.toml.
"""
import json
import os

DEFAULTS = {
    "num-server-rounds": 15, "fraction-train": 1.0, "fraction-evaluate": 1.0,
    "learning-rate": 0.01, "local-epochs": 2, "batch-size": 64, "proximal-mu": 0.0,
    "num-stores": 6, "heterogeneity": 0.7, "data-seed": 42, "run-tag": "fedavg",
    "save-model": True,
}


def get_cfg(run_config) -> dict:
    cfg = dict(DEFAULTS)
    if os.environ.get("ECOMFL_OVERRIDES"):
        cfg.update(json.loads(os.environ["ECOMFL_OVERRIDES"]))
    cfg.update(dict(run_config))
    return cfg
