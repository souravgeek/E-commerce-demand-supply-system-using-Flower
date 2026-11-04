"""Flower ServerApp: FedAvg / FedProx over the store federation + a pooled benchmark."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from ecomfl.config import get_cfg
from ecomfl.data import load_pooled_test
from ecomfl.task import DemandNet, forecast_metrics, get_device, predict_log, to_units

app = ServerApp()


def _results_dir() -> Path:
    d = Path(os.environ.get("ECOMFL_RESULTS", "results")).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_global_evaluate(num_stores: int, seed: int, het: float):
    X, y, units = load_pooled_test(num_stores, seed, het)  # simulation-only central benchmark

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = DemandNet()
        model.load_state_dict(arrays.to_torch_state_dict())
        log_pred = predict_log(model, X, get_device())
        m = forecast_metrics(units, to_units(log_pred), y, log_pred)
        return MetricRecord({"global_mse_log": m["mse_log"], "global_mae": m["mae"],
                             "global_wape": m["wape"]})

    return global_evaluate


def _to_plain(records: dict) -> dict:
    return {int(r): {k: float(v) for k, v in dict(rec).items()} for r, rec in records.items()}


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = get_cfg(context.run_config)
    num_rounds = int(rc["num-server-rounds"])
    lr = float(rc["learning-rate"])
    mu = float(rc["proximal-mu"])
    tag = str(rc["run-tag"])

    arrays = ArrayRecord(DemandNet().state_dict())

    kwargs = dict(fraction_train=float(rc["fraction-train"]),
                  fraction_evaluate=float(rc["fraction-evaluate"]),
                  min_available_nodes=2)
    strategy = FedProx(proximal_mu=mu, **kwargs) if mu > 0 else FedAvg(**kwargs)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=_make_global_evaluate(int(rc["num-stores"]), int(rc["data-seed"]),
                                          float(rc["heterogeneity"])),
    )

    if rc["save-model"]:
        out = _results_dir()
        torch.save(result.arrays.to_torch_state_dict(), out / f"final_model_{tag}.pt")
        history = {
            "tag": tag, "proximal_mu": mu, "config": {k: v for k, v in rc.items()},
            "train_clientapp": _to_plain(result.train_metrics_clientapp),
            "evaluate_clientapp": _to_plain(result.evaluate_metrics_clientapp),
            "evaluate_serverapp": _to_plain(result.evaluate_metrics_serverapp),
        }
        (out / f"history_{tag}.json").write_text(json.dumps(history, indent=2))
        print(f"\nSaved model + history to {out}")
