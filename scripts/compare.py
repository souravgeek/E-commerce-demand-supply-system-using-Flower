"""Compare Naive / Local-only / Centralized / Federated models store-by-store.

Usage:
    python scripts/compare.py                      # trains local+centralized baselines
    python scripts/compare.py --fl fedavg fedprox  # also loads results/final_model_<tag>.pt
Outputs: results/<prefix>comparison.csv, ..._summary.csv, ..._comparison.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ecomfl.task import (DemandNet, evaluate_store, get_device, load_store, make_loader,
                         naive_forecast_metrics, train)

RES = Path("results")


def fit(X, y, epochs, lr=0.01, batch=64, seed=0):
    torch.manual_seed(seed)
    net = DemandNet()
    train(net, make_loader(X, y, batch, True), epochs, lr, get_device())
    return net


def run_comparison(stores=6, seed=42, het=0.7, rounds=15, local_epochs=2,
                   fl_models: dict | None = None) -> pd.DataFrame:
    """fl_models maps a method label -> path of a saved federated state_dict.

    Baselines get the same training budget (rounds*local_epochs epochs over a node's data).
    Returns one row per (method, store)."""
    dev = get_device()
    data = [load_store(i, seed, het) for i in range(stores)]
    budget = rounds * local_epochs
    rows = []
    # Centralized upper bound: pools raw data from all stores (NOT privacy-preserving).
    central = fit(np.concatenate([d.X_train for d in data]),
                  np.concatenate([d.y_train for d in data]), budget)
    fl_nets = {}
    for label, path in (fl_models or {}).items():
        net = DemandNet(); net.load_state_dict(torch.load(path)); fl_nets[label] = net
    for d in data:
        rows.append(dict(method="naive_7d_mean", store=d.name, **naive_forecast_metrics(d)))
        local = fit(d.X_train, d.y_train, budget)
        rows.append(dict(method="local_only", store=d.name, **evaluate_store(local, d, dev)))
        rows.append(dict(method="centralized", store=d.name, **evaluate_store(central, d, dev)))
        for label, net in fl_nets.items():
            rows.append(dict(method=label, store=d.name, **evaluate_store(net, d, dev)))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    s = df.groupby("method")[["mae", "wape", "fill_rate", "avg_inventory", "total_cost"]].mean()
    s["worst_store_wape"] = df.groupby("method")["wape"].max()
    return s.sort_values("wape")


def plot(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a_, col, title in zip(ax, ["wape", "fill_rate", "total_cost"],
                              ["Forecast error (WAPE, lower better)", "Fill rate (higher better)",
                               "Supply cost: holding + shortage (lower better)"]):
        df.pivot(index="store", columns="method", values=col).plot.bar(ax=a_, rot=0)
        a_.set_title(title); a_.set_xlabel(""); a_.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--het", type=float, default=0.7)
    ap.add_argument("--rounds", type=int, default=15, help="FL rounds used (to match baseline budget)")
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--fl", nargs="*", default=[], help="tags of saved FL models in results/")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    fl = {f"federated_{t}": RES / f"final_model_{t}.pt" for t in a.fl}
    df = run_comparison(a.stores, a.seed, a.het, a.rounds, a.local_epochs, fl)
    df.to_csv(RES / f"{a.prefix}comparison.csv", index=False)
    summ = summarize(df); summ.to_csv(RES / f"{a.prefix}comparison_summary.csv")
    print(summ.round(3).to_string())
    plot(df, RES / f"{a.prefix}comparison.png")
    print(f"\nWrote results/{a.prefix}comparison.{{csv,png}} and ..._summary.csv")


if __name__ == "__main__":
    main()
