"""Robustness sweep: heterogeneity x seeds. For every (h, seed) it
  1) trains FedAvg and FedProx through Flower (subprocess of run_simulation.py),
  2) trains local-only / centralized baselines and scores everything,
and finally aggregates mean +- std across seeds.

    python scripts/run_sweep.py --het 0.0 0.5 1.0 --seeds 1 2 --rounds 10
Outputs: results/sweep_raw.csv, results/sweep_summary.csv, results/sweep.png
Runtime: ~1-2 min per FL run on a laptop CPU.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from compare import RES, run_comparison

RAW = RES / "sweep_raw.csv"


def run_fl(tag, mu, seed, het, rounds, stores):
    env = dict(os.environ, ECOMFL_RESULTS=str(RES.resolve()))
    cmd = [sys.executable, str(ROOT / "scripts/run_simulation.py"), "--tag", tag, "--mu", str(mu),
           "--seed", str(seed), "--het", str(het), "--rounds", str(rounds), "--stores", str(stores)]
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--het", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--stores", type=int, default=6)
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    done = pd.read_csv(RAW) if RAW.exists() else pd.DataFrame()
    parts = [done] if len(done) else []
    for h in a.het:
        for s in a.seeds:
            if len(done) and ((done.het == h) & (done.seed == s)).any():
                print(f"skip h={h} seed={s} (already done)"); continue
            print(f"== h={h} seed={s}", flush=True)
            ta, tp = f"sweep_fedavg_h{h}_s{s}", f"sweep_fedprox_h{h}_s{s}"
            run_fl(ta, 0.0, s, h, a.rounds, a.stores)
            run_fl(tp, a.mu, s, h, a.rounds, a.stores)
            df = run_comparison(a.stores, s, h, a.rounds, 2,
                                {"federated_fedavg": RES / f"final_model_{ta}.pt",
                                 "federated_fedprox": RES / f"final_model_{tp}.pt"})
            df["het"], df["seed"] = h, s
            parts.append(df)
            pd.concat(parts).to_csv(RAW, index=False)  # checkpoint after every config
            for f in RES.glob("final_model_sweep_*"): f.unlink()   # keep repo small
            for f in RES.glob("history_sweep_*"): f.unlink()
    raw = pd.concat(parts)
    per_run = raw.groupby(["het", "seed", "method"])[["wape", "fill_rate", "total_cost"]].mean().reset_index()
    per_run["worst_wape"] = raw.groupby(["het", "seed", "method"])["wape"].max().values
    summ = per_run.groupby(["het", "method"]).agg(["mean", "std"]).round(4)
    summ.to_csv(RES / "sweep_summary.csv")
    print(summ[[("wape", "mean"), ("wape", "std"), ("total_cost", "mean"), ("worst_wape", "mean")]].to_string())

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a_, col, title in zip(ax, ["wape", "worst_wape", "total_cost"],
                              ["Mean forecast error (WAPE)", "Worst-store WAPE", "Supply cost"]):
        for m, g in per_run.groupby("method"):
            st = g.groupby("het")[col].agg(["mean", "std"]).fillna(0)
            a_.errorbar(st.index, st["mean"], yerr=st["std"], marker="o", capsize=3, label=m)
        a_.set_title(title); a_.set_xlabel("heterogeneity h")
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(RES / "sweep.png", dpi=140)
    print("Wrote results/sweep_summary.csv, results/sweep.png")


if __name__ == "__main__":
    main()
