"""Plot per-round curves from results/history_<tag>.json (run after `flwr run`)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tags = sys.argv[1:] or ["fedavg", "fedprox"]
panels = [("evaluate_serverapp", "global_wape", "Pooled benchmark WAPE (lower better)"),
          ("evaluate_clientapp", "fill_rate", "Fill rate across stores (higher better)"),
          ("train_clientapp", "train_loss", "Avg local training loss")]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for tag in tags:
    p = Path("results") / f"history_{tag}.json"
    if not p.exists():
        print("missing", p); continue
    h = json.loads(p.read_text())
    for a, (section, key, title) in zip(ax, panels):
        rounds = sorted(int(r) for r in h[section])
        a.plot(rounds, [h[section][str(r)][key] for r in rounds], marker="o", label=tag)
        a.set_title(title); a.set_xlabel("FL round")
for a in ax: a.legend()
plt.tight_layout(); plt.savefig("results/training_curves.png", dpi=140)
print("Wrote results/training_curves.png")
