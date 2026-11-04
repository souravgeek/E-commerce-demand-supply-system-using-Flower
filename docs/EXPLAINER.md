# Explainer: how the system works

## 1. One-minute story

> Six stores each have private sales history. None will share it. We still want a good forecast for each.
> So every round the server sends the same small neural network to all stores, each store improves it on
> *its own* data for a couple of epochs, and sends back only the new weights. The server averages them
> (weighted by how much data each store has) and repeats. At the end each store uses the shared model to
> forecast tomorrow's demand and decide how much stock to order. We then measure fill rate and cost.

## 2. What happens in one round (trace it in the logs)

```
Server (ServerApp.main -> strategy.start)
  1. configure_train   : pick stores, send {global weights (ArrayRecord), lr (+ proximal-mu)}
Store k (ClientApp @train)
  2. rebuild DemandNet, load global weights
  3. load ITS OWN data (load_store(partition_id))
  4. train() for `local-epochs` epochs  (FedProx: + (mu/2)||w-w_global||^2)
  5. reply {new weights, train_loss, num-examples}
Server
  6. aggregate_train   : w_new = sum_k (n_k / sum n) * w_k      <- FedAvg
  7. configure_evaluate: send w_new to stores
Store k (ClientApp @evaluate)
  8. forecast ITS OWN held-out future days, run inventory simulation
  9. reply {eval_loss, wape, fill_rate, total_cost, ... , num-examples}
Server
 10. aggregate_evaluate: weighted mean of metrics
 11. evaluate_fn       : (simulation only) score global model on pooled test set
```
Log lines like `configure_train: Sampled 6 nodes`, `aggregate_train: Received 6 results and 0 failures`,
`Aggregated MetricRecord: {...}` correspond exactly to steps 1, 6, 10.

## 3. Files

| File | Role | Key idea |
|---|---|---|
| `ecomfl/data.py` | Synthetic sales generator + feature builder | Stores share structure (elasticity, weekly pattern, promo effect) but differ in size/season/mix; `heterogeneity` scales the differences. Temporal split. |
| `ecomfl/task.py` | Plain PyTorch | `DemandNet` (23-64-32-1 MLP), `train()` with optional FedProx term, `evaluate_store()`. No Flower code. |
| `ecomfl/inventory.py` | Supply simulator | Base-stock policy, lead time 2, z=1.28; returns fill rate, stock-out rate, avg inventory, cost. |
| `ecomfl/client_app.py` | Flower ClientApp | `@app.train()` / `@app.evaluate()`; converts `ArrayRecord <-> state_dict`. |
| `ecomfl/server_app.py` | Flower ServerApp | Builds initial model, chooses `FedAvg` or `FedProx`, calls `strategy.start`, saves model + history. |
| `ecomfl/config.py` | Config resolution | Same code works under `flwr run` and the Python API. |
| `scripts/run_simulation.py` | Run federation in-process | No per-run dependency install. |
| `scripts/compare.py` | Baselines + scoring | Naive / local-only / centralized / federated per store. |
| `scripts/run_sweep.py` | Robustness study | Heterogeneity x seeds, mean ± std. |
| `scripts/plot_training.py` | Per-round curves | From `results/history_<tag>.json`. |

## 4. The maths in plain words

**Features -> target.** For store s, SKU i, day t: lags (1,2,3,7,14,28 days) of log(1+sales), rolling means,
planned price ratio, promo flag, day-of-week, season (sin/cos), category. Target: log(1+sales_t).
Logs make a store selling 100/day and one selling 5/day comparable.

**FedAvg.**  `w_global = Σ_k (n_k / N) · w_k`, n_k = training rows at store k, N = Σ n_k.

**FedProx.**  local loss `L_k(w) + (μ/2)‖w − w_global‖²`. The extra term is a spring that stops a store
with unusual data from dragging its model far away from the global one. μ = 0 gives FedAvg.

**Base-stock replenishment** (each day, per SKU):

```
protection period   P = L + 1                (lead time + 1 review day)
safety stock        SS = z · σ · sqrt(P)      (σ = std of forecast error on TRAIN data)
order-up-to level   S = P · forecast_tomorrow + SS
order               = max(0, S − (on_hand + on_order))
```
Better forecast -> smaller σ and smaller forecast bias -> less safety stock for the same service level.

**Metrics.**
* WAPE = Σ|forecast − actual| / Σ actual (scale-free, safe with zero days).
* Fill rate = units sold / units demanded.
* Cost = 0.05 · (unit-days held) + 1.00 · (units lost).

## 5. Suggested live demo (5 minutes)

```bash
pytest -q                                              # 1. sanity (5 tests)
python scripts/run_simulation.py --tag fedavg --rounds 5   # 2. watch rounds in the log
python scripts/compare.py --fl fedavg --rounds 5       # 3. table: naive/local/central/federated
```
Then open `results/comparison.png` and point to the worst-store bar.

## 6. Future scope

| Extension | Where to hook in | Reference |
|---|---|---|
| Update clipping + Gaussian noise (DP) | client `train` reply / Flower's DP wrappers around the strategy | Gboard deck; Turazza et al. |
| Secure aggregation | Flower's SecAgg+ workflow | Bonawitz et al. |
| Clustered FL (group similar stores) | server aggregation by cluster | Long et al. (PA-CFL) |
| Personalization (fine-tune global model locally) | after final round, per store | Kairouz et al. |
| GRU / Transformer model | `DemandNet` in `task.py` | Long et al. |
| Real data (M5, Rossmann, Olist) | `load_store()` in `data.py` | Makridakis et al. |
| Multi-echelon / (s,S) policy | `inventory.py` | Silver, Pyke, Thomas |
