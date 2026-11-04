"""Supply side: turn demand forecasts into replenishment decisions and score them.

Policy: daily-review base-stock (order-up-to) with lead time L.
    S_t = (L + 1) * forecast_{t+1} + z * sigma * sqrt(L + 1)
    order_t = max(0, S_t - inventory_position_t)
Better forecasts -> lower safety stock for the same service level -> lower cost.
"""
from __future__ import annotations

from collections import deque

import numpy as np

HOLD_COST = 0.05       # per unit per day
SHORTAGE_COST = 1.00   # per unit of lost sales


def simulate_base_stock(actual: np.ndarray, forecast: np.ndarray, sigma: float,
                        lead_time: int = 2, z: float = 1.28) -> dict:
    """Simulate one SKU. `forecast[t]` is the forecast of demand on day t made on day t-1."""
    n = len(actual)
    pipeline = deque([0.0] * lead_time)
    protect = lead_time + 1
    safety = z * sigma * np.sqrt(protect)
    on_hand = protect * forecast[0] + safety
    sold_total = lost_total = inv_sum = 0.0
    stockout_days = 0
    for t in range(n):
        if lead_time > 0:
            on_hand += pipeline.popleft()
        sold = min(on_hand, actual[t])
        lost = actual[t] - sold
        on_hand -= sold
        sold_total += sold; lost_total += lost; inv_sum += on_hand
        stockout_days += int(lost > 1e-9)
        nxt = forecast[t + 1] if t + 1 < n else forecast[t]
        target = protect * nxt + safety
        order = max(0.0, target - (on_hand + sum(pipeline)))
        if lead_time > 0:
            pipeline.append(order)
        else:
            on_hand += order
    return dict(sold=sold_total, lost=lost_total, inv_sum=inv_sum,
                stockout_days=stockout_days, days=n)


def evaluate_supply(units_true: np.ndarray, units_pred: np.ndarray, sku: np.ndarray,
                    sigma_by_sku: dict[int, float], lead_time: int = 2, z: float = 1.28) -> dict:
    """Aggregate base-stock KPIs over all SKUs of one store."""
    tot = dict(sold=0.0, lost=0.0, inv_sum=0.0, stockout_days=0, days=0)
    for k in np.unique(sku):
        m = sku == k
        r = simulate_base_stock(units_true[m], units_pred[m], sigma_by_sku.get(int(k), 1.0), lead_time, z)
        for key in tot:
            tot[key] += r[key]
    demand = tot["sold"] + tot["lost"]
    return dict(
        fill_rate=tot["sold"] / max(demand, 1e-9),
        stockout_rate=tot["stockout_days"] / max(tot["days"], 1),
        avg_inventory=tot["inv_sum"] / max(tot["days"], 1),
        total_cost=HOLD_COST * tot["inv_sum"] + SHORTAGE_COST * tot["lost"],
    )
