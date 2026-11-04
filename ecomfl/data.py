"""Synthetic multi-store e-commerce sales generator (non-IID by design).

Each *store* (a regional warehouse / seller) owns private daily sales for several SKUs.
Stores share some global structure (category price elasticity, weekly pattern, promo effect)
but differ in size, seasonality phase, category mix, weekend tilt and promo response.
That is exactly the setting where federated learning is attractive: the shared structure
can be learned jointly, while raw sales never leave a store.

Task: one-step-ahead daily demand forecast per (store, SKU).
Target: log1p(units sold on day t). Features use only information known at end of day t-1
plus the *planned* price/promo for day t.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

CATEGORIES = ["electronics", "fashion", "grocery", "home"]
BASE_PRICE = np.array([300.0, 40.0, 8.0, 60.0])
ELASTICITY = np.array([-1.2, -1.5, -0.8, -1.0])      # shared across stores (global signal)
WEEKLY = np.array([0.90, 0.85, 0.90, 0.95, 1.10, 1.30, 1.20])  # Mon..Sun

LAGS = (1, 2, 3, 7, 14, 28)
MAX_LAG = max(LAGS)
FEATURE_NAMES = (
    [f"lag_{k}" for k in LAGS]
    + ["roll_mean_7", "roll_mean_28", "price_ratio", "promo"]
    + [f"dow_{d}" for d in range(7)]
    + ["season_sin", "season_cos"]
    + [f"cat_{c}" for c in CATEGORIES]
)
INPUT_DIM = len(FEATURE_NAMES)  # 23
ROLL7_IDX = FEATURE_NAMES.index("roll_mean_7")


@dataclass
class StoreData:
    store_id: int
    name: str
    X_train: np.ndarray
    y_train: np.ndarray          # log1p(units)
    X_test: np.ndarray
    y_test: np.ndarray           # log1p(units)
    sku_train: np.ndarray        # sku index per row (rows are ordered sku-major, then time)
    sku_test: np.ndarray
    units_train: np.ndarray
    units_test: np.ndarray       # actual units sold on test days


def _profile(rng: np.random.Generator, h: float) -> dict:
    mix_alpha = 20.0 * (1.0 - h) + 0.6
    return dict(
        scale=float(np.exp(rng.normal(0.0, 0.6 * h))),
        season_phase=float(rng.uniform(0, 2 * np.pi) * h),
        season_amp=float(0.20 + 0.30 * h * rng.random()),
        weekend_tilt=float(1.0 + h * rng.uniform(-0.25, 0.35)),
        promo_lift=float(0.40 + 0.60 * h * rng.random()),
        trend=float(h * rng.uniform(-0.0004, 0.0008)),
        cat_mix=rng.dirichlet(np.ones(len(CATEGORIES)) * mix_alpha),
    )


def _simulate_store(store_id: int, seed: int, h: float, n_days: int, n_skus: int):
    rng = np.random.default_rng(seed * 10_007 + store_id)
    p = _profile(rng, h)
    cats = rng.choice(len(CATEGORIES), size=n_skus, p=p["cat_mix"])
    base = np.exp(rng.normal(np.log(10.0 * p["scale"]), 0.5, size=n_skus))
    t = np.arange(n_days)
    dow, doy = t % 7, t % 365
    weekly = WEEKLY.copy()
    weekly[5:] *= p["weekend_tilt"]

    all_units, all_price, all_promo = [], [], []
    for k in range(n_skus):
        c = cats[k]
        # category-specific seasonality (shared shape) shifted by store phase
        if CATEGORIES[c] == "fashion":
            seas = 1 + p["season_amp"] * np.sin(2 * np.pi * doy / 365 + p["season_phase"])
        elif CATEGORIES[c] == "electronics":
            seas = 1 + 0.6 * np.exp(-((doy - 330) ** 2) / (2 * 15**2))  # Nov peak
        elif CATEGORIES[c] == "home":
            seas = 1 + 0.5 * p["season_amp"] * np.cos(2 * np.pi * doy / 365 + p["season_phase"])
        else:
            seas = np.ones(n_days)
        # promo events: random starts, 4 days long
        promo = np.zeros(n_days)
        starts = np.where(rng.random(n_days) < 0.03)[0]
        for s in starts:
            promo[s : s + 4] = 1.0
        discount = 0.15 + 0.15 * rng.random()
        price = BASE_PRICE[c] * (1 - discount * promo) * (1 + 0.02 * rng.normal(size=n_days))
        lam = (
            base[k] * seas * weekly[dow]
            * (price / BASE_PRICE[c]) ** ELASTICITY[c]
            * (1 + p["promo_lift"] * promo)
            * (1 + p["trend"] * t)
        )
        lam = np.clip(lam, 0.05, None)
        units = rng.poisson(lam * rng.gamma(10.0, 0.1, size=n_days))  # over-dispersed counts
        all_units.append(units.astype(float))
        all_price.append(price / BASE_PRICE[c])
        all_promo.append(promo)
    return cats, np.array(all_units), np.array(all_price), np.array(all_promo)


def _build_rows(cats, units, price_ratio, promo, n_days):
    X, y, sku, raw = [], [], [], []
    logs = np.log1p(units)
    t_all = np.arange(MAX_LAG, n_days)
    for k in range(units.shape[0]):
        s = logs[k]
        lag_feats = np.stack([s[t_all - lag] for lag in LAGS], axis=1)
        roll7 = np.array([s[t - 7 : t].mean() for t in t_all])
        roll28 = np.array([s[t - 28 : t].mean() for t in t_all])
        dow = np.eye(7)[t_all % 7]
        doy = t_all % 365
        season = np.stack([np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)], axis=1)
        cat = np.tile(np.eye(len(CATEGORIES))[cats[k]], (len(t_all), 1))
        feats = np.column_stack(
            [lag_feats, roll7, roll28, price_ratio[k][t_all], promo[k][t_all], dow, season, cat]
        )
        X.append(feats)
        y.append(s[t_all])
        raw.append(units[k][t_all])
        sku.append(np.full(len(t_all), k))
    return (np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.float32),
            np.concatenate(sku), np.concatenate(raw))


@lru_cache(maxsize=64)
def load_store(store_id: int, seed: int = 42, heterogeneity: float = 0.7,
               n_days: int = 420, n_skus: int = 8, test_frac: float = 0.2) -> StoreData:
    """Deterministic per-store dataset with a *temporal* train/test split (last 20% of days)."""
    cats, units, price, promo = _simulate_store(store_id, seed, heterogeneity, n_days, n_skus)
    split_day = int(n_days * (1 - test_frac))
    X, y, sku, raw = _build_rows(cats, units, price, promo, n_days)
    day_of_row = np.tile(np.arange(MAX_LAG, n_days), units.shape[0])
    train_mask = day_of_row < split_day
    return StoreData(
        store_id=store_id, name=f"store_{store_id}",
        X_train=X[train_mask], y_train=y[train_mask], sku_train=sku[train_mask], units_train=raw[train_mask],
        X_test=X[~train_mask], y_test=y[~train_mask], sku_test=sku[~train_mask], units_test=raw[~train_mask],
    )


def load_pooled_test(num_stores: int, seed: int = 42, heterogeneity: float = 0.7):
    """Server-side benchmark (simulation only): union of every store's test split."""
    xs, ys, us = [], [], []
    for s in range(num_stores):
        d = load_store(s, seed, heterogeneity)
        xs.append(d.X_test); ys.append(d.y_test); us.append(d.units_test)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(us)
