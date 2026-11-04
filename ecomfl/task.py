"""Plain PyTorch: model, training (with optional FedProx term), evaluation, helpers.

Nothing in this file is Flower specific; the ClientApp / ServerApp wrap it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ecomfl.data import INPUT_DIM, ROLL7_IDX, StoreData, load_store
from ecomfl.inventory import evaluate_supply


class DemandNet(nn.Module):
    """Small MLP: engineered lag/price/promo/calendar features -> log1p(demand)."""

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train(net, loader, epochs, lr, device, proximal_mu: float = 0.0) -> float:
    """Local SGD. If proximal_mu > 0 adds the FedProx term (mu/2)||w - w_global||^2."""
    net.to(device).train()
    global_params = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    crit = nn.MSELoss()
    running, steps = 0.0, 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(net(xb), yb)
            if proximal_mu > 0:
                prox = sum(((p - g) ** 2).sum() for p, g in zip(net.parameters(), global_params))
                loss = loss + (proximal_mu / 2.0) * prox
            loss.backward()
            opt.step()
            running += loss.item(); steps += 1
    return running / max(steps, 1)


@torch.no_grad()
def predict_log(net, X: np.ndarray, device) -> np.ndarray:
    net.to(device).eval()
    return net(torch.from_numpy(X).to(device)).cpu().numpy()


def to_units(log_pred: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(log_pred), 0.0, None)


def forecast_metrics(units_true, units_pred, log_true, log_pred) -> dict:
    err = units_pred - units_true
    return dict(
        mse_log=float(np.mean((log_pred - log_true) ** 2)),
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        wape=float(np.sum(np.abs(err)) / max(np.sum(units_true), 1e-9)),
    )


def sigma_by_sku(units_true, units_pred, sku) -> dict[int, float]:
    return {int(k): float(np.std(units_true[sku == k] - units_pred[sku == k]) + 1e-6)
            for k in np.unique(sku)}


def _score(units_pred_train, log_pred_test, data: StoreData, lead_time, z) -> dict:
    sig = sigma_by_sku(data.units_train, units_pred_train, data.sku_train)
    units_pred = to_units(log_pred_test)
    m = forecast_metrics(data.units_test, units_pred, data.y_test, log_pred_test)
    m.update(evaluate_supply(data.units_test, units_pred, data.sku_test, sig, lead_time, z))
    return m


def evaluate_store(net, data: StoreData, device, lead_time: int = 2, z: float = 1.28) -> dict:
    """Forecast accuracy + supply KPIs for one store. sigma is estimated on TRAIN residuals."""
    tr = to_units(predict_log(net, data.X_train, device))
    return _score(tr, predict_log(net, data.X_test, device), data, lead_time, z)


def naive_forecast_metrics(data: StoreData, lead_time: int = 2, z: float = 1.28) -> dict:
    """Baseline: forecast = trailing 7-day mean (the roll_mean_7 feature)."""
    tr = to_units(data.X_train[:, ROLL7_IDX])
    return _score(tr, data.X_test[:, ROLL7_IDX], data, lead_time, z)
