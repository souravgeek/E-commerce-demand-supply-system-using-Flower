import numpy as np
import torch

from ecomfl.data import INPUT_DIM, load_store
from ecomfl.inventory import simulate_base_stock
from ecomfl.task import DemandNet, make_loader, train


def test_store_data_shapes_and_no_leakage():
    d = load_store(0, 1, 0.7)
    assert d.X_train.shape[1] == INPUT_DIM == d.X_test.shape[1]
    assert len(d.X_train) > len(d.X_test) > 0
    assert np.isfinite(d.X_train).all() and (d.units_train >= 0).all()


def test_stores_are_non_iid():
    m = [load_store(i, 1, 0.9).units_train.mean() for i in range(4)]
    assert max(m) / min(m) > 1.2
    iid = [load_store(i, 1, 0.0).units_train.mean() for i in range(4)]
    assert max(iid) / min(iid) < max(m) / min(m)


def test_perfect_forecast_beats_zero_forecast():
    rng = np.random.default_rng(0)
    actual = rng.poisson(10, 200).astype(float)
    good = simulate_base_stock(actual, actual, sigma=1.0)
    bad = simulate_base_stock(actual, np.zeros_like(actual), sigma=1.0)
    assert good["lost"] < bad["lost"]


def test_training_reduces_loss():
    d = load_store(0, 1, 0.5)
    torch.manual_seed(0)
    net = DemandNet()
    loader = make_loader(d.X_train, d.y_train, 64, True)
    l1 = train(net, loader, 1, 0.01, "cpu")
    l2 = train(net, loader, 5, 0.01, "cpu")
    assert l2 < l1


def test_fedprox_term_keeps_weights_closer():
    d = load_store(0, 1, 0.5)
    loader = make_loader(d.X_train, d.y_train, 64, True)
    def drift(mu):
        torch.manual_seed(0); net = DemandNet()
        w0 = [p.detach().clone() for p in net.parameters()]
        train(net, loader, 3, 0.01, "cpu", proximal_mu=mu)
        return sum(((p - q) ** 2).sum().item() for p, q in zip(net.parameters(), w0))
    assert drift(5.0) < drift(0.0)
