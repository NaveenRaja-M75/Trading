import numpy as np
import pandas as pd

from jevbot import portfolio
from jevbot.data import synthetic


def panel(n=900, coins=5, seed=0, drift=0.0):
    px = {}
    for i in range(coins):
        d = synthetic(n, seed=seed + i, drift=drift, vol=0.03)
        d.index = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
        px[f"C{i}USDT"] = d
    px["BTCUSDT"] = px.pop("C0USDT")
    return portfolio.closes(px)


def test_weights_are_causal():
    px = panel()
    for name, (_, f) in portfolio.candidates().items():
        full, cut = f(px).iloc[:600], f(px.iloc[:600])
        pd.testing.assert_frame_equal(full, cut, check_freq=False, obj=name)


def test_weights_never_exceed_full_investment():
    px = panel()
    for _, (_, f) in portfolio.candidates().items():
        w = f(px)
        assert (w >= 0).all().all() and (w.sum(axis=1) <= 1 + 1e-9).all()


def test_simulate_charges_costs_and_rebalances_weekly():
    px = panel(300)
    w = portfolio.ts_momentum(px, 14)
    free = portfolio.simulate(px, w, 0.0)
    paid = portfolio.simulate(px, w, 0.01)
    assert paid.sum() < free.sum()


def test_noise_picks_nothing_impressive():
    px = panel(1500, coins=6, seed=40)
    r = portfolio.evaluate(px, 0.0012, vault_days=365)
    assert r["vault"].loc[r["pick"], "sharpe"] < 1.5
    assert r["pick"] in r["dev"].index
