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


def test_weekly_paper_rebalances_weekly_and_keeps_value(tmp_path):
    from jevbot import weekly_paper
    full = {}
    for i, sym in enumerate(["BTCUSDT", "ETHUSDT", "SOLUSDT"]):
        d = synthetic(500, seed=60 + i, drift=0.002, vol=0.02)
        d.index = pd.date_range("2025-01-01", periods=500, freq="D", tz="UTC")
        full[sym] = d
    path = tmp_path / "w.json"
    msgs = []
    for day in range(420, 436):
        fetch = lambda s, i, n, day=day: full[s].iloc[:day]
        msgs.append(weekly_paper.run_once(list(full), "ts_momentum_14d", path, 1000, 10, 2, fetch=fetch))
    assert sum("rebalanced" in m for m in msgs) == 3  # day 0, 7, 14
    st = weekly_paper.load(path, 1000)
    assert st["cash"] >= -1e-9 and len(st["history"]) == 16
    assert "Weekly portfolio" in weekly_paper.report(path, "ts_momentum_14d")


def test_walk_forward_picks_never_see_their_test_year():
    px = panel(1600, coins=5, seed=70)
    wf = portfolio.walk_forward(px, 0.0012, first_test="2022-01-01")
    for year, row in wf["years"].iterrows():
        cut = pd.Timestamp(f"{year[:4]}-01-01", tz="UTC")
        only_past = portfolio.walk_forward(px[px.index < cut + pd.Timedelta(days=40)], 0.0012,
                                           first_test=f"{year[:4]}-01-01")
        assert only_past["years"].iloc[0]["pick"] == row["pick"]
    assert set(wf["summary"].index) == {"walk_forward_picks", "equal_hold", "btc_hold"}
