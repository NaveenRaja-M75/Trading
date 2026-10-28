import numpy as np
import pandas as pd

from jevbot import analysis, portfolio


def px_from(prices):
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="D", tz="UTC")
    return pd.DataFrame({"BTCUSDT": prices}, index=idx)


def test_trades_and_stats_by_hand():
    px = px_from([100, 110, 121, 100, 100, 90, 95, 95])
    held = pd.DataFrame({"BTCUSDT": [1, 1, 1, 0, 0, 1, 1, 0]}, index=px.index, dtype=float)
    t = analysis.trades(px, held, 0.0)
    assert list(t.return_pct.round(2)) == [0.0, round((95 / 90 - 1) * 100, 2)]  # 100->100, 90->95
    st = analysis.trade_stats(t)
    assert st["trades"] == 2 and st["win_rate_pct"] == 50.0


def test_scenarios_flat_market_never_trips():
    idx = pd.date_range("2022-01-01", periods=800, freq="D", tz="UTC")
    r = analysis.scenarios(pd.Series(0.0, index=idx), 20, 5, sims=200)
    assert r["median"] == 20 and r["chance_kill_switch_pct"] == 0


def test_scenarios_kill_switch_caps_losses():
    idx = pd.date_range("2022-01-01", periods=800, freq="D", tz="UTC")
    r = analysis.scenarios(pd.Series(-0.01, index=idx), 20, 5, sims=200)
    assert r["chance_kill_switch_pct"] == 100 and r["worst"] >= 14  # stops near 15, one week of slippage


def test_offset_rebalance_and_delay_run():
    rng = np.random.default_rng(1)
    px = px_from(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 600))))
    f = portfolio.candidates()["trend_filter_ma100"][1]
    table = analysis.stress(px, f, 0.0012, px.index[300], every=7)
    assert "trade 1 day late" in table.index and len(table) == 11


def test_trend_band_is_causal_and_buffers():
    rng = np.random.default_rng(2)
    px = px_from(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 700))))
    full = portfolio.trend_band(px, 200, 0.02).iloc[:500]
    cut = portfolio.trend_band(px.iloc[:500], 200, 0.02)
    pd.testing.assert_frame_equal(full, cut, check_freq=False)
    flips = lambda w: int((w.diff().abs() > 0).sum().sum())
    assert flips(portfolio.trend_band(px, 200, 0.03)) <= flips(portfolio.trend_band(px, 200, 0.0))
