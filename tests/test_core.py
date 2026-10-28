import numpy as np
import pandas as pd

from jevbot import backtest
from jevbot.broker import BotBook, PaperBroker
from jevbot.data import synthetic
from jevbot.features import FEATURE_COLUMNS, build_features
from jevbot.policy import Policy, RiskGate
from jevbot.signals import LocalModel, Momentum, Signal, default_signals


def test_features_are_causal():
    df = synthetic(600, seed=1)
    full = build_features(df)
    cut = build_features(df.iloc[:400])
    pd.testing.assert_frame_equal(full[FEATURE_COLUMNS].iloc[:400], cut[FEATURE_COLUMNS])


def test_local_model_has_no_lookahead():
    df = synthetic(1500, seed=2)
    m = LocalModel(window=800, retrain=100, min_train=300)
    full = m.scores(build_features(df))
    cut = m.scores(build_features(df.iloc[:1000]))
    np.testing.assert_allclose(full.iloc[:1000].to_numpy(), cut.to_numpy())


class Oracle(Signal):
    """Cheats by peeking at the next bar: proves the backtest pays out when an edge exists."""
    name = "oracle"

    def scores(self, feats):
        return np.sign(feats["fwd_ret"]).fillna(0.0)


def test_oracle_profits_and_costs_bite():
    df = synthetic(3000, seed=3)
    pol = Policy(enter=0.5, exit=0.0, min_hold=0)
    free, _ = backtest.run(df, Oracle(), pol, cost_per_side=0.0)
    costly, _ = backtest.run(df, Oracle(), pol, cost_per_side=0.001)
    assert free.total_return_pct > 50
    assert costly.total_return_pct < free.total_return_pct
    assert costly.cost_paid_pct > 0


def test_no_edge_on_random_walk_after_costs():
    """On pure noise nothing should look like a winner once costs are charged."""
    table = pd.concat([backtest.compare(synthetic(6000, seed=s), default_signals(), Policy(),
                                        cost_per_side=0.0012, symbol=f"NOISE{s}") for s in range(4)])
    assert not any(line.startswith("PASS") for line in backtest.verdict(table))


def test_policy_hysteresis_and_min_hold():
    s = pd.Series([0.6, -0.1, -0.1, -0.1, 0.2, 0.7])
    pos = Policy(enter=0.5, exit=0.0, min_hold=3).targets(s).tolist()
    assert pos == [1, 1, 1, 0, 0, 1]


def test_risk_gate_latches_on_loss_but_not_on_stale_data():
    g = RiskGate(max_position_usdt=50, daily_loss_limit_usdt=5)
    assert g.check(-1, 10) == ""
    assert "stale" in g.check(0, 999)
    assert g.check(0, 10) == ""
    assert "daily loss" in g.check(-6, 10)
    assert "daily loss" in g.check(+100, 10)  # stays tripped


def test_paper_broker_round_trip(tmp_path):
    b = PaperBroker(1000, fee_bps=10, slippage_bps=0, state=tmp_path / "p.json")
    b.buy_usdt(100, price=50_000)
    assert b.book.qty > 0 and abs(b.usdt - 900) < 1e-9
    b.sell_qty(b.book.qty, price=50_000)
    assert b.book.qty == 0
    assert -0.25 < b.book.realized_usdt < -0.15  # two 10 bps fees on ~100 USDT
    again = PaperBroker(1000, 10, 0, tmp_path / "p.json")  # state survives restart
    assert abs(again.usdt - b.usdt) < 1e-9


def test_paper_broker_tiny_and_empty_orders_dont_crash(tmp_path):
    b = PaperBroker(0.5, 10, 0, tmp_path / "p.json")
    assert b.buy_usdt(100, 50_000).startswith("skip")
    assert b.sell_qty(1.0, 50_000).startswith("skip")


def test_book_never_sells_more_than_it_owns():
    book = BotBook()
    book.on_buy(1.0, 100.0, 100.0)
    book.on_sell(5.0, 120.0, 120.0)
    assert book.qty == 0 and abs(book.realized_usdt - 20.0) < 1e-9


def test_momentum_scores_bounded():
    s = Momentum().scores(build_features(synthetic(500)))
    assert s.between(-1, 1).all()


def test_chart_signals_have_no_lookahead():
    from jevbot.signals import Donchian, PullbackRSI2, SRBreakout, Structure, Trend
    df = synthetic(1500, seed=9)
    for sig in (Trend(), Donchian(), Structure(), SRBreakout(), PullbackRSI2()):
        full = sig.scores(build_features(df)).iloc[:1000].to_numpy()
        cut = sig.scores(build_features(df.iloc[:1000])).to_numpy()
        np.testing.assert_array_equal(full, cut, err_msg=sig.name)


def test_swings_are_confirmed_late_not_early():
    from jevbot.chart import swings
    close = np.array([10, 11, 12, 13, 20, 13, 12, 11, 10, 9, 8, 8, 8], dtype=float)
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0},
                      index=pd.date_range("2024", periods=len(close), freq="h"))
    s = swings(df, k=3)
    assert np.isnan(s.last_sh.iloc[6])      # peak at bar 4 not yet confirmed (needs 3 bars after)
    assert s.last_sh.iloc[7] == 20          # confirmed exactly k bars later


def test_chart_description_reads_trend():
    from jevbot.chart import describe
    up = synthetic(400, seed=1, drift=0.002)
    text = describe(up)
    assert "above the 200-bar average" in text and "Price" in text
