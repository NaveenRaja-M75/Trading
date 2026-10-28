import numpy as np
import pandas as pd

from jevbot import journal
from jevbot.data import synthetic


def test_learner_is_causal():
    df = synthetic(1500, seed=11)
    full = journal.replay(df, horizon=4).p_up.iloc[:1000].to_numpy()
    cut = journal.replay(df.iloc[:1000], horizon=4).p_up.to_numpy()
    np.testing.assert_allclose(full, cut, equal_nan=True)


def test_no_skill_on_noise():
    led = journal.replay(synthetic(6000, seed=12), horizon=4).iloc[500:]
    card = journal.scorecard(led)
    assert card["hit_rate_pct"] < 53
    assert card["brier_skill_pct"] < 1.0


def test_learner_finds_a_real_pattern():
    """Plant an edge (trends persist) and check the learner picks the right experts."""
    rng = np.random.default_rng(13)
    n = 6000
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.6 * r[t - 1] + rng.normal(0, 0.002)  # strong momentum
    close = 100 * np.exp(np.cumsum(r))
    idx = pd.date_range("2024", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"open": np.r_[100, close[:-1]], "high": close * 1.001, "low": close * 0.999,
                       "close": close, "volume": 1.0}, index=idx)
    card = journal.scorecard(journal.replay(df, horizon=1).iloc[500:])
    assert card["hit_rate_pct"] > 53  # base rate here is ~51%
    assert card["brier_skill_pct"] > 0


def test_situation_labels():
    s = journal.situation(synthetic(800, seed=14))
    assert s.str.count(r"\|").eq(2).all()


def test_live_journal_writes_then_grades(tmp_path):
    df = synthetic(900, seed=15)
    df.index = pd.date_range("2025-01-01", periods=len(df), freq="h", tz="UTC")
    path = tmp_path / "journal.csv"
    j = journal.live_update("BTCUSDT", df.iloc[:800], path, horizon=4)
    assert len(j) == 1 and pd.isna(j.correct.iloc[0])
    journal.live_update("BTCUSDT", df.iloc[:800], path, horizon=4)  # same bar: no duplicate
    assert len(pd.read_csv(path)) == 1
    j = journal.live_update("BTCUSDT", df.iloc[:805], path, horizon=4)  # 5 bars later: first call graded
    first = j.iloc[0]
    assert first.outcome_price == df.close.iloc[803]
    assert len(j) == 2
    assert "graded" in journal.journal_report(j)
