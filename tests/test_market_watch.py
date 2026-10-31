import numpy as np
import pandas as pd

from jevbot import market_watch


def series(values, end="2026-10-30"):
    return pd.Series(values, index=pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=len(values), freq="D"))


def test_alerts_once_per_new_level_and_reset_on_recovery(tmp_path):
    st = tmp_path / "m.json"
    base = np.full(300, 100.0)
    a = market_watch.drawdown_alerts({"Gold": series(np.r_[base, 88.0])}, st)
    assert len(a) == 1 and "-10%" in a[0]
    assert market_watch.drawdown_alerts({"Gold": series(np.r_[base, 87.0])}, st) == []      # same level: quiet
    a = market_watch.drawdown_alerts({"Gold": series(np.r_[base, 75.0])}, st)
    assert len(a) == 1 and "-20%" in a[0]
    assert market_watch.drawdown_alerts({"Gold": series(np.r_[base, 97.0])}, st) == []      # recovered: reset
    a = market_watch.drawdown_alerts({"Gold": series(np.r_[base, 89.0])}, st)
    assert len(a) == 1 and "-10%" in a[0]


def test_table_renders():
    t = market_watch.table({"Nifty 50": series(np.linspace(100, 120, 400))})
    assert "Nifty 50" in t and "vs 200-day avg" in t
