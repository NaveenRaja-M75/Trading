import json

import numpy as np
import pandas as pd

from jevbot import alerts, tax
from jevbot.broker import BotBook


def frame(prices, end="2026-10-28"):
    idx = pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=len(prices), freq="D")
    return pd.DataFrame({"close": prices}, index=idx)


def test_alerts_fire_on_target_change_crash_and_bad_data(tmp_path):
    up = 100 * np.exp(np.cumsum(np.full(300, 0.004)))
    now = pd.Timestamp("2026-10-29 01:00", tz="UTC")
    st = tmp_path / "s.json"
    assert alerts.check({"BTCUSDT": frame(up)}, st, None, now) == []          # first run: just remembers
    crash = up.copy()
    crash[-1] = crash[-2] * 0.85                                              # -15% day, trend breaks
    found = alerts.check({"BTCUSDT": frame(crash)}, st, None, now)
    assert any("fell -15.0% in a day" in a for a in found)
    stale = alerts.check({"BTCUSDT": frame(up, end="2026-10-20")}, st, None, now)
    assert any("data problem" in a for a in stale)


def test_paper_drawdown_alert(tmp_path):
    paper = tmp_path / "p.json"
    paper.write_text(json.dumps({"history": [{"equity": 20}, {"equity": 25}, {"equity": 20}]}))
    up = 100 * np.exp(np.cumsum(np.full(300, 0.004)))
    found = alerts.check({"BTCUSDT": frame(up)}, tmp_path / "s.json", paper, pd.Timestamp("2026-10-29 01:00", tz="UTC"))
    assert any("below its peak" in a for a in found)


def test_tax_report_taxes_gains_not_losses(tmp_path):
    b = BotBook()
    b.on_buy(1.0, 100.0, 100.0)
    b.on_sell(0.5, 80.0, 160.0)      # gain 30
    b.on_sell(0.5, 40.0, 80.0)       # loss -10: not deductible
    path = tmp_path / "live_weekly_main_BTCUSDT.json"
    b.save(path)
    n, total = tax.write([path], 88.0, tmp_path / "vda.csv")
    assert n == 2
    assert total == round(30 * 0.312 * 88)
    text = (tmp_path / "vda.csv").read_text()
    assert "BTCUSDT" in text and "-10.0" in text
