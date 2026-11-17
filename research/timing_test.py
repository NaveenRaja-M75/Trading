"""Does it matter WHEN the daily check runs, or if the computer misses days?

The live rule (vote_voltgt, BTC, 5% trade band, 0.12% per side) decides from the daily close at
00:00 UTC (05:30 IST). In real life the order goes in later: 5 minutes later on a PC that is on,
about 7 hours later on GitHub's delayed schedule, or the next day if the PC was off.
  A  same signal, order placed k hours after the close (k = 0, 1, 3, 7, 12, 23)
  B  the PC only runs every n days (n = 1, 2, 3, 7)
  C  the PC misses a random 30% of days (5 different random draws)
Hourly Binance data, so each late order is filled at the real price at that hour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_regime_test import BAND, COST
from india_test import perf
from jevbot.data import load_or_fetch
from jevbot.portfolio import vote_voltgt


def hold_series(target: pd.Series, act: np.ndarray) -> pd.Series:
    """Position after applying the 5% band, acting only on days where `act` is True."""
    out, held = [], 0.0
    for x, ok in zip(target.fillna(0.0).to_numpy(), act):
        if ok and (abs(x - held) >= BAND or (x == 0 and held > 0)):
            held = x
        out.append(held)
    return pd.Series(out, index=target.index)


def run(h: pd.Series, daily_pos: pd.Series, lag_hours: int) -> pd.Series:
    """daily_pos[d] is decided at the close of day d (= 00:00 of d+1) and filled lag_hours later."""
    when = daily_pos.index + pd.Timedelta(days=1, hours=lag_hours)
    pos = pd.Series(daily_pos.to_numpy(), index=when).reindex(h.index, method="ffill").fillna(0.0)
    r = h.pct_change().fillna(0.0)
    net = pos.shift().fillna(0.0) * r - pos.diff().abs().fillna(0.0) * COST
    return (1 + net).groupby(net.index.floor("D")).prod() - 1


def main() -> None:
    hourly = load_or_fetch("BTCUSDT", "1h", 3300)
    h = hourly.close.copy()
    h.index = h.index + pd.Timedelta(hours=1)                 # label each bar by its close time
    daily = h[h.index.hour == 0]
    daily.index = daily.index - pd.Timedelta(days=1)          # the day that just closed
    target = vote_voltgt(daily.to_frame("BTCUSDT")).BTCUSDT
    every_day = np.ones(len(target), bool)
    start = target.index[260]
    rows = []
    for k in (0, 1, 3, 7, 12, 23):
        net = run(h, hold_series(target, every_day), k)[start:]
        rows.append({**perf(f"A order {k:2d}h after the close", net, 365)})
    for n in (2, 3, 7):
        act = (np.arange(len(target)) % n) == 0
        rows.append({**perf(f"B PC runs every {n} days", run(h, hold_series(target, act), 0)[start:], 365)})
    for seed in range(5):
        act = np.random.default_rng(seed).random(len(target)) > 0.30
        rows.append({**perf(f"C misses 30% of days (draw {seed})", run(h, hold_series(target, act), 0)[start:], 365)})
    hold = h.pct_change().fillna(0.0)
    rows.append(perf("hold BTC", ((1 + hold).groupby(hold.index.floor("D")).prod() - 1)[start:], 365))
    with pd.option_context("display.width", 200):
        print(f"BTC hourly {h.index[0]:%Y-%m-%d} -> {h.index[-1]:%Y-%m-%d}; scored from {start:%Y-%m-%d}\n")
        print(pd.DataFrame(rows).set_index("name").to_string())


if __name__ == "__main__":
    main()
