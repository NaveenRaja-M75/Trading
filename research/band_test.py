"""Remove weekday luck: check the 100-day trend every day, with a buffer band to avoid flip-flopping.

Compared against the weekly version on all 7 possible rebalance weekdays, BTC + ETH, after costs.
Then: how different loss limits change next-year outcomes for 20 USDT.
"""
import numpy as np
import pandas as pd

from jevbot import analysis, chart, portfolio
from jevbot.data import load_or_fetch

COST = 0.0012
px = portfolio.closes({s: load_or_fetch(s, "1d", 2555) for s in ("BTCUSDT", "ETHUSDT")})
n = px.notna().sum(axis=1)


def band_weights(ma: int, band: float) -> pd.DataFrame:
    w = {}
    for s in px.columns:
        c = px[s]
        avg = c.rolling(ma).mean()
        w[s] = chart.latch(c > avg * (1 + band), c < avg * (1 - band)).clip(lower=0)
    return pd.DataFrame(w).div(n, axis=0).fillna(0.0)


def run(w, every, offset=0, since="2022-01-01"):
    net = portfolio.simulate(px, w, COST, every=every, offset=offset)
    return net[net.index >= pd.Timestamp(since, tz="UTC")]


rows = []
for since in ("2020-06-01", "2022-01-01"):
    for ma in (50, 100, 200):
        weekly = [portfolio.perf("", run(band_weights(ma, 0.0), 7, off, since), 0) for off in range(7)]
        rows.append({"since": since, "setup": f"weekly ma{ma} (7 weekdays)",
                     "total_min": min(p.total_pct for p in weekly), "total_max": max(p.total_pct for p in weekly),
                     "total_avg": round(np.mean([p.total_pct for p in weekly]), 1),
                     "dd_worst": min(p.max_dd_pct for p in weekly),
                     "sharpe_avg": round(np.mean([p.sharpe for p in weekly]), 2)})
        for band in (0.0, 0.02, 0.03, 0.05):
            w = band_weights(ma, band)
            p = portfolio.perf("", run(w, 1, 0, since), 0)
            held = portfolio.held_weights(w, every=1)
            t = analysis.trade_stats(analysis.trades(px, held, COST))
            rows.append({"since": since, "setup": f"daily ma{ma} band {band:.0%}", "total_min": p.total_pct,
                         "total_max": p.total_pct, "total_avg": p.total_pct, "dd_worst": p.max_dd_pct,
                         "sharpe_avg": p.sharpe, "trades": t["trades"], "win_rate": t["win_rate_pct"],
                         "payoff": t["payoff_ratio"]})
pd.set_option("display.width", 220)
print("Weekly (range over the 7 possible rebalance weekdays) vs daily check with a buffer band, BTC + ETH:\n")
print(pd.DataFrame(rows).to_string(index=False))

print("\nNext 12 months for 20 USDT with different loss limits (daily ma100 band 3%, since 2022):")
net = run(band_weights(100, 0.03), 1)
for ml in (4, 5, 7, 10, 20):
    r = analysis.scenarios(net, 20, ml)
    print(f"  loss limit {ml:>2} USDT: chance up {r['chance_up_after_1y_pct']:5.1f}%  median {r['median']:6.2f}  "
          f"bad 5% {r['bad_5pct']:6.2f}  good 5% {r['good_5pct']:6.2f}  limit hit {r['chance_kill_switch_pct']:5.1f}%")
