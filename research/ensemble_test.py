"""Last loop step: average many trend + volatility rules instead of picking one (no selection = no overfitting).

Ensembles (equal-weight average of the exposures, then the same 5-point trading band):
  ens_trendvol   all 44 trend/momentum x volatility-target rules from generation 2
  ens_ma_vol     the 24 moving-average x volatility rules only
  ens_core4      4 classic rules: 50/100/200 vote, 50/200 vote, 90-day momentum, 200-day; all at 60% vol target
Compared with the current rule and holding on the same test years, on BTC (2017+) and ETH (2019+).
"""
import numpy as np
import pandas as pd

from india_test import perf, yahoo
from jevbot.data import load_or_fetch
from search_loop import CURRENT, STEP, generations, make, simulate


def ensemble(c, pool):
    w = sum(make(c, p) for p in pool) / len(pool)
    return (np.floor(w / STEP) * STEP).clip(0, 1)


def main():
    btc = yahoo("BTC-USD", max_move=0.45)
    eth = load_or_fetch("ETHUSDT", "1d", 3300).close
    eth.index = eth.index.normalize()
    g2 = generations()[2]
    pools = {
        "ens_trendvol (44 rules)": g2,
        "ens_ma_vol (24 rules)": [p for p in g2 if p["sig"][0] == "ma"],
        "ens_core4 (4 classic rules)": [{"sig": ("ma", (50, 100, 200)), "vol": (0.6, 60)},
                                        {"sig": ("ma", (50, 200)), "vol": (0.6, 60)},
                                        {"sig": ("mom", (90,)), "vol": (0.6, 60)},
                                        {"sig": ("ma", (200,)), "vol": (0.6, 60)}],
    }
    for title, c, first in (("BTC", btc, "2017-01-01"), ("ETH", eth, "2019-01-01")):
        rows, yearly = [], {}
        m = c.index >= pd.Timestamp(first, tz="UTC")
        cands = {"current rule": make(c, CURRENT), **{k: ensemble(c, v) for k, v in pools.items()}}
        for k, w in cands.items():
            net = simulate(c, w)[m]
            rows.append(perf(k, net, 365) | {"avg_exposure_pct": round(float(w[m].mean()) * 100),
                                             "trades_per_year": round(float((w[m].diff().abs() >= STEP).sum()) / (m.sum() / 365), 1)})
            yearly[k] = ((1 + net).groupby(net.index.year).prod() - 1) * 100
        hold = c.pct_change().fillna(0)[m]
        rows.append(perf("hold", hold, 365))
        yearly["hold"] = ((1 + hold).groupby(hold.index.year).prod() - 1) * 100
        with pd.option_context("display.width", 230, "display.max_columns", 20):
            print(f"\n######## {title}, test years {first[:4]} -> now")
            print(pd.DataFrame(rows).set_index("name").to_string())
            print(pd.DataFrame(yearly).round(1).to_string())


if __name__ == "__main__":
    main()
