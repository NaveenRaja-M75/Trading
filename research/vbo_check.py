"""Why did the BTC volatility breakout make ~+67%/yr from daily candles but ~+15%/yr from hourly candles?

Same rule (k = 0.7, no filter, 0.12% per side), computed four ways on overlapping days:
  A  hourly path, Binance hourly (enter at the first hour that reaches the trigger)
  B  daily-OHLC method on Binance hourly candles aggregated to days
  C  daily-OHLC method on Binance daily candles
  D  daily-OHLC method on Yahoo BTC-USD daily candles
"""
import numpy as np
import pandas as pd

from india_test import perf
from intraday_india import ohlc, vbo
from intraday_search import hourly_matrix, to_daily
from jevbot.data import load_or_fetch

K, COST = 0.7, 0.0012


def main():
    h = load_or_fetch("BTCUSDT", "1h", 3300)
    days, O, H, L, C = hourly_matrix(h)
    rng_prev = np.r_[np.nan, (H.max(1) - L.min(1))[:-1]]
    trig = O[:, 0] + K * rng_prev
    hit = H >= trig[:, None]
    first = np.where(hit.any(1), hit.argmax(1), -1)
    entry = np.full(len(days), np.nan)
    ok = (first >= 0) & np.isfinite(trig)
    idx = np.where(ok)[0]
    entry[idx] = np.maximum(trig[idx], O[idx, first[idx]])
    A = to_daily(days, entry, C[:, 23], COST)

    agg = pd.DataFrame({"open": O[:, 0], "high": H.max(1), "low": L.min(1), "close": C[:, 23]}, index=days)
    B, _ = vbo(agg, K, "none", COST)
    d = load_or_fetch("BTCUSDT", "1d", 3300)
    d.index = d.index.normalize()
    Cd, _ = vbo(d[["open", "high", "low", "close"]], K, "none", COST)
    Dy, _ = vbo(ohlc("BTC-USD"), K, "none", COST)

    common = A.index.intersection(B.index).intersection(Cd.index).intersection(Dy.index)
    common = common[common >= common[0] + pd.Timedelta(days=5)]
    tab = pd.DataFrame({"A hourly path": A[common], "B daily-OHLC, Binance hourly": B[common],
                        "C daily-OHLC, Binance daily": Cd[common], "D daily-OHLC, Yahoo": Dy[common]})
    print(f"{len(common)} common days {common[0]:%Y-%m-%d} -> {common[-1]:%Y-%m-%d}\n")
    for c in tab:
        print(f"{c:32s}", perf(c, tab[c], 365))
    print("\nyear by year (%):")
    print(((1 + tab).groupby(tab.index.year).prod() - 1).mul(100).round(1).to_string())
    diff = (tab["A hourly path"] - tab["B daily-OHLC, Binance hourly"]).abs()
    print(f"\nA vs B differ by >0.1% on {(diff > 0.001).sum()} days (same data, two methods)")
    yb = (tab["D daily-OHLC, Yahoo"] - tab["C daily-OHLC, Binance daily"]).abs()
    print(f"Yahoo vs Binance differ by >1% on {(yb > 0.01).sum()} days; largest:")
    for dt in yb.sort_values(ascending=False).head(8).index:
        y = Dy.name if hasattr(Dy, "name") else "D"
        print(f"  {dt:%Y-%m-%d}  Binance {tab.loc[dt, 'C daily-OHLC, Binance daily'] * 100:+.2f}%  Yahoo {tab.loc[dt, 'D daily-OHLC, Yahoo'] * 100:+.2f}%")
    yo = ohlc("BTC-USD")
    j = yo.index.intersection(d.index)
    for col in ("open", "high", "low", "close"):
        rel = (yo.loc[j, col] / d.loc[j, col] - 1).abs()
        print(f"Yahoo vs Binance daily {col}: median gap {rel.median() * 100:.2f}%, days >1% apart: {(rel > 0.01).sum()}")


if __name__ == "__main__":
    main()
