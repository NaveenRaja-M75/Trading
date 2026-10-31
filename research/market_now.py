"""Where gold, Nifty and Bitcoin stand today, and what history says about gold slumps like this one."""
import numpy as np
import pandas as pd

from india_test import yahoo


def status(name, s):
    s = s.dropna()
    last, d = s.iloc[-1], s.index[-1]
    ch = lambda days: (last / s.asof(d - pd.Timedelta(days=days)) - 1) * 100
    hi52, ath = s[s.index >= d - pd.Timedelta(days=365)].max(), s.max()
    ma50, ma200 = s.rolling(50).mean().iloc[-1], s.rolling(200).mean().iloc[-1]
    vol = s.pct_change().rolling(30).std().iloc[-1] * np.sqrt(252) * 100
    print(f"{name:16s} {d:%Y-%m-%d}  1w {ch(7):+6.1f}%  1m {ch(30):+6.1f}%  3m {ch(91):+6.1f}%  6m {ch(182):+6.1f}%  "
          f"1y {ch(365):+6.1f}% | from 1y high {(last / hi52 - 1) * 100:+6.1f}%  from all-time high {(last / ath - 1) * 100:+6.1f}% "
          f"({s.idxmax():%Y-%m-%d}) | vs 50d {(last / ma50 - 1) * 100:+5.1f}%  vs 200d {(last / ma200 - 1) * 100:+5.1f}% | "
          f"30d vol {vol:.0f}%/yr")


def main():
    fx = yahoo("INR=X")
    gold_usd = yahoo("GC=F")
    gold_inr = (gold_usd * fx.reindex(fx.index.union(gold_usd.index)).ffill().reindex(gold_usd.index)).dropna()
    nifty = yahoo("^NSEI")
    btc = yahoo("BTC-USD", max_move=0.45)
    print("\nTODAY")
    for name, s in (("gold (USD)", gold_usd), ("gold (rupees)", gold_inr), ("Nifty 50", nifty), ("Bitcoin (USD)", btc), ("USD/INR", fx)):
        status(name, s)

    print("\nGOLD'S BIGGEST DAILY MOVES IN THE LAST 12 MONTHS (USD)")
    r = gold_usd.pct_change()
    r = r[r.index >= r.index[-1] - pd.Timedelta(days=365)]
    for dt, v in r.abs().sort_values(ascending=False).head(6).items():
        print(f"  {dt:%Y-%m-%d} {r[dt] * 100:+.1f}%")

    print("\nHISTORY: after gold (in rupees) was this far below its 200-day average, what came next?")
    g = gold_inr[gold_inr.index >= "2004-01-01"]
    gap = g / g.rolling(200).mean() - 1
    now_gap = gap.iloc[-1]
    rows = []
    for label, mask in (("all days since 2005", gap.notna()),
                        (f"days >= {abs(now_gap) * 100 * 0.7:.0f}% below the 200d avg (like now)", gap <= now_gap * 0.7)):
        idx = g.index[mask & (g.index <= g.index[-1] - pd.Timedelta(days=365))]
        row = {"situation": label, "days": len(idx)}
        for m in (3, 6, 12):
            fwd = np.array([g.asof(d + pd.Timedelta(days=int(m * 30.4))) / g[d] - 1 for d in idx]) * 100
            row[f"next {m}m median"] = f"{np.median(fwd):+.1f}%"
            row[f"next {m}m up"] = f"{(fwd > 0).mean() * 100:.0f}%"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("situation").to_string())
    print(f"\nPlan impact: gold is 35% of the plan, so every -10% in gold is about -3.5% on the whole plan.")


if __name__ == "__main__":
    main()
