"""Plain SIP vs smart SIP, same monthly budget (Rs 1,000), Nifty 50, gold (rupees), Bitcoin (rupees), 2008+.

  plain   invest the full Rs 1,000 every month
  smart   invest more when the asset is cheap vs its 200-day average, less when expensive:
          >=15% below: 2x | 5-15% below: 1.5x | within -5%..+10%: 1x | 10-25% above: 0.75x | >25% above: 0.5x
          Unspent money waits in a liquid fund (6%/yr); extra buys come from that fund, never borrowed.
  pause   skip buying while the asset is below its 200-day average (cash waits at 6%), buy it all back
          when it is above again: what "do not buy a falling asset" looks like in practice
Same total money in. Scored on what it is worth at the end (holdings + cash) and on the worst drop.
Exchange data for Bitcoin; Yahoo for the index and gold, guarded against bad ticks.
"""
import numpy as np
import pandas as pd

from india_test import yahoo
from jevbot.data import load_or_fetch

BUDGET, CASH_RATE = 1000.0, 0.06


def tier(gap):
    return 2.0 if gap <= -0.15 else 1.5 if gap <= -0.05 else 1.0 if gap <= 0.10 else 0.75 if gap <= 0.25 else 0.5


def run(px: pd.Series, mode: str, start: str):
    ma = px.rolling(200).mean()
    months = px[px.index >= pd.Timestamp(start, tz="UTC")].groupby(lambda d: (d.year, d.month)).head(1).index
    units, cash, put_in, path = 0.0, 0.0, 0.0, []
    m_rate = (1 + CASH_RATE) ** (1 / 12) - 1
    for d in months:
        cash *= 1 + m_rate
        cash += BUDGET
        put_in += BUDGET
        p, gap = px[d], px[d] / ma[d] - 1
        if mode == "plain":
            spend = cash
        elif mode == "smart":
            spend = min(cash, BUDGET * tier(gap))
        else:
            spend = cash if gap >= 0 else 0.0
        units += spend / p
        cash -= spend
        path.append((d, units * p + cash, put_in))
    df = pd.DataFrame(path, columns=["day", "worth", "put_in"]).set_index("day")
    final = df.worth.iloc[-1]
    peak = df.worth.cummax()
    return {"put_in": round(df.put_in.iloc[-1]), "worth_at_end": round(final), "x": round(final / df.put_in.iloc[-1], 2),
            "worst_drop_pct": round(float((df.worth / peak - 1).min()) * 100, 1),
            "cash_left_pct": round(cash / final * 100, 1)}


def main():
    fx = yahoo("INR=X")
    g = yahoo("GC=F")
    gold = (g * fx.reindex(fx.index.union(g.index)).ffill().reindex(g.index)).dropna()
    nifty = yahoo("^NSEI")
    btc = load_or_fetch("BTCUSDT", "1d", 3300).close
    btc.index = btc.index.normalize()
    btc_inr = (btc * fx.reindex(fx.index.union(btc.index)).ffill().reindex(btc.index)).dropna()
    for name, px, starts in (("Nifty 50", nifty, ("2008-07-01", "2013-01-01", "2018-01-01")),
                             ("Gold (rupees)", gold, ("2008-07-01", "2013-01-01", "2018-01-01")),
                             ("Bitcoin (rupees)", btc_inr, ("2018-06-01", "2020-01-01", "2022-01-01"))):
        print(f"\n######## {name}  (Rs {BUDGET:,.0f} a month)")
        rows = []
        for st in starts:
            for mode in ("plain", "smart", "pause"):
                rows.append({"start": st[:7], "sip": mode, **run(px, mode, st)})
        with pd.option_context("display.width", 200):
            print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
