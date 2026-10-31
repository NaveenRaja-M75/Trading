"""Intraday volatility breakout on Indian indices (tradeable via NIFTYBEES / BANKBEES), vs crypto.

Rule (Larry Williams): buy when price reaches today's open + k x yesterday's (high - low); sell at the
close (intraday, squared off the same day). Needs only daily OHLC: entered if the day's high reached
the trigger; fill at the trigger, or at the open if the day gapped above it. Filters: none / previous
close above its 5-day average / daily trend vote (50/100/200) >= 2/3, all known before the open.
Costs per side: 0.05% and 0.08% (discount-broker intraday charges + slippage on an ETF).
Tax, India: intraday equity is speculative business income, taxed at your slab with losses offsetting
gains (confirm with a CA). Shown at a 30% slab (yearly net gains taxed, losses carried forward) and
at 0% (income under the rebate limit). Walk-forward yearly pick from k x filter; test years 2012+.
"""
import json
import urllib.request

import numpy as np
import pandas as pd

from india_test import perf


def ohlc(sym: str) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.request.quote(sym)}?period1=0&period2=9999999999&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        res = json.loads(r.read())["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")},
                      index=pd.to_datetime(res["timestamp"], unit="s", utc=True).normalize()).dropna()
    df = df[(df > 0).all(axis=1)]
    df = df[~df.index.duplicated(keep="last")]
    # data quality: Yahoo's older Indian index rows sometimes have open == previous close or high == low
    flat = (df.high == df.low).mean() * 100
    stale_open = (df.open == df.close.shift()).mean() * 100
    print(f"{sym}: {len(df)} days {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}; high==low {flat:.1f}%, "
          f"open==prev close {stale_open:.1f}%", flush=True)
    good = ~((df.high == df.low) | (df.open == df.close.shift()))
    first_good = good.rolling(250).mean()
    start = first_good[first_good > 0.98].index.min()
    return df[df.index >= start] if pd.notna(start) else df


def vbo(df, k, filt, cost):
    rng = (df.high - df.low).shift(1)
    trig = df.open + k * rng
    entered = df.high >= trig
    fill = np.maximum(trig, df.open)
    close = df.close
    vote = sum((close > close.rolling(n).mean()).astype(float) for n in (50, 100, 200)) / 3
    allow = {"none": pd.Series(True, index=df.index),
             "ma5": (close > close.rolling(5).mean()).shift(1).fillna(False),
             "trend": (vote >= 2 / 3).shift(1).fillna(False)}[filt]
    take = entered & allow & rng.notna()
    net = ((close / fill) * (1 - cost) ** 2 - 1).where(take, 0.0)
    return net, take


def after_tax(net: pd.Series, rate: float) -> float:
    eq, carry = 1.0, 0.0
    for y, r in net.groupby(net.index.year):
        start = eq
        eq *= float(np.prod(1 + r))
        gain = eq - start + carry
        if gain > 0:
            eq -= rate * gain
            carry = 0.0
        else:
            carry = gain
    years = (net.index[-1] - net.index[0]).days / 365.25
    return round((eq ** (1 / years) - 1) * 100, 1)


def study(name, df, cost, first_test="2012-01-01"):
    grid = {f"vbo_{k}_{f}": vbo(df, k, f, cost) for k in (0.3, 0.4, 0.5, 0.6, 0.7) for f in ("none", "ma5", "trend")}
    nets = pd.DataFrame({n: v[0] for n, v in grid.items()})
    warm = 220
    cuts = list(pd.date_range(pd.Timestamp(first_test, tz="UTC"), nets.index[-1], freq="YS"))
    parts, picks = [], []
    for i, c in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else nets.index[-1] + pd.Timedelta(days=1)
        train = nets[(nets.index >= nets.index[warm]) & (nets.index < c)]
        best = (train.mean() / train.std()).idxmax()
        picks.append(f"{c.year}:{best}")
        parts.append(nets.loc[(nets.index >= c) & (nets.index < end), best])
    wf = pd.concat(parts)
    hold = df.close.pct_change().fillna(0.0)[wf.index]
    print(f"\n######## {name}, cost {cost * 100:.2f}% per side, test years {first_test[:4]} -> now")
    print("  walk-forward pick :", perf("wf", wf, 252), f"| after tax: 30% slab {after_tax(wf, 0.30)}%/yr, 0% {after_tax(wf, 0.0)}%/yr")
    print("  hold the index    :", perf("hold", hold, 252))
    print("  picks:", ", ".join(picks))
    rows = []
    for n_, (net, take) in grid.items():
        s = net[net.index >= wf.index[0]]
        t = s[take[s.index]]
        rows.append({"rule": n_, **{k: v for k, v in perf(n_, s, 252).items() if k != "name"},
                     "trades_per_year": round(len(t) / ((s.index[-1] - s.index[0]).days / 365.25)),
                     "win_pct": round((t > 0).mean() * 100, 1), "avg_trade_pct": round(t.mean() * 100, 3),
                     "after_tax_30pct_cagr": after_tax(s, 0.30)})
    with pd.option_context("display.width", 230, "display.max_columns", 20):
        print(pd.DataFrame(rows).set_index("rule").sort_values("sharpe", ascending=False).head(8).to_string())
    yr = (1 + wf).groupby(wf.index.year).prod() - 1
    print("  walk-forward pick by year: " + ", ".join(f"{y} {v * 100:+.1f}%" for y, v in yr.items()))


def main():
    for sym, name in (("^NSEI", "Nifty 50 (trade via NIFTYBEES)"), ("^NSEBANK", "Bank Nifty (trade via BANKBEES)")):
        df = ohlc(sym)
        for cost in (0.0005, 0.0008):
            study(name, df, cost)
    btc = ohlc("BTC-USD")
    study("Bitcoin (cross-check vs hourly result)", btc, 0.0012, "2017-01-01")


if __name__ == "__main__":
    main()
