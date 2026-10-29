"""Does the free FRED "morning risk dial" improve our live strategy?

Dial (as proposed): share of green lights among
  liquidity  Fed net liquidity = WALCL/1000 - WTREGEN - RRPONTSYD (all in $bn), 4-week change > 0
  credit     high-yield spread BAMLH0A0HYM2, 4-week change <= 0
  btc        BTC close above its 200-day average
  (+ vix variant: VIXCLS < 22)
Rule (pre-set, not tuned): stand down when the dial is under 0.50.

Look-ahead guard: each FRED series is shifted by its publication lag before use
(weekly H.4.1 data +2 days, daily VIX/RRP +1 day, high-yield spread +2 days).
"""
from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd

from jevbot import portfolio
from jevbot.data import load_or_fetch

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
LAG_DAYS = {"WALCL": 2, "WTREGEN": 2, "RRPONTSYD": 1, "VIXCLS": 1, "BAMLH0A0HYM2": 2}
COST = 0.0012


def fred(series: str) -> pd.Series:
    req = urllib.request.Request(FRED.format(series), headers={"User-Agent": "Mozilla/5.0 research"})
    with urllib.request.urlopen(req, timeout=60) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    date_col = df.columns[0]
    s = pd.to_numeric(df[df.columns[1]], errors="coerce")
    s.index = pd.to_datetime(df[date_col]).dt.tz_localize("UTC")
    s = s.dropna()
    s.index = s.index + pd.Timedelta(days=LAG_DAYS[series])  # when it was actually public
    return s


def daily(s: pd.Series, idx) -> pd.Series:
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    px = portfolio.closes({s: load_or_fetch(s, "1d", 2555) for s in ("BTCUSDT", "ETHUSDT")})
    idx = px.index
    raw = {k: daily(fred(k), idx) for k in LAG_DAYS}
    for k, v in raw.items():
        print(f"{k:14s} last value {v.dropna().iloc[-1]:.3f} (as public on {v.dropna().index[-1]:%Y-%m-%d})")
    net_liq = raw["WALCL"] / 1000 - raw["WTREGEN"] - raw["RRPONTSYD"]
    green = pd.DataFrame({
        "liquidity": net_liq.diff(28) > 0,
        "credit": raw["BAMLH0A0HYM2"].diff(28) <= 0,
        "btc": px.BTCUSDT > px.BTCUSDT.rolling(200).mean(),
        "vix": raw["VIXCLS"] < 22,
    }, index=idx)
    dial3 = green[["liquidity", "credit", "btc"]].mean(axis=1)
    dial4 = green[["liquidity", "credit", "btc", "vix"]].mean(axis=1)
    print("\ndial today:", round(float(dial3.iloc[-1]), 2), "(3-light),", round(float(dial4.iloc[-1]), 2), "(with VIX)")
    print("share of days each light was green:", (green.mean() * 100).round(1).to_dict())

    base = portfolio.trend_band(px, 200, 0.02)
    hold = portfolio.equal_hold(px)
    variants = {
        "trend_band (live setup)": base,
        "trend_band x stand-down dial<0.5": base.mul((dial3 >= .5).astype(float), axis=0),
        "trend_band x dial (scaled)": base.mul(dial3, axis=0),
        "trend_band x stand-down, VIX dial": base.mul((dial4 >= .5).astype(float), axis=0),
        "dial alone on BTC+ETH (>=0.5)": hold.mul((dial3 >= .5).astype(float), axis=0),
        "hold BTC+ETH": hold,
    }
    rows = []
    fig, ax = plt.subplots(figsize=(11, 5))
    for since in ("2020-06-01", "2022-01-01"):
        for name, w in variants.items():
            net = portfolio.simulate(px, w, COST, every=1)
            net = net[net.index >= pd.Timestamp(since, tz="UTC")]
            p = portfolio.perf(name, net, float(w[w.index >= pd.Timestamp(since, tz="UTC")].sum(axis=1).mean()))
            rows.append({"since": since, **p.__dict__})
            if since == "2022-01-01":
                ax.plot((1 + net).cumprod(), label=name, lw=1.4 if "live" in name else .9)
    ax.set_yscale("log")
    ax.set_title("BTC+ETH since 2022: does the FRED dial help the 200-day strategy? (log scale, after costs)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("reports/macro_dial.png", dpi=80)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("\n" + pd.DataFrame(rows).set_index(["since", "name"]).to_string())


if __name__ == "__main__":
    main()
