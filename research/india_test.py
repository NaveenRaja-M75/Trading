"""The 200-day trend rule on Indian ETFs (Nifty 50, gold) and a combined INR portfolio with BTC + ETH.

Same rule as the live crypto setup, NOT re-tuned: hold while price is 2% above its 200-day
average, sell when 2% below, checked daily. Nearby settings shown to check it is a plateau.

Data: Yahoo Finance daily, dividend/split-adjusted (NIFTYBEES.NS, GOLDBEES.NS, INR=X), Binance for
crypto. Costs: 0.15% per side for ETFs (spread + exchange charges, zero-brokerage broker), 0.12%
for crypto. Cash earns nothing while out of the market (conservative: a liquid fund pays ~6%).
Tax (simplified; confirm with a CA): ETFs -> yearly net realised gain taxed 20%, losses carried
forward; buy-and-hold taxed 12.5% once at the end. Crypto -> 31.2% on every winning trade.
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

from jevbot import portfolio
from jevbot.data import load_or_fetch

# explicit period1/period2: with range=max Yahoo silently switches to MONTHLY bars
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1=0&period2=9999999999&interval=1d&events=div%2Csplit"
ETF_COST, CRYPTO_COST = 0.0015, 0.0012


def yahoo(sym: str) -> pd.Series:
    for host in ("query1", "query2"):
        try:
            req = urllib.request.Request(YAHOO.format(sym).replace("query1", host),
                                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                res = json.loads(r.read())["chart"]["result"][0]
            idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).normalize()
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or res["indicators"]["quote"][0]["close"]
            s = pd.Series(adj, index=idx, dtype=float).dropna()
            s = s[~s.index.duplicated(keep="last")]
            gap = float(pd.Series(s.index).diff().dt.days.median())
            print(f"{sym}: {len(s)} bars {s.index[0]:%Y-%m-%d} -> {s.index[-1]:%Y-%m-%d}, median gap {gap:.0f} day(s)",
                  flush=True)
            if gap > 3:  # guard: refuse anything that is not daily data
                raise RuntimeError(f"{sym} is not daily data (median gap {gap} days)")
            return s
        except Exception as e:
            print(f"{sym} via {host}: {e!r}", flush=True)
    raise RuntimeError(f"no data for {sym}")


def perf(name, net: pd.Series, per_year: float) -> dict:
    eq = (1 + net).cumprod()
    years = (net.index[-1] - net.index[0]).days / 365.25
    sd = net.std()
    yearly = (1 + net).groupby(net.index.year).prod() - 1
    return {"name": name, "total_pct": round((eq.iloc[-1] - 1) * 100, 1),
            "cagr_pct": round((eq.iloc[-1] ** (1 / years) - 1) * 100, 1),
            "max_dd_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 1),
            "sharpe": round(float(net.mean() / sd * np.sqrt(per_year)) if sd > 0 else 0.0, 2),
            "losing_years": f"{int((yearly < 0).sum())}/{len(yearly)}",
            "worst_year_pct": round(float(yearly.min()) * 100, 1)}


def after_tax_trend(px: pd.Series, w: pd.Series, cost: float, crypto: bool) -> float:
    """Sequential trades on one asset, taxed the Indian way. Returns final multiple of 1."""
    held = w > 0
    eq, carry, start = 1.0, 0.0, None
    year_gain: dict[int, float] = {}
    for i, (t, on) in enumerate(held.items()):
        if on and start is None:
            start = i
        if start is not None and (not on or i == len(held) - 1):
            r = px.iloc[i] / px.iloc[start] * (1 - cost) ** 2 - 1
            gain = eq * r
            if crypto:
                eq += gain * (1 - 0.312) if gain > 0 else gain
            else:
                eq += gain
                year_gain[t.year] = year_gain.get(t.year, 0.0) + gain
            start = None
    if not crypto:
        for y in sorted(year_gain):  # tax paid at year end on net gains, losses carried forward
            g = year_gain[y] + carry
            if g > 0:
                eq -= 0.20 * g
                carry = 0.0
            else:
                carry = g
    return eq


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nifty, gold = yahoo("NIFTYBEES.NS"), yahoo("GOLDBEES.NS")
    usdinr = yahoo("INR=X")

    rows, sweep = [], []
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for ax, (name, s) in zip(axes, (("Nifty 50 (NIFTYBEES)", nifty), ("Gold (GOLDBEES)", gold))):
        px = s.to_frame("x")
        for since in ("2008-01-01", "2015-01-01", "2020-01-01"):
            m = px.index >= pd.Timestamp(since, tz="UTC")
            w = portfolio.trend_band(px, 200, 0.02)
            trend = portfolio.simulate(px, w, ETF_COST, every=1)[m]
            hold = px.x.pct_change().fillna(0.0)[m]
            rows.append({"asset": name, "since": since, **perf("hold", hold, 252)})
            rows.append({"asset": name, "since": since, **perf("200-day rule", trend, 252),
                         "time_in_market_pct": round(float((w.x[m] > 0).mean()) * 100)})
            if since == "2008-01-01":
                tax_hold = 1 + ((1 + hold).prod() - 1) * (1 - 0.125)
                tax_trend = after_tax_trend(px.x[m], w.x[m], ETF_COST, crypto=False)
                rows[-1]["after_tax_x"] = round(tax_trend, 2)
                rows[-2]["after_tax_x"] = round(tax_hold, 2)
                ax.plot((1 + hold).cumprod(), color="#999", lw=.9, label="hold")
                ax.plot((1 + trend).cumprod(), color="#2c7bb6", label="200-day rule")
                ax.set_yscale("log")
                ax.set_title(f"{name}: growth of 1 since 2008 (log scale, after costs, before tax)")
                ax.legend(loc="upper left")
        for ma in (100, 150, 200, 250):
            for band in (0.0, 0.02, 0.03):
                net = portfolio.simulate(px, portfolio.trend_band(px, ma, band), ETF_COST, every=1)
                net = net[net.index >= pd.Timestamp("2008-01-01", tz="UTC")]
                p = perf("", net, 252)
                sweep.append({"asset": name, "ma": ma, "band": band, "cagr_pct": p["cagr_pct"],
                              "max_dd_pct": p["max_dd_pct"], "sharpe": p["sharpe"]})
    fig.tight_layout()
    fig.savefig("reports/india_assets.png", dpi=80)
    plt.close(fig)

    # combined INR portfolio: Nifty, gold, BTC, ETH; 200-day rule on each, 25% slots, cash otherwise
    btc, eth = (load_or_fetch(s, "1d", 3300).close for s in ("BTCUSDT", "ETHUSDT"))
    btc.index, eth.index = btc.index.normalize(), eth.index.normalize()
    cal = pd.date_range(max(btc.index[0], eth.index[0]), min(btc.index[-1], nifty.index[-1]), freq="D", tz="UTC")
    fx = usdinr.reindex(usdinr.index.union(cal)).ffill().reindex(cal)
    panel = pd.DataFrame({
        "NIFTY": nifty.reindex(nifty.index.union(cal)).ffill().reindex(cal),
        "GOLD": gold.reindex(gold.index.union(cal)).ffill().reindex(cal),
        "BTC_INR": btc.reindex(cal).ffill() * fx, "ETH_INR": eth.reindex(cal).ffill() * fx})
    w_all = portfolio.trend_band(panel, 200, 0.02)
    cost = pd.Series({"NIFTY": ETF_COST, "GOLD": ETF_COST, "BTC_INR": CRYPTO_COST, "ETH_INR": CRYPTO_COST})
    rets = panel.pct_change(fill_method=None).fillna(0.0)
    turnover = w_all.diff().abs().fillna(w_all.abs())
    combos = {
        "200-day rule on all 4 (25% each)": (w_all.shift() * rets).sum(axis=1) - (turnover * cost).sum(axis=1).shift().fillna(0),
        "200-day rule, Nifty + gold only (50/50)": None,
        "hold all 4 (25% each, rebalanced daily)": (0.25 * rets).sum(axis=1),
        "hold Nifty + gold 50/50": (0.5 * rets[["NIFTY", "GOLD"]]).sum(axis=1),
        "hold BTC (INR)": rets.BTC_INR,
        "hold Nifty": rets.NIFTY,
    }
    ng = panel[["NIFTY", "GOLD"]]
    w_ng = portfolio.trend_band(ng, 200, 0.02)
    combos["200-day rule, Nifty + gold only (50/50)"] = portfolio.simulate(ng, w_ng, ETF_COST, every=1)
    combo_rows = []
    fig, ax = plt.subplots(figsize=(11, 5))
    start = cal[200]
    for name, net in combos.items():
        net = net[net.index >= start]
        combo_rows.append(perf(name, net, 365))
        ax.plot((1 + net).cumprod(), label=name, lw=1.6 if name.startswith("200-day rule on all") else .9)
    ax.set_yscale("log")
    ax.set_title(f"In rupees since {start:%Y-%m-%d}: growth of 1 (log scale, after costs, before tax)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("reports/india_portfolio.png", dpi=80)

    now = {k: (round(float(panel[k].iloc[-1] / panel[k].rolling(200).mean().iloc[-1] * 100 - 100), 1),
               "HOLD" if w_all[k].iloc[-1] > 0 else "CASH") for k in panel}
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 100):
        print("\nEACH ASSET: 200-day rule vs holding (after costs; after_tax_x = final multiple of 1 after Indian tax)")
        print(pd.DataFrame(rows).to_string(index=False))
        print("\nNEARBY SETTINGS since 2008 (a real effect is a plateau, not one lucky number)")
        print(pd.DataFrame(sweep).pivot_table(index=["asset", "ma"], columns="band",
                                              values=["cagr_pct", "max_dd_pct"]).to_string())
        print(f"\nCOMBINED PORTFOLIO IN RUPEES, since {start:%Y-%m-%d}")
        print(pd.DataFrame(combo_rows).to_string(index=False))
        print("\nYEAR BY YEAR, combined 200-day rule vs hold all 4 (%)")
        yr = pd.DataFrame({k: ((1 + combos[k][combos[k].index >= start]).groupby(lambda d: d.year).prod() - 1) * 100
                           for k in ("200-day rule on all 4 (25% each)", "hold all 4 (25% each, rebalanced daily)",
                                     "hold Nifty")}).round(1)
        print(yr.to_string())
        print("\nTODAY: % above/below the 200-day average, and what the rule says")
        for k, (pct, act) in now.items():
            print(f"  {k:8s} {pct:+6.1f}%  -> {act}")


if __name__ == "__main__":
    main()
