"""Can anything beat the core-satellite mix? Six strategy families on Nifty, gold, BTC and ETH in rupees.

Families (each with three nearby settings, so a winner must not hang on one lucky number):
  core_sat_rule200   40/40 Nifty/gold held, crypto share on the 200-day rule      crypto 10/20/30%
  core_sat_ensemble  same, crypto on a 50/100/200-day vote x volatility target    crypto 10/20/30%
  fixed_mix          40/40 + crypto held, rebalanced (sell high / buy low)         crypto 10/20/30%
  dual_momentum      monthly: top 2 of 4 by past return, only if it was positive   lookback 90/180/365d
  risk_parity_trend  inverse-volatility weights, crypto only above its 200-day    vol window 30/60/120d
  trend_ensemble_all every asset 25% x its own moving-average vote                 averages short/mid/long

Realistic simulation: holdings drift with prices; trades happen when a target changes by more than
2 points or, monthly, when drift exceeds 5 points; every trade pays 0.15%. Cash earns nothing.
Locked test: the last 730 days are hidden; the family with the best WORST-neighbour Sharpe on the
older data is picked by rule, then scored once on the locked years.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from india_test import yahoo, perf
from jevbot.data import load_or_fetch

COST = 0.0015
VAULT_DAYS = 730


def panel() -> pd.DataFrame:
    nifty, usdinr, gold_usd = yahoo("^NSEI"), yahoo("INR=X"), yahoo("GC=F")
    btc, eth = (load_or_fetch(s, "1d", 3300).close for s in ("BTCUSDT", "ETHUSDT"))
    btc.index, eth.index = btc.index.normalize(), eth.index.normalize()
    cal = pd.date_range(max(btc.index[0], eth.index[0]), min(btc.index[-1], nifty.index[-1]), freq="D", tz="UTC")
    ff = lambda s: s.reindex(s.index.union(cal)).ffill().reindex(cal)
    fx = ff(usdinr)
    return pd.DataFrame({"NIFTY": ff(nifty), "GOLD": ff(gold_usd * ff(usdinr).reindex(gold_usd.index)).ffill(),
                         "BTC": ff(btc) * fx, "ETH": ff(eth) * fx}).dropna()


def simulate(px: pd.DataFrame, target: pd.DataFrame, cost=COST, band=0.05, jump=0.02) -> pd.Series:
    """Daily net returns with drifting holdings, signal trades and monthly band rebalancing."""
    r = px.pct_change(fill_method=None).fillna(0.0).to_numpy()
    tg = target.reindex(px.index).fillna(0.0).to_numpy()
    n, k = r.shape
    h = np.zeros(k)          # value held per asset (equity starts at 1, all cash)
    cash, prev_t = 1.0, np.zeros(k)
    out = np.zeros(n)
    for t in range(n):
        eq0 = cash + h.sum()
        h = h * (1 + r[t])
        eq = cash + h.sum()
        w_now = h / eq if eq > 0 else h
        changed = np.abs(tg[t] - prev_t).max() > jump
        drifted = (t % 30 == 0) and np.abs(w_now - tg[t]).max() > band
        if changed or drifted or (t == 0 and tg[t].sum() > 0):
            new_h = tg[t] * eq
            fee = np.abs(new_h - h).sum() * cost
            h = new_h * (1 - fee / max(eq, 1e-12))
            cash = eq - h.sum() - fee
            prev_t = tg[t].copy()
        out[t] = (cash + h.sum()) / eq0 - 1
    return pd.Series(out, index=px.index)


def above(px, n, band=0.0):
    return (px > px.rolling(n).mean() * (1 + band)).astype(float)


def rule200(s):
    from jevbot.chart import latch
    avg = s.rolling(200).mean()
    return latch(s > avg * 1.02, s < avg * 0.98).clip(lower=0)


def vol(px, n):
    return px.pct_change(fill_method=None).rolling(n).std() * np.sqrt(365)


def families(px: pd.DataFrame) -> dict:
    F = {}
    core = lambda c: pd.DataFrame({"NIFTY": 0.5 - c / 2, "GOLD": 0.5 - c / 2}, index=px.index)
    for c in (0.10, 0.20, 0.30):
        w = core(c)
        w["BTC"], w["ETH"] = c / 2 * rule200(px.BTC), c / 2 * rule200(px.ETH)
        F[f"core_sat_rule200_c{int(c * 100)}"] = ("core_sat_rule200", w)

        w = core(c)
        for a in ("BTC", "ETH"):
            vote = (above(px[a], 50) + above(px[a], 100) + above(px[a], 200)) / 3
            scale = (0.60 / vol(px[a], 60)).clip(upper=1.0).fillna(0)
            w[a] = c / 2 * (vote * scale).round(2)          # rounding stops daily micro-trades
        F[f"core_sat_ensemble_c{int(c * 100)}"] = ("core_sat_ensemble", w)

        w = core(c)
        w["BTC"], w["ETH"] = c / 2, c / 2
        F[f"fixed_mix_c{int(c * 100)}"] = ("fixed_mix", w)

    month = pd.Series(np.arange(len(px)) // 30, index=px.index)
    for lb in (90, 180, 365):
        mom = px / px.shift(lb) - 1
        rank = mom.where(mom > 0).rank(axis=1, ascending=False)
        w = (rank <= 2).astype(float) / 2
        w = w.groupby(month).transform("first")             # decide once a month
        F[f"dual_momentum_{lb}d"] = ("dual_momentum", w)

    for vn in (30, 60, 120):
        iv = 1 / vol(px, vn)
        ok = pd.DataFrame({"NIFTY": 1.0, "GOLD": 1.0, "BTC": rule200(px.BTC), "ETH": rule200(px.ETH)})
        raw = iv * ok
        w = raw.div(iv.sum(axis=1), axis=0).fillna(0)       # out-of-trend crypto share stays in cash
        w = np.floor(w.groupby(month).transform("first") * 100) / 100   # round down: never above 100%
        F[f"risk_parity_trend_{vn}d"] = ("risk_parity_trend", w)

    for name, ns in (("short", (20, 50, 100)), ("mid", (50, 100, 200)), ("long", (100, 200, 300))):
        vote = sum(above(px, n) for n in ns) / len(ns)
        F[f"trend_ensemble_all_{name}"] = ("trend_ensemble_all", 0.25 * vote)

    hold = lambda d: ("benchmark", pd.DataFrame(d, index=px.index))
    F["hold_nifty_gold_50_50"] = hold({"NIFTY": .5, "GOLD": .5, "BTC": 0.0, "ETH": 0.0})
    F["hold_all_4_equal"] = hold({"NIFTY": .25, "GOLD": .25, "BTC": .25, "ETH": .25})
    F["hold_btc"] = hold({"NIFTY": 0.0, "GOLD": 0.0, "BTC": 1.0, "ETH": 0.0})
    F["hold_nifty"] = hold({"NIFTY": 1.0, "GOLD": 0.0, "BTC": 0.0, "ETH": 0.0})
    return F


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    px = panel()
    start = px.index[300]                                   # every indicator warmed up
    cut = px.index[-VAULT_DAYS]
    print(f"data {px.index[0]:%Y-%m-%d} -> {px.index[-1]:%Y-%m-%d}; scored from {start:%Y-%m-%d}; "
          f"LOCKED from {cut:%Y-%m-%d}\n", flush=True)
    F = families(px)
    nets, turns = {}, {}
    for name, (fam, w) in F.items():
        nets[name] = simulate(px, w)

    def table(lo, hi):
        rows = []
        for name, (fam, _) in F.items():
            s = nets[name][(nets[name].index >= lo) & (nets[name].index < hi)]
            rows.append({"family": fam, **perf(name, s, 365)})
        return pd.DataFrame(rows).set_index("name")

    dev = table(start, cut)
    fam = dev[dev.family != "benchmark"].groupby("family").sharpe.agg(["min", "median"])
    best = fam.sort_values(["min", "median"], ascending=False).index[0]
    members = dev[dev.family == best].sort_values("sharpe")
    pick = members.index[len(members) // 2]
    vault = table(cut, px.index[-1] + pd.Timedelta(days=1))
    full = table(start, px.index[-1] + pd.Timedelta(days=1))

    with pd.option_context("display.width", 230, "display.max_columns", 20, "display.max_rows", 100):
        print(f"OLDER DATA ({start:%Y-%m-%d} -> {cut:%Y-%m-%d}), sorted by Sharpe")
        print(dev.sort_values("sharpe", ascending=False).to_string())
        print("\nFAMILIES: worst and median Sharpe across their three settings")
        print(fam.sort_values(["min", "median"], ascending=False).to_string())
        print(f"\nPICKED BY RULE before opening the locked years: {pick}  (family {best})")
        print(f"\nLOCKED YEARS ({cut:%Y-%m-%d} -> {px.index[-1]:%Y-%m-%d}), opened once")
        keep = [pick, "core_sat_rule200_c20", "hold_nifty_gold_50_50", "hold_all_4_equal", "hold_btc", "hold_nifty"]
        print(vault.loc[list(dict.fromkeys(keep))].drop(columns="family").to_string())
        print("\n(every candidate on the locked years, for transparency; NOT used for picking)")
        print(vault.sort_values("sharpe", ascending=False).drop(columns="family").to_string())
        print(f"\nWHOLE PERIOD ({start:%Y-%m-%d} -> {px.index[-1]:%Y-%m-%d})")
        print(full.sort_values("sharpe", ascending=False).drop(columns="family").to_string())
        yr = pd.DataFrame({k: ((1 + nets[k][nets[k].index >= start]).groupby(lambda d: d.year).prod() - 1) * 100
                           for k in dict.fromkeys(keep)}).round(1)
        print("\nYEAR BY YEAR (%)")
        print(yr.to_string())

    fig, ax = plt.subplots(figsize=(11, 5))
    for k in dict.fromkeys(keep):
        s = nets[k][nets[k].index >= start]
        ax.plot((1 + s).cumprod(), label=k, lw=1.8 if k == pick else .9)
    ax.axvline(cut, color="#c00", ls="--", lw=.8)
    ax.text(cut, ax.get_ylim()[1], " locked test", color="#c00", va="top", fontsize=8)
    ax.set_yscale("log")
    ax.set_title("Growth of 1 rupee, after costs (log scale)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("reports/better_than_200.png", dpi=80)
    pd.DataFrame({k: v for k, v in nets.items()}).to_csv("reports/better_than_200_daily.csv")


if __name__ == "__main__":
    main()
