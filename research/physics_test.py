"""Crypto-sleeve signals, old and new, including two physics / engineering ideas.

  rule200          live setup: 200-day average with 2% buffer
  vote             50/100/200-day averages vote (0, 1/3, 2/3, 1)
  vote_voltgt      vote x volatility target (60%/yr): the crypto sleeve of the new winning mix
  kalman           Kalman filter (rocket-guidance tracking) estimates the trend's slope; hold while > 0
  hurst_vote       vote, but only while the Hurst exponent (fractal physics) says the market is
                   trending (H > 0.5) rather than mean-reverting
  hold             buy and hold

Three nearby settings per idea. Judged by a yearly walk-forward: each year's pick is made only from
earlier years (best worst-neighbour Sharpe), then traded that year. Note: these years were already
seen in earlier research, so the walk-forward procedure is the protection, not a fresh vault.
BTC + ETH from Binance (2017+), and BTC alone from Yahoo (2014+) for a longer history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from better_than_200 import simulate
from india_test import perf, yahoo
from jevbot.chart import latch
from jevbot.data import load_or_fetch


def rule200(s, n=200, band=0.02):
    avg = s.rolling(n).mean()
    return latch(s > avg * (1 + band), s < avg * (1 - band)).clip(lower=0)


def vote(s, ns=(50, 100, 200)):
    return sum((s > s.rolling(n).mean()).astype(float) for n in ns) / len(ns)


def voltgt(s, target=0.60, n=60):
    v = s.pct_change(fill_method=None).rolling(n).std() * np.sqrt(365)
    return (target / v).clip(upper=1.0).fillna(0.0)


def kalman_slope(s, k):
    """Local linear trend Kalman filter on log price. k = slope noise / measurement noise."""
    y = np.log(s.to_numpy())
    r = np.nanvar(np.diff(y[:365])) if len(y) > 365 else np.nanvar(np.diff(y))
    q = np.diag([r * 0.01, r * k])
    x, P = np.array([y[0], 0.0]), np.eye(2) * r
    F, H = np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([1.0, 0.0])
    slope = np.zeros(len(y))
    for t in range(len(y)):
        x, P = F @ x, F @ P @ F.T + q                       # predict
        S = H @ P @ H + r
        K = P @ H / S
        x = x + K * (y[t] - H @ x)                          # update with today's close only
        P = P - np.outer(K, H @ P)
        slope[t] = x[1]
    slope[:365] = np.nan          # noise level was calibrated on year one: stay silent until it is past
    return pd.Series(slope, index=s.index)


def hurst(s, w, q=10):
    """Variance-ratio Hurst estimate over a rolling window: >0.5 trending, <0.5 mean-reverting."""
    lr = np.log(s).diff()
    v1 = lr.rolling(w).var()
    vq = lr.rolling(q).sum().rolling(w).var()
    return 0.5 + 0.5 * np.log((vq / (q * v1)).clip(lower=1e-6)) / np.log(q)


def candidates(px: pd.DataFrame) -> dict:
    n = len(px.columns)
    each = lambda sig: pd.DataFrame({c: sig(px[c]) / n for c in px.columns}).fillna(0.0)
    C = {}
    for nm, (a, b) in {"tight": (150, .02), "mid": (200, .02), "wide": (250, .03)}.items():
        C[f"rule200_{nm}"] = ("rule200", each(lambda s, a=a, b=b: rule200(s, a, b)))
    for nm, ns in {"short": (20, 50, 100), "mid": (50, 100, 200), "long": (100, 200, 300)}.items():
        C[f"vote_{nm}"] = ("vote", each(lambda s, ns=ns: vote(s, ns)))
        C[f"vote_voltgt_{nm}"] = ("vote_voltgt", each(lambda s, ns=ns: (vote(s, ns) * voltgt(s)).round(2)))
    for k in (1e-4, 1e-3, 1e-2):
        C[f"kalman_{k:g}"] = ("kalman", each(lambda s, k=k: (kalman_slope(s, k) > 0).astype(float)))
    for w in (64, 128, 256):
        C[f"hurst_vote_{w}"] = ("hurst_vote", each(lambda s, w=w: vote(s) * (hurst(s, w) > 0.5)))
    C["hold"] = ("benchmark", pd.DataFrame(1.0 / n, index=px.index, columns=px.columns))
    return C


def table(nets, F, lo, hi):
    rows = []
    for name, (fam, _) in F.items():
        s = nets[name][(nets[name].index >= lo) & (nets[name].index < hi)]
        if len(s) > 60:
            rows.append({"family": fam, **perf(name, s, 365)})
    return pd.DataFrame(rows).set_index("name")


def select(dev):
    fam = dev[dev.family != "benchmark"].groupby("family").sharpe.agg(["min", "median"])
    best = fam.sort_values(["min", "median"], ascending=False).index[0]
    m = dev[dev.family == best].sort_values("sharpe")
    return m.index[len(m) // 2]


def study(title, px, first_test):
    F = candidates(px)
    nets = {k: simulate(px, w, cost=0.0012) for k, (_, w) in F.items()}
    start = px.index[300]
    cuts = list(pd.date_range(pd.Timestamp(first_test, tz="UTC"), px.index[-1], freq="YS"))
    rows, stitched = [], []
    for i, c in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else px.index[-1] + pd.Timedelta(days=1)
        pick = select(table(nets, F, start, c))
        t = table(nets, F, c, end)
        rows.append({"year": c.year, "pick": pick, "pick_pct": t.loc[pick, "total_pct"],
                     "pick_dd": t.loc[pick, "max_dd_pct"], "rule200_mid_pct": t.loc["rule200_mid", "total_pct"],
                     "vote_voltgt_mid_pct": t.loc["vote_voltgt_mid", "total_pct"], "hold_pct": t.loc["hold", "total_pct"]})
        m = (px.index >= c) & (px.index < end)
        stitched.append(nets[pick][m])
    wf = pd.concat(stitched)
    period = (wf.index[0], wf.index[-1] + pd.Timedelta(days=1))
    summ = pd.DataFrame([perf("walk-forward picks", wf, 365)] +
                        [perf(k, nets[k][(nets[k].index >= period[0]) & (nets[k].index < period[1])], 365)
                         for k in ("rule200_mid", "vote_voltgt_mid", "kalman_0.001", "hurst_vote_128", "hold")])
    full = table(nets, F, start, px.index[-1] + pd.Timedelta(days=1))
    with pd.option_context("display.width", 230, "display.max_columns", 20, "display.max_rows", 100):
        print(f"\n######## {title}: {px.index[0]:%Y-%m-%d} -> {px.index[-1]:%Y-%m-%d}")
        print("\nWALK-FORWARD (each year's pick uses only earlier years)")
        print(pd.DataFrame(rows).set_index("year").to_string())
        print(f"\nSTITCHED TEST YEARS {period[0]:%Y} -> now, vs fixed choices over the same years")
        print(summ.set_index("name").to_string())
        print("\nFAMILIES over the whole period: worst and median Sharpe of the three settings")
        print(full[full.family != "benchmark"].groupby("family").agg(
            worst_sharpe=("sharpe", "min"), median_sharpe=("sharpe", "median"),
            median_cagr=("cagr_pct", "median"), median_dd=("max_dd_pct", "median")).sort_values(
            "worst_sharpe", ascending=False).to_string())
    return nets


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    btc_eth = pd.DataFrame({s[:3]: load_or_fetch(s, "1d", 3300).close for s in ("BTCUSDT", "ETHUSDT")}).dropna()
    nets = study("BTC + ETH (your bot)", btc_eth, "2020-01-01")
    btc_long = yahoo("BTC-USD", max_move=0.45).to_frame("BTC")   # BTC really fell 37% on 2020-03-12
    study("BTC only, longer history", btc_long, "2017-01-01")

    fig, ax = plt.subplots(figsize=(11, 5))
    start = btc_eth.index[300]
    for k in ("rule200_mid", "vote_voltgt_mid", "kalman_0.001", "hurst_vote_128", "hold"):
        s = nets[k][nets[k].index >= start]
        ax.plot((1 + s).cumprod(), label=k, lw=1.6 if k in ("vote_voltgt_mid", "rule200_mid") else .9)
    ax.set_yscale("log")
    ax.set_title("BTC + ETH: growth of 1, after costs (log scale)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("reports/physics_test.png", dpi=80)


if __name__ == "__main__":
    main()
