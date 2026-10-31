"""Intraday strategy search on hourly BTC (2017+), replayed on ETH. Structural rules, not forecasts.

Families (all decided with information available at the time; one position at a time):
  vbo_k[_filter]      volatility breakout: buy when price reaches today's open + k x yesterday's range,
                      sell at the day's close (k 0.3-0.7; filters: none / above 5-day average / daily trend vote)
  tod_aH_L            hold from hour a for L hours each day (UTC); 200+ windows, chosen walk-forward
  imom_n[_filter]     intraday momentum: if open -> hour n is up, hold from hour n to the close
  dip_t_h[_filter]    buy after a 1-hour drop of t%, sell h hours later
  dow_d               hold one weekday from open to close
Scoring: daily net returns after costs (0.12% or 0.08% per side + fills at the trigger price or the
next hour's open if it gapped), walk-forward yearly picks (best in-sample Sharpe, and 3rd best),
PBO (CSCV, 16 blocks) and deflated Sharpe over the pool; India: 31.2% on every winning trade.
"""
import math
import sys

import numpy as np
import pandas as pd

from india_test import perf
from jevbot.data import load_or_fetch
from search_loop import deflated_sharpe, pbo, walk_forward

TAX = 0.312


def hourly_matrix(h: pd.DataFrame):
    h = h.copy()
    h["day"], h["hour"] = h.index.floor("D"), h.index.hour
    full = h.groupby("day").hour.count() == 24
    h = h[h.day.isin(full[full].index)]
    piv = {c: h.pivot(index="day", columns="hour", values=c).to_numpy() for c in ("open", "high", "low", "close")}
    days = h.day.unique()
    return pd.DatetimeIndex(days), piv["open"], piv["high"], piv["low"], piv["close"]


def daily_filters(days, C):
    close = pd.Series(C[:, 23], index=days)
    vote = sum((close > close.rolling(n).mean()).astype(float) for n in (50, 100, 200)) / 3
    return {"none": np.ones(len(days), bool),
            "ma5": (close > close.rolling(5).mean()).shift(1).fillna(False).to_numpy(bool),
            "trend": (vote >= 2 / 3).shift(1).fillna(False).to_numpy(bool)}


def candidates(days, O, H, L, C, hourly_close):
    n = len(days)
    filt = daily_filters(days, C)
    out = {}                                       # name -> (family, entry price array, exit price array) per day
    rng_prev = np.r_[np.nan, (H.max(1) - L.min(1))[:-1]]
    for k in (0.3, 0.4, 0.5, 0.6, 0.7):
        trig = O[:, 0] + k * rng_prev
        hit = H >= trig[:, None]
        first = np.where(hit.any(1), hit.argmax(1), -1)
        entry = np.full(n, np.nan)
        ok = (first >= 0) & (first < 23) & np.isfinite(trig)
        idx = np.where(ok)[0]
        entry[idx] = np.maximum(trig[idx], O[idx, first[idx]])         # gapped above: fill at that hour's open
        for f, m in filt.items():
            e = np.where(m, entry, np.nan)
            out[f"vbo_{k}_{f}"] = ("vbo", e, C[:, 23])
    for a in range(0, 24):
        for Lh in (1, 2, 3, 4, 6, 8, 12):
            b = a + Lh - 1
            if b > 23:
                continue
            out[f"tod_{a:02d}h_{Lh}"] = ("tod", O[:, a].copy(), C[:, b].copy())
    for nh in (4, 8, 12):
        up = C[:, nh - 1] / O[:, 0] - 1 > 0
        for f, m in filt.items():
            out[f"imom_{nh}_{f}"] = ("imom", np.where(up & m, C[:, nh - 1], np.nan), C[:, 23])
    wd = days.weekday
    for d in range(7):
        out[f"dow_{d}"] = ("dow", np.where(wd == d, O[:, 0], np.nan), C[:, 23])
    return out, filt


def dip_trades(hc: pd.Series, thr, hold, allowed_day: pd.Series):
    """Buy at the close of an hour that fell >= thr, sell hold hours later; returns per-exit-day gross returns."""
    r = hc.pct_change().to_numpy()
    px = hc.to_numpy()
    days = hc.index.floor("D")
    ok_day = allowed_day.reindex(days).fillna(False).to_numpy(bool)
    rets, exit_days, t = [], [], 0
    while t < len(px) - hold:
        if r[t] <= -thr and ok_day[t]:
            rets.append(px[t + hold] / px[t] - 1)
            exit_days.append(days[t + hold])
            t += hold
        t += 1
    return pd.Series(rets, index=pd.DatetimeIndex(exit_days))


def to_daily(days, entry, exit_, cost):
    g = exit_ / entry - 1
    net = np.where(np.isfinite(entry), (1 + g) * (1 - cost) ** 2 - 1, 0.0)
    return pd.Series(net, index=days)


def trade_stats(net: pd.Series, traded: np.ndarray):
    t = net[traded]
    if len(t) == 0:
        return {}
    wins, losses = t[t > 0], t[t <= 0]
    taxed = np.where(t > 0, t * (1 - TAX), t)
    years = (net.index[-1] - net.index[0]).days / 365.25
    return {"trades_per_year": round(len(t) / years), "win_pct": round((t > 0).mean() * 100, 1),
            "avg_win_pct": round(wins.mean() * 100, 2) if len(wins) else 0, "avg_loss_pct": round(losses.mean() * 100, 2),
            "per_trade_after_india_tax_pct": round(taxed.mean() * 100, 3),
            "after_india_tax_cagr_pct": round((np.prod(1 + taxed) ** (1 / years) - 1) * 100, 1)}


def study(sym, cost, first_test, verbose=True):
    h = load_or_fetch(sym, "1h", 3300)
    days, O, H, L, C = hourly_matrix(h)
    cands, filt = candidates(days, O, H, L, C, h.close)
    nets, traded = {}, {}
    for name, (fam, e, x) in cands.items():
        nets[name] = to_daily(days, e, x, cost)
        traded[name] = np.isfinite(e)
    for thr in (0.02, 0.03, 0.04):
        for hold in (4, 12, 24):
            for f in ("none", "trend"):
                allowed = pd.Series(filt[f], index=days)
                tr = dip_trades(h.close, thr, hold, allowed)
                net = ((1 + tr) * (1 - cost) ** 2 - 1).groupby(level=0).sum().reindex(days).fillna(0.0)
                name = f"dip_{int(thr * 100)}_{hold}_{f}"
                nets[name] = net
                traded[name] = (net != 0).to_numpy()
    N = pd.DataFrame(nets)
    warm = 230
    scored = N.iloc[warm:]
    wf_best, wf_robust, picks = walk_forward(N, first_test, warm)
    dsr, j, best_sr = deflated_sharpe(scored.to_numpy())
    pb = pbo(scored.to_numpy())
    # benchmarks over the same test years
    close = pd.Series(C[:, 23], index=days)
    hold_r = close.pct_change().fillna(0.0)
    vote = sum((close > close.rolling(n).mean()).astype(float) for n in (50, 100, 200)) / 3
    vol = close.pct_change().rolling(60).std() * math.sqrt(365)
    w = (np.floor((vote * (0.6 / vol).clip(upper=1)).fillna(0) / 0.05) * 0.05).shift(1).fillna(0)
    daily_rule = w * hold_r - w.diff().abs().fillna(0) * cost
    m = N.index >= pd.Timestamp(first_test, tz="UTC")
    fam_of = lambda n_: n_.split("_")[0]
    full_rows = []
    for name in N.columns:
        p = perf(name, N[name].iloc[warm:], 365)
        full_rows.append({"family": fam_of(name), **p, **trade_stats(N[name].iloc[warm:], traded[name][warm:])})
    full = pd.DataFrame(full_rows).set_index("name")
    if verbose:
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 60):
            print(f"\n######## {sym} hourly, cost {cost * 100:.2f}% per side, {days[0]:%Y-%m-%d} -> {days[-1]:%Y-%m-%d}, "
                  f"{N.shape[1]} intraday rules")
            print("WALK-FORWARD (each year's rule picked only from earlier years), test years from", first_test[:4])
            print("  best in-sample pick :", perf("wf_best", wf_best, 365))
            print("  3rd-best pick       :", perf("wf_robust", wf_robust, 365))
            print("  daily trend rule    :", perf("daily_rule", daily_rule[m], 365))
            print("  hold                :", perf("hold", hold_r[m], 365))
            print("  yearly picks:", ", ".join(f"{y}:{p}" for y, p in picks))
            print(f"OVERFITTING: PBO {pb:.2f} (above 0.5 = mostly luck); deflated Sharpe of the full-period best "
                  f"({N.columns[j]}, Sharpe {best_sr:.2f}) = {dsr:.3f}")
            print("\nBEST 3 OF EACH FAMILY over the whole period (after costs; tax columns for India):")
            cols = ["family", "cagr_pct", "max_dd_pct", "sharpe", "trades_per_year", "win_pct", "avg_win_pct",
                    "avg_loss_pct", "per_trade_after_india_tax_pct", "after_india_tax_cagr_pct"]
            top = full.sort_values("sharpe", ascending=False).groupby("family").head(3)
            print(top[cols].sort_values(["family", "sharpe"], ascending=[True, False]).to_string())
    return full


def main():
    for cost in (0.0012, 0.0008):
        study("BTCUSDT", cost, "2019-01-01")
        study("ETHUSDT", cost, "2019-01-01")


if __name__ == "__main__":
    main()
