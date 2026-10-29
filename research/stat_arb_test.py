"""Test the "Medallion" Ornstein-Uhlenbeck stat-arb rule from master_ai_trading_workflow.md on real data.

Entry rule, copied from the proposal (MedallionStatArbEngine.analyze_pair):
  rolling 60-bar OLS of ln(laggard) on ln(BTC), spread z-score, AR(1) half-life;
  trade when half-life in [0.5, 18] bars, |z| >= 2.2 and (|z|-0.5)*sd*1e4 - costs >= 15 bps.
The proposal has no exit, so: exit when z returns to +/-0.5, or after 3x half-life (2..48 bars).

Variants:
  long_only  buy the laggard when it is cheap (z <= -2.2). The only version a spot account can do.
  pair       buy laggard / short beta x BTC (and the reverse). Needs shorting; shown as the
             theoretical best case.
Controls: the same number of trades with the same holding times at random moments (500 draws).
Costs: 0.10% fee + 0.02% slippage per side per leg. India: 31.2% tax on every winning trade,
losses not deductible (Section 115BBH, as commonly read; confirm with a CA).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from jevbot.data import load_or_fetch

COST = 0.0012
TAX = 0.312
PAIRS = ["ETHUSDT", "SOLUSDT", "BNBUSDT"]
SETS = [("1h", 1095), ("15m", 365)]


def rolling_signal(ly, lx, w=60):
    n = len(ly)
    z, hl, beta, sd = (np.full(n, np.nan) for _ in range(4))
    for t in range(w - 1, n):
        X, Y = lx[t - w + 1:t + 1], ly[t - w + 1:t + 1]
        xm, ym, vx = X.mean(), Y.mean(), X.var()
        if vx < 1e-12:
            continue
        b = ((X - xm) * (Y - ym)).mean() / vx
        S = Y - (ym - b * xm + b * X)
        s = S.std(ddof=1)
        if s < 1e-12:
            continue
        z[t] = (S[-1] - S.mean()) / s
        sl, dS = S[:-1], np.diff(S)
        bs = np.cov(sl, dS)[0, 1] / max(np.var(sl), 1e-12)  # as written in the proposal
        hl[t] = 999.0 if (bs >= -0.01 or 1 + bs <= 0.05) else math.log(2) / max(-math.log(1 + bs), 1e-6)
        beta[t], sd[t] = b, s
    return z, hl, beta, sd


def backtest(py, px, z, hl, beta, sd, mode, delay):
    n, t, out = len(z), 0, []
    while t < n - 2 - delay:
        ok = (np.isfinite(z[t]) and 0.5 <= hl[t] <= 18 and abs(z[t]) >= 2.2
              and (abs(z[t]) - 0.5) * sd[t] * 1e4 - (2 * COST * 1e4) * (2 if mode == "pair" else 1) >= 15)
        if mode == "long_only":
            ok = ok and z[t] <= -2.2
        if not ok:
            t += 1
            continue
        side = 1 if z[t] < 0 else -1
        e = t + delay
        cap = int(min(48, max(2, math.ceil(3 * hl[t]))))
        x = min(n - 1, e + cap)
        for u in range(e + 1, min(n, e + cap + 1)):
            if (side == 1 and z[u] >= -0.5) or (side == -1 and z[u] <= 0.5):
                x = u
                break
        xf = min(n - 1, x + delay)
        ry, rx = py[xf] / py[e] - 1, px[xf] / px[e] - 1
        if mode == "long_only":
            pnl = ry - 2 * COST
        else:
            b = beta[t]
            pnl = (side * (ry - b * rx) - 2 * COST * (1 + abs(b))) / (1 + abs(b))
        out.append({"entry": e, "exit": xf, "bars": xf - e, "pnl": pnl, "z": z[t], "hl": hl[t]})
        t = xf + 1
    return pd.DataFrame(out)


def stats(tr, years, idx):
    if tr.empty:
        return {"trades": 0}
    eq = np.cumprod(1 + tr.pnl.to_numpy())
    taxed = np.where(tr.pnl > 0, tr.pnl * (1 - TAX), tr.pnl)
    eq_tax = np.cumprod(1 + taxed)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    by_year = tr.groupby(idx[tr.entry].year).pnl.apply(lambda p: (np.prod(1 + p) - 1) * 100).round(1).to_dict()
    return {"trades": len(tr), "per_year": round(len(tr) / years), "win_pct": round((tr.pnl > 0).mean() * 100, 1),
            "avg_bps": round(tr.pnl.mean() * 1e4, 1), "total_pct": round((eq[-1] - 1) * 100, 1),
            "after_india_tax_pct": round((eq_tax[-1] - 1) * 100, 1), "max_dd_pct": round(dd * 100, 1),
            "by_year_pct": by_year}


def random_control(py, tr, sims=500, seed=0):
    """Same count and holding times, random entry bars, long the laggard."""
    rng = np.random.default_rng(seed)
    bars = tr.bars.to_numpy()
    n = len(py)
    means = []
    for _ in range(sims):
        e = rng.integers(60, n - bars.max() - 1, len(bars))
        means.append(np.mean(py[e + bars] / py[e] - 1 - 2 * COST))
    return np.array(means)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = []
    for interval, days in SETS:
        btc = load_or_fetch("BTCUSDT", interval, days)
        fig, axes = plt.subplots(len(PAIRS), 1, figsize=(11, 3.2 * len(PAIRS)))
        for ax, sym in zip(axes, PAIRS):
            alt = load_or_fetch(sym, interval, days)
            df = pd.concat([alt.close.rename("y"), btc.close.rename("x")], axis=1).dropna()
            py, px = df.y.to_numpy(), df.x.to_numpy()
            years = (df.index[-1] - df.index[0]).days / 365
            z, hl, beta, sd = rolling_signal(np.log(py), np.log(px))
            print(f"\n=== {sym} vs BTC, {interval}, {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}, "
                  f"{len(df)} bars", flush=True)
            hold = (py[-1] / py[60] - 1) * 100
            for mode in ("long_only", "pair"):
                for delay in (0, 1):
                    tr = backtest(py, px, z, hl, beta, sd, mode, delay)
                    st = stats(tr, years, df.index)
                    row = {"interval": interval, "pair": f"{sym[:-4]}/BTC", "mode": mode,
                           "fill": "same bar" if delay == 0 else "1 bar late", **st}
                    if mode == "long_only" and delay == 0 and len(tr) > 5:
                        ctrl = random_control(py, tr)
                        row["beats_random_pct"] = round((ctrl < tr.pnl.mean()).mean() * 100, 1)
                        ax.plot(df.index[tr.exit], np.cumprod(1 + tr.pnl) * 100 - 100, label="stat-arb long-only")
                        ax.plot(df.index, py / py[60] * 100 - 100, color="#999", lw=.8, label=f"hold {sym[:-4]}")
                        ax.set_title(f"{sym[:-4]}/BTC {interval}: stat-arb long-only vs holding (%)")
                        ax.legend(loc="upper left")
                    summary.append(row)
                    print(f"  {mode:9s} {row['fill']:10s} {st}" +
                          (f"  beats random entries: {row.get('beats_random_pct')}%" if "beats_random_pct" in row else ""))
            print(f"  hold {sym[:-4]} over the same period: {hold:+.1f}%")
        fig.tight_layout()
        fig.savefig(f"reports/stat_arb_{interval}.png", dpi=80)
        plt.close(fig)

    s = pd.DataFrame(summary)
    cols = ["interval", "pair", "mode", "fill", "trades", "per_year", "win_pct", "avg_bps", "total_pct",
            "after_india_tax_pct", "max_dd_pct", "beats_random_pct"]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("\nSUMMARY")
        print(s.reindex(columns=cols).to_string(index=False))
    s.to_csv("reports/stat_arb_summary.csv", index=False)


if __name__ == "__main__":
    main()
