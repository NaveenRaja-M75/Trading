"""Generation-by-generation strategy search that measures its own overfitting.

Each generation adds candidates to the pool. For every pool we report:
  WALK-FORWARD  each year, pick from the pool using ONLY earlier years, trade that year; the stitched
                test years are the honest score of the search procedure itself
                (two pickers: best in-sample Sharpe, and the median of the in-sample top 5)
  PBO           probability of backtest overfitting (Bailey, Borwein, Lopez de Prado, Zhu 2017),
                CSCV with 16 blocks: how often the in-sample winner lands in the bottom half out-of-sample
  DSR           deflated Sharpe ratio (Bailey & Lopez de Prado 2014) of the full-period best, given the
                number of candidates tried: the chance its Sharpe is real rather than the luck of many tries
Generation 5 mutates the best candidates, ranked ONLY on data before the first test year.
Searched on BTC (Yahoo, 2014+); the same procedure is then replayed on ETH, which it never saw.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from india_test import perf, yahoo
from jevbot.chart import latch
from jevbot.data import load_or_fetch
from physics_test import hurst, kalman_slope

COST = 0.0012
STEP = 0.05


# ------------------------------------------------------------------ building blocks
def ma_vote(c, ns):
    return sum((c > c.rolling(n).mean()).astype(float) for n in ns) / len(ns)


def mom_vote(c, lbs):
    return sum((c / c.shift(lb) > 1).astype(float) for lb in lbs) / len(lbs)


def donchian(c, up, dn):
    return latch(c > c.rolling(up).max().shift(1), c < c.rolling(dn).min().shift(1)).clip(lower=0)


def voltgt(c, target, n):
    v = c.pct_change(fill_method=None).rolling(n).std() * math.sqrt(365)
    return (target / v).clip(upper=1.0).fillna(0.0)


def make(c: pd.Series, p: dict) -> pd.Series:
    kind = p["sig"][0]
    if kind == "ma":
        w = ma_vote(c, p["sig"][1])
    elif kind == "mom":
        w = mom_vote(c, p["sig"][1])
    elif kind == "kalman":
        w = (kalman_slope(c, p["sig"][1]) > 0).astype(float)
    elif kind == "donchian":
        w = donchian(c, *p["sig"][1])
    else:
        w = pd.Series(1.0, index=c.index)
    if p.get("vol"):
        w = w * voltgt(c, *p["vol"])
    if p.get("regime") == "hurst":
        w = w * (hurst(c, 128) > 0.5)
    elif p.get("regime") == "ma200":
        w = w * (c > c.rolling(200).mean())
    if p.get("smooth"):
        w = w.ewm(span=p["smooth"], adjust=False).mean()
    return (np.floor(w.fillna(0.0) / STEP) * STEP).clip(0, 1)


def name(p: dict) -> str:
    s = p["sig"]
    out = f"{s[0]}{list(s[1]) if isinstance(s[1], tuple) else s[1]}".replace(" ", "")
    if p.get("vol"):
        out += f"|vt{int(p['vol'][0] * 100)}/{p['vol'][1]}"
    if p.get("regime"):
        out += f"|{p['regime']}"
    if p.get("smooth"):
        out += f"|ema{p['smooth']}"
    return out


def simulate(c: pd.Series, w: pd.Series) -> pd.Series:
    """Decide at today's close, earn from tomorrow; trade only when the target moves >= 5 points."""
    r = c.pct_change(fill_method=None).fillna(0.0).to_numpy()
    tw = w.to_numpy()
    out, h = np.zeros(len(r)), 0.0
    for t in range(len(r)):
        out[t] = h * r[t]
        if abs(tw[t] - h) >= STEP - 1e-9:
            out[t] -= abs(tw[t] - h) * COST
            h = tw[t]
    return pd.Series(out, index=c.index)


# ------------------------------------------------------------------ the generations
CURRENT = {"sig": ("ma", (50, 100, 200)), "vol": (0.60, 60)}
MA_SETS = [(20, 50, 100), (50, 100, 200), (100, 200, 300), (50, 200), (30, 90, 180, 360), (200,)]
MOM_SETS = [(30, 90, 180), (60, 120, 240), (90, 180, 365), (28,), (90,)]
VOLS = [(0.40, 60), (0.60, 60), (0.80, 60), (0.60, 30)]


def generations() -> list[list[dict]]:
    g0 = [CURRENT, {"sig": ("ma", (200,))}, {"sig": ("hold", 0)}]
    g1 = [{"sig": ("ma", s)} for s in MA_SETS] + [{"sig": ("mom", s)} for s in MOM_SETS]
    g2 = [{"sig": b["sig"], "vol": v} for b in g1 for v in VOLS]
    g3 = [{"sig": b["sig"], "vol": (0.60, 60), "regime": rg} for b in g1 for rg in ("hurst", "ma200")]
    g3 += [{"sig": ("kalman", k), "vol": v} for k in (1e-4, 1e-3, 1e-2) for v in (None, (0.60, 60))]
    g4 = [{**b, "smooth": 5} for b in g2] + [{"sig": ("donchian", d), "vol": v}
                                             for d in ((55, 20), (100, 50), (20, 10)) for v in (None, (0.60, 60))]
    return [g0, g1, g2, g3, g4]


def mutate(top: list[dict], rng, n=40) -> list[dict]:
    out = []
    for _ in range(n):
        p = dict(top[rng.integers(len(top))])
        kind, arg = p["sig"]
        if kind in ("ma", "mom") and isinstance(arg, tuple):
            arg = tuple(sorted({max(5, int(round(a * rng.uniform(0.75, 1.3)))) for a in arg}))
            p["sig"] = (kind, arg)
        if p.get("vol"):
            t, n_ = p["vol"]
            p["vol"] = (round(float(np.clip(t * rng.uniform(0.75, 1.3), 0.2, 1.5)), 2), int(rng.choice([30, 60, 90])))
        elif rng.random() < 0.5:
            p["vol"] = (round(float(rng.uniform(0.3, 1.0)), 2), 60)
        if rng.random() < 0.3:
            p["smooth"] = int(rng.choice([3, 5, 10]))
        out.append(p)
    return out


# ------------------------------------------------------------------ overfitting maths
def sharpe(x: np.ndarray) -> float:
    sd = x.std()
    return float(x.mean() / sd) if sd > 0 else 0.0


def pbo(R: np.ndarray, S=16, samples=2000, seed=0) -> float:
    """CSCV probability of backtest overfitting. R: days x candidates daily returns."""
    T, N = R.shape
    blocks = np.array_split(np.arange(T), S)
    s1 = np.stack([R[b].sum(0) for b in blocks])
    s2 = np.stack([(R[b] ** 2).sum(0) for b in blocks])
    cnt = np.array([len(b) for b in blocks])
    rng = np.random.default_rng(seed)
    combos = list(itertools.combinations(range(S), S // 2))
    pick = rng.choice(len(combos), min(samples, len(combos)), replace=False)
    lam = []
    for i in pick:
        ins = np.zeros(S, bool)
        ins[list(combos[i])] = True

        def sr(mask):
            n = cnt[mask].sum()
            m = s1[mask].sum(0) / n
            v = s2[mask].sum(0) / n - m ** 2
            return m / np.sqrt(np.maximum(v, 1e-18))
        best = np.argmax(sr(ins))
        oos = sr(~ins)
        w = (oos < oos[best]).sum() / (N - 1) if N > 1 else 0.5
        w = min(max(w, 1e-6), 1 - 1e-6)
        lam.append(math.log(w / (1 - w)))
    return float((np.array(lam) <= 0).mean())


def deflated_sharpe(R: np.ndarray) -> tuple[float, str, float]:
    """DSR of the full-period best candidate, given N trials (daily Sharpe units)."""
    srs = np.array([sharpe(R[:, j]) for j in range(R.shape[1])])
    j = int(np.argmax(srs))
    N, T = R.shape[1], R.shape[0]
    g = 0.5772156649
    sr0 = math.sqrt(max(srs.var(), 1e-12)) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * math.e))) if N > 1 else 0.0
    x = R[:, j]
    sk, ku = skew(x), kurtosis(x, fisher=False)
    den = math.sqrt(max(1 - sk * srs[j] + (ku - 1) / 4 * srs[j] ** 2, 1e-12))
    return float(norm.cdf((srs[j] - sr0) * math.sqrt(T - 1) / den)), j, srs[j] * math.sqrt(365)


# ------------------------------------------------------------------ the loop
def walk_forward(nets: pd.DataFrame, first_test: str, warm: int):
    start = nets.index[warm]
    cuts = list(pd.date_range(pd.Timestamp(first_test, tz="UTC"), nets.index[-1], freq="YS"))
    best_parts, robust_parts, picks = [], [], []
    for i, c in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else nets.index[-1] + pd.Timedelta(days=1)
        train = nets[(nets.index >= start) & (nets.index < c)]
        srs = train.mean() / train.std().replace(0, np.nan)
        order = srs.sort_values(ascending=False).index
        best, robust = order[0], order[min(2, len(order) - 1)]   # median of the top 5
        m = (nets.index >= c) & (nets.index < end)
        best_parts.append(nets.loc[m, best])
        robust_parts.append(nets.loc[m, robust])
        picks.append((c.year, best))
    return pd.concat(best_parts), pd.concat(robust_parts), picks


def run(c: pd.Series, title: str, pools: list[list[dict]], first_test: str, warm: int, verbose=True):
    rows, nets_all, keep = [], {}, {}
    pool: list[dict] = []
    for g, new in enumerate(pools):
        pool += [p for p in new if name(p) not in {name(q) for q in pool}]
        for p in pool:
            if name(p) not in nets_all:
                nets_all[name(p)] = simulate(c, make(c, p))
        nets = pd.DataFrame({name(p): nets_all[name(p)] for p in pool})
        wf_best, wf_robust, picks = walk_forward(nets, first_test, warm)
        scored = nets[nets.index >= nets.index[warm]].to_numpy()
        dsr, j, best_sr = deflated_sharpe(scored)
        pb = pbo(scored)
        b, r = perf("best", wf_best, 365), perf("robust", wf_robust, 365)
        rows.append({"gen": g, "pool": len(pool), "wf_best_cagr": b["cagr_pct"], "wf_best_dd": b["max_dd_pct"],
                     "wf_best_sharpe": b["sharpe"], "wf_robust_cagr": r["cagr_pct"], "wf_robust_dd": r["max_dd_pct"],
                     "wf_robust_sharpe": r["sharpe"], "PBO": round(pb, 2), "DSR": round(dsr, 3),
                     "full_best": nets.columns[j], "full_best_sharpe": round(best_sr, 2)})
        keep[g] = (wf_best, wf_robust, picks)
    tab = pd.DataFrame(rows).set_index("gen")
    if verbose:
        with pd.option_context("display.width", 260, "display.max_columns", 30, "display.max_colwidth", 60):
            print(f"\n######## {title}: {c.index[0]:%Y-%m-%d} -> {c.index[-1]:%Y-%m-%d}, test years from {first_test[:4]}")
            print(tab.to_string())
            for g in (0, len(pools) - 1):
                print(f"  gen {g} yearly picks:", ", ".join(f"{y}:{p}" for y, p in keep[g][2]))
    return tab, keep, nets_all


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    btc = yahoo("BTC-USD", max_move=0.45)
    gens = generations()

    # generation 5: mutate the best of gens 0-4, ranked ONLY before the first test year (2017)
    pool = [p for g in gens for p in g]
    early = btc[btc.index < pd.Timestamp("2017-01-01", tz="UTC")]
    if len(early) > 500:
        scores = []
        for p in pool:
            x = simulate(early, make(early, p)).iloc[365:].to_numpy()
            scores.append(sharpe(x))
        top = [pool[i] for i in np.argsort(scores)[::-1][:10]]
        gens.append(mutate(top, np.random.default_rng(7)))
        print("gen 5 parents (best before 2017):", [name(p) for p in top[:5]])

    tab_btc, keep_btc, nets_btc = run(btc, "BTC (searched)", gens, "2017-01-01", 365)

    eth = load_or_fetch("ETHUSDT", "1d", 3300).close
    eth.index = eth.index.normalize()
    tab_eth, keep_eth, _ = run(eth, "ETH (never searched: same pools, same procedure)", gens, "2019-01-01", 365)

    # the fixed current rule on the same test years, for reference
    for title, c, first in (("BTC", btc, "2017-01-01"), ("ETH", eth, "2019-01-01")):
        cur = simulate(c, make(c, CURRENT))
        cur = cur[cur.index >= pd.Timestamp(first, tz="UTC")]
        hold = c.pct_change().fillna(0)[c.index >= pd.Timestamp(first, tz="UTC")]
        print(f"\n{title} same test years: current rule {perf('current', cur, 365)}")
        print(f"{title} same test years: hold         {perf('hold', hold, 365)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (title, keep, c, first) in zip(axes, (("BTC (searched)", keep_btc, btc, "2017-01-01"),
                                                    ("ETH (never searched)", keep_eth, eth, "2019-01-01"))):
        for g, (wb, wr, _) in keep.items():
            ax.plot((1 + wr).cumprod(), label=f"gen {g} (robust pick)", lw=.9 + .25 * g)
        cur = simulate(c, make(c, CURRENT))
        cur = cur[cur.index >= pd.Timestamp(first, tz="UTC")]
        ax.plot((1 + cur).cumprod(), color="k", ls="--", lw=1, label="current rule")
        ax.set_yscale("log")
        ax.set_title(f"{title}: walk-forward test years, growth of 1")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig("reports/search_loop.png", dpi=80)


if __name__ == "__main__":
    main()
