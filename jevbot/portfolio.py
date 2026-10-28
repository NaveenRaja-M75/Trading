"""Weekly multi-coin strategies on daily candles: momentum, rotation, trend filters.

Weights are decided at the close of a rebalance day and earn the following days' returns.
Every change in weights pays `cost_per_side` on the traded amount.

Honesty rules built in:
- The last `vault_days` are a locked final test. Strategies are compared and the winner is
  picked on the older "dev" data only, by a rule fixed in code (best dev Sharpe among
  candidates that also survive nearby settings); the vault is then scored once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def closes(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily close per coin, aligned; NaN before a coin was listed."""
    return pd.DataFrame({s: d["close"] for s, d in dfs.items()}).sort_index()


# --------------------------------------------------------------------------- strategies
# Each takes a close-price panel and returns target weights (rows sum to <= 1, rest is cash).

def equal_hold(px: pd.DataFrame) -> pd.DataFrame:
    w = px.notna().astype(float)
    return w.div(w.sum(axis=1), axis=0).fillna(0.0)


def trend_filter(px: pd.DataFrame, ma: int = 100) -> pd.DataFrame:
    """Equal weight, but only coins trading above their `ma`-day average. Cash otherwise."""
    n = px.notna().sum(axis=1)
    up = (px > px.rolling(ma).mean()).astype(float)
    return up.div(n, axis=0).fillna(0.0)


def ts_momentum(px: pd.DataFrame, lookback: int = 28) -> pd.DataFrame:
    """Each coin: hold its 1/N slice if it rose over the last `lookback` days."""
    n = px.notna().sum(axis=1)
    up = (px / px.shift(lookback) > 1).astype(float)
    return up.div(n, axis=0).fillna(0.0)


def rotation(px: pd.DataFrame, lookback: int = 28, top: int = 3, ma: int = 100) -> pd.DataFrame:
    """Dual momentum: hold the `top` strongest coins, but only those also above their average."""
    mom = px / px.shift(lookback) - 1
    ok = (px > px.rolling(ma).mean()) & (mom > 0)
    rank = mom.where(ok).rank(axis=1, ascending=False)
    pick = (rank <= top).astype(float)
    return (pick / top).fillna(0.0)


def vol_target(w: pd.DataFrame, px: pd.DataFrame, target: float = 0.4, lookback: int = 30) -> pd.DataFrame:
    """Scale exposure down when the held basket is volatile (annualised `target` vol, never > 1x)."""
    rets = px.pct_change(fill_method=None).fillna(0.0)
    port_vol = (w.shift() * rets).sum(axis=1).rolling(lookback).std() * np.sqrt(365)
    coin_vol = rets.rolling(lookback).std() * np.sqrt(365)
    est = np.sqrt(((w * coin_vol) ** 2).sum(axis=1))  # ignores correlation -> conservative when used as cap
    vol = pd.concat([port_vol, est], axis=1).max(axis=1).replace(0, np.nan)
    scale = (target / vol).clip(upper=1.0).fillna(0.0)
    return w.mul(scale, axis=0)


# --------------------------------------------------------------------------- backtest

@dataclass
class Perf:
    name: str
    total_pct: float
    cagr_pct: float
    sharpe: float
    max_dd_pct: float
    weeks_positive_pct: float
    exposure_pct: float


def simulate(px: pd.DataFrame, w: pd.DataFrame, cost_per_side: float, every: int = 7) -> pd.Series:
    """Daily net returns. Weights only change on rebalance days (every `every` days)."""
    rets = px.pct_change(fill_method=None).fillna(0.0)
    held = w.copy()
    mask = np.arange(len(w)) % every != 0
    held.iloc[mask] = np.nan
    held = held.ffill().fillna(0.0)
    # between rebalances, weights drift with prices; approximate cost on the rebalance trades
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    gross = (held.shift() * rets).sum(axis=1)
    return gross - turnover.shift().fillna(0.0) * cost_per_side


def perf(name: str, net: pd.Series, w_exposure: float) -> Perf:
    eq = (1 + net).cumprod()
    years = max(len(net) / 365, 1e-9)
    weekly = (1 + net).resample("W").prod() - 1
    sd = net.std()
    return Perf(name, round((eq.iloc[-1] - 1) * 100, 1), round((eq.iloc[-1] ** (1 / years) - 1) * 100, 1),
                round(float(net.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0, 2),
                round(float((eq / eq.cummax() - 1).min()) * 100, 1),
                round(float((weekly > 0).mean()) * 100, 1), round(w_exposure * 100, 1))


def candidates() -> dict:
    """Strategy families with nearby settings. A winner must work across its neighbours."""
    c = {}
    for ma in (50, 100, 200):
        c[f"trend_filter_ma{ma}"] = ("trend_filter", lambda px, ma=ma: trend_filter(px, ma))
    for lb in (14, 28, 56):
        c[f"ts_momentum_{lb}d"] = ("ts_momentum", lambda px, lb=lb: ts_momentum(px, lb))
    for lb in (14, 28, 56):
        for top in (2, 3, 5):
            c[f"rotation_{lb}d_top{top}"] = ("rotation", lambda px, lb=lb, top=top: rotation(px, lb, top))
    for name, (fam, f) in list(c.items()):
        c[name + "_voltgt"] = (fam + "_voltgt", lambda px, f=f: vol_target(f(px), px))
    return c


def evaluate(px: pd.DataFrame, cost_per_side: float, vault_days: int = 365) -> dict:
    """Dev-period table, the pre-committed pick, and its single vault test."""
    cut = px.index[-vault_days]
    warm = 250  # indicators need history; score from here on
    rows, nets = [], {}
    cands = candidates()
    base = {"equal_hold": ("benchmark", equal_hold), "btc_hold": ("benchmark", lambda p: equal_hold(p[["BTCUSDT"]]).reindex(columns=p.columns, fill_value=0.0))}
    for name, (fam, f) in {**base, **cands}.items():
        w = f(px)
        net = simulate(px, w, cost_per_side)
        nets[name] = (fam, net, w)
        dev = net[(net.index < cut)].iloc[warm:]
        expo = float(w[(w.index < cut)].iloc[warm:].sum(axis=1).mean())
        rows.append({"family": fam, **perf(name, dev, expo).__dict__})
    dev_table = pd.DataFrame(rows).set_index("name")

    # pre-committed selection rule: family whose WORST neighbour Sharpe is highest, then its median member
    fam_stats = dev_table[dev_table.family != "benchmark"].groupby("family").sharpe.agg(["min", "median"])
    best_fam = fam_stats.sort_values(["min", "median"], ascending=False).index[0]
    members = dev_table[dev_table.family == best_fam].sort_values("sharpe")
    pick = members.index[len(members) // 2]

    vault_rows = []
    for name in ("equal_hold", "btc_hold", pick):
        fam, net, w = nets[name]
        v = net[net.index >= cut]
        expo = float(w[w.index >= cut].sum(axis=1).mean())
        vault_rows.append(perf(name, v, expo).__dict__)
    return {"dev": dev_table, "family_stats": fam_stats, "pick": pick, "family": best_fam,
            "vault": pd.DataFrame(vault_rows).set_index("name"), "cut": cut,
            "all_vault": pd.DataFrame([perf(n, nets[n][1][nets[n][1].index >= cut], 0).__dict__
                                       for n in nets]).set_index("name")}
