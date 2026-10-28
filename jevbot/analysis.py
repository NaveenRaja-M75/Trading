"""Everything about one chosen weekly setup: trades, win rate, stress tests, future scenarios."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import portfolio


def trades(px: pd.DataFrame, held: pd.DataFrame, cost_per_side: float) -> pd.DataFrame:
    """One row per position: a coin bought on a rebalance day and later sold."""
    rows = []
    for s in px.columns:
        on = held[s] > 0
        start = None
        for i, (t, flag) in enumerate(on.items()):
            if flag and start is None:
                start = i
            if start is not None and (not flag or i == len(on) - 1):
                end = i if not flag else i  # sold at this day's close (or still open at the end)
                p0, p1 = px[s].iloc[start], px[s].iloc[end]
                ret = (p1 / p0) * (1 - cost_per_side) ** 2 - 1
                rows.append({"symbol": s, "entry": px.index[start], "exit": px.index[end],
                             "days": end - start, "return_pct": ret * 100, "open": bool(flag)})
                start = None
    return pd.DataFrame(rows)


def trade_stats(t: pd.DataFrame) -> dict:
    closed = t[~t.open]
    wins, losses = closed[closed.return_pct > 0], closed[closed.return_pct <= 0]
    streak = longest = 0
    for r in closed.sort_values("exit").return_pct:
        streak = streak + 1 if r <= 0 else 0
        longest = max(longest, streak)
    gross_win, gross_loss = wins.return_pct.sum(), -losses.return_pct.sum()
    return {
        "trades": len(closed),
        "win_rate_pct": round(len(wins) / max(len(closed), 1) * 100, 1),
        "avg_win_pct": round(wins.return_pct.mean(), 2) if len(wins) else 0.0,
        "avg_loss_pct": round(losses.return_pct.mean(), 2) if len(losses) else 0.0,
        "biggest_win_pct": round(closed.return_pct.max(), 1),
        "biggest_loss_pct": round(closed.return_pct.min(), 1),
        "payoff_ratio": (round(wins.return_pct.mean() / -losses.return_pct.mean(), 2)
                         if len(wins) and len(losses) and losses.return_pct.mean() < 0 else np.nan),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else np.inf,
        "avg_trade_pct": round(closed.return_pct.mean(), 2),
        "longest_losing_streak": longest,
        "avg_days_held": round(closed.days.mean(), 1),
    }


def period_win_rates(net: pd.Series) -> dict:
    out = {}
    for name, rule in (("weeks", "W"), ("months", "ME"), ("quarters", "QE"), ("years", "YE")):
        r = (1 + net).resample(rule).prod() - 1
        active = r[r != 0]
        out[f"{name}_up_pct"] = round((r > 0).mean() * 100, 1)
        out[f"{name}_up_when_invested_pct"] = round((active > 0).mean() * 100, 1) if len(active) else np.nan
    return out


def stress(px: pd.DataFrame, f, cost_per_side: float, since, every: int = 7) -> pd.DataFrame:
    """Does it survive worse conditions than the backtest assumes?"""
    w = f(px)
    rows = []
    cases = {"as tested": (w, cost_per_side, 0),
             "trade 1 day late": (w.shift(1).fillna(0.0), cost_per_side, 0),
             "trade 2 days late": (w.shift(2).fillna(0.0), cost_per_side, 0),
             "double fees+slippage": (w, cost_per_side * 2, 0),
             "4x fees+slippage": (w, cost_per_side * 4, 0)}
    for off in range(1, every):
        cases[f"rebalance day +{off}"] = (w, cost_per_side, off)
    for name, (ww, c, off) in cases.items():
        net = portfolio.simulate(px, ww, c, every=every, offset=off)
        rows.append(portfolio.perf(name, net[net.index >= since], 0.0).__dict__)
    return pd.DataFrame(rows).set_index("name").drop(columns="exposure_pct")


def scenarios(net: pd.Series, budget: float, max_loss: float, weeks: int = 52, sims: int = 10_000,
              block: int = 8, seed: int = 0) -> dict:
    """Block-bootstrap real weekly returns into many possible next years for the given budget."""
    wk = ((1 + net).resample("W").prod() - 1).to_numpy()
    rng = np.random.default_rng(seed)
    n_blocks = -(-weeks // block)
    finals, tripped = np.empty(sims), 0
    for k in range(sims):
        starts = rng.integers(0, len(wk) - block, n_blocks)
        path = np.concatenate([wk[s:s + block] for s in starts])[:weeks]
        eq = budget * np.cumprod(1 + path)
        hit = np.nonzero(eq <= budget - max_loss)[0]
        if len(hit):  # kill switch sells and stops at that level
            tripped += 1
            finals[k] = eq[hit[0]]
        else:
            finals[k] = eq[-1]
    q = np.percentile(finals, [5, 25, 50, 75, 95])
    return {"budget": budget, "chance_up_after_1y_pct": round((finals > budget).mean() * 100, 1),
            "bad_5pct": round(q[0], 2), "q25": round(q[1], 2), "median": round(q[2], 2),
            "q75": round(q[3], 2), "good_5pct": round(q[4], 2),
            "chance_kill_switch_pct": round(tripped / sims * 100, 1),
            "worst": round(finals.min(), 2), "best": round(finals.max(), 2)}
