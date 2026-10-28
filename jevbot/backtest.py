"""Vectorised backtest with fees + slippage, compared against coin-flip and buy-and-hold.

Timing: the target decided at the close of bar t earns bar t+1's return. Every change in
position pays `cost_per_side` on the traded fraction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .features import build_features
from .policy import Policy
from .signals import Signal

BARS_PER_YEAR = {"1m": 525_600, "3m": 175_200, "5m": 105_120, "15m": 35_040,
                 "30m": 17_520, "1h": 8_760, "4h": 2_190, "1d": 365}


@dataclass
class Result:
    name: str
    total_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    trades: int
    win_rate_pct: float
    exposure_pct: float
    cost_paid_pct: float

    def row(self) -> dict:
        return asdict(self)


def run(df: pd.DataFrame, signal: Signal, policy: Policy, cost_per_side: float,
        interval: str = "1m", warmup: int = 50) -> tuple[Result, pd.DataFrame]:
    feats = build_features(df)
    scores = signal.scores(feats)
    pos = policy.targets(scores)
    pos.iloc[:warmup] = 0.0

    bar_ret = np.exp(feats["fwd_ret"].fillna(0.0)) - 1  # return from close t to close t+1
    turnover = pos.diff().abs().fillna(pos.abs())
    gross = pos * bar_ret
    cost = turnover * cost_per_side
    net = gross - cost
    equity = (1 + net).cumprod()

    # trade-level stats: a trade runs from entry to the bar the position goes back to 0 / flips
    # (the exit cost lands on the flat bar after the trade, so attribute that bar to the trade)
    trade_id = (pos != pos.shift()).cumsum()
    exits = (pos == 0) & (pos.shift().fillna(0) != 0)
    owner = trade_id.where(pos != 0).fillna(trade_id.shift().where(exits))
    trade_pnl = np.log1p(net[owner.notna()]).groupby(owner.dropna()).sum()

    k = BARS_PER_YEAR.get(interval, 525_600)
    sd = net.std()
    dd = equity / equity.cummax() - 1
    res = Result(
        name=signal.name,
        total_return_pct=round((equity.iloc[-1] - 1) * 100, 3),
        sharpe=round(float(net.mean() / sd * np.sqrt(k)) if sd > 0 else 0.0, 2),
        max_drawdown_pct=round(float(dd.min()) * 100, 3),
        trades=int(len(trade_pnl)),
        win_rate_pct=round(float((trade_pnl > 0).mean() * 100) if len(trade_pnl) else 0.0, 1),
        exposure_pct=round(float((pos != 0).mean() * 100), 1),
        cost_paid_pct=round(float(cost.sum()) * 100, 3),
    )
    detail = pd.DataFrame({"close": df["close"], "score": scores, "position": pos,
                           "net_ret": net, "equity": equity})
    return res, detail


def buy_and_hold(df: pd.DataFrame, cost_per_side: float) -> Result:
    r = df["close"].pct_change().fillna(0.0)
    equity = (1 + r).cumprod() * (1 - cost_per_side) ** 2
    dd = equity / equity.cummax() - 1
    return Result("buy_and_hold", round((equity.iloc[-1] - 1) * 100, 3), 0.0,
                  round(float(dd.min()) * 100, 3), 1, 100.0 if equity.iloc[-1] > 1 else 0.0, 100.0,
                  round(2 * cost_per_side * 100, 3))


def compare(df: pd.DataFrame, signals: dict[str, Signal], policy: Policy, cost_per_side: float,
            interval: str = "1m", split: float = 0.5) -> pd.DataFrame:
    """Score every signal on an in-sample and an out-of-sample half. Trust only out-of-sample."""
    cut = int(len(df) * split)
    rows = []
    for part, chunk in (("in_sample", df.iloc[:cut]), ("out_of_sample", df.iloc[cut:])):
        for sig in signals.values():
            r, _ = run(chunk, sig, policy, cost_per_side, interval)
            rows.append({"period": part, **r.row()})
        rows.append({"period": part, **buy_and_hold(chunk, cost_per_side).row()})
    return pd.DataFrame(rows)


def verdict(table: pd.DataFrame) -> list[str]:
    """Plain-English go/no-go per signal, based only on out-of-sample results."""
    oos = table[table.period == "out_of_sample"].set_index("name")
    base = oos.loc["coinflip", "total_return_pct"] if "coinflip" in oos.index else 0.0
    lines = []
    for name, r in oos.iterrows():
        if name in {"coinflip", "buy_and_hold"}:
            continue
        ok = r.total_return_pct > 0 and r.total_return_pct > base and r.sharpe > 1.0 and r.trades >= 30
        lines.append(f"{'PASS' if ok else 'FAIL'}  {name:10s} oos return {r.total_return_pct:+.2f}% "
                     f"sharpe {r.sharpe:.2f} trades {int(r.trades)} (coinflip {base:+.2f}%)")
    return lines
