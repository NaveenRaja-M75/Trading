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


def buy_and_hold(df: pd.DataFrame, cost_per_side: float, interval: str = "1m") -> Result:
    r = df["close"].pct_change().fillna(0.0)
    equity = (1 + r).cumprod() * (1 - cost_per_side) ** 2
    dd = equity / equity.cummax() - 1
    sd = r.std()
    sharpe = float(r.mean() / sd * np.sqrt(BARS_PER_YEAR.get(interval, 525_600))) if sd > 0 else 0.0
    return Result("buy_and_hold", round((equity.iloc[-1] - 1) * 100, 3), round(sharpe, 2),
                  round(float(dd.min()) * 100, 3), 1, 100.0 if equity.iloc[-1] > 1 else 0.0, 100.0,
                  round(2 * cost_per_side * 100, 3))


def compare(df: pd.DataFrame, signals: dict[str, Signal], policy: Policy, cost_per_side: float,
            interval: str = "1m", split: float = 0.5, symbol: str = "") -> pd.DataFrame:
    """Score every signal on an in-sample and an out-of-sample half. Trust only out-of-sample.

    Parameters are fixed up front (never tuned on this data), so in-sample is just context.
    """
    cut = int(len(df) * split)
    rows = []
    for part, chunk in (("in_sample", df.iloc[:cut]), ("out_of_sample", df.iloc[cut:])):
        for sig in signals.values():
            r, _ = run(chunk, sig, policy, cost_per_side, interval)
            rows.append({"symbol": symbol, "period": part, **r.row()})
        rows.append({"symbol": symbol, "period": part, **buy_and_hold(chunk, cost_per_side, interval).row()})
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample results averaged across coins, one row per signal."""
    oos = table[table.period == "out_of_sample"]
    g = oos.groupby("name")
    out = pd.DataFrame({
        "coins": g.size(),
        "coins_profitable": g.total_return_pct.apply(lambda x: int((x > 0).sum())),
        "avg_return_pct": g.total_return_pct.mean().round(2),
        "avg_sharpe": g.sharpe.mean().round(2),
        "avg_max_dd_pct": g.max_drawdown_pct.mean().round(2),
        "total_trades": g.trades.sum(),
    })
    return out.sort_values("avg_sharpe", ascending=False)


def verdict(table: pd.DataFrame) -> list[str]:
    """Plain-English go/no-go per signal, from out-of-sample results across all coins.

    PASS needs ALL of: average return > 0, profitable on >= 75% of coins, average Sharpe > 0.8,
    beats coin-flip, smaller average drawdown than simply holding, and >= 20 trades in total.
    """
    sm = summarize(table)
    base = sm.loc["coinflip", "avg_return_pct"] if "coinflip" in sm.index else -np.inf
    hold_dd = sm.loc["buy_and_hold", "avg_max_dd_pct"] if "buy_and_hold" in sm.index else -100.0
    hold_ret = sm.loc["buy_and_hold", "avg_return_pct"] if "buy_and_hold" in sm.index else 0.0
    lines = []
    for name, r in sm.iterrows():
        if name in {"coinflip", "buy_and_hold"}:
            continue
        checks = {
            "positive": r.avg_return_pct > 0,
            "most coins": r.coins_profitable >= 0.75 * r.coins,
            "sharpe>0.8": r.avg_sharpe > 0.8,
            "beats coinflip": r.avg_return_pct > base,
            "less drawdown than holding": r.avg_max_dd_pct > hold_dd,
            "enough trades": r.total_trades >= 20,
        }
        failed = [k for k, v in checks.items() if not v]
        lines.append(f"{'PASS' if not failed else 'FAIL'}  {name:11s} avg {r.avg_return_pct:+7.2f}% "
                     f"sharpe {r.avg_sharpe:5.2f} dd {r.avg_max_dd_pct:7.2f}% "
                     f"profitable {int(r.coins_profitable)}/{int(r.coins)} trades {int(r.total_trades)}"
                     + (f"   missing: {', '.join(failed)}" if failed else ""))
    lines.append(f"(holding: avg {hold_ret:+.2f}%, drawdown {hold_dd:.2f}%)")
    return lines


def plot(df: pd.DataFrame, signal: Signal, policy: Policy, cost_per_side: float, interval: str,
         path: str, title: str) -> None:
    """Price with entries/exits on top, strategy equity vs holding below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, d = run(df, signal, policy, cost_per_side, interval)
    hold = df["close"] / df["close"].iloc[0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[3, 2])
    ax1.plot(d.index, d.close, lw=0.8, color="#555")
    chg = d.position.diff().fillna(d.position)
    ax1.scatter(d.index[chg > 0], d.close[chg > 0], marker="^", color="#1a9850", s=28, label="buy", zorder=3)
    ax1.scatter(d.index[chg < 0], d.close[chg < 0], marker="v", color="#d73027", s=28, label="sell", zorder=3)
    ax1.fill_between(d.index, d.close.min(), d.close.max(), where=d.position > 0, color="#1a9850", alpha=0.07)
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax2.plot(d.index, d.equity, label=signal.name, color="#2c7bb6")
    ax2.plot(hold.index, hold, label="buy and hold", color="#999", ls="--")
    ax2.axhline(1, color="#ccc", lw=0.6)
    ax2.set_ylabel("growth of 1")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
