"""Chart reading: the things a human trader looks at, computed causally.

Swing highs/lows only count once they are *confirmed*: a swing high at bar i is known at
bar i+k, after k lower highs follow it. Nothing here can see the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def swings(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Last two confirmed swing highs/lows as of each bar."""
    hi, lo, idx = df["high"], df["low"], df.index
    win = 2 * k + 1
    sh = hi.shift(k).where(hi.shift(k) == hi.rolling(win).max())   # pivot high at t-k, confirmed at t
    sl = lo.shift(k).where(lo.shift(k) == lo.rolling(win).min())

    def last_two(points: pd.Series) -> tuple[pd.Series, pd.Series]:
        pts = points.dropna()
        return pts.reindex(idx).ffill(), pts.shift(1).reindex(idx).ffill()

    last_sh, prev_sh = last_two(sh)
    last_sl, prev_sl = last_two(sl)
    return pd.DataFrame({"last_sh": last_sh, "prev_sh": prev_sh,
                         "last_sl": last_sl, "prev_sl": prev_sl}, index=idx)


def structure(df: pd.DataFrame, k: int = 5) -> pd.Series:
    """+1 uptrend (higher highs AND higher lows), -1 downtrend (lower highs AND lower lows), 0 range."""
    s = swings(df, k)
    up = (s.last_sh > s.prev_sh) & (s.last_sl > s.prev_sl)
    down = (s.last_sh < s.prev_sh) & (s.last_sl < s.prev_sl)
    return pd.Series(np.where(up, 1, np.where(down, -1, 0)), index=df.index)


def latch(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """Turn entry/exit events into a held state: +1 while in a trade, -1 while out."""
    e, x = enter.fillna(False).to_numpy(), exit_.fillna(False).to_numpy()
    out = np.empty(len(e))
    state = -1.0
    for i in range(len(e)):
        if state < 0 and e[i]:
            state = 1.0
        elif state > 0 and x[i]:
            state = -1.0
        out[i] = state
    return pd.Series(out, index=enter.index)


def describe(df: pd.DataFrame, k: int = 5) -> str:
    """One-paragraph plain-English read of the latest chart, for logs and an AI judge's prompt."""
    c = df["close"]
    s = swings(df, k).iloc[-1]
    st = structure(df, k).iloc[-1]
    ma50, ma200 = sma(c, 50).iloc[-1], sma(c, 200).iloc[-1]
    a = atr(df).iloc[-1]
    vol_ratio = df["volume"].iloc[-1] / df["volume"].tail(20).mean()
    trend = {1: "uptrend (higher highs and higher lows)", -1: "downtrend (lower highs and lower lows)",
             0: "range / no clear structure"}[int(st)]
    parts = [f"Price {c.iloc[-1]:.2f}.", f"Structure: {trend}."]
    if np.isfinite(s.last_sh):
        parts.append(f"Nearest resistance (last swing high) {s.last_sh:.2f}, support (last swing low) {s.last_sl:.2f}.")
    if np.isfinite(ma200):
        parts.append(f"Price is {'above' if c.iloc[-1] > ma200 else 'below'} the 200-bar average "
                     f"({ma200:.2f}); 50-bar average is {'above' if ma50 > ma200 else 'below'} it.")
    parts.append(f"Typical bar range (ATR) {a:.2f} = {a / c.iloc[-1] * 100:.2f}% of price. "
                 f"Volume is {vol_ratio:.1f}x its 20-bar average.")
    return " ".join(parts)
