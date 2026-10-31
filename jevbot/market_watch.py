"""Daily snapshot of every part of the plan (Nifty, gold, Bitcoin), and alerts when gold or Nifty
fall through -10%, -20%, -30% from their 1-year high. The plan holds Nifty and gold on purpose
(timing them did worse than holding in testing), so these alerts inform; they do not trade."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

WATCH_LEVELS = (10, 20, 30, 40)


def table(series: dict[str, pd.Series]) -> str:
    lines = ["| asset | last close | 1 month | 6 months | from 1-year high | vs 200-day avg |",
             "|---|---|---|---|---|---|"]
    for name, s in series.items():
        s = s.dropna()
        d, last = s.index[-1], s.iloc[-1]
        ch = lambda days: (last / s.asof(d - pd.Timedelta(days=days)) - 1) * 100
        hi = s[s.index >= d - pd.Timedelta(days=365)].max()
        ma = s.rolling(200).mean().iloc[-1]
        lines.append(f"| {name} | {d:%Y-%m-%d} | {ch(30):+.1f}% | {ch(182):+.1f}% | {(last / hi - 1) * 100:+.1f}% | "
                     f"{(last / ma - 1) * 100:+.1f}% |")
    return "\n".join(lines)


def drawdown_alerts(series: dict[str, pd.Series], state_path: Path) -> list[str]:
    """Alert once per new level reached; the level resets when the asset recovers to within 5% of its high."""
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    out = []
    for name, s in series.items():
        s = s.dropna()
        hi = s[s.index >= s.index[-1] - pd.Timedelta(days=365)].max()
        dd = (s.iloc[-1] / hi - 1) * 100
        level = max([lv for lv in WATCH_LEVELS if dd <= -lv], default=0)
        prev = state.get(name, 0)
        if level > prev:
            out.append(f"**{name} is {dd:.1f}% below its 1-year high** (crossed -{level}%). The plan keeps holding it: "
                       f"monthly money buys it cheaper, and the April check rebalances if it is more than 5 points "
                       f"under target. Every -10% here is about -3.5% on the whole plan.")
        state[name] = 0 if dd > -5 else max(level, prev if dd <= -prev else level)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1))
    return out
