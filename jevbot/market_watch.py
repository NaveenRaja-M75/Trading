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
            out.append(f"**{name} is {abs(dd):.1f}% below its 1-year high** (crossed -{level}%). The plan keeps holding it: "
                       f"monthly money buys it cheaper, and the April check rebalances if it is more than 5 points "
                       f"under target. Every -10% here is about -3.5% on the whole plan.")
        state[name] = 0 if dd > -5 else max(level, prev if dd <= -prev else level)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1))
    return out


def stress_line(vix: pd.Series) -> str:
    v = float(vix.iloc[-1])
    level = "calm" if v < 20 else "nervous" if v < 30 else "fear" if v < 40 else "panic"
    return (f"**Fear index (VIX): {v:.1f} ({level}).** Under 20 is calm, 20-30 nervous, 30-40 fear, over 40 "
            f"panic (March 2020 reached 82). It is the market's own read of crash risk in US stocks.")


def vix_alert(vix: pd.Series, state_path: Path) -> list[str]:
    """Alert when the fear index crosses 30 and again at 40; resets once it is back under 25."""
    state = json.loads(state_path.read_text()) if state_path.exists() else {"level": 0}
    v = float(vix.iloc[-1])
    level = 40 if v >= 40 else 30 if v >= 30 else 0
    out = []
    if level > state.get("level", 0):
        out.append(f"**Market fear is high: VIX {v:.1f}** (crossed {level}). This is when crashes and war or crisis "
                   f"headlines show up in prices. The bot already holds less crypto when markets are wild; Nifty and "
                   f"gold stay held. No action needed; do not sell in panic.")
    state["level"] = 0 if v < 25 else max(level, state.get("level", 0))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state))
    return out
