"""Daily checks that deserve a human's attention. Run by the weekly-paper GitHub job, which opens a
GitHub issue (email + app notification) whenever this returns anything."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import portfolio
from .weekly_live import data_problem


def check(closes: dict[str, pd.DataFrame], state_path: Path, paper_path: Path | None, now: pd.Timestamp) -> list[str]:
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    alerts, new_state = [], {}
    good = {}
    for sym, df in closes.items():
        problem = data_problem(df, now)
        if problem:
            alerts.append(f"**{sym} data problem:** {problem}. The bot will not trade {sym} until it clears.")
        else:
            good[sym] = df
    if good:
        px = portfolio.closes(good)
        w = portfolio.vote_voltgt(px).iloc[-1]
        for sym in px.columns:
            tgt = round(float(w[sym]) * 100)
            new_state[f"target_{sym}"] = tgt
            prev = state.get(f"target_{sym}")
            if prev is not None and abs(tgt - prev) >= 15:
                alerts.append(f"**{sym} target changed:** {prev}% -> {tgt}% of the bot's budget. "
                              f"The bot buys or sells at its next run.")
            c = px[sym]
            d1, d7 = c.iloc[-1] / c.iloc[-2] - 1, c.iloc[-1] / c.iloc[-8] - 1
            if d1 <= -0.08:
                alerts.append(f"**{sym} fell {d1 * 100:.1f}% in a day.** No action needed: the bot reacts at its "
                              f"daily check, and the exchange stop guards the loss limit.")
            if d7 <= -0.15:
                alerts.append(f"**{sym} fell {d7 * 100:.1f}% in a week.**")
    if paper_path and paper_path.exists():
        h = [x["equity"] for x in json.loads(paper_path.read_text()).get("history", [])]
        if len(h) > 1:
            dd = h[-1] / max(h) - 1
            if dd <= -0.15:
                alerts.append(f"**Paper account is {dd * 100:.1f}% below its peak.** Compare with RESULTS.md: "
                              f"drops of 20-30% happened in testing; past -45% would be outside anything seen.")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(new_state | {"checked": now.isoformat()}, indent=1))
    return alerts
