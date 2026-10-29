"""Monthly money planner for the tested mix, and the "$10 a day" goal maths.

Default mix (research/better_than_200.py, picked by rule, then won on the locked years):
35% Nifty 50, 35% gold, 30% crypto (the bot: BTC + ETH on vote_voltgt).

Monthly: new money is split so holdings move toward the targets (buy-only: no selling, no tax).
Yearly (REBALANCE_MONTH): if any part is more than 5 points off target, a full rebalance is proposed.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

TARGETS = {"nifty": 0.35, "gold": 0.35, "crypto": 0.30}
NAMES = {"nifty": "Nifty 50 index fund / ETF", "gold": "Gold ETF", "crypto": "Crypto bot (BTC + ETH, Binance)"}
REBALANCE_MONTH = 4          # April: start of the Indian financial year
BAND = 0.05


@dataclass
class Plan:
    buys: dict
    rebalance: dict
    total_after: float
    weights_after: dict
    notes: list


def monthly(holdings: dict, deposit: float, month: int, targets: dict = TARGETS) -> Plan:
    keys = list(targets)
    cur = {k: float(holdings.get(k, 0.0)) for k in keys}
    total = sum(cur.values()) + deposit
    gap = {k: max(0.0, targets[k] * total - cur[k]) for k in keys}
    g = sum(gap.values())
    buys = {k: (deposit * gap[k] / g if g > 0 else deposit * targets[k]) for k in keys}
    after = {k: cur[k] + buys[k] for k in keys}
    w = {k: after[k] / total if total else 0.0 for k in keys}
    notes, reb = [], {}
    off = max(abs(w[k] - targets[k]) for k in keys)
    if month == REBALANCE_MONTH and off > BAND:
        reb = {k: targets[k] * total - after[k] for k in keys}   # + buy / - sell
        notes.append(f"Yearly check: the mix is {off * 100:.0f} points off target, so a rebalance is proposed. "
                     "Selling may create tax; holdings over 12 months pay the lower long-term rate.")
    elif month == REBALANCE_MONTH:
        notes.append("Yearly check: the mix is within 5 points of target. Nothing to sell this year.")
    return Plan(buys, reb, total, w, notes)


def capital_for_daily(target_per_day: float, annual_return: float) -> float:
    return target_per_day * 365 / annual_return


def years_to_goal(start: float, monthly_add: float, annual_return: float, goal: float) -> float | None:
    """Months of compounding (monthly) until `goal`, as years. None if it never gets there in 60 years."""
    m = (1 + annual_return) ** (1 / 12) - 1
    v = start
    for month in range(1, 60 * 12 + 1):
        v = v * (1 + m) + monthly_add
        if v >= goal:
            return month / 12
    return None


def report(holdings: dict, deposit: float, month: int, usd_inr: float, goal_usd_day: float = 10.0) -> str:
    p = monthly(holdings, deposit, month)
    inr = lambda x: f"Rs {x:,.0f}"
    lines = [f"# Your plan this month", "", f"New money: **{inr(deposit)}**", "",
             "| put into | amount | share after |", "|---|---|---|"]
    for k, v in p.buys.items():
        lines.append(f"| {NAMES[k]} | **{inr(v)}** | {p.weights_after[k] * 100:.0f}% (target {TARGETS[k] * 100:.0f}%) |")
    if p.rebalance:
        lines += ["", "**Yearly rebalance proposed:**"]
        lines += [f"- {NAMES[k]}: {'buy' if v > 0 else 'sell'} {inr(abs(v))}" for k, v in p.rebalance.items() if abs(v) >= 1]
    lines += [""] + [f"_{n}_" for n in p.notes]
    goal_inr = goal_usd_day * usd_inr
    lines += ["", f"## Goal: ${goal_usd_day:.0f} a day ({inr(goal_inr)}/day)", "",
              f"Portfolio after this month: **{inr(p.total_after)}**", "",
              "| if it earns | money needed | you are at | years to get there adding this much monthly |",
              "|---|---|---|---|"]
    for r in (0.10, 0.15, 0.20):
        need = capital_for_daily(goal_inr, r)
        yrs = years_to_goal(p.total_after, deposit, r, need)
        lines.append(f"| {r * 100:.0f}% a year | {inr(need)} | {p.total_after / need * 100:.1f}% | "
                     f"{'over 60' if yrs is None else f'{yrs:.1f}'} |")
    lines += ["", "_Past results are not a promise. The tested mix made about 26% a year since 2018 with a "
                  "worst drop of -22%; plan with 10-15%._"]
    return "\n".join(lines)


def load_holdings(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"nifty": 0, "gold": 0, "crypto": 0}
