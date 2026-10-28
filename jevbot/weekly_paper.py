"""Paper portfolio for the weekly strategy: marked to market daily, rebalanced every 7 days."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from . import portfolio
from .data import fetch_recent

STRATEGIES = portfolio.live_strategies()


def load(path: Path, start_usdt: float) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"cash": start_usdt, "start": start_usdt, "qty": {}, "last_rebalance": "", "history": [], "trades": []}


def equity(state: dict, prices: dict[str, float]) -> float:
    return state["cash"] + sum(q * prices.get(s, 0.0) for s, q in state["qty"].items())


def run_once(symbols: list[str], strategy: str, path: Path, start_usdt: float,
             fee_bps: float, slippage_bps: float, every_days: int = 7, fetch=fetch_recent) -> str:
    state = load(path, start_usdt)
    dfs = {}
    for s in symbols:
        try:
            dfs[s] = fetch(s, "1d", 400)
        except Exception as e:  # a coin that fails today simply isn't traded today
            print(f"{s}: no data ({e})")
    px = portfolio.closes(dfs)
    day = px.index[-1]
    prices = {s: float(px[s].iloc[-1]) for s in px.columns if pd.notna(px[s].iloc[-1])}
    eq = equity(state, prices)
    msg = f"{day:%Y-%m-%d} equity {eq:.2f}"

    due = not state["last_rebalance"] or (day - pd.Timestamp(state["last_rebalance"])).days >= every_days
    if due:
        w = STRATEGIES[strategy](px).iloc[-1].fillna(0.0)
        cost = (fee_bps + slippage_bps) / 1e4
        for s in sorted(prices, key=lambda s: w.get(s, 0.0)):  # sells first, so buys have cash
            target_usdt = w.get(s, 0.0) * eq
            held_usdt = state["qty"].get(s, 0.0) * prices[s]
            delta = target_usdt - held_usdt
            if abs(delta) < max(1.0, 0.01 * eq):
                continue
            if delta > 0:
                delta = min(delta, state["cash"])
                state["cash"] -= delta
                state["qty"][s] = state["qty"].get(s, 0.0) + delta * (1 - cost) / prices[s]
            else:
                q = min(-delta / prices[s], state["qty"].get(s, 0.0))
                state["qty"][s] = state["qty"].get(s, 0.0) - q
                state["cash"] += q * prices[s] * (1 - cost)
            state["trades"].append({"day": f"{day:%Y-%m-%d}", "symbol": s, "usdt": round(delta, 2)})
        state["qty"] = {s: q for s, q in state["qty"].items() if q * prices.get(s, 0.0) >= 0.01}
        state["last_rebalance"] = day.isoformat()
        held = ", ".join(f"{s[:-4]} {w[s] * 100:.0f}%" for s in w.index if w[s] > 0) or "all cash"
        msg += f" | rebalanced -> {held}"
        eq = equity(state, prices)

    hist = [h for h in state["history"] if h["day"] != f"{day:%Y-%m-%d}"]
    hist.append({"day": f"{day:%Y-%m-%d}", "equity": round(eq, 2),
                 "hold_equal_px": {s: prices[s] for s in prices}})
    state["history"] = hist[-2000:]
    state["trades"] = state["trades"][-2000:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=1))
    return msg


def report(path: Path, strategy: str) -> str:
    st = json.loads(path.read_text())
    h = st["history"]
    first, last = h[0], h[-1]
    common = [s for s in first["hold_equal_px"] if s in last["hold_equal_px"]]
    hold = sum(last["hold_equal_px"][s] / first["hold_equal_px"][s] for s in common) / max(len(common), 1)
    ret = last["equity"] / st["start"] - 1
    eqs = pd.Series([x["equity"] for x in h])
    dd = float((eqs / eqs.cummax() - 1).min())
    held = ", ".join(f"{s[:-4]}" for s in st["qty"]) or "all cash"
    return (f"## Weekly portfolio: {strategy} (paper, daily check, weekly rebalance)\n\n"
            f"Since {first['day']}: **{ret * 100:+.2f}%** (equal-weight holding of the same coins: "
            f"{(hold - 1) * 100:+.2f}%), worst drop {dd * 100:.2f}%. Holding now: {held}. "
            f"Updated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}.\n")
