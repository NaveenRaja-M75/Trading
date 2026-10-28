"""Real-money version of the weekly strategy on Binance spot.

Safety:
- Dry run unless --execute AND LIVE_TRADING=true. Testnet unless BINANCE_TESTNET=false.
- Trades at most WEEKLY_BUDGET_USDT, and only coins the bot itself bought (per-coin BotBook).
- Kill switch: if bot equity falls WEEKLY_MAX_LOSS_USDT below the budget, sell everything the
  bot owns and refuse to trade again until the latch file is deleted by a human.
- Orders under Binance's minimum (~5 USDT) are skipped, never forced.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from . import portfolio
from .data import fetch_recent

STRATEGIES = portfolio.live_strategies()


MIN_SLOT_USDT = 8.0  # >= 5 USDT exchange minimum plus room for a ~35% drop before a sell is impossible


def plan_orders(weights: dict[str, float], prices: dict[str, float], held_usdt: dict[str, float],
                equity: float, min_trade: float = 5.0) -> list[tuple[str, str, float]]:
    """(symbol, 'SELL'|'BUY', usdt) to move from current holdings to target weights. Sells first."""
    sells, buys = [], []
    for s in prices:
        delta = weights.get(s, 0.0) * equity - held_usdt.get(s, 0.0)
        if delta <= -min_trade or (weights.get(s, 0.0) == 0 and held_usdt.get(s, 0.0) > 0):
            sells.append((s, "SELL", round(-delta, 2)))
        elif delta >= min_trade:
            buys.append((s, "BUY", round(delta, 2)))
    return sells + buys


class WeeklyLive:
    def __init__(self, symbols: list[str], strategy: str, budget: float, max_loss: float,
                 make_broker, state_dir: Path = Path("state"), fetch=fetch_recent):
        self.symbols, self.strategy, self.budget, self.max_loss = symbols, strategy, budget, max_loss
        self.make_broker, self.dir, self.fetch = make_broker, state_dir, fetch
        self.meta_path = state_dir / "weekly_live.json"
        self.latch = state_dir / "WEEKLY_KILL_SWITCH"

    def _meta(self) -> dict:
        return json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {"last_rebalance": "", "log": []}

    def run(self, execute: bool, every_days: int = 7, force: bool = False) -> list[str]:
        out = []
        if self.latch.exists():
            return [f"KILL SWITCH is on ({self.latch.read_text().strip()}). Delete {self.latch} to re-enable."]
        dfs = {}
        for s in self.symbols:
            try:
                dfs[s] = self.fetch(s, "1d", 400)
            except Exception as e:
                out.append(f"{s}: no data, not traded today ({e})")
        px = portfolio.closes(dfs)
        day = px.index[-1]
        prices = {s: float(px[s].iloc[-1]) for s in px.columns if pd.notna(px[s].iloc[-1])}
        brokers = {s: self.make_broker(s) for s in prices}
        held = {s: b.book.qty * prices[s] for s, b in brokers.items()}
        cost = sum(b.book.cost_usdt for b in brokers.values())
        realized = sum(b.book.realized_usdt for b in brokers.values())
        equity = self.budget + realized + (sum(held.values()) - cost)  # uninvested budget + P&L
        out.append(f"{day:%Y-%m-%d} bot equity {equity:.2f} of budget {self.budget:.2f} "
                   f"(holding {sum(held.values()):.2f})")

        if equity <= self.budget - abs(self.max_loss):
            out.append(f"LOSS LIMIT: equity {equity:.2f} <= {self.budget - abs(self.max_loss):.2f}; selling everything")
            for s, b in brokers.items():
                if b.book.qty > 0:
                    out.append(f"  {s}: " + (b.sell_qty(b.book.qty, prices[s]) if execute else "would sell all"))
            if execute:
                self.dir.mkdir(parents=True, exist_ok=True)
                self.latch.write_text(f"tripped {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}, equity {equity:.2f}")
            return out

        meta = self._meta()
        due = force or not meta["last_rebalance"] or (day - pd.Timestamp(meta["last_rebalance"])).days >= every_days
        if not due:
            return out + [f"next rebalance after {pd.Timestamp(meta['last_rebalance']) + pd.Timedelta(days=every_days):%Y-%m-%d}"]

        w = STRATEGIES[self.strategy](px).iloc[-1].fillna(0.0).to_dict()
        slot = equity / max(len(prices), 1)
        if slot < MIN_SLOT_USDT:
            return out + [f"REFUSED: {slot:.2f} USDT per coin is too close to Binance's 5 USDT minimum order; "
                          f"positions could not be sold after a dip. Use at most "
                          f"{max(1, int(equity // MIN_SLOT_USDT))} coin(s) for this budget, or a bigger budget."]
        orders = plan_orders(w, prices, held, equity)
        targets = ", ".join(f"{s[:-4]} {v * 100:.0f}%" for s, v in w.items() if v > 0) or "all cash"
        out.append(f"targets: {targets}")
        for s, side, usdt in orders:
            b = brokers[s]
            if not execute:
                out.append(f"  DRY RUN {side} {s} {usdt:.2f} USDT")
            elif side == "SELL":
                qty = b.book.qty if w.get(s, 0.0) == 0 else usdt / prices[s]
                out.append(f"  {s}: " + b.sell_qty(qty, prices[s]))
            else:
                out.append(f"  {s}: " + b.buy_usdt(usdt, prices[s]))
        if execute:
            meta["last_rebalance"] = day.isoformat()
            meta["log"] = (meta["log"] + out)[-500:]
            self.dir.mkdir(parents=True, exist_ok=True)
            self.meta_path.write_text(json.dumps(meta, indent=1))
        return out
