"""How often does the live bot actually trade, and what does compounding do?

Replays the live logic day by day on real BTC (and ETH) candles from 2018: vote_voltgt targets,
orders only when the change is at least 5 USDT (Binance minimum), 0.12% cost per side, all gains
reinvested (equity = budget + profit). Optional monthly deposits added to the budget.
"""
import numpy as np
import pandas as pd

from jevbot import portfolio
from jevbot.data import load_or_fetch

COST, MIN_ORDER = 0.0012, 5.0


def replay(px: pd.DataFrame, start_usdt: float, monthly_add: float = 0.0):
    w = portfolio.vote_voltgt(px)
    start = px.index[230]
    px, w = px[px.index >= start], w[w.index >= start]
    cash, qty = start_usdt, {c: 0.0 for c in px.columns}
    deposits, orders, eq_path = start_usdt, [], []
    last_month = px.index[0].month
    for t in range(len(px)):
        day = px.index[t]
        p = px.iloc[t]
        if monthly_add and day.month != last_month:
            cash += monthly_add
            deposits += monthly_add
            last_month = day.month
        equity = cash + sum(qty[c] * p[c] for c in px.columns)
        for c in sorted(px.columns, key=lambda c: w[c].iloc[t]):          # sells first
            delta = w[c].iloc[t] * equity - qty[c] * p[c]
            if abs(delta) >= MIN_ORDER or (w[c].iloc[t] == 0 and qty[c] * p[c] >= MIN_ORDER):
                if delta < 0:
                    sell = min(-delta, qty[c] * p[c])
                    qty[c] -= sell / p[c]
                    cash += sell * (1 - COST)
                    orders.append((day, c, "SELL", sell))
                elif cash >= MIN_ORDER:
                    buy = min(delta, cash)
                    qty[c] += buy * (1 - COST) / p[c]
                    cash -= buy
                    orders.append((day, c, "BUY", buy))
        eq_path.append(cash + sum(qty[c] * p[c] for c in px.columns))
    eq = pd.Series(eq_path, index=px.index)
    return eq, pd.DataFrame(orders, columns=["day", "coin", "side", "usdt"]), deposits


def main():
    btc = load_or_fetch("BTCUSDT", "1d", 3300).close
    eth = load_or_fetch("ETHUSDT", "1d", 3300).close
    btc.index, eth.index = btc.index.normalize(), eth.index.normalize()
    sets = {"$20, BTC only": (btc.to_frame("BTC"), 20, 0), "$60, BTC + ETH": (pd.concat([btc.rename("BTC"), eth.rename("ETH")], axis=1).dropna(), 60, 0),
            "$200, BTC + ETH": (pd.concat([btc.rename("BTC"), eth.rename("ETH")], axis=1).dropna(), 200, 0),
            "$20 + $17 every month, BTC only": (btc.to_frame("BTC"), 20, 17)}
    for name, (px, start, add) in sets.items():
        eq, orders, dep = replay(px, start, add)
        days = len(eq)
        years = days / 365
        per_month = orders.groupby(orders.day.dt.to_period("M")).size().reindex(
            pd.period_range(eq.index[0], eq.index[-1], freq="M"), fill_value=0)
        dd = (eq / eq.cummax() - 1).min()
        print(f"\n######## {name}: {eq.index[0]:%Y-%m-%d} -> {eq.index[-1]:%Y-%m-%d}")
        print(f"  orders: {len(orders)} in {years:.1f} years = {len(orders) / years:.1f} a year, "
              f"{len(orders) / (years * 12):.1f} a month on average")
        print(f"  months with 0 orders: {(per_month == 0).mean() * 100:.0f}%   1-2 orders: {per_month.between(1, 2).mean() * 100:.0f}%   "
              f"3+: {(per_month >= 3).mean() * 100:.0f}%   busiest month: {per_month.max()} orders")
        print(f"  days with any order: {orders.day.nunique() / days * 100:.1f}% of days (never more than "
              f"{orders.groupby('day').size().max()} order(s) in a day)")
        print(f"  average order size: ${orders.usdt.mean():.2f}; smallest ${orders.usdt.min():.2f}")
        print(f"  money put in: ${dep:,.0f}  ->  ended with ${eq.iloc[-1]:,.0f}  "
              f"(x{eq.iloc[-1] / dep:.1f}); worst drop {dd * 100:.0f}%")
        yearly = eq.resample("YE").last()
        print("  end of each year: " + ", ".join(f"{d:%Y} ${v:,.0f}" for d, v in yearly.items()))


if __name__ == "__main__":
    main()
