"""Should the bot react to sudden crashes between its daily checks?

Base: the live rule (50/100/200 vote x 60% volatility target) decided once a day at 00:00 UTC.
Guards, checked every hour on BTC hourly candles (Binance 2017+):
  drop_g       sell everything if price falls g% below the price at the last daily decision
  trail_g      sell everything if price falls g% from its highest hourly close since the last decision
After a guard fires, the bot waits for the next daily decision (which may buy back).
Costs 0.12% per side on every change. Also lists every guard firing with what happened next.
"""
import numpy as np
import pandas as pd

from india_test import perf
from jevbot.data import load_or_fetch
from search_loop import CURRENT, make

COST = 0.0012


def run(h: pd.Series, daily_w: pd.Series, kind: str, g: float):
    """h: hourly closes. daily_w: target decided at each day's close, applied from the next hour."""
    r = h.pct_change().fillna(0.0).to_numpy()
    day = h.index.floor("D")
    target_by_hour = daily_w.shift(1).reindex(day).fillna(0.0).to_numpy()   # yesterday's decision
    px = h.to_numpy()
    out = np.zeros(len(h))
    hold, ref, peak, fired_today, last_day = 0.0, np.nan, np.nan, False, None
    fires = []
    for t in range(len(h)):
        out[t] = hold * r[t]
        if day[t] != last_day:                           # new day: apply the daily decision
            last_day, fired_today = day[t], False
            tgt = target_by_hour[t]
            if abs(tgt - hold) >= 0.05:
                out[t] -= abs(tgt - hold) * COST
                hold = tgt
            ref = peak = px[t]
        peak = max(peak, px[t])
        if kind != "none" and hold > 0 and not fired_today:
            base = ref if kind == "drop" else peak
            if px[t] <= base * (1 - g):
                out[t] -= hold * COST
                fires.append((h.index[t], hold, px[t]))
                hold, fired_today = 0.0, True
    return pd.Series(out, index=h.index), fires


def main():
    h = load_or_fetch("BTCUSDT", "1h", 3300).close
    d = h.resample("D").last().dropna()
    w = make(d, CURRENT)
    start = d.index[365]
    m = h.index >= start
    rows, details = [], {}
    for kind, gs in (("none", [0]), ("drop", [0.05, 0.07, 0.10, 0.15]), ("trail", [0.05, 0.07, 0.10, 0.15])):
        for g in gs:
            net, fires = run(h, w, kind, g)
            net = net[m]
            daily = (1 + net).resample("D").prod() - 1
            name = "daily only (live)" if kind == "none" else f"{kind} {int(g * 100)}%"
            rows.append({**perf(name, daily, 365), "guard_fired": len(fires)})
            details[name] = fires
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(f"BTC hourly {h.index[0]:%Y-%m-%d} -> {h.index[-1]:%Y-%m-%d}; scored from {start:%Y-%m-%d}\n")
        print(pd.DataFrame(rows).set_index("name").to_string())
    for name in ("drop 10%", "drop 15%"):
        print(f"\n{name}: every firing, and BTC 7 and 30 days later (positive = selling was a mistake)")
        for t, held, p in details[name]:
            later7 = h[h.index >= t + pd.Timedelta(days=7)]
            later30 = h[h.index >= t + pd.Timedelta(days=30)]
            f7 = f"{(later7.iloc[0] / p - 1) * 100:+.1f}%" if len(later7) else "n/a"
            f30 = f"{(later30.iloc[0] / p - 1) * 100:+.1f}%" if len(later30) else "n/a"
            print(f"  {t:%Y-%m-%d %H:%M}  held {held:.2f}  sold at {p:,.0f}  7d {f7}  30d {f30}")


if __name__ == "__main__":
    main()
