"""Do classic candlestick patterns predict anything? 13 patterns, BTC & ETH (Binance daily and 4h), Nifty daily.

For each pattern: average return over the next 1, 3, 5 and 10 bars after it appears, minus the
average return of ALL bars over the same period (so a rising market does not count as skill),
t-statistic, and whether it held in both halves of the data. With 13 patterns x 4 horizons x 5 data
sets = 260 tests, |t| > 3.5 is needed (Bonferroni, 5%) before anything counts as real.
Trading check for each pattern's best horizon: enter at the pattern's close, hold, 0.12%/0.08% costs.
"""
import json
import math
import urllib.request

import numpy as np
import pandas as pd

from jevbot.data import load_or_fetch

HORIZONS = (1, 3, 5, 10)
T_NEEDED = 3.5


def patterns(d: pd.DataFrame) -> dict[str, pd.Series]:
    o, h, l, c = d.open, d.high, d.low, d.close
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    up, dn = c > o, c < o
    lower = np.minimum(o, c) - l
    upper = h - np.maximum(o, c)
    avg_body = body.rolling(20).mean()
    trend_down = c < c.rolling(10).mean()
    trend_up = c > c.rolling(10).mean()
    p = {
        "bullish_engulfing": dn.shift() & up & (o <= c.shift()) & (c >= o.shift()),
        "bearish_engulfing": up.shift() & dn & (o >= c.shift()) & (c <= o.shift()),
        "hammer (after a fall)": trend_down & (lower >= 2 * body) & (upper <= body) & (body > 0),
        "shooting_star (after a rise)": trend_up & (upper >= 2 * body) & (lower <= body) & (body > 0),
        "doji": body <= 0.1 * rng,
        "morning_star": dn.shift(2) & (body.shift(2) > avg_body) & (body.shift() < 0.3 * avg_body)
                        & up & (c > (o.shift(2) + c.shift(2)) / 2),
        "evening_star": up.shift(2) & (body.shift(2) > avg_body) & (body.shift() < 0.3 * avg_body)
                        & dn & (c < (o.shift(2) + c.shift(2)) / 2),
        "three_white_soldiers": up & up.shift() & up.shift(2) & (c > c.shift()) & (c.shift() > c.shift(2)),
        "three_black_crows": dn & dn.shift() & dn.shift(2) & (c < c.shift()) & (c.shift() < c.shift(2)),
        "inside_bar": (h < h.shift()) & (l > l.shift()),
        "outside_bar_up": (h > h.shift()) & (l < l.shift()) & up,
        "long_lower_wick": lower >= 0.6 * rng,
        "big_range_day (2x avg)": (h - l) >= 2 * (h - l).rolling(20).mean(),
    }
    return {k: v.fillna(False) for k, v in p.items()}


def yahoo_ohlc(sym):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.request.quote(sym)}?period1=0&period2=9999999999&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        res = json.loads(r.read())["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")},
                      index=pd.to_datetime(res["timestamp"], unit="s", utc=True)).dropna()
    df = df[(df > 0).all(axis=1) & (df.high >= df.low)]
    return df[df.index >= "2010-01-01"]


def study(name, d, cost):
    pats = patterns(d)
    lc = np.log(d.close)
    rows, best_trades = [], []
    mid = d.index[len(d) // 2]
    for pn, mask in pats.items():
        for hz in HORIZONS:
            fwd = (lc.shift(-hz) - lc)
            base = fwd.mean()
            x = (fwd[mask] - base).dropna()
            if len(x) < 30:
                continue
            t = x.mean() / (x.std() / math.sqrt(len(x))) if x.std() > 0 else 0.0
            first, second = x[x.index < mid].mean(), x[x.index >= mid].mean()
            rows.append({"pattern": pn, "h": hz, "n": len(x), "excess_bps": round(x.mean() * 1e4, 1), "t": round(t, 2),
                         "same_sign_both_halves": bool(np.sign(first) == np.sign(second)),
                         "counts": abs(t) >= T_NEEDED and np.sign(first) == np.sign(second)})
    tab = pd.DataFrame(rows)
    print(f"\n######## {name}: {len(d)} bars {d.index[0]:%Y-%m-%d} -> {d.index[-1]:%Y-%m-%d}")
    strongest = tab.reindex(tab.t.abs().sort_values(ascending=False).index).head(10)
    with pd.option_context("display.width", 200):
        print(strongest.to_string(index=False))
    real = tab[tab.counts]
    print(f"  patterns passing |t| >= {T_NEEDED} AND same direction in both halves: {len(real)} of {len(tab)}")
    # trading check: bullish patterns, best horizon by t (this choice is itself optimistic)
    for pn in ("bullish_engulfing", "hammer (after a fall)", "morning_star", "three_white_soldiers", "long_lower_wick"):
        sub = tab[tab.pattern == pn]
        if sub.empty:
            continue
        hz = int(sub.sort_values("t", ascending=False).h.iloc[0])
        mask = pats[pn].to_numpy()
        px = d.close.to_numpy()
        i, rets = 0, []
        while i < len(px) - hz:
            if mask[i]:
                rets.append(px[i + hz] / px[i] * (1 - cost) ** 2 - 1)
                i += hz
            i += 1
        r = np.array(rets)
        hold = (px[-1] / px[0]) ** (1 / max(len(px), 1)) - 1
        best_trades.append(f"{pn} hold {hz}: {len(r)} trades, avg {r.mean() * 100:+.2f}%, won {(r > 0).mean() * 100:.0f}% "
                           f"(holding averages {((1 + hold) ** hz - 1) * 100:+.2f}% per {hz} bars)")
    print("  trading the bullish patterns (best horizon, after costs):")
    for line in best_trades:
        print("   ", line)
    return len(real), len(tab)


def main():
    total_real, total = 0, 0
    sets = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        dd = load_or_fetch(sym, "1d", 3300)
        sets.append((f"{sym} daily", dd, 0.0012))
    sets.append(("BTCUSDT 4h", load_or_fetch("BTCUSDT", "4h", 3300), 0.0012))
    sets.append(("Nifty 50 daily", yahoo_ohlc("^NSEI"), 0.0008))
    sets.append(("Bank Nifty daily", yahoo_ohlc("^NSEBANK"), 0.0008))
    for name, d, cost in sets:
        r, t = study(name, d[["open", "high", "low", "close"]], cost)
        total_real += r
        total += t
    print(f"\nOVERALL: {total_real} of {total} pattern/horizon tests show a real, stable effect.")


if __name__ == "__main__":
    main()
