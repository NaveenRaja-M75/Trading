"""The "crypto clock": is there an edge in Polymarket's 15-minute Bitcoin Up/Down markets?

For every window in the last N days:
  1. Polymarket: the Up price 1-5 minutes before the close (prices-history, ~1 point/minute) and
     the actual result (outcomePrices of the resolved market).
  2. Binance 1-minute candles: the fair chance the leading side wins, from how far BTC already
     moved and how volatile it is. Two models, because the rules say Chainlink TWAP:
       final: last price vs open
       twap:  average over the window vs open (most of it is already fixed late in the window)
  3. Buy the side whose fair chance beats its price by a margin, pay the fee, and score it.

Read-only. Places no bets. Caveat: price history is a midpoint/last price, not the ask you
would actually pay, so a 1-cent haircut is charged on every simulated buy.
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from jevbot.data import fetch_history

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DAYS = float(sys.argv[1]) if len(sys.argv) > 1 else 14
HAIRCUT = 0.01


def get(url, params=None, tries=3):
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    for k in range(tries):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": "Mozilla/5.0 research"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(1 + 2 * k)
    return None


def window(ts: int) -> dict | None:
    ev = get(f"{GAMMA}/events", {"slug": f"btc-updown-15m-{ts}"})
    if not ev:
        return None
    m = ev[0]["markets"][0]
    if not m.get("closed"):
        return None
    prices = [float(x) for x in json.loads(m["outcomePrices"])]
    if sorted(prices) != [0.0, 1.0]:
        return None
    up_token = json.loads(m["clobTokenIds"])[0]
    hist = get(f"{CLOB}/prices-history", {"market": up_token, "startTs": ts - 60, "endTs": ts + 900, "fidelity": 1})
    pts = (hist or {}).get("history") or []
    return {"ts": ts, "up_won": prices[0] == 1.0, "hist": [(p["t"], p["p"]) for p in pts],
            "fee": {k: m.get(k) for k in ("feesEnabled", "feeType", "feeSchedule", "takerBaseFee")},
            "volume": float(m.get("volumeNum") or 0)}


def price_at(hist, t):
    before = [p for (s, p) in hist if s <= t]
    return before[-1] if before else None


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main():
    now = int(time.time())
    last = now - now % 900 - 900
    starts = list(range(last - int(DAYS * 86400) // 900 * 900, last + 1, 900))
    print(f"{len(starts)} windows, {DAYS:g} days", flush=True)
    with ThreadPoolExecutor(6) as ex:
        wins = [w for w in ex.map(window, starts) if w]
    print(f"{len(wins)} resolved windows with data", flush=True)
    print("fee fields as published:", json.dumps(wins[-1]["fee"])[:600])

    btc = fetch_history("BTCUSDT", "1m", DAYS + 1)
    close = btc["close"]
    opn = btc["open"]
    lr = np.log(close).diff()

    rows = []
    for w in wins:
        t0 = pd.Timestamp(w["ts"], unit="s", tz="UTC")
        if t0 not in opn.index:
            continue
        p0 = opn[t0]
        for m in (5, 4, 3, 2, 1):
            t = t0 + pd.Timedelta(minutes=15 - m)          # decision moment
            last_bar = t - pd.Timedelta(minutes=1)          # last fully closed minute
            if last_bar not in close.index:
                continue
            pt = close[last_bar]
            sig = lr.loc[:last_bar].tail(60).std()
            if not np.isfinite(sig) or sig <= 0:
                continue
            q_final = phi(math.log(pt / p0) / (sig * math.sqrt(m)))
            known = close.loc[t0:last_bar]                  # minutes already in the average
            k = len(known)
            var = sig ** 2 * (m + 1) * (2 * m + 1) / (6 * m)
            need = (15 * p0 - known.sum() - m * pt) / (m * pt)  # future mean return needed for Up
            q_twap = 1 - phi(need / math.sqrt(var)) if k == 15 - m else np.nan
            up_px = price_at(w["hist"], w["ts"] + 60 * (15 - m))
            if up_px is None:
                continue
            rows.append({"ts": w["ts"], "m": m, "up_won": w["up_won"], "up_px": up_px,
                         "q_final": q_final, "q_twap": q_twap, "volume": w["volume"]})
    d = pd.DataFrame(rows).dropna()
    print(f"{len(d)} (window, minutes-left) rows\n")

    def logloss(q, y):
        q = q.clip(1e-4, 1 - 1e-4)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())

    y = d.up_won.astype(float)
    print("WHO PREDICTS THE RESULT BEST (lower log-loss is better; accuracy of the favourite)")
    for name, q in (("polymarket price", d.up_px), ("model: final price", d.q_final), ("model: twap", d.q_twap)):
        print(f"  {name:20s} log-loss {logloss(q, y):.4f}   accuracy {((q > .5) == (y == 1)).mean() * 100:5.1f}%")
    for m in (5, 3, 1):
        s = d[d.m == m]
        yy = s.up_won.astype(float)
        print(f"   {m} min left: polymarket {logloss(s.up_px, yy):.4f}  final {logloss(s.q_final, yy):.4f}  "
              f"twap {logloss(s.q_twap, yy):.4f}")

    best = "q_twap" if logloss(d.q_twap, y) < logloss(d.q_final, y) else "q_final"
    print(f"\nusing {best} as the fair chance\n")

    print("IS POLYMARKET CALIBRATED? (leading side's price vs how often it actually won)")
    lead_px = np.where(d.up_px >= .5, d.up_px, 1 - d.up_px)
    lead_won = np.where(d.up_px >= .5, d.up_won, ~d.up_won.astype(bool))
    cal = pd.DataFrame({"px": lead_px, "won": lead_won.astype(float)})
    cal["bucket"] = pd.cut(cal.px, [.5, .6, .7, .8, .9, .95, .99, 1.0])
    print(cal.groupby("bucket", observed=True).agg(n=("won", "size"), avg_price=("px", "mean"),
                                                  won_pct=("won", "mean")).round(3).to_string())

    def fee(p, rate):
        return rate * p * (1 - p)

    print("\nSIMULATED BETS ($1 per share, 1-cent haircut, fee = rate x p x (1-p) per share)")
    out = []
    for rate in (0.0, 0.07):
        for m in (5, 4, 3, 2, 1):
            s = d[d.m == m]
            q_up = s[best]
            for margin in (0.0, 0.03, 0.08):
                side_up = q_up >= 0.5
                q = np.where(side_up, q_up, 1 - q_up)
                p = np.where(side_up, s.up_px, 1 - s.up_px) + HAIRCUT
                won = np.where(side_up, s.up_won, ~s.up_won.astype(bool)).astype(float)
                f = fee(p, rate)
                take = (q - p - f >= margin) & (p < 0.99)
                pnl = (won - p - f)[take]
                cost = (p + f)[take]
                days = pd.to_datetime(s.ts[take], unit="s").dt.date
                daily = pd.Series(pnl, index=days).groupby(level=0).sum() if take.any() else pd.Series(dtype=float)
                out.append({"fee_rate": rate, "min_left": m, "margin": margin, "bets": int(take.sum()),
                            "win_pct": round(won[take].mean() * 100, 1) if take.any() else np.nan,
                            "avg_price": round(p[take].mean(), 3) if take.any() else np.nan,
                            "pnl_per_share": round(pnl.sum(), 2), "roi_pct": round(pnl.sum() / cost.sum() * 100, 2) if take.any() else np.nan,
                            "green_days": f"{int((daily > 0).sum())}/{len(daily)}"})
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(pd.DataFrame(out).to_string(index=False))

    print("\nTHE INFLUENCER RULE: buy the leading side when it costs 75-92c, 1-4 min left (fee 7%)")
    for m in (4, 3, 2, 1):
        s = d[d.m == m]
        side_up = s.up_px >= .5
        p = np.where(side_up, s.up_px, 1 - s.up_px) + HAIRCUT
        won = np.where(side_up, s.up_won, ~s.up_won.astype(bool)).astype(float)
        take = (p >= .75) & (p <= .92)
        pnl = (won - p - fee(p, .07))[take]
        print(f"  {m} min left: {int(take.sum())} bets, won {won[take].mean() * 100:.1f}%, "
              f"avg price {p[take].mean():.3f}, P&L per $1 share {pnl.sum():+.2f}, "
              f"ROI {pnl.sum() / (p[take] + fee(p[take], .07)).sum() * 100:+.2f}%")


if __name__ == "__main__":
    main()
