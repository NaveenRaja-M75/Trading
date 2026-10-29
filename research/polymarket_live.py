"""Live test of the crypto clock on REAL order-book prices (what a buyer actually pays).

Every ~10 s for `minutes`: the best ask/bid and size for Up and Down in the current 15-minute
Bitcoin market, plus the Binance BTC price. Then waits for the windows to resolve and scores
"buy the side the model favours when its ask (plus fee) is below the model's chance".

Read-only. Places no bets.
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
BINANCE = "https://data-api.binance.vision/api/v3"
MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 60
RATE = 0.07


def get(url, params=None, tries=3):
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    for k in range(tries):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": "Mozilla/5.0 research"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(0.5 + k)
    return None


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def best(book):
    asks = [(float(a["price"]), float(a["size"])) for a in (book or {}).get("asks", [])]
    bids = [(float(b["price"]), float(b["size"])) for b in (book or {}).get("bids", [])]
    ask = min(asks) if asks else (None, 0.0)
    bid = max(bids) if bids else (None, 0.0)
    return ask, bid


markets, klines_cache = {}, {}


def market(ts):
    if ts not in markets:
        ev = get(f"{GAMMA}/events", {"slug": f"btc-updown-15m-{ts}"})
        if not ev:
            return None
        m = ev[0]["markets"][0]
        markets[ts] = json.loads(m["clobTokenIds"])
    return markets[ts]


def klines(minute):
    """Last 61 closed 1m candles as of `minute` (cached per minute)."""
    if minute not in klines_cache:
        k = get(f"{BINANCE}/klines", {"symbol": "BTCUSDT", "interval": "1m", "limit": 62})
        if not k:
            return None
        df = pd.DataFrame([r[:5] for r in k], columns=["t", "o", "h", "l", "c"]).astype(float)
        klines_cache.clear()
        klines_cache[minute] = df.iloc[:-1]  # drop the forming candle
    return klines_cache[minute]


def record():
    rows, end = [], time.time() + MINUTES * 60
    while time.time() < end:
        now = time.time()
        ts = int(now - now % 900)
        left = ts + 900 - now
        toks = market(ts)
        k = klines(int(now // 60))
        px = get(f"{BINANCE}/ticker/price", {"symbol": "BTCUSDT"})
        if toks and k is not None and px:
            up_ask, up_bid = best(get(f"{CLOB}/book", {"token_id": toks[0]}))
            dn_ask, dn_bid = best(get(f"{CLOB}/book", {"token_id": toks[1]}))
            opens = k[k.t == ts * 1000]
            p0 = float(opens.o.iloc[0]) if len(opens) else np.nan
            sig = np.log(k.c).diff().tail(60).std()
            p = float(px["price"])
            q_up = phi(math.log(p / p0) / (sig * math.sqrt(max(left, 1) / 60))) if np.isfinite(p0) else np.nan
            rows.append({"t": now, "ts": ts, "left_s": left, "btc": p, "p0": p0, "q_up": q_up,
                         "up_ask": up_ask[0], "up_ask_size": up_ask[1], "up_bid": up_bid[0],
                         "dn_ask": dn_ask[0], "dn_ask_size": dn_ask[1], "dn_bid": dn_bid[0]})
        time.sleep(max(0.0, 10 - (time.time() - now)))
    return pd.DataFrame(rows)


def resolve(ts_list):
    out, deadline = {}, time.time() + 20 * 60
    pending = set(ts_list)
    while pending and time.time() < deadline:
        for ts in list(pending):
            ev = get(f"{GAMMA}/events", {"slug": f"btc-updown-15m-{ts}"})
            if ev:
                m = ev[0]["markets"][0]
                pr = [float(x) for x in json.loads(m["outcomePrices"])]
                if m.get("closed") and sorted(pr) == [0.0, 1.0]:
                    out[ts] = pr[0] == 1.0
                    pending.discard(ts)
        if pending:
            time.sleep(30)
    return out


def main():
    d = record()
    print(f"{len(d)} snapshots over {d.ts.nunique()} windows", flush=True)
    res = resolve(sorted(d.ts.unique()))
    d["up_won"] = d.ts.map(res)
    d = d.dropna(subset=["up_won", "q_up"])
    d.to_csv("reports/polymarket_live.csv", index=False)
    print(f"{d.ts.nunique()} windows resolved\n")

    late = d[(d.left_s <= 300) & (d.left_s >= 30)]
    print("REAL ORDER BOOK, last 5 minutes of each window")
    spread_up = (late.up_ask - late.up_bid)
    print(f"  median Up spread {spread_up.median():.3f}, median ask size {late.up_ask_size.median():.0f} shares")
    print(f"  Up ask + Down ask (should be >= 1.00): median {(late.up_ask + late.dn_ask).median():.3f}\n")

    side_up = late.q_up >= .5
    q = np.where(side_up, late.q_up, 1 - late.q_up)
    ask = np.where(side_up, late.up_ask, late.dn_ask).astype(float)
    won = np.where(side_up, late.up_won, ~late.up_won.astype(bool)).astype(float)
    fee = RATE * ask * (1 - ask)
    print("BUY THE MODEL'S SIDE AT THE REAL ASK (first qualifying snapshot per window)")
    for margin in (0.0, 0.03, 0.08):
        ok = (q - ask - fee >= margin) & (ask < .99) & np.isfinite(ask)
        first = late[ok].groupby("ts").head(1).index
        sel = late.index.isin(first)
        pnl = (won - ask - fee)[sel]
        n = int(sel.sum())
        if n:
            print(f"  margin {margin:.2f}: {n} bets of {late.ts.nunique()} windows, won {won[sel].mean() * 100:.1f}%, "
                  f"avg ask {ask[sel].mean():.3f}, model said {q[sel].mean():.3f}, "
                  f"ROI {pnl.sum() / (ask + fee)[sel].sum() * 100:+.2f}%")
        else:
            print(f"  margin {margin:.2f}: no bets (the real ask never left that much room)")
    print("\nHOW OFTEN THE REAL ASK LEFT ANY ROOM (model chance - ask - fee > 0):",
          f"{((q - ask - fee) > 0).mean() * 100:.1f}% of late snapshots")
    gap = (q - ask - fee)
    print("  distribution of (model chance - ask - fee):",
          {k: round(float(v), 3) for k, v in pd.Series(gap).describe(percentiles=[.1, .5, .9]).items()})


if __name__ == "__main__":
    main()
