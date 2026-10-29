"""Polymarket soccer, second half: is the favourite underpriced? (the "late closer" idea)

For resolved soccer markets of the last N days: the price of the leading side between the 60th
and 85th match minute (kick-off + 75..100 minutes of real time, half-time included), versus how
often that side actually won, after the market's own fee (rate x p x (1-p) per share).
Read-only. Prints its discovery steps so the API shape is visible.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 60


def get(url, params=None, tries=3):
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    for k in range(tries):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": "Mozilla/5.0 research"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(1 + 2 * k)
    return None


def soccer_tag():
    sports = get(f"{GAMMA}/sports") or []
    print("sports endpoint:", len(sports), "entries; sample:", json.dumps(sports[:2])[:400])
    tags = set()
    for s in sports:
        if str(s.get("sport", "")).lower() in {"epl", "lal", "sea", "bun", "fl1", "ucl", "uel", "mls", "soccer"}:
            tags.update(t for t in str(s.get("tags", "")).split(",") if t and t != "1")
    print("soccer-ish tag ids from /sports:", sorted(tags)[:20])
    return sorted(tags)


def events_for(tag, since_iso):
    out, offset = [], 0
    while offset < 2000:
        page = get(f"{GAMMA}/events", {"tag_id": tag, "closed": "true", "end_date_min": since_iso,
                                         "limit": 100, "offset": offset})
        if not page:
            break
        out += page
        offset += 100
        if len(page) < 100:
            break
    return out


def kickoff(m, ev):
    for v in (m.get("gameStartTime"), ev.get("startTime"), m.get("eventStartTime")):
        if v:
            try:
                return pd.Timestamp(v).tz_convert("UTC") if pd.Timestamp(v).tzinfo else pd.Timestamp(v, tz="UTC")
            except Exception:
                pass
    return None


def sample(args):
    m, ko = args
    try:
        prices = [float(x) for x in json.loads(m["outcomePrices"])]
        tokens = json.loads(m["clobTokenIds"])
    except Exception:
        return []
    if sorted(prices) != [0.0, 1.0] or len(tokens) != 2:
        return []
    t0 = int(ko.timestamp())
    hist = (get(f"{CLOB}/prices-history", {"market": tokens[0], "startTs": t0 + 75 * 60,
                                           "endTs": t0 + 100 * 60, "fidelity": 1}) or {}).get("history") or []
    fs = m.get("feeSchedule") or {}
    rate = float(fs.get("rate", 0.0)) if m.get("feesEnabled") else 0.0
    rows = []
    for minute in (60, 70, 80, 85):
        t = t0 + (minute + 15) * 60
        pts = [p["p"] for p in hist if p["t"] <= t]
        if pts:
            rows.append({"market": m.get("slug"), "minute": minute, "yes_px": float(pts[-1]),
                         "yes_won": prices[0] == 1.0, "fee_rate": rate})
    return rows


def main():
    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    tags = soccer_tag() or ["100350"]
    evs = {}
    for tag in tags[:12]:
        for e in events_for(tag, since):
            evs[e["id"]] = e
    print(f"{len(evs)} closed soccer events since {since[:10]}")
    jobs = []
    for e in evs.values():
        for m in e.get("markets", []):
            q = (m.get("question") or "").lower()
            if "win" not in q and "draw" not in q:
                continue  # match-result markets only
            ko = kickoff(m, e)
            if ko is not None:
                jobs.append((m, ko))
    print(f"{len(jobs)} match-result markets with a kick-off time")
    if jobs:
        print("example:", json.dumps({k: jobs[0][0].get(k) for k in ("question", "gameStartTime", "feesEnabled",
                                                                     "feeSchedule", "sportsMarketType")})[:400])
    with ThreadPoolExecutor(6) as ex:
        rows = [r for rs in ex.map(sample, jobs) for r in rs]
    d = pd.DataFrame(rows)
    if d.empty:
        print("no price history found; nothing to score")
        return
    print(f"{d.market.nunique()} markets with in-game prices, fee rates seen: {sorted(d.fee_rate.unique())}\n")

    fav_yes = d.yes_px >= .5
    d["px"] = np.where(fav_yes, d.yes_px, 1 - d.yes_px)
    d["won"] = np.where(fav_yes, d.yes_won, ~d.yes_won.astype(bool)).astype(float)
    d["fee"] = d.fee_rate * d.px * (1 - d.px)
    d["bucket"] = pd.cut(d.px, [.5, .6, .7, .75, .8, .85, .9, .92, .95, .99, 1.0])
    print("CALIBRATION: price of the leading side in the 60th-85th minute vs how often it won")
    print(d.groupby("bucket", observed=True).agg(n=("won", "size"), avg_price=("px", "mean"),
                                                won_pct=("won", "mean"), avg_fee=("fee", "mean")).round(3).to_string())
    band = d[(d.px >= .75) & (d.px <= .92)]
    for minute in (60, 70, 80, 85):
        b = band[band.minute == minute]
        if len(b):
            pnl = b.won - b.px - b.fee - 0.01  # 1c to cross the spread
            print(f"  minute {minute}: {len(b)} buys at 75-92c, won {b.won.mean() * 100:.1f}% "
                  f"vs avg price {b.px.mean() * 100:.1f}c -> ROI {pnl.sum() / (b.px + b.fee).sum() * 100:+.2f}% "
                  f"after fee and 1c spread")
    d.drop(columns="bucket").to_csv("reports/polymarket_sports.csv", index=False)


if __name__ == "__main__":
    main()
