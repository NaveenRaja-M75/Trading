"""Does Jev (TypeSafe AI, via the Vercel AI Gateway) predict crypto direction better than a coin flip?

800 random past moments on 10 coins (80 each). At each moment Jev gets a full, multi-timeframe chart
read built only from data up to that moment, and answers: probability the price is higher 4 hours
later, and 24 hours later. Graded against what happened, next to the base rate, our learner, and a
shuffled control (the same descriptions given to the wrong moments: if Jev scores the same, it is
not reading the chart).

Fairness: the description has NO dates, years or price levels, only percentages. A model that has
read price history could otherwise recognise "BTC at 69,000" and remember what came next; a pass
earned that way would be fake.

Needs AI_GATEWAY_API_KEY (GitHub secret). The key is never printed. Read-only: no orders.
usage: python research/jev_test.py [moments_per_coin=80] [model=typesafe-ai/jev]
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from jevbot import chart, journal
from jevbot.data import load_or_fetch
from jevbot.features import rsi

COINS = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT,LTCUSDT".split(",")
PER_COIN = int(sys.argv[1]) if len(sys.argv) > 1 else 80
MODEL = sys.argv[2] if len(sys.argv) > 2 else "typesafe-ai/jev"
BASE = "https://ai-gateway.vercel.sh/typesafe/v1"
KEY = os.environ.get("AI_GATEWAY_API_KEY", "")
WARM = 24 * 210                      # enough hourly history for 200-day averages
HORIZONS = (4, 24)
QUESTIONS = {f"up_{h}h": {"type": "noul", "instructions":
                          f"Will this coin's price be higher {h} hours from now than it is right now?"} for h in HORIZONS}
QUESTION = "Questions (yes/no probability each): " + " | ".join(q["instructions"] for q in QUESTIONS.values())
COST = []
TREND = {1: "uptrend (higher highs and higher lows)", -1: "downtrend (lower highs and lower lows)", 0: "range, no clear trend"}


def pct(a: float, b: float) -> float:
    return (a / b - 1) * 100


def frame_read(df: pd.DataFrame, label: str, mas: tuple[int, ...]) -> str:
    c = df["close"]
    p = c.iloc[-1]
    s = chart.swings(df, 5).iloc[-1]
    out = [f"{label}: {TREND[int(chart.structure(df, 5).iloc[-1])]}."]
    if np.isfinite(s.last_sh) and np.isfinite(s.last_sl):
        out.append(f"Last swing high is {pct(s.last_sh, p):+.2f}% from price ({'broken above' if p > s.last_sh else 'acting as resistance'}); "
                   f"last swing low is {pct(s.last_sl, p):+.2f}% from price ({'broken below' if p < s.last_sl else 'acting as support'}).")
    avg = [f"{pct(p, c.rolling(n).mean().iloc[-1]):+.1f}% vs {n}-bar average" for n in mas if len(c) >= n]
    if avg:
        out.append("Price is " + ", ".join(avg) + ".")
    a = chart.atr(df).iloc[-1]
    vr = df["volume"].iloc[-1] / df["volume"].tail(20).mean()
    out.append(f"RSI(14) {rsi(c).iloc[-1]:.0f}. Typical bar range {a / p * 100:.2f}% of price. Volume {vr:.1f}x its 20-bar average.")
    return " ".join(out)


def to_bars(h: pd.DataFrame, rule: str) -> pd.DataFrame:
    return h.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def context(sym: str, h: pd.DataFrame) -> str:
    """Daily picture: trend votes, drawdown, volatility, multi-period returns."""
    d = to_bars(h, "1D")
    c = d.close
    p = h.close.iloc[-1]
    votes = sum(p > c.rolling(n).mean().iloc[-1] for n in (50, 100, 200))
    dd = pct(p, h.high.tail(24 * 90).max())
    vol = c.pct_change().tail(30).std() * np.sqrt(365) * 100
    rets = ", ".join(f"{pct(p, h.close.iloc[-1 - n]):+.1f}% over {lbl}" for n, lbl in
                     ((24 * 7, "7 days"), (24 * 30, "30 days"), (24 * 90, "90 days")))
    return (f"{sym} daily view: price is above {votes} of its 50/100/200-day averages; {dd:+.1f}% from its 90-day high; "
            f"30-day volatility {vol:.0f}% a year; {rets}.")


def rich_state(sym: str, h: pd.DataFrame, btc: pd.DataFrame | None) -> str:
    c = h.close
    p = c.iloc[-1]
    moves = ", ".join(f"{pct(p, c.iloc[-1 - n]):+.2f}% over {n}h" for n in (1, 4, 12, 24, 72))
    last12 = ", ".join(f"{x:+.2f}" for x in (c.pct_change().tail(12) * 100))
    d24 = h.tail(24)
    pos = (p - d24.low.min()) / max(d24.high.max() - d24.low.min(), 1e-12) * 100
    ts = h.index[-1]
    lines = [
        f"Coin: {sym}. Time of day {ts.hour:02d}:00 UTC, {ts.day_name()}.",
        context(sym, h),
        frame_read(to_bars(h.tail(24 * 60), "4h"), "4-hour chart", (20, 50, 100)),
        frame_read(h.tail(400), "1-hour chart", (20, 50, 200)),
        f"Recent moves: {moves}. Last 12 hourly changes in % (oldest to newest): {last12}. "
        f"Price sits at {pos:.0f}% of its 24-hour high-low range.",
    ]
    if btc is not None:
        b = btc.close
        bv = sum(b.iloc[-1] > b.rolling(24 * n).mean().iloc[-1] for n in (50, 100, 200))
        lines.append(f"Market leader BTC: {pct(b.iloc[-1], b.iloc[-25]):+.2f}% over 24h, "
                     f"{pct(b.iloc[-1], b.iloc[-169]):+.2f}% over 7 days, above {bv} of its 50/100/200-day averages.")
    return "\n".join(lines)


def call(state: str, retries: int = 4) -> tuple[tuple[float | None, float | None], str]:
    """Jev through the gateway's TypeSafe-compatible API: state + typed noul questions -> P(yes) each."""
    body = json.dumps({"model": MODEL, "state": state, "questions": QUESTIONS}).encode()
    for i in range(retries):
        req = urllib.request.Request(f"{BASE}/systemone", data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode()
            j = json.loads(raw)
            COST.append(float(j.get("provider_metadata", {}).get("gateway", {}).get("cost", 0) or 0))
            ans = j["answers"]
            vals = tuple(float(ans[f"up_{h}h"]["noul"]) for h in HORIZONS)
            return tuple(v if 0 <= v <= 1 else None for v in vals), raw[:700]   # type: ignore[return-value]
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:400]
            if e.code in (429, 500, 502, 503) and i < retries - 1:
                time.sleep(2 ** (i + 1))
                continue
            return (None, None), f"HTTP {e.code}: {msg}"
        except Exception as e:                                   # noqa: BLE001
            if i < retries - 1:
                time.sleep(2 ** (i + 1))
                continue
            return (None, None), f"error: {type(e).__name__}"
    return (None, None), "gave up"


def main() -> None:
    if not KEY:
        sys.exit("AI_GATEWAY_API_KEY is not set (add it as a GitHub secret)")

    rng = np.random.default_rng(0)
    data = {sym: load_or_fetch(sym, "1h", 1095) for sym in COINS}
    btc_all = data["BTCUSDT"]
    rows = []
    for sym in COINS:
        df = data[sym]
        led = journal.replay(df, horizon=HORIZONS[0])
        picks = np.sort(rng.choice(np.arange(WARM, len(df) - max(HORIZONS)), PER_COIN, replace=False))
        for t in picks:
            h = df.iloc[: t + 1]
            btc = None if sym == "BTCUSDT" else btc_all[btc_all.index <= h.index[-1]]
            row = {"symbol": sym, "time": df.index[t], "state": rich_state(sym, h, btc), "learner_p": led.p_up.iloc[t]}
            for hz in HORIZONS:
                row[f"up_{hz}h"] = bool(df.close.iloc[t + hz] > df.close.iloc[t])
            rows.append(row)
        print(f"{sym}: {PER_COIN} moments prepared", flush=True)
    d = pd.DataFrame(rows)
    leak = d.state.str.contains(r"20[12]\d|\d{4,}\.\d", regex=True)
    print(f"descriptions containing a year or a price level: {leak.sum()} (must be 0)")
    print("\nExample description shown to Jev:\n" + d.state.iloc[len(d) // 2] + "\n\n" + QUESTION + "\n", flush=True)

    (p4, p24), raw = call(d.state.iloc[0])
    print(f"Probe call to {MODEL}: p_up_4h={p4}, p_up_24h={p24}\nraw reply: {raw}\n", flush=True)
    if p4 is None:
        sys.exit("Jev did not return usable probabilities; stopping before the full test.")

    shuffled = d.state.sample(frac=1, random_state=1).tolist()
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        real = [r[0] for r in ex.map(call, d.state.tolist())]
        shuf = [r[0] for r in ex.map(call, shuffled)]
    for i, hz in enumerate(HORIZONS):
        d[f"jev_{hz}h"] = [r[i] for r in real]
        d[f"jev_shuffled_{hz}h"] = [r[i] for r in shuf]
    print(f"Jev answered {2 * len(d)} descriptions in {time.time() - t0:.0f}s, cost ${sum(COST):.4f}; unusable replies: "
          f"{d.jev_4h.isna().sum()} real + {d.jev_shuffled_4h.isna().sum()} shuffled\n")

    for hz in HORIZONS:
        up = d[f"up_{hz}h"]
        base = up.mean()

        def grade(p: pd.Series) -> str:
            p = p.astype(float)
            m = p.notna()
            hit = ((p[m] > 0.5) == up[m]).mean() * 100
            brier = ((p[m] - up[m]) ** 2).mean()
            skill = (1 - brier / ((base - up[m]) ** 2).mean()) * 100
            return (f"correct {hit:5.1f}%   calls up {(p[m] > 0.5).mean() * 100:5.1f}%   avg p {p[m].mean():.3f}   "
                    f"spread {p[m].std():.3f}   brier skill {skill:+6.2f}%   (n={m.sum()})")

        j = d[f"jev_{hz}h"].astype(float)
        n = j.notna().sum()
        print(f"######## {hz} hours ahead: {len(d)} moments, price went up {base * 100:.1f}% of the time")
        print(f"a real edge needs about {50 + 100 / np.sqrt(max(n, 1)):.1f}% correct (2 standard errors above 50%)")
        print(f"  always 'up'          correct {base * 100:5.1f}%")
        print(f"  Jev                  {grade(j)}")
        print(f"  Jev, shuffled        {grade(d[f'jev_shuffled_{hz}h'])}")
        if hz == HORIZONS[0]:
            print(f"  our learner          {grade(d.learner_p)}")
        if n > 20 and j.std() > 0:
            lo, hi = j.quantile([0.2, 0.8])
            print(f"  Jev most bullish (top 20%): went up {up[j >= hi].mean() * 100:.1f}%; "
                  f"most bearish (bottom 20%): went up {up[j <= lo].mean() * 100:.1f}%")
            for cut in (0.6, 0.7, 0.8):
                sure = (j >= cut) | (j <= 1 - cut)
                if sure.sum():
                    print(f"  confidence >= {cut:.0%}: {sure.sum()} calls, correct {((j[sure] > 0.5) == up[sure]).mean() * 100:.1f}%")
        print("  per coin, correct %: " + ", ".join(
            f"{s[:-4]} {v:.0f}" for s, v in d.assign(ok=(j > 0.5) == up)[j.notna()].groupby("symbol").ok.mean().mul(100).items()))
        print()
    os.makedirs("reports", exist_ok=True)
    d.drop(columns="state").to_csv("reports/jev_test.csv", index=False)


if __name__ == "__main__":
    main()
