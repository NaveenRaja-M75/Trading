"""Does Jev (TypeSafe AI, via the Vercel AI Gateway) predict crypto direction better than a coin flip?

Same design as research/laya_test.py: random past moments on 10 coins, Jev reads the bot's
plain-English chart description and gives the probability that the price is higher 4 hours later.
Graded against what happened, next to the base rate, our learner, and a shuffled-text control
(the same texts given to the wrong moments: if Jev scores the same, the chart told it nothing).

Needs AI_GATEWAY_API_KEY (GitHub secret). The key is never printed. Read-only: no orders.
usage: python research/jev_test.py [moments_per_coin=80] [model=typesafe-ai/jev]
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from jevbot import journal
from jevbot.data import load_or_fetch
from laya_test import COINS, HORIZON, state_text

PER_COIN = int(sys.argv[1]) if len(sys.argv) > 1 else 80
MODEL = sys.argv[2] if len(sys.argv) > 2 else "typesafe-ai/jev"
BASE = "https://ai-gateway.vercel.sh/v1"
KEY = os.environ.get("AI_GATEWAY_API_KEY", "")
QUESTION = (f"Will the price be higher {HORIZON} hours from now than it is now? "
            'Reply only with JSON {"p_up": <probability between 0 and 1 that it will be higher>}.')


def call(state: str, retries: int = 4) -> tuple[float | None, str]:
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 60,
                       "messages": [{"role": "user", "content": state + "\n\n" + QUESTION}]}).encode()
    for i in range(retries):
        req = urllib.request.Request(f"{BASE}/chat/completions", data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode()
            text = json.loads(raw)["choices"][0]["message"]["content"] or ""
            return parse(text), raw[:600]
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:400]
            if e.code in (429, 500, 502, 503) and i < retries - 1:
                time.sleep(2 ** (i + 1))
                continue
            return None, f"HTTP {e.code}: {msg}"
        except Exception as e:                                   # noqa: BLE001
            if i < retries - 1:
                time.sleep(2 ** (i + 1))
                continue
            return None, f"error: {type(e).__name__}"
    return None, "gave up"


def parse(text: str) -> float | None:
    try:
        v = float(json.loads(text[text.index("{"): text.rindex("}") + 1])["p_up"])
    except Exception:                                            # noqa: BLE001
        nums = [float(x) for x in re.findall(r"\d*\.?\d+", text)]
        nums = [x for x in nums if 0 <= x <= 1]
        if not nums:
            return None
        v = nums[0]
    return v if 0 <= v <= 1 else None


def main() -> None:
    if not KEY:
        sys.exit("AI_GATEWAY_API_KEY is not set (add it as a GitHub secret)")
    with urllib.request.urlopen(urllib.request.Request(f"{BASE}/models", headers={"User-Agent": "Mozilla/5.0"}),
                                timeout=60) as r:
        meta = [m for m in json.loads(r.read())["data"] if "typesafe" in m["id"]]
    print("Gateway entries for TypeSafe:\n" + json.dumps(meta, indent=1)[:3000] + "\n", flush=True)

    p, raw = call("BTCUSDT hourly chart. Price is above its 50, 100 and 200-hour averages.")
    print(f"Probe call to {MODEL}: parsed p_up={p}\nraw reply: {raw}\n", flush=True)
    if p is None:
        sys.exit("Jev did not return a usable probability; stopping before the full test.")

    rng = np.random.default_rng(0)
    rows = []
    for sym in COINS:
        df = load_or_fetch(sym, "1h", 1095)
        led = journal.replay(df, horizon=HORIZON)
        picks = np.sort(rng.choice(np.arange(600, len(df) - HORIZON), PER_COIN, replace=False))
        for t in picks:
            rows.append({"symbol": sym, "time": df.index[t], "state": state_text(sym, df.iloc[: t + 1].tail(400)),
                         "went_up": bool(df.close.iloc[t + HORIZON] > df.close.iloc[t]),
                         "learner_p": led.p_up.iloc[t]})
        print(f"{sym}: {PER_COIN} moments prepared", flush=True)
    d = pd.DataFrame(rows)
    shuffled = d.state.sample(frac=1, random_state=1).tolist()
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        d["jev_p"] = [r[0] for r in ex.map(call, d.state.tolist())]
        d["jev_shuffled_p"] = [r[0] for r in ex.map(call, shuffled)]
    print(f"\nJev answered {2 * len(d)} questions in {time.time() - t0:.0f}s; "
          f"unusable replies: {d.jev_p.isna().sum()} + {d.jev_shuffled_p.isna().sum()}\n")

    up = d.went_up
    base = up.mean()

    def grade(p: pd.Series) -> str:
        p = p.astype(float)
        m = p.notna()
        hit = ((p[m] > 0.5) == up[m]).mean() * 100
        brier = ((p[m] - up[m]) ** 2).mean()
        skill = (1 - brier / ((base - up[m]) ** 2).mean()) * 100
        return (f"correct {hit:5.1f}%   calls up {(p[m] > 0.5).mean() * 100:5.1f}%   avg p {p[m].mean():.3f}   "
                f"spread {p[m].std():.3f}   brier skill {skill:+6.2f}%   (n={m.sum()})")

    n = d.jev_p.notna().sum()
    print(f"{len(d)} moments, {d.symbol.nunique()} coins, price went up {base * 100:.1f}% of the time")
    print(f"a real edge needs about {50 + 100 / np.sqrt(max(n, 1)):.1f}% correct (2 standard errors above 50%)\n")
    print(f"  always 'up'          correct {base * 100:5.1f}%")
    print(f"  Jev                  {grade(d.jev_p)}")
    print(f"  Jev, shuffled text   {grade(d.jev_shuffled_p)}")
    print(f"  our learner          {grade(d.learner_p)}")
    j = d.jev_p.astype(float)
    if j.notna().sum() > 20 and j.std() > 0:
        lo, hi = j.quantile([0.2, 0.8])
        print(f"\n  when Jev was most bullish (top 20%):  went up {up[j >= hi].mean() * 100:.1f}%")
        print(f"  when Jev was most bearish (bottom 20%): went up {up[j <= lo].mean() * 100:.1f}%")
        for cut in (0.6, 0.7, 0.8):
            sure = (j >= cut) | (j <= 1 - cut)
            if sure.sum():
                print(f"  confidence >= {cut:.0%}: {sure.sum()} calls, correct "
                      f"{((j[sure] > 0.5) == up[sure]).mean() * 100:.1f}%")
    print("\nPer coin, Jev correct %:")
    print(d.assign(ok=(j > 0.5) == d.went_up)[j.notna()].groupby("symbol").ok.mean().mul(100).round(1).to_string())
    print("\nExample state shown to Jev:\n  " + d.state.iloc[0])
    os.makedirs("reports", exist_ok=True)
    d.drop(columns="state").to_csv("reports/jev_test.csv", index=False)


if __name__ == "__main__":
    main()
