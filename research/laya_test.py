"""Does Laya (open-source Jev alternative) predict crypto direction better than a coin flip?

At random past moments we show Laya the plain-English chart read the bot already produces
(trend structure, support/resistance, averages, volatility, volume, recent moves) and ask:
"Will the price be higher N hours from now?". Answers are graded against what happened,
alongside the base rate and our own learner on the exact same moments.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from jevbot import chart, journal
from jevbot.data import load_or_fetch

COINS = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT,LTCUSDT".split(",")
PER_COIN = int(sys.argv[1]) if len(sys.argv) > 1 else 300
HORIZON = 4


def state_text(sym: str, hist: pd.DataFrame) -> str:
    c = hist["close"]
    moves = ", ".join(f"{(c.iloc[-1] / c.iloc[-1 - n] - 1) * 100:+.2f}% over {n}h" for n in (1, 4, 24, 168))
    return f"{sym} hourly chart. {chart.describe(hist)} Recent moves: {moves}."


def main() -> None:
    from laya import Router
    router = Router()
    q = {"up": {"type": "noul", "instructions":
                f"Based on this chart, will the price be higher {HORIZON} hours from now than it is now?"}}
    rng = np.random.default_rng(0)
    rows = []
    for sym in COINS:
        df = load_or_fetch(sym, "1h", 1095)
        led = journal.replay(df, horizon=HORIZON)
        picks = np.sort(rng.choice(np.arange(600, len(df) - HORIZON), PER_COIN, replace=False))
        for t in picks:
            hist = df.iloc[: t + 1]
            rows.append({"symbol": sym, "time": df.index[t], "state": state_text(sym, hist.tail(400)),
                         "went_up": bool(df.close.iloc[t + HORIZON] > df.close.iloc[t]),
                         "learner_p": led.p_up.iloc[t]})
        print(f"{sym}: {PER_COIN} moments prepared", flush=True)
    d = pd.DataFrame(rows)
    # control: same texts shuffled across moments. If Laya scores the same, the chart text told it nothing.
    shuffled = d.state.sample(frac=1, random_state=1).to_numpy()
    t0 = time.time()
    for col, texts in (("laya_p", d.state.tolist()), ("laya_shuffled_p", list(shuffled))):
        out = router.predict_batch([{"state": s, "questions": q, "model": "english"} for s in texts], batch_size=32)
        d[col] = [float(r["answers"]["up"]["noul"]) for r in out]
    print(f"\nLaya answered {2 * len(d)} questions in {time.time() - t0:.0f}s\n")

    up = d.went_up
    base = up.mean()

    def grade(p: pd.Series) -> str:
        p = p.astype(float)
        m = p.notna()
        hit = ((p[m] > 0.5) == up[m]).mean() * 100
        brier = ((p[m] - up[m]) ** 2).mean()
        skill = (1 - brier / ((base - up[m]) ** 2).mean()) * 100
        return f"correct {hit:5.1f}%   calls up {(p[m] > 0.5).mean() * 100:5.1f}%   " \
               f"avg p {p[m].mean():.3f}   spread {p[m].std():.3f}   brier skill {skill:+6.2f}%"

    print(f"{len(d)} moments, {d.symbol.nunique()} coins, price went up {base * 100:.1f}% of the time\n")
    print(f"  always 'up'          correct {base * 100:5.1f}%")
    print(f"  Laya                 {grade(d.laya_p)}")
    print(f"  Laya, shuffled text  {grade(d.laya_shuffled_p)}")
    print(f"  our learner          {grade(d.learner_p)}")
    lo, hi = d.laya_p.quantile([0.2, 0.8])
    print(f"\n  when Laya was most bullish (top 20%):  went up {up[d.laya_p >= hi].mean() * 100:.1f}%")
    print(f"  when Laya was most bearish (bottom 20%): went up {up[d.laya_p <= lo].mean() * 100:.1f}%")
    print("\nPer coin, Laya correct %:")
    print(d.assign(ok=(d.laya_p > 0.5) == d.went_up).groupby("symbol").ok.mean().mul(100).round(1).to_string())
    print("\nExample state shown to Laya:\n  " + d.state.iloc[0])
    d.drop(columns="state").to_csv("reports/laya_test.csv", index=False)


if __name__ == "__main__":
    main()
