"""Does the "crypto regime score" from tradermonty/claude-trading-skills improve our BTC rule?

Their crypto-regime-analyzer scores the market 0-100 from six parts (BTC trend 25%, alt breadth 20%,
BTC dominance 15%, funding 15%, drawdown & volatility 15%, momentum thrust 10%) and labels it
RISK_ON (80+), NEUTRAL (40-79) or RISK_OFF (<40). Its own docs say the bands were never backtested.
Here every part except dominance (no market-cap history on Binance; its weight is redistributed, as
their scorer does) is rebuilt from their published thresholds, day by day, using only data up to
that day. Thresholds are theirs, not tuned here.

Variants, all BTC only, checked daily, 5% trade band, 0.12% per side:
  A  vote_voltgt (our live rule)
  B  regime zones alone: RISK_ON 100%, NEUTRAL 50%, RISK_OFF 0%
  C  regime score alone: exposure = score / 100
  D  A, but nothing while RISK_OFF
  E  A, halved while funding is crowded (BTC+ETH 8h average above +0.03%)
Caveat: the alt list is coins that still trade today (survivorship), which flatters breadth a little.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from funding_carry_test import funding
from india_test import perf
from jevbot.data import load_or_fetch
from jevbot.portfolio import vote_voltgt

ALTS = ("ETHUSDT BNBUSDT XRPUSDT ADAUSDT LTCUSDT EOSUSDT XLMUSDT TRXUSDT ETCUSDT NEOUSDT LINKUSDT "
        "DOGEUSDT ATOMUSDT SOLUSDT AVAXUSDT DOTUSDT BCHUSDT").split()
COST, BAND = 0.0012, 0.05
W = {"btc_trend": 0.25, "alt_breadth": 0.20, "funding": 0.15, "drawdown_vol": 0.15, "momentum": 0.10}


def btc_trend(c: pd.Series) -> pd.Series:
    m50, m200 = c.rolling(50).mean(), c.rolling(200).mean()
    prev = m200.shift(20)
    base = np.select([(c > m50) & (m50 > m200), (c > m200) & (c <= m50) & (m50 > m200),
                      (c <= m200) & (c <= m50) & (m50 > m200), (c > m50) & (m50 <= m200)],
                     [90, 65, 55, 45], 15).astype(float)
    slope = np.where(m200 > prev, 10, np.where(m200 < prev, -10, 0))
    return pd.Series(np.clip(base + slope, 0, 100), index=c.index).where(prev.notna())


def alt_breadth(px: pd.DataFrame) -> pd.Series:
    alts = px.drop(columns="BTCUSDT")
    ok = alts.notna().rolling(200).sum() >= 200
    a200 = (alts > alts.rolling(200).mean()) & ok
    a50 = (alts > alts.rolling(50).mean()) & ok
    n = ok.sum(axis=1)
    p200, p50 = a200.sum(axis=1) / n * 100, a50.sum(axis=1) / n * 100
    base = np.select([p200 >= 80, p200 >= 65, p200 >= 50, p200 >= 35, p200 >= 20], [95, 80, 65, 45, 25], 10)
    mod = np.where(p50 >= p200 + 15, 5, np.where(p50 <= p200 - 15, -5, 0))
    return pd.Series(np.clip(base + mod, 0, 100), index=px.index).where(n >= 5)


def funding_score(avg: pd.Series) -> pd.Series:
    s = np.select([avg <= -0.0001, avg < 0, avg <= 0.0001, avg <= 0.0003, avg <= 0.0006], [80, 65, 75, 55, 30], 10)
    return pd.Series(s, index=avg.index, dtype=float).where(avg.notna())


def drawdown_vol(c: pd.Series) -> pd.Series:
    lr = np.log(c).diff()
    vol = lr.rolling(29).std() * math.sqrt(365)            # their window: last 29 log returns of 30 closes
    out = pd.Series(np.nan, index=c.index)
    vals, cv = c.to_numpy(), vol.to_numpy()
    for i in range(364, len(c)):
        dd = 1 - vals[i] / vals[i - 364: i + 1].max()
        score = 90 if dd <= .10 else 75 if dd <= .20 else 55 if dd <= .35 else 35 if dd <= .50 else 20 if dd <= .65 else 10
        sample = cv[i - 364 + 30: i + 1: 7]
        sample = sample[np.isfinite(sample)]
        pct = (sample <= cv[i]).mean() if len(sample) else 0.5
        if pct <= 1 / 3:
            score += 10
        elif pct >= 2 / 3:
            score -= 10
        if dd > .65 and pct >= 2 / 3:
            score = max(score, 15)
        out.iloc[i] = min(100, max(0, score))
    return out


def momentum(px: pd.DataFrame) -> pd.Series:
    r30 = px / px.shift(30) - 1
    n = r30.notna().sum(axis=1)
    p = (r30 > 0).sum(axis=1) / n * 100
    s = np.select([p >= 85, p >= 65, p >= 45, p >= 25, p >= 10], [90, 75, 55, 35, 20], 35)
    return pd.Series(s, index=px.index, dtype=float).where(n >= 5)


def composite(parts: dict[str, pd.Series]) -> pd.Series:
    df = pd.DataFrame(parts)
    w = pd.Series(W)
    avail = df.notna()
    wsum = avail.mul(w, axis=1).sum(axis=1)
    score = df.fillna(0).mul(w, axis=1).sum(axis=1) / wsum
    enough = (avail.sum(axis=1) >= 4) & (wsum >= 0.65)       # their rule (out of the full 1.00)
    return score.where(enough)


def banded(w: pd.Series) -> pd.Series:
    out, held = [], 0.0
    for x in w.fillna(0.0).to_numpy():
        if abs(x - held) >= BAND or (x == 0 and held > 0):
            held = x
        out.append(held)
    return pd.Series(out, index=w.index)


def run(c: pd.Series, target: pd.Series) -> tuple[pd.Series, int]:
    held = banded(target).shift().fillna(0.0)                # decided at today's close, held from tomorrow
    r = c.pct_change().fillna(0.0)
    turn = held.diff().abs().fillna(held.abs())
    return held * r - turn * COST, int((turn > 0).sum())


def sharpe_diff_boot(a: pd.Series, b: pd.Series, n: int = 2000, block: int = 20, seed: int = 0) -> float:
    """Share of block-bootstrap resamples where b's Sharpe beats a's."""
    rng = np.random.default_rng(seed)
    x, y = a.to_numpy(), b.to_numpy()
    T, wins = len(x), 0
    for _ in range(n):
        starts = rng.integers(0, T - block, T // block + 1)
        idx = (starts[:, None] + np.arange(block)).ravel()[:T]
        sa, sb = x[idx].mean() / (x[idx].std() + 1e-12), y[idx].mean() / (y[idx].std() + 1e-12)
        wins += sb > sa
    return wins / n


def main() -> None:
    dfs = {}
    for s in ("BTCUSDT",) + tuple(ALTS):
        try:
            d = load_or_fetch(s, "1d", 3300)
            d.index = d.index.normalize()
            dfs[s] = d.close
        except Exception as e:                                   # noqa: BLE001
            print(f"{s}: skipped ({e!r})")
    px = pd.DataFrame(dfs).sort_index()
    print(f"universe: {px.shape[1]} coins, {px.index[0]:%Y-%m-%d} -> {px.index[-1]:%Y-%m-%d}", flush=True)
    btc = px.BTCUSDT
    fr = pd.DataFrame({s: funding(s) for s in ("BTCUSDT", "ETHUSDT")})
    avg_f = fr.mean(axis=1).groupby(fr.index.floor("D")).last()
    avg_f.index = avg_f.index.normalize()
    avg_f = avg_f.reindex(px.index).ffill(limit=2)

    parts = {"btc_trend": btc_trend(btc), "alt_breadth": alt_breadth(px), "funding": funding_score(avg_f),
             "drawdown_vol": drawdown_vol(btc), "momentum": momentum(px)}
    score = composite(parts)
    zone = pd.cut(score, [-1, 39.999, 79.999, 101], labels=["RISK_OFF", "NEUTRAL", "RISK_ON"])
    start = score.first_valid_index()
    print(f"regime score available from {start:%Y-%m-%d}; time in each zone: "
          + ", ".join(f"{k} {v * 100:.0f}%" for k, v in zone[start:].value_counts(normalize=True).items()))

    base = vote_voltgt(px[["BTCUSDT"]]).BTCUSDT
    targets = {
        "A vote_voltgt (live)": base,
        "B zones alone": zone.map({"RISK_ON": 1.0, "NEUTRAL": 0.5, "RISK_OFF": 0.0}).astype(float),
        "C score/100 alone": (score / 100).round(2),
        "D A, off in RISK_OFF": base.where(zone != "RISK_OFF", 0.0),
        "E A, halved if funding crowded": base.where(~(avg_f > 0.0003), base / 2),
        "hold BTC": pd.Series(1.0, index=px.index),
    }
    nets, orders = {}, {}
    for k, t in targets.items():
        nets[k], orders[k] = run(btc, t.reindex(px.index))
    T = pd.DataFrame(nets)[start:]
    years = (T.index[-1] - T.index[0]).days / 365.25
    halves = [(T.index[0], pd.Timestamp("2022-01-01", tz="UTC")), (pd.Timestamp("2022-01-01", tz="UTC"), T.index[-1])]
    rows = []
    for k in T:
        row = perf(k, T[k], 365)
        row["orders_per_year"] = round(orders[k] / years, 1)
        for a, b in halves:
            row[f"sharpe {a.year}-{b.year if b.year < 2026 else 'now'}"] = perf(k, T[k][a:b], 365)["sharpe"]
        row["P(beats A)"] = "-" if k.startswith("A ") else round(sharpe_diff_boot(T["A vote_voltgt (live)"], T[k]), 2)
        rows.append(row)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(f"\nBTC, {T.index[0]:%Y-%m-%d} -> {T.index[-1]:%Y-%m-%d} ({years:.1f} years), after 0.12% per side")
        print(pd.DataFrame(rows).set_index("name").to_string())
        yr = (1 + T).groupby(T.index.year).prod().sub(1).mul(100).round(1)
        print("\nyear by year, %:\n" + yr.to_string())
    print("\nP(beats A) = share of 2,000 block-bootstrap resamples where the variant's Sharpe is higher. "
          "Below 0.95 means no reliable improvement. 4 variants were tried, so the bar is higher still.")
    now = {k: (round(float(v.dropna().iloc[-1]), 1) if v.notna().any() else None) for k, v in parts.items()}
    print(f"\nToday: score {score.dropna().iloc[-1]:.0f} ({zone.dropna().iloc[-1]}), parts {now}")


if __name__ == "__main__":
    main()
