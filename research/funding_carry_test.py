"""Delta-neutral funding carry: buy spot BTC, short the same amount of the BTC perpetual, collect funding.

No direction needed: price moves cancel; when funding is positive (the usual case) the short receives
funding every 8 hours. Measured on Binance's real funding history (2019+), BTC and ETH:
  always_on       hold the hedge all the time
  only_when_paid  hold it only while the last 7 days of funding averaged above 5%/yr (switching costs counted)
Costs: spot 0.10% + perp 0.05% per side, so ~0.30% to open and close the pair.

India: crypto gains are taxed 31.2% per leg and a loss on one leg cannot offset the other (Section
115BBH as commonly read; whether perp profits count as VDA is not settled - confirm with a CA).
In a hedge one leg always gains and the other loses, so the tax lands on the winning leg every year.
The after-tax column closes and reopens the hedge each 31 March (Indian financial year end).
"""
import io
import json
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from jevbot.data import load_or_fetch

OPEN_CLOSE = 0.003      # spot 0.10% + perp 0.05%, both legs, open and close
TAX = 0.312


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def funding(sym: str) -> pd.Series:
    rows, start = [], int(pd.Timestamp("2019-09-01", tz="UTC").timestamp() * 1000)
    try:
        while True:
            page = json.loads(get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&startTime={start}&limit=1000"))
            if not page:
                break
            rows += page
            start = page[-1]["fundingTime"] + 1
            if len(page) < 1000:
                break
            time.sleep(0.2)
        s = pd.Series({pd.Timestamp(r["fundingTime"], unit="ms", tz="UTC"): float(r["fundingRate"]) for r in rows})
        print(f"{sym}: {len(s)} funding prints from the API")
        return s.sort_index()
    except Exception as e:
        print(f"{sym}: API unavailable ({e!r}); using Binance monthly archives")
    parts = []
    for m in pd.date_range("2019-09-01", pd.Timestamp.now(), freq="MS"):
        url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m:%Y-%m}.zip"
        try:
            z = zipfile.ZipFile(io.BytesIO(get(url)))
            df = pd.read_csv(z.open(z.namelist()[0]))
            parts.append(pd.Series(df.iloc[:, -1].astype(float).to_numpy(),
                                   index=pd.to_datetime(df.iloc[:, 0].astype("int64"), unit="ms", utc=True)))
        except Exception:
            continue
    s = pd.concat(parts).sort_index()
    print(f"{sym}: {len(s)} funding prints from archives")
    return s[~s.index.duplicated()]


def main():
    for sym in ("BTCUSDT", "ETHUSDT"):
        f = funding(sym)
        spot = load_or_fetch(sym, "1d", 3300).close
        spot.index = spot.index.normalize()
        f = f[f.index >= spot.index[0]]
        daily = f.groupby(f.index.normalize()).sum()                 # funding earned per day (fraction)
        print(f"\n######## {sym} funding carry, {daily.index[0]:%Y-%m-%d} -> {daily.index[-1]:%Y-%m-%d}")
        print(f"  funding prints negative (shorts PAY): {(f < 0).mean() * 100:.0f}% of the time; "
              f"average {f.mean() * 3 * 365 * 100:.1f}%/yr")
        # strategy returns per day
        always = daily.copy()
        always.iloc[0] -= OPEN_CLOSE / 2
        always.iloc[-1] -= OPEN_CLOSE / 2
        trail = daily.rolling(7).mean() * 365
        on = (trail.shift(1) > 0.05).astype(float)                   # decided on yesterday's info
        switches = on.diff().abs().fillna(on.iloc[0])
        gated = on * daily - switches * OPEN_CLOSE / 2
        rows = []
        for name, r in (("always_on", always), ("only_when_paid", gated)):
            yearly = r.groupby(r.index.year).sum() * 100
            monthly = r.groupby(r.index.to_period("M")).sum() * 100
            rows.append({"strategy": name, "per_year_avg_pct": round(r.mean() * 365 * 100, 1),
                         "worst_month_pct": round(monthly.min(), 2), "losing_months_pct": round((monthly < 0).mean() * 100),
                         **{str(y): round(v, 1) for y, v in yearly.items()}})
        with pd.option_context("display.width", 230, "display.max_columns", 30):
            print(pd.DataFrame(rows).set_index("strategy").to_string())

        # India: each financial year (Apr-Mar) close and reopen; tax each leg's gain, no offset
        print(f"\n  INDIA, per financial year (hedge closed each 31 March), per 100 of notional:")
        fy = lambda d: d.year if d.month >= 4 else d.year - 1
        for y in sorted({fy(d) for d in daily.index}):
            days = [d for d in daily.index if fy(d) == y]
            p0, p1 = spot.asof(days[0]), spot.asof(days[-1])
            move = p1 / p0 - 1
            fund = daily.loc[days].sum()
            spot_gain, perp_gain = move, -move + fund                 # perp short receives funding
            tax = TAX * (max(spot_gain, 0) + max(perp_gain, 0))
            pre = fund - OPEN_CLOSE
            print(f"   FY{y}-{str(y + 1)[2:]}: BTC/ETH move {move * 100:+6.1f}%  funding {fund * 100:+5.1f}  "
                  f"before tax {pre * 100:+5.1f}  tax {-tax * 100:6.1f}  after tax {(pre - tax) * 100:+6.1f}")
    print("\nMinimum order: Binance's BTCUSDT perpetual needs about 100 USDT of notional per order, "
          "so a 20 USDT account cannot hold this hedge.")


if __name__ == "__main__":
    main()
