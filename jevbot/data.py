"""Market data: Binance public klines (no API key needed), CSV cache, synthetic data for tests."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BINANCE_DATA_URL = "https://data-api.binance.vision/api/v3/klines"
COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]
INTERVAL_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
               "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def _get_json(url: str, params: dict, timeout: float = 10.0):
    full = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(full, timeout=timeout) as resp:
        return json.loads(resp.read())


def _to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame([r[:6] for r in rows], columns=COLUMNS)
    df[COLUMNS[1:]] = df[COLUMNS[1:]].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    return df.set_index("open_time")


def fetch_recent(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Latest `limit` (<=1000) closed+forming candles. Drops the still-forming last candle."""
    rows = _get_json(BINANCE_DATA_URL, {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)})
    df = _to_frame(rows)
    now = pd.Timestamp.now(tz="UTC")
    if len(df) and df.index[-1] + pd.Timedelta(milliseconds=INTERVAL_MS[interval]) > now:
        df = df.iloc[:-1]
    return df


def fetch_history(symbol: str, interval: str, days: float) -> pd.DataFrame:
    """Page backwards through Binance klines to collect `days` of history."""
    step = INTERVAL_MS[interval]
    end = int(time.time() * 1000)
    start = end - int(days * 86_400_000)
    frames, cursor = [], start
    while cursor < end:
        rows = _get_json(BINANCE_DATA_URL, {"symbol": symbol, "interval": interval,
                                            "startTime": cursor, "limit": 1000})
        if not rows:
            break
        frames.append(_to_frame(rows))
        cursor = int(rows[-1][0]) + step
        time.sleep(0.1)  # stay well under rate limits
    df = pd.concat(frames)
    df = df[~df.index.duplicated()].sort_index()
    return df.iloc[:-1]  # last candle may still be forming


def cache_path(symbol: str, interval: str, days: float) -> Path:
    return Path("data") / f"{symbol}_{interval}_{days:g}d.csv"


def load_or_fetch(symbol: str, interval: str, days: float, refresh: bool = False) -> pd.DataFrame:
    path = cache_path(symbol, interval, days)
    if path.exists() and not refresh:
        return load_csv(path)
    df = fetch_history(symbol, interval, days)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[COLUMNS[1:]].astype(float)


def synthetic(n: int = 5000, seed: int = 0, drift: float = 0.0, vol: float = 0.001,
              start_price: float = 60_000.0) -> pd.DataFrame:
    """Random-walk candles. With drift=0 there is NO edge: any strategy should lose ~costs."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wiggle = np.abs(rng.normal(0, vol / 2, n)) * close
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) + wiggle,
        "low": np.minimum(open_, close) - wiggle, "close": close,
        "volume": rng.lognormal(3, 0.5, n),
    }, index=idx)
