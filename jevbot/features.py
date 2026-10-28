"""Causal features: every value at row t uses only data up to and including bar t."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["ret_1", "ret_5", "ret_15", "vol_15", "z_20", "rsi_14", "vol_ratio", "range_pct"]


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    logc = np.log(c)
    f = df[["open", "high", "low", "close", "volume"]].copy()  # raw candles for chart-reading signals
    f["ret_1"] = logc.diff(1)
    f["ret_5"] = logc.diff(5)
    f["ret_15"] = logc.diff(15)
    f["vol_15"] = f["ret_1"].rolling(15).std()
    mean20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    f["z_20"] = (c - mean20) / std20.replace(0, np.nan)
    f["rsi_14"] = rsi(c)
    f["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    f["range_pct"] = (df["high"] - df["low"]) / c
    # forward return of the NEXT bar: label only, never used as an input
    f["fwd_ret"] = logc.shift(-1) - logc
    return f.replace([np.inf, -np.inf], np.nan)
