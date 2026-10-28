"""Signals ("judges"). Each maps features -> score in [-1, 1] per bar, causally.

score > 0 means "want long", < 0 "want short/flat". The policy, not the signal, decides trades.
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

from . import chart
from .features import FEATURE_COLUMNS, rsi


class Signal:
    name = "base"

    def scores(self, feats: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class CoinFlip(Signal):
    """Baseline. A real signal must beat this after costs or it is worthless."""
    name = "coinflip"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def scores(self, feats):
        rng = np.random.default_rng(self.seed)
        return pd.Series(rng.choice([-1.0, 1.0], len(feats)), index=feats.index)


class Momentum(Signal):
    """Follow the recent move, normalised by recent volatility."""
    name = "momentum"

    def __init__(self, lookback: str = "ret_15", scale: float = 2.0):
        self.lookback, self.scale = lookback, scale

    def scores(self, feats):
        vol = feats["vol_15"] * np.sqrt(int(self.lookback.split("_")[1]))
        return np.tanh(feats[self.lookback] / vol.replace(0, np.nan) / self.scale).fillna(0.0)


class MeanReversion(Signal):
    """Fade stretched moves: short when far above the 20-bar mean, long when far below."""
    name = "meanrev"

    def __init__(self, scale: float = 2.0):
        self.scale = scale

    def scores(self, feats):
        return np.tanh(-feats["z_20"] / self.scale).fillna(0.0)


class LocalModel(Signal):
    """Walk-forward ridge regression on the feature set, predicting next-bar return.

    At bar t it is trained only on rows whose label was already known (rows < t),
    refit every `retrain` bars on the trailing `window` rows. No look-ahead.
    """
    name = "model"

    def __init__(self, window: int = 2000, retrain: int = 250, ridge: float = 10.0, min_train: int = 500):
        self.window, self.retrain, self.ridge, self.min_train = window, retrain, ridge, min_train

    def scores(self, feats):
        X_all = feats[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_all = feats["fwd_ret"].to_numpy(dtype=float)
        out = np.zeros(len(feats))
        w = mu = sd = None
        y_scale = 1.0
        for t in range(len(feats)):
            if t >= self.min_train and (w is None or t % self.retrain == 0):
                lo = max(0, t - self.window)
                X, y = X_all[lo:t - 1], y_all[lo:t - 1]  # label of row t-1 needs bar t's close: known at t
                ok = np.isfinite(X).all(1) & np.isfinite(y)
                if ok.sum() >= self.min_train // 2:
                    X, y = X[ok], y[ok]
                    mu, sd = X.mean(0), X.std(0) + 1e-12
                    Z = (X - mu) / sd
                    w = np.linalg.solve(Z.T @ Z + self.ridge * np.eye(Z.shape[1]), Z.T @ y)
                    y_scale = (np.std(Z @ w) + 1e-12)
            if w is not None and np.isfinite(X_all[t]).all():
                out[t] = np.tanh(((X_all[t] - mu) / sd) @ w / y_scale)
        return pd.Series(out, index=feats.index)


class Trend(Signal):
    """Classic trend filter: long only while price > 200-bar average and 50 > 200 ("golden cross")."""
    name = "trend"

    def __init__(self, fast: int = 50, slow: int = 200):
        self.fast, self.slow = fast, slow

    def scores(self, feats):
        c = feats["close"]
        fast, slow = chart.sma(c, self.fast), chart.sma(c, self.slow)
        return pd.Series(np.where((c > slow) & (fast > slow), 1.0, -1.0), index=feats.index)


class Donchian(Signal):
    """Turtle breakout: buy a close above the 55-bar high, sell a close below the 20-bar low."""
    name = "breakout"

    def __init__(self, entry: int = 55, exit: int = 20):
        self.entry, self.exit = entry, exit

    def scores(self, feats):
        hh = feats["high"].rolling(self.entry).max().shift(1)
        ll = feats["low"].rolling(self.exit).min().shift(1)
        return chart.latch(feats["close"] > hh, feats["close"] < ll)


class Structure(Signal):
    """Chart structure: in while making higher highs + higher lows, out when a swing low breaks."""
    name = "structure"

    def __init__(self, k: int = 5):
        self.k = k

    def scores(self, feats):
        s = chart.swings(feats, self.k)
        st = chart.structure(feats, self.k)
        enter = (st > 0) & (feats["close"] > s.last_sh)          # uptrend + new high confirms it
        exit_ = (feats["close"] < s.last_sl) | (st < 0)          # support broke or trend flipped
        return chart.latch(enter, exit_)


class SRBreakout(Signal):
    """Resistance breakout confirmed by volume, stop below the last support, trend-filtered."""
    name = "sr_volume"

    def __init__(self, k: int = 5, vol_mult: float = 1.5):
        self.k, self.vol_mult = k, vol_mult

    def scores(self, feats):
        s = chart.swings(feats, self.k)
        c = feats["close"]
        loud = feats["volume"] > self.vol_mult * feats["volume"].rolling(20).mean()
        enter = (c > s.last_sh) & (c.shift() <= s.last_sh) & loud & (c > chart.sma(c, 200))
        return chart.latch(enter, c < s.last_sl)


class PullbackRSI2(Signal):
    """Buy sharp dips inside an uptrend (RSI(2) < 10 above the 200-bar average), exit on bounce."""
    name = "pullback"

    def scores(self, feats):
        c = feats["close"]
        r2 = rsi(c, 2)
        enter = (c > chart.sma(c, 200)) & (r2 < 10)
        exit_ = (c > chart.sma(c, 5)) | (c < chart.sma(c, 200))
        return chart.latch(enter, exit_)


class Ensemble(Signal):
    name = "ensemble"

    def __init__(self, members: list[Signal], weights: list[float] | None = None):
        self.members = members
        self.weights = weights or [1.0] * len(members)

    def scores(self, feats):
        total = sum(w * m.scores(feats) for m, w in zip(self.members, self.weights))
        return (total / sum(abs(w) for w in self.weights)).clip(-1, 1)


class LLMJudge(Signal):
    """Optional AI judge via any OpenAI-compatible chat endpoint (e.g. Jev, a Claude gateway).

    Live/paper only: it scores just the latest bar. Any error or timeout -> 0 (stay flat).
    """
    name = "llm"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 3.0):
        self.base_url, self.api_key, self.model, self.timeout = base_url.rstrip("/"), api_key, model, timeout

    def latest_score(self, feats: pd.DataFrame) -> float:
        snapshot = feats[FEATURE_COLUMNS].tail(10).round(6).to_dict(orient="records")
        prompt = ("You judge short-term crypto momentum. Given the last 10 bars of features "
                  "(log returns, volatility, z-score, RSI, volume ratio), reply ONLY with JSON "
                  '{"score": <float between -1 and 1>} where positive means expect price up next bar.\n'
                  + json.dumps(snapshot))
        body = json.dumps({"model": self.model, "temperature": 0,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                text = json.loads(r.read())["choices"][0]["message"]["content"]
            score = float(json.loads(text[text.index("{"): text.rindex("}") + 1])["score"])
            return float(np.clip(score, -1, 1))
        except Exception:
            return 0.0

    def scores(self, feats):
        s = pd.Series(0.0, index=feats.index)
        s.iloc[-1] = self.latest_score(feats)
        return s


def default_signals() -> dict[str, Signal]:
    sigs = {
        "coinflip": CoinFlip(),
        "momentum": Momentum(),
        "meanrev": MeanReversion(),
        "model": LocalModel(),
        "ensemble": Ensemble([Momentum(), MeanReversion(), LocalModel()]),
        # slow, chart-reading signals: few trades, so fees matter far less
        "trend": Trend(),
        "breakout": Donchian(),
        "structure": Structure(),
        "sr_volume": SRBreakout(),
        "pullback": PullbackRSI2(),
        "chart_vote": Ensemble([Trend(), Donchian(), Structure()]),
    }
    for key, sig in sigs.items():
        sig.name = key  # results are reported under the dictionary key
    return sigs
