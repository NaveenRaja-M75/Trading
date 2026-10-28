"""Signals ("judges"). Each maps features -> score in [-1, 1] per bar, causally.

score > 0 means "want long", < 0 "want short/flat". The policy, not the signal, decides trades.
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS


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
    return {
        "coinflip": CoinFlip(),
        "momentum": Momentum(),
        "meanrev": MeanReversion(),
        "model": LocalModel(),
        "ensemble": Ensemble([Momentum(), MeanReversion(), LocalModel()]),
    }
