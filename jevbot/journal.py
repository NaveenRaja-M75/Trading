"""Predict -> check -> learn.

Every bar, each "expert" (a signal) votes up/down on the price `horizon` bars ahead. The
learner combines the votes using weights it keeps separately for each market *situation*
(trend structure x volatility x above/below the 200-bar average). When a prediction's
outcome becomes known (horizon bars later) the experts that were wrong in that situation
lose weight, so the learner learns which experts to trust in which kind of market.

Strictly causal: at bar t the learner only uses outcomes of predictions made at t-horizon or
earlier, which are fully known by t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import chart
from .features import build_features
from .signals import Signal, default_signals

EXPERTS = ["trend", "breakout", "structure", "sr_volume", "momentum", "meanrev", "pullback"]


def situation(df: pd.DataFrame) -> pd.Series:
    """Label each bar with a market situation, e.g. 'up|calm|above'."""
    c = df["close"]
    st = chart.structure(df).map({1: "up", -1: "down", 0: "range"})
    atr_pct = chart.atr(df) / c
    vol = np.where(atr_pct > atr_pct.rolling(500, min_periods=50).median(), "wild", "calm")
    ma = np.where(c > chart.sma(c, 200), "above", "below")
    return st.astype(str) + "|" + pd.Series(vol, index=df.index) + "|" + pd.Series(ma, index=df.index)


def expert_votes(df: pd.DataFrame) -> pd.DataFrame:
    feats = build_features(df)
    sigs = default_signals()
    votes = pd.DataFrame({name: np.sign(sigs[name].scores(feats)) for name in EXPERTS}, index=df.index)
    votes["always_up"] = 1.0    # lets the learner fall back on "markets drift up" if nothing beats it
    votes["always_down"] = -1.0
    return votes.fillna(0.0)


def replay(df: pd.DataFrame, horizon: int = 4, eta: float = 0.15, share: float = 0.02,
           prior: float = 20.0, decay: float = 0.998) -> pd.DataFrame:
    """Run the predict/check/learn loop over history. Returns one ledger row per bar."""
    votes = expert_votes(df)
    names = list(votes.columns)
    V = votes.to_numpy()
    ctx = situation(df).to_numpy()
    close = df["close"].to_numpy()
    n, k = V.shape
    weights: dict[str, np.ndarray] = {}
    record: dict[str, list] = {}           # own track record per situation: [correct, total]
    direction = np.zeros(n)                # +1 / -1 call made at each bar (0 = no call)
    p_up = np.full(n, np.nan)
    for t in range(n):
        # 1) check: the prediction made at t-horizon has now resolved -> punish the wrong experts
        s = t - horizon
        if s >= 0:
            went_up = close[t] > close[s]
            w = weights.setdefault(ctx[s], np.ones(k))
            wrong = (V[s] != 0) & ((V[s] > 0) != went_up)
            w *= np.exp(-eta * wrong)
            w[:] = (1 - share) * w + share * w.mean()  # never fully forget an expert: markets change
            w /= w.sum()
            if direction[s] != 0:
                rec = record.setdefault(ctx[s], [0.0, 0.0])
                rec[0] = rec[0] * decay + ((direction[s] > 0) == went_up)
                rec[1] = rec[1] * decay + 1
        # 2) predict: direction from the weighted vote, confidence from our own track record here
        w = weights.setdefault(ctx[t], np.ones(k))
        active = V[t] != 0
        if active.any():
            frac_up = (w * (V[t] > 0) * active).sum() / (w * active).sum()
            if frac_up != 0.5:
                direction[t] = 1.0 if frac_up > 0.5 else -1.0
                correct, total = record.get(ctx[t], [0.0, 0.0])
                acc = (correct + prior) / (total + 2 * prior)  # starts at 50%: no evidence, no confidence
                p_up[t] = acc if direction[t] > 0 else 1 - acc
            else:
                p_up[t] = 0.5
    fwd = pd.Series(close, index=df.index).shift(-horizon) / df["close"] - 1
    led = pd.DataFrame({"situation": ctx, "p_up": p_up, "fwd_ret": fwd}, index=df.index)
    led["went_up"] = (led.fwd_ret > 0).where(led.fwd_ret.notna())
    for name in names:
        led[f"v_{name}"] = votes[name]
    return led


def scorecard(led: pd.DataFrame, confident: float = 0.03) -> dict:
    """How good were the forecasts? Compared against the dumb base rate."""
    d = led.dropna(subset=["p_up", "went_up"])
    up = d.went_up.astype(bool)
    pred_up = d.p_up > 0.5
    conf = (d.p_up - 0.5).abs() >= confident
    base = up.mean()
    brier = ((d.p_up - up) ** 2).mean()
    brier_base = ((base - up) ** 2).mean()
    out = {
        "forecasts": len(d),
        "base_rate_up_pct": round(base * 100, 1),
        "hit_rate_pct": round((pred_up == up).mean() * 100, 2),
        "best_dumb_rule_pct": round(max(base, 1 - base) * 100, 2),
        "confident_share_pct": round(conf.mean() * 100, 1),
        "confident_hit_rate_pct": round((pred_up[conf] == up[conf]).mean() * 100, 2) if conf.any() else np.nan,
        "brier_skill_pct": round((1 - brier / brier_base) * 100, 2),  # >0 means better than base rate
    }
    for c in [c for c in d.columns if c.startswith("v_")]:
        v = d[c]
        m = v != 0
        out[f"expert_{c[2:]}_hit_pct"] = round(((v[m] > 0) == up[m]).mean() * 100, 2) if m.any() else np.nan
    return out


def by_situation(led: pd.DataFrame, confident: float = 0.03) -> pd.DataFrame:
    """Where does it get things right / wrong? The 'what went wrong' report."""
    d = led.dropna(subset=["p_up", "went_up"]).copy()
    d["correct"] = (d.p_up > 0.5) == d.went_up.astype(bool)
    d["confident"] = (d.p_up - 0.5).abs() >= confident
    g = d.groupby("situation")
    return pd.DataFrame({
        "bars": g.size(),
        "up_pct": (g.went_up.mean() * 100).round(1),
        "hit_pct": (g.correct.mean() * 100).round(1),
        "confident_hit_pct": d[d.confident].groupby("situation").correct.mean().mul(100).round(1),
        "avg_move_pct": (g.fwd_ret.apply(lambda x: x.abs().mean()) * 100).round(2),
    }).sort_values("bars", ascending=False)


class Learner(Signal):
    """Trade the learner: in only when it is confident the price goes up."""
    name = "learner"

    def __init__(self, horizon: int = 4, edge: float = 0.03):
        self.horizon, self.edge = horizon, edge

    def scores(self, feats):
        led = replay(feats[["open", "high", "low", "close", "volume"]], self.horizon)
        return pd.Series(np.where(led.p_up > 0.5 + self.edge, 1.0, -1.0), index=feats.index)


JOURNAL_COLUMNS = ["time", "symbol", "price", "p_up", "call", "situation", "chart_read",
                   "due", "outcome_price", "went_up", "correct"]


def live_update(symbol: str, df: pd.DataFrame, path, horizon: int = 4) -> pd.DataFrame:
    """Write down this bar's prediction, then grade every earlier prediction that has come due."""
    from pathlib import Path
    path = Path(path)
    j = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=JOURNAL_COLUMNS)
    j = j.astype({"went_up": object, "correct": object, "outcome_price": float})
    led = replay(df, horizon)
    t = df.index[-1]
    step = df.index[-1] - df.index[-2]
    stamp = t.isoformat()
    if not ((j["symbol"] == symbol) & (j["time"] == stamp)).any():
        p = float(led.p_up.iloc[-1]) if pd.notna(led.p_up.iloc[-1]) else 0.5
        j.loc[len(j)] = {
            "time": stamp, "symbol": symbol, "price": float(df.close.iloc[-1]), "p_up": round(p, 4),
            "call": "up" if p > 0.5 else ("down" if p < 0.5 else "none"),
            "situation": led.situation.iloc[-1], "chart_read": chart.describe(df),
            "due": (t + horizon * step).isoformat(), "outcome_price": np.nan, "went_up": np.nan, "correct": np.nan,
        }
    closes = df["close"]
    closes.index = [i.isoformat() for i in closes.index]
    for i in j.index[(j.symbol == symbol) & j.outcome_price.isna()]:
        due = j.at[i, "due"]
        if due in closes.index:
            out = float(closes[due])
            up = out > float(j.at[i, "price"])
            j.at[i, "outcome_price"] = out
            j.at[i, "went_up"] = up
            j.at[i, "correct"] = (j.at[i, "call"] == "up") == up if j.at[i, "call"] != "none" else np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    j.to_csv(path, index=False)
    return j


def journal_report(j: pd.DataFrame, last_mistakes: int = 5) -> str:
    g = j.dropna(subset=["correct"]).copy()
    lines = ["## Prediction journal (guess, check, learn)", ""]
    pending = int(j.outcome_price.isna().sum())
    if g.empty:
        return "\n".join(lines + [f"{len(j)} predictions written, {pending} waiting for their result."])
    g["correct"] = g.correct.astype(str).str.lower().isin(["true", "1", "1.0"])
    conf = (g.p_up - 0.5).abs() >= 0.03
    lines.append(f"{len(g)} graded, {pending} waiting. Correct: **{g.correct.mean() * 100:.1f}%**"
                 + (f"; when confident: **{g.correct[conf].mean() * 100:.1f}%** of {int(conf.sum())}" if conf.any() else ""))
    sit = g.groupby("situation").correct.agg(["size", "mean"]).sort_values("size", ascending=False)
    lines += ["", "| situation | graded | correct |", "|---|---|---|"]
    lines += [f"| {k} | {int(r['size'])} | {r['mean'] * 100:.0f}% |" for k, r in sit.iterrows()]
    wrong = g[~g.correct].tail(last_mistakes)
    if len(wrong):
        lines += ["", "Latest mistakes:"]
        for _, r in wrong.iterrows():
            move = (r.outcome_price / r.price - 1) * 100
            lines.append(f"- {r.time[:16]} {r.symbol}: called **{r.call}** ({r.p_up:.0%} up) in `{r.situation}`, "
                         f"price moved {move:+.2f}%. Chart then: {r.chart_read}")
    return "\n".join(lines)
