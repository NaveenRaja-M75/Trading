"""Can a trained model predict the next week well enough to beat the live rule?

Model: gradient-boosted trees (scikit-learn HistGradientBoostingClassifier), the workhorse of tabular
prediction contests. Target: is the price higher 7 days from now? 16 causal features: distance from
20/50/100/200-day averages, 7/30/90/180-day returns, 30/90-day volatility and their ratio, drawdown
from the 1-year high, Hurst exponent, Kalman trend slope, RSI(14), 50/100/200 vote.
Walk-forward: retrained every January on all earlier data, with a 7-day gap so no training label
overlaps the test year. Trading uses the predicted probability:
  ml_switch      hold 100% when P(up) > 0.55, else cash
  ml_scaled      hold min(1, max(0, (P - 0.5) x 5))
  ml_x_voltgt    ml_scaled x the 60% volatility target
  rule_ml_filter the live rule, but only while P(up) > 0.5
Scored on BTC (Yahoo, 2014+, test years 2018+) and ETH (Binance, test years 2020+), each trained on itself.
"""
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from india_test import perf, yahoo
from jevbot.data import load_or_fetch
from jevbot.features import rsi
from physics_test import hurst, kalman_slope
from search_loop import CURRENT, STEP, make, simulate, voltgt

H = 7


def features(c: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=c.index)
    for n in (20, 50, 100, 200):
        f[f"dist_{n}"] = c / c.rolling(n).mean() - 1
    for n in (7, 30, 90, 180):
        f[f"ret_{n}"] = c / c.shift(n) - 1
    lr = np.log(c).diff()
    f["vol_30"], f["vol_90"] = lr.rolling(30).std(), lr.rolling(90).std()
    f["vol_ratio"] = f.vol_30 / f.vol_90
    f["drawdown_1y"] = c / c.rolling(365, min_periods=60).max() - 1
    f["hurst"] = hurst(c, 128)
    f["kalman"] = kalman_slope(c, 1e-3)
    f["rsi"] = rsi(c, 14)
    f["vote"] = sum((c > c.rolling(n).mean()).astype(float) for n in (50, 100, 200)) / 3
    return f


def walk_forward_proba(c: pd.Series, first_test: str) -> tuple[pd.Series, list]:
    X = features(c)
    y = (c.shift(-H) / c - 1 > 0).astype(float).where(c.shift(-H).notna())
    proba = pd.Series(np.nan, index=c.index)
    report = []
    years = pd.date_range(pd.Timestamp(first_test, tz="UTC"), c.index[-1], freq="YS")
    for i, start in enumerate(years):
        end = years[i + 1] if i + 1 < len(years) else c.index[-1] + pd.Timedelta(days=1)
        train = (X.index < start - pd.Timedelta(days=H)) & X.notna().all(axis=1) & y.notna()
        test = (X.index >= start) & (X.index < end) & X.notna().all(axis=1)
        model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                               l2_regularization=1.0, random_state=0)
        model.fit(X[train], y[train])
        p = model.predict_proba(X[test])[:, 1]
        proba[test] = p
        yt = y[test].dropna()
        pt = proba[yt.index]
        report.append({"year": start.year, "train_days": int(train.sum()),
                       "accuracy_pct": round(float(((pt > 0.5) == (yt == 1)).mean()) * 100, 1),
                       "always_up_pct": round(float(yt.mean()) * 100, 1),
                       "auc": round(float(roc_auc_score(yt, pt)), 3) if yt.nunique() > 1 else np.nan})
    return proba, report


def main():
    btc = yahoo("BTC-USD", max_move=0.45)
    eth = load_or_fetch("ETHUSDT", "1d", 3300).close
    eth.index = eth.index.normalize()
    for title, c, first in (("BTC", btc, "2018-01-01"), ("ETH", eth, "2020-01-01")):
        p, rep = walk_forward_proba(c, first)
        q = lambda w: (np.floor(w.fillna(0) / STEP) * STEP).clip(0, 1)
        live = make(c, CURRENT)
        cands = {
            "live rule (vote x vol target)": live,
            "ml_switch": q((p > 0.55).astype(float)),
            "ml_scaled": q(((p - 0.5) * 5).clip(0, 1)),
            "ml_x_voltgt": q(((p - 0.5) * 5).clip(0, 1) * voltgt(c, 0.60, 60)),
            "rule_ml_filter": q(live * (p > 0.5)),
        }
        m = c.index >= pd.Timestamp(first, tz="UTC")
        rows = [perf(k, simulate(c, w)[m], 365) for k, w in cands.items()]
        rows.append(perf("hold", c.pct_change().fillna(0)[m], 365))
        with pd.option_context("display.width", 220, "display.max_columns", 20):
            print(f"\n######## {title}: model retrained each January, test years {first[:4]} -> now")
            print("PREDICTION QUALITY (accuracy of 'higher in 7 days'; AUC 0.5 = coin flip, 1.0 = perfect)")
            print(pd.DataFrame(rep).set_index("year").to_string())
            yt = (c.shift(-H) / c - 1 > 0)[m & p.notna() & c.shift(-H).notna()]
            print(f"  all test years: accuracy {((p[yt.index] > .5) == yt).mean() * 100:.1f}% vs always-up "
                  f"{yt.mean() * 100:.1f}%, AUC {roc_auc_score(yt, p[yt.index]):.3f}")
            print("\nTRADING ON IT (after costs)")
            print(pd.DataFrame(rows).set_index("name").to_string())


if __name__ == "__main__":
    main()
