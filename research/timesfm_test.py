"""Does Google's TimesFM foundation model forecast prices well enough to trade?

TimesFM (github.com/google-research/timesfm): a decoder-only model pretrained on ~100 billion real
time-points, used zero-shot here (no tuning, no fitting to these markets). At each test point it
sees only the past 512 closes and forecasts ahead; the forecast is scored against what happened.
  daily:   BTC, gold (rupees), Nifty 50; a forecast every 5 trading days since 2019, 5 days ahead
  hourly:  BTC; a forecast every 6 hours over the last year, 1 and 4 hours ahead (the intraday case)
Trading test: long for the horizon when the forecast is up, otherwise cash; 0.12% (crypto) or
0.15% (ETF) per side. Compared with always-long and with the live trend rule on the same windows.
"""
import importlib.metadata
import time

import numpy as np
import pandas as pd

from india_test import yahoo
from jevbot.data import load_or_fetch

CONTEXT = 512


def load_model(max_h: int):
    import timesfm
    print("timesfm", importlib.metadata.version("timesfm"), flush=True)
    if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        m.compile(timesfm.ForecastConfig(max_context=CONTEXT, max_horizon=max(128, max_h), normalize_inputs=True,
                                         use_continuous_quantile_head=True, fix_quantile_crossing=True))

        def fc(inputs, h):
            point, _ = m.forecast(horizon=h, inputs=inputs)
            return np.asarray(point)[:, :h]
        print("using TimesFM 2.5 (200M)", flush=True)
        return fc
    m = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(backend="cpu", per_core_batch_size=32, horizon_len=128,
                                       context_len=CONTEXT, num_layers=50, use_positional_embedding=False),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))

    def fc(inputs, h):
        point, _ = m.forecast(inputs, freq=[0] * len(inputs))
        return np.asarray(point)[:, :h]
    print("using TimesFM 2.0 (500M)", flush=True)
    return fc


def forecasts(fc, s: pd.Series, points, h):
    """Predicted % move at horizon h for each test point (uses only data up to that point)."""
    arr = s.to_numpy(dtype=float)
    out = {}
    for i in range(0, len(points), 64):
        batch = points[i:i + 64]
        inputs = [arr[p - CONTEXT + 1:p + 1] for p in batch]
        pred = fc(inputs, h)
        for p, row in zip(batch, pred):
            out[p] = row[h - 1] / arr[p] - 1
    return pd.Series(out)


def score(name, s, preds, h, cost, trend_ok):
    arr = s.to_numpy(dtype=float)
    pts = preds.index.to_numpy()
    actual = arr[pts + h] / arr[pts] - 1
    p = preds.to_numpy()
    up = actual > 0
    conf = np.abs(p) >= np.quantile(np.abs(p), 0.8)
    strat = np.where(p > 0, actual - 2 * cost, 0.0)
    hold = actual
    trend = np.where(trend_ok[pts], actual - 2 * cost, 0.0)
    comp = lambda r: (np.prod(1 + r) ** (1 / max(len(r) * h / (365 if "hourly" not in name else 8760), 1e-9)) - 1) * 100
    print(f"\n######## {name}: {len(p)} forecasts, {s.index[pts[0]]:%Y-%m-%d} -> {s.index[pts[-1]]:%Y-%m-%d}")
    print(f"  direction right: {((p > 0) == up).mean() * 100:.1f}%   always-up: {up.mean() * 100:.1f}%   "
          f"top-20% most confident: {((p[conf] > 0) == up[conf]).mean() * 100:.1f}% of {conf.sum()}")
    print(f"  correlation of forecast with what happened: {np.corrcoef(p, actual)[0, 1]:+.3f}  (0 = no skill)")
    print(f"  trade on it (long when forecast up): avg {strat.mean() * 100:+.3f}% per window, "
          f"won {(strat[p > 0] > 0).mean() * 100:.0f}% of {int((p > 0).sum())} trades")
    print(f"  always long same windows:            avg {hold.mean() * 100:+.3f}% per window")
    print(f"  live trend rule same windows:        avg {trend.mean() * 100:+.3f}% per window")


def main():
    t0 = time.time()
    fc = load_model(5)
    fx, g = yahoo("INR=X"), yahoo("GC=F")
    daily = {
        "BTC daily": (yahoo("BTC-USD", max_move=0.45), 0.0012),
        "Gold (rupees) daily": ((g * fx.reindex(fx.index.union(g.index)).ffill().reindex(g.index)).dropna(), 0.0015),
        "Nifty 50 daily": (yahoo("^NSEI"), 0.0015),
    }
    for name, (s, cost) in daily.items():
        s = s.dropna()
        start = max(CONTEXT, int(np.searchsorted(s.index, pd.Timestamp("2019-01-01", tz="UTC"))))
        pts = np.arange(start, len(s) - 5, 5)
        trend_ok = (sum((s > s.rolling(n).mean()).astype(float) for n in (50, 100, 200)) / 3 >= 2 / 3).to_numpy()
        preds = forecasts(fc, s, pts, 5)
        score(name, s, preds, 5, cost, trend_ok)
        now = fc([s.to_numpy(dtype=float)[-CONTEXT:]], 20)[0]
        print(f"  TimesFM's forecast from the last close ({s.index[-1]:%Y-%m-%d}): 5 days {now[4] / s.iloc[-1] * 100 - 100:+.1f}%, "
              f"20 days {now[19] / s.iloc[-1] * 100 - 100:+.1f}%  (judge it by the accuracy above)", flush=True)
    h = load_or_fetch("BTCUSDT", "1h", 400).close
    trend_h = (sum((h > h.rolling(n).mean()).astype(float) for n in (50, 100, 200)) / 3 >= 2 / 3).to_numpy()
    pts = np.arange(len(h) - 24 * 365, len(h) - 4, 6)
    for hz in (1, 4):
        preds = forecasts(fc, h, pts, hz)
        score(f"BTC hourly, {hz}h ahead (intraday)", h, preds, hz, 0.0012, trend_h)
    print(f"\nfinished in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
