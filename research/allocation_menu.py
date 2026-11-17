"""How much of the plan should be the crypto bot? The risk/return menu.

The earlier search (better_than_200.py) only tried crypto shares of 10/20/30% and picked 30%, the
top of its grid. This prints the whole menu: crypto share 20% -> 100% (BTC + ETH on vote_voltgt),
the rest split equally between Nifty 50 and gold, in rupees, same realistic simulation (drifting
holdings, 0.15% per trade).
For each mix: return, worst drop, Sharpe (whole period and the locked last 2 years), an approximate
after-tax return (31.2% of each year's crypto gain; real tax is a little higher because losing
trades cannot offset winning ones), and a block bootstrap of the next 5 years.
More crypto = more return AND bigger drops. This is a choice about risk, not a new edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from better_than_200 import VAULT_DAYS, above, panel, simulate, vol
from india_test import perf

SHARES = (0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00)
TAX = 0.312


def weights(px: pd.DataFrame, c: float, core: bool = True) -> pd.DataFrame:
    w = pd.DataFrame({"NIFTY": (0.5 - c / 2) if core else 0.0, "GOLD": (0.5 - c / 2) if core else 0.0}, index=px.index)
    for a in ("BTC", "ETH"):
        vote = (above(px[a], 50) + above(px[a], 100) + above(px[a], 200)) / 3
        scale = (0.60 / vol(px[a], 60)).clip(upper=1.0).fillna(0)
        w[a] = c / 2 * (vote * scale).round(2)
    return w


def after_tax_cagr(net: pd.Series, crypto_only: pd.Series) -> float:
    eq = 1.0
    for y, r in net.groupby(net.index.year):
        gain = float(np.prod(1 + crypto_only[r.index]) - 1)
        eq *= float(np.prod(1 + r)) - TAX * max(0.0, gain)
    years = (net.index[-1] - net.index[0]).days / 365.25
    return round((eq ** (1 / years) - 1) * 100, 1)


def bootstrap(net: pd.Series, years: int = 5, n: int = 3000, block: int = 30, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x, T = net.to_numpy(), years * 365
    finals, dds, lose1 = [], [], 0
    for _ in range(n):
        starts = rng.integers(0, len(x) - block, T // block + 1)
        path = np.cumprod(1 + x[(starts[:, None] + np.arange(block)).ravel()[:T]])
        finals.append(path[-1])
        dds.append((path / np.maximum.accumulate(path) - 1).min())
        lose1 += path[364] < 1
    f = np.array(finals)
    return {"P(down after 1y)": f"{lose1 / n * 100:.0f}%", "5y: Rs100 -> typical": round(float(np.median(f)) * 100),
            "5y: bad case (1 in 20)": round(float(np.quantile(f, 0.05)) * 100),
            "5y: typical worst drop": f"{np.median(dds) * 100:.0f}%", "5y: bad worst drop": f"{np.quantile(dds, 0.05) * 100:.0f}%"}


def main() -> None:
    px = panel()
    cut = px.index[-1] - pd.Timedelta(days=VAULT_DAYS)
    start = px.index[220]
    rows, boots = [], []
    for c in SHARES:
        net = simulate(px, weights(px, c))[start:]
        crypto = simulate(px, weights(px, c, core=False))[start:]
        p, lk = perf(f"{int(c * 100)}% crypto bot", net, 365), perf("", net[cut:], 365)
        roll = (1 + net).rolling(365).apply(np.prod, raw=True) - 1
        rows.append({"mix": p["name"], "per_year_%": p["cagr_pct"], "after_tax_%": after_tax_cagr(net, crypto),
                     "worst_drop_%": p["max_dd_pct"], "sharpe": p["sharpe"], "worst_12_months_%": round(float(roll.min()) * 100, 1),
                     "losing_years": p["losing_years"], "locked_2y_per_year_%": lk["cagr_pct"],
                     "locked_2y_drop_%": lk["max_dd_pct"], "locked_2y_sharpe": lk["sharpe"]})
        boots.append({"mix": p["name"], **bootstrap(net)})
    hold = simulate(px, pd.DataFrame({"NIFTY": .5, "GOLD": .5, "BTC": 0.0, "ETH": 0.0}, index=px.index))[start:]
    p = perf("0% (Nifty + gold only)", hold, 365)
    rows.insert(0, {"mix": p["name"], "per_year_%": p["cagr_pct"], "after_tax_%": p["cagr_pct"], "worst_drop_%": p["max_dd_pct"],
                    "sharpe": p["sharpe"], "losing_years": p["losing_years"]})
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(f"Rupee portfolios {start:%Y-%m-%d} -> {px.index[-1]:%Y-%m-%d}; locked years from {cut:%Y-%m-%d}\n")
        print(pd.DataFrame(rows).set_index("mix").to_string())
        print("\nNext 5 years, 3,000 reshuffles of the past in 30-day blocks (before tax):")
        print(pd.DataFrame(boots).set_index("mix").to_string())
    print("\nThe past here includes a strong crypto and gold run; plan with less. The bootstrap can only "
          "reshuffle what already happened, so it cannot show a future that is worse than the past.")


if __name__ == "__main__":
    main()
