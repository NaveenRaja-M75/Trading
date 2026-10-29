"""Schedule VDA helper: every sale the bot made, with cost, gain and Indian crypto tax.

India (Section 115BBH, as commonly applied; confirm with a CA): 30% tax + 4% cess on each gain,
no deduction other than the cost of the coins, and a loss on one sale cannot reduce tax on another.
Exchanges may also deduct 1% TDS on sales, which you can claim back in your return.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

TAX = 0.30 * 1.04


def rows_from_book(path: Path, usd_inr: float) -> list[dict]:
    book = json.loads(path.read_text())
    symbol = path.stem.split("_")[-1]
    out = []
    for f in book.get("fills", []):
        if f.get("side") != "SELL":
            continue
        sale, gain = float(f["usdt"]), float(f.get("pnl", 0.0))
        cost = sale - gain
        tax = max(gain, 0.0) * TAX
        out.append({"date": time.strftime("%Y-%m-%d", time.gmtime(f["t"])), "asset": symbol, "qty": round(f["qty"], 8),
                    "sale_usdt": round(sale, 2), "cost_usdt": round(cost, 2), "gain_usdt": round(gain, 2),
                    "sale_inr": round(sale * usd_inr), "cost_inr": round(cost * usd_inr), "gain_inr": round(gain * usd_inr),
                    "tax_inr_30pct_plus_cess": round(tax * usd_inr)})
    return out


def write(paths: list[Path], usd_inr: float, out: Path) -> tuple[int, float]:
    rows = sorted((r for p in paths if p.exists() for r in rows_from_book(p, usd_inr)), key=lambda r: r["date"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        cols = ["date", "asset", "qty", "sale_usdt", "cost_usdt", "gain_usdt", "sale_inr", "cost_inr", "gain_inr",
                "tax_inr_30pct_plus_cess"]
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    return len(rows), float(sum(r["tax_inr_30pct_plus_cess"] for r in rows))
