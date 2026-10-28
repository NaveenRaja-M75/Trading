"""CLI: python -m jevbot {fetch,backtest,paper,live}"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import backtest
from .config import Settings
from .data import load_csv, load_or_fetch, synthetic
from .policy import Policy
from .signals import LLMJudge, default_signals


def _policy(a) -> Policy:
    return Policy(enter=a.enter, exit=a.exit, min_hold=a.min_hold)


def cmd_fetch(s: Settings, a) -> None:
    df = load_or_fetch(s.symbol, s.interval, a.days, refresh=True)
    print(f"saved {len(df)} candles {df.index[0]} -> {df.index[-1]}")


def cmd_backtest(s: Settings, a) -> None:
    if a.synthetic:
        df = synthetic(n=a.bars, seed=a.seed)
        print(f"SYNTHETIC random walk, {len(df)} bars (no real edge exists here)")
    elif a.csv:
        df = load_csv(a.csv)
    else:
        df = load_or_fetch(s.symbol, s.interval, a.days)
    print(f"{s.symbol} {s.interval}: {len(df)} bars, cost per side {s.cost_per_side * 1e4:.1f} bps\n")
    table = backtest.compare(df, default_signals(), _policy(a), s.cost_per_side, s.interval)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))
    print("\nVerdict (out-of-sample only):")
    for line in backtest.verdict(table):
        print("  " + line)
    Path("reports").mkdir(exist_ok=True)
    table.to_csv("reports/backtest.csv", index=False)
    print("\nsaved reports/backtest.csv")


def _llm(s: Settings, a):
    if not a.llm:
        return None
    if not (s.llm_base_url and s.llm_model):
        sys.exit("--llm needs LLM_BASE_URL and LLM_MODEL in .env")
    return LLMJudge(s.llm_base_url, s.llm_api_key, s.llm_model)


def cmd_paper(s: Settings, a) -> None:
    from .broker import PaperBroker
    from .runner import loop
    broker = PaperBroker(s.paper_start_usdt, s.fee_bps, s.slippage_bps, Path("state/paper.json"))
    loop(s, default_signals()[a.signal], _policy(a), broker, _llm(s, a))


def cmd_live(s: Settings, a) -> None:
    from .broker import BinanceBroker
    from .runner import loop
    if not s.live_trading:
        sys.exit("Refusing: set LIVE_TRADING=true in .env to enable real orders.")
    if not a.i_understand_i_can_lose_money:
        sys.exit("Refusing: pass --i-understand-i-can-lose-money.")
    where = "TESTNET" if s.binance_testnet else "REAL MONEY"
    print(f"*** {where} *** max position {s.max_position_usdt} USDT, daily loss limit {s.daily_loss_limit_usdt}")
    broker = BinanceBroker(s.symbol, s.binance_api_key, s.binance_api_secret, s.binance_testnet,
                           Path(f"state/live_{'testnet' if s.binance_testnet else 'main'}.json"))
    loop(s, default_signals()[a.signal], _policy(a), broker, _llm(s, a))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="jevbot")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--enter", type=float, default=0.5)
        sp.add_argument("--exit", type=float, default=0.0)
        sp.add_argument("--min-hold", type=int, default=3)

    f = sub.add_parser("fetch", help="download history from Binance")
    f.add_argument("--days", type=float, default=14)

    b = sub.add_parser("backtest", help="compare signals vs coin-flip, after costs")
    b.add_argument("--days", type=float, default=14)
    b.add_argument("--csv")
    b.add_argument("--synthetic", action="store_true")
    b.add_argument("--bars", type=int, default=20_000)
    b.add_argument("--seed", type=int, default=0)
    common(b)

    for name in ("paper", "live"):
        sp = sub.add_parser(name)
        sp.add_argument("--signal", default="ensemble", choices=list(default_signals()))
        sp.add_argument("--llm", action="store_true", help="blend in the LLM judge from .env")
        common(sp)
    sub.choices["live"].add_argument("--i-understand-i-can-lose-money", action="store_true")

    a = p.parse_args(argv)
    s = Settings.from_env()
    {"fetch": cmd_fetch, "backtest": cmd_backtest, "paper": cmd_paper, "live": cmd_live}[a.cmd](s, a)


if __name__ == "__main__":
    main()
