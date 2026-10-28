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
    symbols = [x.strip().upper() for x in (a.symbols or s.symbol).split(",") if x.strip()]
    signals, pol = default_signals(), _policy(a)
    tables, data = [], {}
    for i, sym in enumerate(symbols):
        if a.synthetic:
            df = synthetic(n=a.bars, seed=a.seed + i)
        elif a.csv:
            df = load_csv(a.csv)
        else:
            df = load_or_fetch(sym, s.interval, a.days)
        data[sym] = df
        print(f"{sym} {s.interval}: {len(df)} bars {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        tables.append(backtest.compare(df, signals, pol, s.cost_per_side, s.interval, symbol=sym))
    if a.synthetic:
        print("SYNTHETIC random walks: no real edge exists, everything should FAIL")
    table = pd.concat(tables, ignore_index=True)
    print(f"cost per side {s.cost_per_side * 1e4:.1f} bps; parameters fixed in advance, not tuned\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_rows", 500):
        print("Out-of-sample, per coin:")
        print(table[table.period == "out_of_sample"].drop(columns="period").to_string(index=False))
        print("\nOut-of-sample, averaged over coins:")
        print(backtest.summarize(table).to_string())
    print("\nVerdict:")
    for line in backtest.verdict(table):
        print("  " + line)
    Path("reports").mkdir(exist_ok=True)
    table.to_csv(f"reports/backtest_{s.interval}.csv", index=False)
    if a.plot:
        best = [n for n in backtest.summarize(table).index if n not in {"coinflip", "buy_and_hold"}][:a.plot]
        for sym, df in data.items():
            oos = df.iloc[len(df) // 2:]
            for name in best:
                path = f"reports/{s.interval}_{sym}_{name}.png"
                backtest.plot(oos, signals[name], pol, s.cost_per_side, s.interval, path,
                              f"{sym} {s.interval} out-of-sample: {name}")
                print("chart", path)


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


def cmd_paper_once(s: Settings, a) -> None:
    """One decision per coin, then exit. Meant to be run on a schedule (see .github/workflows/paper.yml)."""
    import time
    from .broker import PaperBroker
    from .data import INTERVAL_MS
    from .policy import RiskGate
    from .runner import log, status_report, step
    sig, pol = default_signals()[a.signal], _policy(a)
    books = {}
    for sym in [x.strip().upper() for x in a.symbols.split(",") if x.strip()]:
        broker = PaperBroker(s.paper_start_usdt, s.fee_bps, s.slippage_bps, Path(f"state/paper_{sym}.json"))
        gate = RiskGate(s.max_position_usdt, s.daily_loss_limit_usdt,
                        max_data_age_s=INTERVAL_MS[s.interval] / 1000)  # scheduled runs can start late
        try:
            log(f"{sym} {sig.name}: " + step(s, sig, pol, broker, gate, symbol=sym))
        except Exception as e:
            log(f"{sym} error: {e!r}")
        price = broker.book.fills[-1]["price"] if broker.book.fills else 0.0
        try:
            from .data import fetch_recent
            price = float(fetch_recent(sym, s.interval, 2)["close"].iloc[-1])
        except Exception:
            pass
        books[sym] = (broker.book, price)
    report = status_report(books, s.paper_start_usdt)
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    Path("state/STATUS.md").write_text(f"# Paper trading: {sig.name} on {s.interval}\n\nUpdated {stamp}\n\n{report}\n")
    print(report)


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
        sp.add_argument("--min-hold", type=int, default=1)

    f = sub.add_parser("fetch", help="download history from Binance")
    f.add_argument("--days", type=float, default=14)

    b = sub.add_parser("backtest", help="compare signals vs coin-flip, after costs")
    b.add_argument("--days", type=float, default=14)
    b.add_argument("--csv")
    b.add_argument("--synthetic", action="store_true")
    b.add_argument("--bars", type=int, default=20_000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--symbols", help="comma list, e.g. BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    b.add_argument("--plot", type=int, default=0, help="draw charts for the N best signals")
    common(b)

    for name in ("paper", "live"):
        sp = sub.add_parser(name)
        sp.add_argument("--signal", default="ensemble", choices=list(default_signals()))
        sp.add_argument("--llm", action="store_true", help="blend in the LLM judge from .env")
        common(sp)
    po = sub.add_parser("paper-once", help="one paper decision per coin (for schedulers)")
    po.add_argument("--signal", default="chart_vote", choices=list(default_signals()))
    po.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    common(po)
    sub.choices["live"].add_argument("--i-understand-i-can-lose-money", action="store_true")

    a = p.parse_args(argv)
    s = Settings.from_env()
    {"fetch": cmd_fetch, "backtest": cmd_backtest, "paper": cmd_paper,
     "paper-once": cmd_paper_once, "live": cmd_live}[a.cmd](s, a)


if __name__ == "__main__":
    main()
