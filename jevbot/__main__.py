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


def chart_vote_variants() -> dict:
    """chart_vote with every lookback scaled up/down: a real edge survives nearby settings."""
    from .signals import Donchian, Ensemble, Structure, Trend
    out = {}
    for f in (0.6, 0.8, 1.0, 1.25, 1.5):
        sig = Ensemble([Trend(round(50 * f), round(200 * f)), Donchian(round(55 * f), round(20 * f)),
                        Structure(max(2, round(5 * f)))])
        sig.name = f"chart_vote_x{f}"
        out[sig.name] = sig
    return out


def cmd_robust(s: Settings, a) -> None:
    from .data import load_or_fetch
    dfs = {}
    for sym in [x.strip().upper() for x in a.symbols.split(",") if x.strip()]:
        try:
            dfs[sym] = load_or_fetch(sym, s.interval, a.days)
            print(f"{sym}: {len(dfs[sym])} bars from {dfs[sym].index[0]:%Y-%m-%d}")
        except Exception as e:
            print(f"{sym}: skipped ({e})")
    variants = chart_vote_variants()
    if a.with_baselines:
        base = default_signals()
        variants |= {k: base[k] for k in ("trend", "breakout", "structure", "sr_volume", "coinflip")}
    r = backtest.robustness(dfs, variants, _policy(a), s.cost_per_side, s.interval, a.windows)
    Path("reports").mkdir(exist_ok=True)
    r.to_csv(f"reports/robust_{s.interval}.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 1000):
        print("\nPer coin x period (chart_vote_x1.0):")
        print(r[r.variant == "chart_vote_x1.0"].pivot(index="symbol", columns="start", values="ret_pct").to_string())
        print("\nHolding, same cells:")
        print(r[r.variant == "chart_vote_x1.0"].pivot(index="symbol", columns="start", values="hold_pct").to_string())
        print("\nSummary over all coin x period cells:")
        print(backtest.robustness_summary(r).to_string())


def cmd_learn(s: Settings, a) -> None:
    """Replay predict -> check -> learn over history on many coins, then trade what it learned."""
    from . import journal
    from .data import load_or_fetch
    dfs = {}
    for sym in [x.strip().upper() for x in a.symbols.split(",") if x.strip()]:
        try:
            dfs[sym] = load_or_fetch(sym, s.interval, a.days)
        except Exception as e:
            print(f"{sym}: skipped ({e})")
    Path("reports").mkdir(exist_ok=True)
    for h in [int(x) for x in a.horizons.split(",")]:
        cards, ledgers = [], []
        for sym, df in dfs.items():
            led = journal.replay(df, horizon=h)
            led = led.iloc[500:]  # let the learner see some history before it is graded
            cards.append({"symbol": sym, **journal.scorecard(led)})
            ledgers.append(led)
        cards = pd.DataFrame(cards).set_index("symbol")
        pooled = pd.concat(ledgers)
        with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200):
            print(f"\n=============== horizon {h} bars ({s.interval}) ===============")
            print("Forecast scorecard per coin:")
            print(cards[["forecasts", "base_rate_up_pct", "hit_rate_pct", "best_dumb_rule_pct",
                         "confident_share_pct", "confident_hit_rate_pct", "brier_skill_pct"]].to_string())
            print("\nAll coins pooled:")
            for k, v in journal.scorecard(pooled).items():
                print(f"  {k:32s} {v}")
            print("\nWhere it is right / wrong (pooled, by market situation):")
            print(journal.by_situation(pooled).to_string())
        variants = {f"learner_h{h}": journal.Learner(h)}
        variants[f"learner_h{h}"].name = f"learner_h{h}"
        variants["chart_vote"] = default_signals()["chart_vote"]
        r = backtest.robustness(dfs, variants, _policy(a), s.cost_per_side, s.interval, a.windows)
        r.to_csv(f"reports/learn_h{h}_{s.interval}.csv", index=False)
        print("\nTrading what it learned (10 coins x periods, after costs):")
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(backtest.robustness_summary(r).to_string())
            print(r[r.variant == f"learner_h{h}"].pivot(index="symbol", columns="start", values="ret_pct").to_string())


def cmd_weekly(s: Settings, a) -> None:
    """Weekly momentum / rotation / trend-filter strategies with a locked final-test year."""
    from . import portfolio
    from .data import load_or_fetch
    dfs = {}
    for sym in [x.strip().upper() for x in a.symbols.split(",") if x.strip()]:
        try:
            dfs[sym] = load_or_fetch(sym, "1d", a.days)
        except Exception as e:
            print(f"{sym}: skipped ({e})")
    px = portfolio.closes(dfs)
    r = portfolio.evaluate(px, s.cost_per_side, a.vault_days)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 100):
        print(f"{len(dfs)} coins, daily, {px.index[0]:%Y-%m-%d} -> {px.index[-1]:%Y-%m-%d}; "
              f"weekly rebalance; cost {s.cost_per_side * 1e4:.0f} bps per side")
        print(f"DEV period: up to {r['cut']:%Y-%m-%d}. The final {a.vault_days} days are LOCKED until the pick is made.\n")
        print(r["dev"].sort_values("sharpe", ascending=False).to_string())
        print("\nFamilies (worst and median Sharpe across nearby settings):")
        print(r["family_stats"].sort_values("min", ascending=False).to_string())
        print(f"\nPRE-COMMITTED PICK (rule: best worst-neighbour Sharpe, median member): {r['pick']}")
        print(f"\n=== LOCKED YEAR, opened once: {r['cut']:%Y-%m-%d} -> {px.index[-1]:%Y-%m-%d} ===")
        print(r["vault"].to_string())
        print("\n(for transparency: every candidate on the locked year; NOT used for picking)")
        print(r["all_vault"].sort_values("sharpe", ascending=False).to_string())


def cmd_weekly_paper(s: Settings, a) -> None:
    from . import weekly_paper
    path = Path(f"state/weekly_{a.strategy}.json")
    syms = [x.strip().upper() for x in a.symbols.split(",") if x.strip()]
    print(weekly_paper.run_once(syms, a.strategy, path, a.start_usdt, s.fee_bps, s.slippage_bps))
    rep = weekly_paper.report(path, a.strategy)
    Path("state/WEEKLY.md").write_text(rep)
    print(rep)


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
    books, journals = {}, []
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
            from . import journal
            from .data import fetch_recent
            df = fetch_recent(sym, s.interval, 1000)
            price = float(df["close"].iloc[-1])
            journals.append(journal.live_update(sym, df, Path("state/journal.csv"), a.horizon))
        except Exception as e:
            log(f"{sym} journal error: {e!r}")
        books[sym] = (broker.book, price)
    report = status_report(books, s.paper_start_usdt)
    if journals:
        from .journal import journal_report
        report += "\n\n" + journal_report(journals[-1])
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
    rb = sub.add_parser("robust", help="stress-test chart_vote across coins, periods, settings")
    rb.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT,LTCUSDT")
    rb.add_argument("--days", type=float, default=1095)
    rb.add_argument("--windows", type=int, default=6)
    rb.add_argument("--with-baselines", action="store_true")
    common(rb)
    ln = sub.add_parser("learn", help="replay predict/check/learn over history")
    ln.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT,LTCUSDT")
    ln.add_argument("--days", type=float, default=1095)
    ln.add_argument("--horizons", default="4,24")
    ln.add_argument("--windows", type=int, default=6)
    common(ln)
    wk = sub.add_parser("weekly", help="weekly momentum/rotation with a locked final-test year")
    wk.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT,LTCUSDT")
    wk.add_argument("--days", type=float, default=2555)
    wk.add_argument("--vault-days", type=int, default=365)
    wp = sub.add_parser("weekly-paper", help="daily mark-to-market, weekly rebalance of the paper portfolio")
    wp.add_argument("--strategy", default="ts_momentum_14d")
    wp.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT,LTCUSDT")
    wp.add_argument("--start-usdt", type=float, default=1000.0)
    po = sub.add_parser("paper-once", help="one paper decision per coin (for schedulers)")
    po.add_argument("--signal", default="chart_vote", choices=list(default_signals()))
    po.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    po.add_argument("--horizon", type=int, default=4, help="bars ahead the journal predicts")
    common(po)
    sub.choices["live"].add_argument("--i-understand-i-can-lose-money", action="store_true")

    a = p.parse_args(argv)
    s = Settings.from_env()
    {"fetch": cmd_fetch, "backtest": cmd_backtest, "paper": cmd_paper,
     "paper-once": cmd_paper_once, "robust": cmd_robust, "learn": cmd_learn, "weekly": cmd_weekly, "weekly-paper": cmd_weekly_paper, "live": cmd_live}[a.cmd](s, a)


if __name__ == "__main__":
    main()
