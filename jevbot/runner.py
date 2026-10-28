"""Live/paper loop: every closed bar -> features -> signal -> policy -> risk gate -> broker."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .config import Settings
from .data import INTERVAL_MS, fetch_recent
from .features import build_features
from .policy import Policy, RiskGate
from .signals import LLMJudge, Signal

LOG = Path("state") / "trades.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}Z  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def decide(df: pd.DataFrame, signal: Signal, policy: Policy, llm: LLMJudge | None = None,
           llm_weight: float = 0.5) -> tuple[float, float]:
    """Return (score, target_position) for the latest closed bar."""
    feats = build_features(df)
    scores = signal.scores(feats)
    if llm is not None:
        blended = (1 - llm_weight) * scores.iloc[-1] + llm_weight * llm.latest_score(feats)
        scores.iloc[-1] = blended
    return float(scores.iloc[-1]), float(policy.targets(scores).iloc[-1])


def step(settings: Settings, signal: Signal, policy: Policy, broker, gate: RiskGate,
         llm: LLMJudge | None = None) -> str:
    df = fetch_recent(settings.symbol, settings.interval, limit=1000)
    price = float(df["close"].iloc[-1])
    bar_close = df.index[-1] + pd.Timedelta(milliseconds=INTERVAL_MS[settings.interval])
    age = (pd.Timestamp.now(tz="UTC") - bar_close).total_seconds()

    book = broker.book
    book.roll_day()
    halt = gate.check(book.day_pnl(price), age)
    if halt:
        if gate.tripped and book.qty > 0:
            return f"HALT ({halt}) -> flatten: " + broker.sell_qty(book.qty, price)
        return f"HALT ({halt})"

    score, target = decide(df, signal, policy, llm)
    held_usdt = book.qty * price
    status = f"price {price:.2f} score {score:+.3f} target {target:.0f} held {held_usdt:.2f} USDT"
    if target > 0 and held_usdt < 0.5 * settings.max_position_usdt:
        return status + " -> " + broker.buy_usdt(settings.max_position_usdt - held_usdt, price)
    if target <= 0 and book.qty > 0:
        return status + " -> " + broker.sell_qty(book.qty, price)
    return status + " -> hold"


def loop(settings: Settings, signal: Signal, policy: Policy, broker, llm: LLMJudge | None = None) -> None:
    gate = RiskGate(settings.max_position_usdt, settings.daily_loss_limit_usdt)
    period = INTERVAL_MS[settings.interval] / 1000
    log(f"start {type(broker).__name__} {settings.symbol} {settings.interval} signal={signal.name} "
        f"max_pos={settings.max_position_usdt} daily_loss_limit={settings.daily_loss_limit_usdt}")
    while True:
        try:
            log(step(settings, signal, policy, broker, gate, llm))
        except KeyboardInterrupt:
            raise
        except Exception as e:  # network blips etc.: log and keep going, the gate protects us
            log(f"error: {e!r}")
        if gate.tripped:
            log("risk gate tripped; stopping. Review state/trades.log before restarting.")
            return
        time.sleep(period - (time.time() % period) + 2)  # wake just after the next bar closes
