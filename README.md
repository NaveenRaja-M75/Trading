# jevbot

A crypto trading bot built around one rule: **signals judge, deterministic code executes.**
Signals (momentum, mean-reversion, a walk-forward ML model, optional LLM judge such as Jev)
only produce a score. A fixed policy turns scores into positions, and hard risk limits
sit in front of every order.

Every signal is measured against a **coin-flip baseline after fees and slippage**. If a
signal can't beat that out-of-sample, it doesn't trade.

## Quick start

```bash
pip install numpy pandas pytest
cp .env.example .env          # edit symbol, costs, limits
python -m pytest -q           # run the safety/correctness tests

python -m jevbot backtest --days 30          # downloads real Binance candles, compares signals
python -m jevbot backtest --synthetic        # random-walk sanity check: everything should FAIL
```

The backtest prints in-sample and out-of-sample results plus a PASS/FAIL verdict. Only
out-of-sample counts. A PASS needs: positive return, better than coin-flip, Sharpe > 1, ≥ 30 trades.

## Paper → testnet → real (in that order)

```bash
python -m jevbot paper --signal ensemble     # simulated fills on live prices, state in state/
# after days of paper results you trust:
LIVE_TRADING=true BINANCE_TESTNET=true  python -m jevbot live --i-understand-i-can-lose-money
# only then, with small limits:
LIVE_TRADING=true BINANCE_TESTNET=false python -m jevbot live --i-understand-i-can-lose-money
```

Add `--llm` to blend in an LLM judge configured via `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`
(any OpenAI-compatible endpoint). Timeouts or errors count as "no opinion" (stay flat).

## Safety built in

- Live mode needs `LIVE_TRADING=true` **and** a CLI flag; defaults to Binance **testnet**.
- Position capped at `MAX_POSITION_USDT`; daily loss limit flattens and stops the bot.
- The bot only ever sells coins **it** bought (tracked in `state/`), never your existing holdings.
- Orders below exchange minimums are skipped, not crashed on.
- Stale market data pauses trading.

## Layout

| file | role |
|---|---|
| `jevbot/data.py` | Binance public klines, CSV cache, synthetic data |
| `jevbot/features.py` | causal features (no look-ahead, tested) |
| `jevbot/signals.py` | coin-flip, momentum, mean-reversion, walk-forward model, ensemble, LLM judge |
| `jevbot/policy.py` | score → position (hysteresis, min hold) and risk gate |
| `jevbot/backtest.py` | cost-aware backtest, in/out-of-sample comparison, verdict |
| `jevbot/broker.py` | paper broker and Binance spot broker |
| `jevbot/runner.py` | live/paper loop |

## Honest expectations

Short-timeframe trading is dominated by costs: at 0.1% fee per side, a strategy needs to
win more than 0.2% per round trip just to break even. Most simple signals don't. Expect the
first real backtests to say FAIL. That is the tool doing its job and keeping your money safe.
Nothing here guarantees profit.
