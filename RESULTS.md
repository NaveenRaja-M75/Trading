# Results so far (2026-10-28)

All numbers are after 0.1% fee + 0.02% slippage per side, on real Binance data.
Raw output: `backtest-results` branch. Live paper account: `paper-state` branch, `STATUS.md`.

## 1. Fast trading (1m / 15m candles) loses
Every strategy lost money; fees dominate. Coin-flip lost 96–99.98%.

## 2. Slow, chart-reading strategies
Best single result: `chart_vote` on 1h candles, last 6 months, 4 coins: +12.3% vs holding +11.8%,
worst drop -10.8% vs -32.8%. The only PASS out of 30 strategy/timeframe combinations.

## 3. Stress test of chart_vote (1h, 10 coins x six 6-month periods, 5 nearby settings)
- Profitable in only ~48% of coin x period cells; median result about 0%.
- Average +4.5% per period, carried by a few big trends (XRP +162%, DOGE +75%).
- Every setting from 0.6x to 1.5x behaves the same: no fragile fine-tuning, but no reliable edge either.
- Consistent strength: median worst drop -19% vs -43% for holding, and it beat holding in
  85% of periods where holding lost money.

## Conclusion
No strategy here makes money reliably. `chart_vote` acts as a crash shield (it loses much
less than holding when markets fall) rather than a money machine. It loses small amounts in
choppy markets. Real money stays off unless the hourly paper test is clearly positive after
several weeks.

## 4. Predict, check, learn (10 coins, 3 years of 1h, ~515,000 graded predictions)
Every hour the bot predicted up/down 4h and 24h ahead, graded each prediction when it came
due, and down-weighted the methods that were wrong in that market situation.

- Correct: **50.4%** (4h) and **50.6%** (24h). Always guessing "up" scores 50.0–50.1%.
- No single method beat ~51.7% (best: mean-reversion), and no market situation beat ~52.5%.
- Trading on what it learned lost money: median -22% (4h) and -33% (24h) per 6 months after
  fees, because acting on near-coin-flip calls means paying fees on every switch.

Learning from mistakes works (a test with a planted pattern shows it finds it), but there is
nothing to learn here: short-term direction of these coins from price charts is a coin flip.
A 52% hit rate on an average 1.3% move is worth ~0.05% per trade, less than the 0.24%
round-trip cost.

## Decision
- Stop adding strategies: searching more old data until something looks good is how
  backtests lie.
- Keep the hourly paper account and live prediction journal running as an independent check.
- No real money unless the live paper account is clearly positive after fees.
