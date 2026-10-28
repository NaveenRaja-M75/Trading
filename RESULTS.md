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
