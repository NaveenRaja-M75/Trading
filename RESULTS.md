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

## 5. Laya (open-source Jev alternative) as a direction predictor
Laya is a text decision model (reads text, answers typed questions in one pass). We showed it
the bot's plain-English chart read at 800 random past moments (10 coins, 1h) and asked
"will the price be higher 4 hours from now?" (`research/laya_test.py`, raw output on the
`laya-results` branch).

| | correct | note |
|---|---|---|
| always "up" | 50.0% | |
| Laya | 49.8% | says "up" 57.5% of the time; very overconfident (Brier skill -32%) |
| Laya, chart texts shuffled to the wrong moments | 50.0% | same as real: the chart told it nothing |
| our learner | 51.1% | honest confidence (Brier skill ~0%) |

When Laya was most bullish the price went up only 46.6% of the time. Laya is fast and
useful for classifying text (e.g. tagging news), but it has no market knowledge, and price
charts contain no 4h direction signal for it to find. It does not improve our results.

## 6. Weekly momentum / rotation (10 coins, daily candles, 2019-2026), locked final year
Rebalanced once a week, so fees barely matter. The last 365 days (2025-10-28 -> 2026-10-27)
were locked; 36 strategy variants were compared on the older data and the winner was picked
by a rule fixed in code (family with the best *worst-neighbour* Sharpe, median member).

- Dev period: every momentum/trend family beat holding on risk (Sharpe 1.3-1.7 vs 1.31;
  worst drop -35% to -75% vs -79%). Caveat: the 10 coins were chosen because they are big
  today (survivorship), which flatters every long strategy in the old data.
- Pick: **ts_momentum_14d** (hold each coin only if it rose over the last 14 days).
- Locked year, opened once: **+1.4%**, worst drop -15.9%. Holding the same coins: -46.3%
  (worst drop -60.7%); holding BTC: -29.1%.
- Some other variants did better on the locked year (rotation_14d_top3 +24%), but switching
  to them now would be picking with hindsight, so we don't.

"Weeks positive" is low (32%) because the strategy sits in cash in bad markets: its value
is losing far less in crashes and catching big up-trends, not winning most weeks.

Now paper-traded daily (rebalanced weekly) by `.github/workflows/weekly.yml`; see
`WEEKLY.md` on the `paper-state` branch.

## 7. Weekly strategy, walk-forward 2022-2026 (each year picked only from earlier years)
Stitched out-of-sample track record, after fees:

| | 10 survivor coins | 30 coins incl. crashed/delisted | BTC hold | hold all 30 |
|---|---|---|---|---|
| total 2022 -> 2026-10 | +76% | +74% | +75% | -67% |
| per year | +12.5% | +12.2% | +12.3% | -20.5% |
| worst drop | -41% | -40% | -67% | -79% |
| losing years | 2 of 5 | 2 of 5 | 3 of 5 | |

- Same long-run return as holding Bitcoin with ~40% smaller worst drop; far better than
  holding a basket of coins, most of which faded.
- Adding 20 failed/faded coins changed almost nothing: the result is not survivor bias.
- The yearly pick moves between trend-filter and momentum variants; the family (stay in
  coins that are trending up, cash otherwise) is what works, not one exact setting.
- It still loses in some years (2022 -27%, 2025 -15% to -20%), just much less than holding.

## Real money
`python -m jevbot weekly-live` runs this on Binance: dry run by default, budget cap
(WEEKLY_BUDGET_USDT), total-loss kill switch (WEEKLY_MAX_LOSS_USDT) that sells and latches,
never touches coins the user already owns, skips orders under Binance's 5 USDT minimum
(so a position that crashes below 5 USDT cannot be sold by any bot).
