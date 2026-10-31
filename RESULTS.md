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

## 8. Small budget (10-20 USDT): BTC + ETH, walk-forward 2022-2026
| | trend filter, BTC + ETH | BTC only (rule's picks) | hold BTC |
|---|---|---|---|
| total | **+199%** (+25%/yr) | +65% | +75% |
| worst drop | **-36%** | -36% | -67% |
| losing years | 1 of 5 (2022 -22%) | 2 of 5 | 3 of 5 |

Caveat: BTC + ETH was chosen after also testing 1, 10 and 30 coins; all four were positive
with roughly half the drawdown of holding, so the conclusion does not hinge on that choice.
Live setup and steps: `docs/GO_LIVE.md`. The paper portfolio now runs this exact setup.

## 9. Final live setup: daily 200-day trend with 2% buffer, BTC + ETH, 20 USDT
Why it replaced the weekly version: weekly results depended on the rebalance weekday alone
(+27% to +103% since 2022). Checking daily removes that; the 200-day rule was the most
stable across buffer sizes (since 2022: +104% to +166%) with the smallest drops (-29% to -33%).

Since 2022 (partly in-sample: the setup was chosen after seeing these years):
- 14 trades, **win rate 50%**, average win +38.6%, average loss -7.4% (wins 5.2x losses),
  biggest loss -10%, longest losing streak 4, average hold 111 days.
- Total +136%, worst drop -31%; with double fees +132%; **traded 1 day late +78%** (run at 05:35 IST).
- Next 12 months for 20 USDT (10,000 resampled futures, 5 USDT loss limit): 70% chance of
  ending up, median 23.82, bad case (5%) 14.53, good case (5%) 44.53, loss limit hit 13%.

## 10. Polymarket "crypto clock" (15-minute Bitcoin Up/Down), 1,345 windows, 14 days
Tested the most concrete idea from popular "AI trading desk" posts: late in each window, buy the
side Bitcoin is already leading. Read-only research (`research/polymarket_clock.py`).

- First pass looked like a money machine: +10% to +54% ROI after fees, every day green.
- Cause: the model used a Bitcoin price ~38 s newer than the Polymarket price it was compared
  with. Given only data as old as Polymarket's price, the model is **less** accurate than
  Polymarket (log-loss 0.383 vs 0.342) and betting on it **loses 1% to 13%** after fees.
- Polymarket is well calibrated: its 85c side won 86.5%, its 92.7c side won 92.0%.
- The posts' rule (buy the leader at 75-92c with 1-4 min left) lost about 3% after the 7% fee
  curve; break-even only at 1 minute left.
- Any real edge here is a speed race against professional market makers, and these markets are
  marked restricted by Polymarket. Real-money betting is also restricted in India.

Verdict: NO EDGE. A one-line timing mistake is enough to produce screenshots like the ones in
those posts.

## 11. Testing a second AI's "v5.0 Claude + Jev + Medallion" plan
The plan's own "verification" typed the Jev answers in by hand (0.68 / 0.92) and forced the
spread on synthetic data; no market data and no Jev call were involved. Tested here instead:

**"Medallion" OU stat-arb** (`research/stat_arb_test.py`, its exact entry rule, BTC vs ETH/SOL/BNB,
3 years of 1h and 1 year of 15m, fees 0.12% per side per leg): all 24 variants lost,
**-37% to -84%**, **-48% to -90% after Indian tax**, 90-420 trades a year, win rates 23-53%.
The hedged pair version (needs shorting) lost the most, because it pays fees on four legs.
Random entries did about as well (11-90th percentile): no consistent signal.

**Jev ">90% confidence is reliable"**: Laya (open-source Jev alternative) on 800 graded BTC calls:
when it said >90% it was right **47.6%**; 80-90% -> 47.9%. Confidence measured concentration,
not correctness, as Jev's own guide warns.

**Polymarket "late closer"** (`research/polymarket_sports.py`, 6,513 resolved soccer markets,
last 60 days, prices in the 60th-85th minute): Polymarket is well calibrated. Buying the leader
at 75-92c returned **-2.2% to -2.8%** after the fee and a 1c spread, in both halves of the period.
92-99c favourites returned about +1% (both halves): the known favourite-longshot bias, but it
risks ~$9.6 to make ~$0.10 per $10 bet, and real-money betting is restricted in India.
50-60c looked like +7.7%, then +2.1% on the later half: shrinking, likely noise.

**FRED macro dial**: not tested. fred.stlouisfed.org timed out from GitHub's servers on every
attempt; `python research/macro_dial_test.py` runs it from a home connection.

**Indian tax and trading speed**: 31.2% on every winning trade, no loss offset. A strategy that
wins 55% with equal-size wins and losses makes +0.10% per trade before tax and loses -0.07%
after it. High-velocity trading needs a far bigger edge in India than elsewhere; the live
200-day setup (wins +38.6% vs losses -7.4%) stays clearly positive after tax.

## 12. Nifty 50 and gold (India), and a rupee portfolio with crypto (`research/india_test.py`)
Data: Nifty 50 index (^NSEI, price only: holding also earns ~1.3%/yr dividends not counted),
gold = gold futures x USD/INR, crypto from Binance x USD/INR. Daily data, guarded against
non-daily bars and bad ticks (two earlier runs were rejected: Yahoo's range=max returned monthly
bars, and the NIFTYBEES/GOLDBEES series carried mis-adjusted 2019-20 unit splits).

**The 200-day rule is a crypto tool, not a universal one.** Since 2008:
- Nifty: rule 5.3%/yr, worst drop -31%; holding 7.3%/yr (+dividends), worst drop -60%.
- Gold: rule 8.3%/yr, worst drop -42%; holding 14.0%/yr, worst drop -30%. Hold gold.

**Best mix found, since May 2018 (one pre-set mix, not searched):**
40% Nifty + 40% gold held, 20% BTC/ETH on the 200-day rule: **+21.3%/yr, worst drop -23.8%,
Sharpe 1.42 (best of all tested), worst year -4.3%, 2 small losing years of 9.**
Holding Nifty + gold 50/50: +15.0%/yr, worst drop -20%. Hold all four: +29.7%/yr but -52.8%.

Caveats: 8.5 years only, with an exceptional gold run (+20%/yr since 2020) and a crypto bull
market; expect less. The mix was chosen after seeing that holding suits Nifty/gold and the rule
suits crypto. Tax is far kinder on the core: equity/gold ETFs held > 12 months pay 12.5% on
gains (equity: first Rs 1.25 lakh/yr exempt), and losses offset gains; crypto pays 31.2%.

## 13. Beating the 200-day rule: six families, physics ideas, three independent checks
Research: `research/better_than_200.py` (rupee portfolios, drift-aware simulation, last 2 years
locked) and `research/physics_test.py` (crypto signals, yearly walk-forward).

**Winner: vote x volatility target** (`portfolio.vote_voltgt`): a coin is held in thirds according
to how many of its 50/100/200-day averages it is above, scaled down when it is wilder than 60%/yr.
1. Portfolio, locked years Oct 2024 -> Oct 2026 (picked by rule before opening them):
   35% Nifty + 35% gold + 30% crypto on vote_voltgt: **+19.3%/yr, worst drop -11.0%, Sharpe 1.31**;
   old plan (40/40/20 on the 200-day rule) +13.6%, -13.5%, 0.97. Whole period since 2018:
   +25.9%/yr, worst drop -22.3%, worst year -3.9%.
2. BTC + ETH, 2020 -> now: +48.3%/yr, worst drop -42.8%, Sharpe 1.26 vs the 200-day rule
   +48.3%, -47.6%, 1.06.
3. BTC alone, 2017 -> now (Yahoo, from 2014): **+60.5%/yr, worst drop -47.4%, worst year -27.9%**
   vs 200-day rule +57.8%, -67.2%, -54.8%, and holding +56.8%, -83.4%, -73.6%. From 2019 the
   walk-forward rule itself picked vote_voltgt every year.

Also tested and rejected: dual momentum (big drawdowns), risk parity with trend (weak on the
locked years), trend votes on Nifty/gold too (worse than holding them), **Kalman-filter trend**
(rocket-guidance tracking: no better than the 200-day rule, deeper drops) and **Hurst-exponent
regime filter** (fractal physics: Sharpe 0.35-0.64, kills most of the profit).
These years had been seen in earlier research, so the walk-forward and the pre-set selection rule
are the protection; future results will be weaker than these.

## 14. The search loop: can more searching beat the live rule? (`research/search_loop.py`)
Six generations widened the pool from 3 to 174 rules (trend and momentum votes, volatility targets,
Hurst / Kalman / 200-day regime filters, smoothing, breakouts, a genetic mutation step ranked only
on pre-test data). Each pool was scored by the walk-forward result of the search itself, the
probability of backtest overfitting (PBO, CSCV with 16 blocks) and the deflated Sharpe ratio (DSR).
Both detectors were validated first: pure noise -> PBO 0.70 / DSR 0.37; one planted edge among 59
noise rules -> PBO 0.01 / DSR 0.999, and it was found.

BTC test years 2017 -> now: the live rule (50/100/200 vote x 60% volatility target) made
**+57.3%/yr, worst drop -46%, Sharpe 1.37**. The searched picks: gen 0 Sharpe 1.13, gen 2 1.18,
gen 5 1.32 with only +32.6%/yr. ETH (never searched), test years 2019 -> now: live rule
**+47.5%/yr, Sharpe 1.14**; searched picks fell from 1.03 to 0.83 as the pool grew, while PBO
rose from 0.21 to 0.59. Averaging many rules (`research/ensemble_test.py`, 4 / 24 / 44 rules)
also fell short on both coins (Sharpe 1.28-1.36 BTC, 1.04-1.12 ETH) and traded more.

Conclusion: the trend + volatility family is real (DSR 0.95-1.00 after 174 tries); which exact
setting is "best" is luck. The live rule sits on the plateau, and more searching on this data
only adds overfitting. `.github/workflows/yearly-research.yml` re-runs both tests every January;
switch only if a rule beats the live one on BOTH coins with PBO < 0.30 and DSR > 0.95.

## 15. A trained model predicting the next week (`research/ml_test.py`)
Gradient-boosted trees, 16 causal features (trend distances, returns, volatility, drawdown, Hurst,
Kalman slope, RSI, the trend vote), target "higher in 7 days", retrained every January on earlier
data only with a 7-day gap (leakage check passed).
- BTC 2018 -> now: accuracy **50.4%** vs 51.9% for always guessing up, **AUC 0.496** (coin flip).
- ETH 2020 -> now: accuracy **50.9%** vs 52.6%, **AUC 0.495**.
- Trading on its forecasts: -8% to +6%/yr with drops to -86%; as a filter on the live rule it
  cut returns. The live rule over the same years: BTC +36.9%/yr (Sharpe 1.08, worst drop -46%),
  ETH +49.9%/yr (Sharpe 1.17, worst drop -43%).

Across sections 2-15 the result is the same: nothing here can forecast short-term direction.
What works is not forecasting but reacting: hold while the trend is up, hold less when the market
is wild, step aside in downtrends. That is the live rule.

## 16. Stress tests and crash guards
**Every real crisis since 2018, whole plan (35/35/30) vs holding Bitcoin:** 2018 crypto winter
-2.1% vs -47.3%; COVID crash -21.5% vs -32.6% (the plan's worst; new high 153 days later);
May 2021 -5.6% vs -48.0%; LUNA -2.8% vs -51.0%; FTX -2.5% vs -26.4%; all of 2022 -3.9% vs -60.2%;
Aug 2024 -2.0% vs -19.0%. Worst single day -6.5%, worst 30 days -21.4%.

**Hourly crash guards** (`research/crash_guard_test.py`, BTC hourly 2018 -> now): the daily-only
live rule made +46.3%/yr, Sharpe 1.24. Every guard did worse: sell on a 5/7/10/15% drop within the
day +41.2/43.8/42.1/43.7%; trailing 5-15% +32.6 to 43.7%. They mostly sold panics that bounced
(e.g. 26 Nov 2020: sold, +47.5% thirty days later). Kept: daily check only, plus a disaster stop at
the loss-limit price resting on the exchange (works with the computer off).

## 17. How often the live bot trades, and compounding (`research/trade_frequency.py`)
Replay of the live logic from June 2018, gains reinvested. A flaw was found and fixed here: with
only the 5 USDT minimum, a growing account placed ~21 orders a month for tiny volatility tweaks;
orders now also need >= 5% of equity, as in every backtest.
- $20, BTC only: **3.2 orders a month** on average, at most 1 a day, no order on 89% of days;
  $20 -> $421 (x21), worst drop -46%.
- $200, BTC + ETH: 6.5 orders a month; $200 -> $4,469 (x22), worst drop -41%.
- $20 + $17 added every month: $1,720 put in -> $10,173 (x5.9), worst drop -39%.
These years include one of crypto's biggest booms; plan for far less.

## 18. Delta-neutral funding carry (`research/funding_carry_test.py`)
Buy spot, short the same amount of the perpetual, collect funding: no prediction needed.
Real Binance funding history, Jan 2020 -> Sep 2026:
- Before tax it works: BTC **+11.7%/yr** on average (ETH +13.8%), losing months only 7%. But it is
  shrinking: BTC 2020 +17.1%, 2021 +30.6%, 2024 +12.0%, 2025 +5.1%, 2026 so far +2.0%.
  Holding it only when recent funding was high did worse (+8.2%/yr).
- **In India it fails.** Each leg is taxed on its own and a loss on one leg cannot offset the
  other (if perps count as VDA - not settled; confirm with a CA). One leg always gains, so tax hits
  every year: per 100 of notional, BTC FY2020-21 funding +27.7 but tax -244.7 (BTC rose 784%);
  FY2023-24 +11.4 vs -47.0. After tax it lost money in 5 of 8 financial years on BTC, 5 of 8 on ETH.
- Binance's BTC perpetual needs ~100 USDT per order, so a 20 USDT account cannot hold the hedge.
Verdict: a real strategy for other countries and bigger accounts; not usable here.

## 19. Google TimesFM foundation model (`research/timesfm_test.py`, TimesFM 2.5, zero-shot)
Each forecast saw only the previous 512 closes; scored against what happened.
| | direction right | always-up | most confident 20% | correlation | trading on it |
|---|---|---|---|---|---|
| BTC daily, 5 days ahead (572) | 46.5% | 52.4% | 36.5% | -0.10 | -0.008%/window vs +0.76% always long |
| Gold daily (394) | 50.0% | 56.1% | 50.6% | +0.02 | +0.07% vs +0.40% always long |
| Nifty daily (386) | 54.7% | 54.7% | 57.7% | -0.15 | -0.09% vs +0.23% always long |
| BTC hourly, 1h ahead (1,460) | 50.0% | 48.9% | 48.3% | +0.01 | -0.137% per trade (intraday) |
| BTC hourly, 4h ahead (1,460) | 51.2% | 49.5% | 50.3% | +0.01 | -0.139% per trade (intraday) |
No forecasting skill on prices, daily or intraday; its most confident BTC calls were wrong 63% of
the time. A strong general-purpose forecaster finds no pattern because the past price path carries
no usable information about the next move. Intraday trading on it loses the fee on every trade.

**vnpy (VeighNa 4.5)** was reviewed, not tested: an MIT-licensed trading platform (event engine,
CTA backtester, GUI, Qlib-style Alpha158 + LightGBM/MLP stock-ranking module). Its broker gateways
are Chinese futures/securities plus Interactive Brokers; none for Indian brokers or Binance spot.
It is plumbing, not an edge, and our own bot already covers what the plan needs.

## 20. Intraday, done properly (`research/intraday_search.py`, `intraday_india.py`, `vbo_check.py`)
Five intraday families, ~190 rules per coin, BTC and ETH hourly 2017 -> 2026, walk-forward picks,
PBO and deflated Sharpe, two fee levels, Indian tax.
- **Time-of-day windows** (hold certain hours daily): -17% to -45%/yr. **Weekday** and **intraday
  momentum**: weak, deep drops. **Dip buying**: tiny (4 trades/yr) and not repeatable on ETH.
- **Volatility breakout** (buy at open + k x yesterday's range, sell at the close) is real before tax:
  picked every year on both coins, PBO 0.00. On Binance data k = 0.7: **+27.1%/yr, Sharpe 1.02**
  (Oct 2017 -> now). But the daily trend rule beat it (BTC +48%/yr, Sharpe 1.27 in the same test),
  and after India's 31.2% tax on every winning trade the breakout averages **-0.04% to -0.11% per
  trade**: -7% to +1%/yr.
- **On Indian indices** (NIFTYBEES / BANKBEES, where intraday losses do offset gains): Nifty
  -4.0% to -6.4%/yr, Bank Nifty -0.6% to -3.1%/yr walk-forward, vs holding +11.4% / +13.9%/yr.
- **Data trap caught**: Yahoo's BTC-USD candles differ from Binance's traded prices by >1% on
  hundreds of days and inflated the same breakout to a fake +62%/yr (2018: "+154%" vs real +64%).
  Backtests here use exchange data only.
Verdict: no intraday bot. The one real intraday edge is weaker than the daily rule before tax and
negative after Indian crypto tax.
