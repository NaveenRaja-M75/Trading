# Runbook: what happens, and what you do

The bot checks once a day (05:35 India time). It reacts to trends, not to single days: tests showed
that reacting to sudden drops within the day sold bottoms and lowered results (RESULTS.md §16).

## Market events

| Event | What the bot does by itself | What you do |
|---|---|---|
| Bitcoin falls 8%+ in a day | Nothing extra that day. Next morning it re-reads the trend; volatility control usually trims the position. You get a GitHub alert. | Nothing. Do not sell by hand. |
| Bitcoin falls 15%+ in a week | Usually already holding less (fewer averages above, higher volatility). Alert. | Nothing. |
| Crash big enough to hit the loss limit | The **stop order resting on Binance** sells, even if your computer is off. Next run records it and turns the kill switch on. | Read the log. Restart only by deleting `state/WEEKLY_KILL_SWITCH`, and only if you still accept the risk. |
| Long bear market (like 2022) | Mostly in cash (USDT). Tested: -3.9% for the whole plan in 2022 while Bitcoin lost 60%. | Keep adding money to Nifty and gold as planned. |
| Bitcoin rockets up | Holds up to 100% of its budget, scaled down if it is very wild. | Nothing. Do not add leverage. |
| Nifty or gold crashes | Nothing: they are held on purpose (holding beat every timing rule on them). | Keep your monthly buys going: new money buys them cheaper. April rebalance moves money back to 35/35/30. |
| Everything falls together (COVID, March 2020) | Crypto part shrinks fast; Nifty and gold are held. Tested: -21.5%, back to a new high in 153 days. | Nothing. This was the worst case in 8 years. |

## Technical events

| Event | What protects you | What you do |
|---|---|---|
| Computer off or no internet on a day | The exchange stop still guards the loss limit. The next run catches up from the latest daily candle. | Turn it back on. One missed day costs little. |
| Binance down or slow | Orders carry a unique id; after a lost reply the bot checks the order instead of sending a duplicate. | Nothing, unless errors repeat for days. |
| Bad or stale price data | The bot refuses to trade a coin with data older than 36h, shorter than 210 days or a >50% one-day move. Alert. | Nothing; it clears itself when the data does. |
| Bot bug | 59 automated tests run on every change; GitHub rehearses the real-money program daily with live prices. | If something looks wrong, set `LIVE_TRADING=false` in `.env`: the bot then refuses to trade. |
| API key leaked | Key is trading-only with withdrawals disabled, so it cannot take money out. | Delete the key on Binance, make a new one. |

## Money and rules

| Event | Plan |
|---|---|
| Exchange collapse (like FTX) | Only the bot's crypto budget sits on Binance. Nifty and gold are in your demat account. Withdraw profits above the budget to your bank a few times a year. |
| Stablecoin (USDT) trouble | Cash waiting on Binance is in USDT. If USDT loses its $1 peg, sell to INR and stop the bot until it is resolved. |
| India changes crypto rules | Follow the law first: stop the bot, withdraw. The Nifty/gold part is unaffected. |
| Tax time | `python -m jevbot tax-report` writes every sale with cost, gain and tax for the crypto (VDA) part of the return. Nifty/gold: your broker's capital-gains statement. Confirm with a CA. |

## When to stop and review the strategy itself

Results will wander. Stop and review only if **live results fall outside anything seen in testing**:
- a 12-month result worse than the 1-in-20 bad case for your budget (about -26% for the crypto bot), or
- a drop past -45% from the peak on the crypto bot, or past -25% on the whole plan.
Every January GitHub re-runs the full research (`yearly-research.yml`); switch rules only if a new
one beats the live rule on both BTC and ETH with PBO < 0.30 and DSR > 0.95.
