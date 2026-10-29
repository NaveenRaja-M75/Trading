# Going live with 10-20 USDT

**Read first:** no strategy wins every trade or every year. In testing this one won half its
trades and lost money in some years. Only use money you can afford to lose completely.

## The setup (chosen by the tests)
- Coin: **BTC only** while the crypto budget is under ~60 USDT (the strategy buys in thirds and
  Binance's minimum order is 5 USDT). From ~60 USDT use `--symbols BTCUSDT,ETHUSDT`.
- Strategy: **vote_voltgt**, checked **daily**: hold 1/3, 2/3 or all of the budget according to how
  many of the 50/100/200-day averages the price is above, and less when the market is very wild.
  It beat the old 200-day rule in three separate tests (RESULTS.md §13).
- Loss limit: `WEEKLY_MAX_LOSS_USDT=5` (sell everything and stop if 20 USDT becomes 15).

## One-time setup on your computer (Windows or Mac)
1. Install Python 3.12 from python.org (Windows: tick "Add python.exe to PATH").
2. `git clone https://github.com/NaveenRaja-M75/Trading` then `cd Trading`
   (or GitHub -> Code -> Download ZIP, unzip, open a terminal in the folder).
3. `pip install numpy pandas`
4. Copy `.env.example` to `.env` and set:
   ```
   WEEKLY_BUDGET_USDT=20
   WEEKLY_MAX_LOSS_USDT=5
   LIVE_TRADING=false
   BINANCE_TESTNET=true
   ```
5. Binance API key (binance.com -> Profile -> API Management -> Create API):
   enable **Spot trading only**, **never enable withdrawals**, restrict to your IP if offered.
   Put key and secret in `.env` (`BINANCE_API_KEY=`, `BINANCE_API_SECRET=`). Never share `.env`.

## The command
```
python -m jevbot weekly-live --strategy vote_voltgt --every-days 1 --symbols BTCUSDT
```
1. **Dry run**: the command as written. Prints what it would buy or sell; places nothing.
2. **Testnet**: set `LIVE_TRADING=true` (keep `BINANCE_TESTNET=true`, testnet keys from
   testnet.binance.vision) and add `--execute --i-understand-i-can-lose-money`.
3. **Real money**: deposit 20 USDT, set `BINANCE_TESTNET=false`, same `--execute` command.

## Run it every day at about 05:35 India time
Daily candles close at 05:30 IST (00:00 UTC). Trading a day late cut the tested result from
+136% to +78% since 2022 for the old rule, so timing matters. Windows: Task Scheduler -> Create Basic Task ->
Daily 05:35 -> program `python`, arguments `-m jevbot weekly-live ... --execute
--i-understand-i-can-lose-money`, "Start in" = the Trading folder. The computer must be on.

## What protects you
- Never spends more than WEEKLY_BUDGET_USDT, whatever is in your account.
- Never sells coins you owned before.
- Loss limit sells and stops until you delete `state/WEEKLY_KILL_SWITCH`.
- Trades on slow 200-day trends, so it is not racing faster bots.

## What to expect
- About 3 orders a month with 20 USDT (never more than 1 a day); most days nothing happens.
- Gains stay in the bot and compound; when you add money, raise `WEEKLY_BUDGET_USDT` by the same
  amount and keep `WEEKLY_MAX_LOSS_USDT` near a quarter of it.
- A stop order rests on Binance at your loss-limit price after every run, so the limit holds even
  with the computer off. Do not cancel it by hand.
- GitHub opens an issue (you get an email) when something needs attention: see `docs/RUNBOOK.md`.
- Tax time: `python -m jevbot tax-report` lists every sale for the crypto (VDA) schedule.
