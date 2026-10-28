# Going live with 10-20 USDT

**Read first:** no strategy wins every trade or every year. This one lost money in 2022
(-22%) in the honest year-by-year test, while holding Bitcoin lost 64%. Only use money you can
afford to lose completely.

## The setup (chosen by the tests, not by hand)
- Coins: **BTC and ETH** only (a 10-20 USDT budget cannot be split further: Binance's
  minimum order is 5 USDT and the bot refuses positions under 8 USDT).
- Strategy: **trend_filter_ma100**: once a week, hold each coin only while its price is above
  its 100-day average; otherwise hold USDT.
- 20 USDT -> 10 per coin. With 10-15 USDT use BTC only (`--symbols BTCUSDT`).

## One-time setup on your computer (Windows or Mac)
1. Install Python 3.12 from python.org (Windows: tick "Add python.exe to PATH").
2. Download the bot: `git clone https://github.com/NaveenRaja-M75/Trading` then `cd Trading`
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
   - enable **Spot trading** only; **never enable withdrawals**;
   - restrict it to your home IP if Binance offers it;
   - put the key and secret in `.env` (`BINANCE_API_KEY=`, `BINANCE_API_SECRET=`). Never share `.env`.

## Every step before real money
1. Dry run (no orders, no keys needed):
   `python -m jevbot weekly-live --strategy trend_filter_ma100 --symbols BTCUSDT,ETHUSDT --force`
   It prints exactly what it would buy or sell.
2. Testnet (fake money on Binance's test exchange; needs testnet keys from testnet.binance.vision):
   set `LIVE_TRADING=true`, keep `BINANCE_TESTNET=true`, then add
   `--execute --i-understand-i-can-lose-money` to the command above.
3. Real money: deposit 20 USDT, set `BINANCE_TESTNET=false`, run the same `--execute` command.

## Weekly routine
Run the `--execute` command **once a day** (it only trades every 7 days; the other days it just
checks the loss limit). Windows Task Scheduler or Mac `cron` can do this automatically.

## What protects you
- It never spends more than WEEKLY_BUDGET_USDT, whatever is in your account.
- It never sells coins you owned before.
- If the bot's 20 USDT falls to 15 (WEEKLY_MAX_LOSS_USDT=5), it sells and stops until you delete
  `state/WEEKLY_KILL_SWITCH`. Note: this can lock in a loss during a normal dip; the worst
  drop in testing was -36%, so a 5 USDT (25%) limit may trigger in a bad year.
- It does not race other bots: it trades once a week on slow trends, so speed does not matter.
