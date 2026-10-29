# The plan (what the tests support)

1. **Core, 80%: hold Nifty 50 and gold, 40% each.** Buy through a demat account
   (Zerodha, Groww, etc.): a Nifty 50 ETF or index fund, and a gold ETF. Add money every month
   (a SIP). Once a year, move money between them so they are back to roughly 50/50.
   No trading, so no fees or short-term tax.
2. **Satellite, 20%: BTC + ETH with the 200-day rule** (the jevbot setup in `docs/GO_LIVE.md`).
   It holds crypto only in uptrends and steps aside in crashes.
3. **Grow it with deposits.** At small sizes, monthly additions add more than any strategy can.

Tested result of this mix since May 2018 (rupees, after costs, before tax): +21.3% a year,
worst drop -23.8%, worst calendar year -4.3%. Past results, not a promise; see RESULTS.md §12.

What every test says to avoid: fast trading (fees and the 31.2%-per-win crypto tax), leverage,
meme coins, prediction-market betting, and anything promising a fixed daily profit.
