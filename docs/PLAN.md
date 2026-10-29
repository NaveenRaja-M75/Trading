# The plan (what the tests support)

1. **35% Nifty 50** (index fund or ETF) **and 35% gold** (gold ETF): just hold them.
   Buy through a demat account (Zerodha, Groww, etc.). Holding beat every trading rule on these.
2. **30% crypto, run by the bot** on `vote_voltgt`: it holds BTC (and ETH once the crypto part is
   above ~$60) in thirds, by how many of the 50/100/200-day averages price is above, and holds less
   when the market is wild. Setup: `docs/GO_LIVE.md`.
3. **Every month** run `python -m jevbot plan` (or read `MONTHLY_PLAN.md` on the paper-state branch,
   written on the 1st by GitHub): it says exactly how many rupees go to each part. Keep
   `my/holdings.json` up to date with what you hold and how much you add each month.
4. **Every April** the plan checks whether a real rebalance is needed (more than 5 points off).

Tested result of this mix: locked years Oct 2024 -> Oct 2026 +19.3%/yr with a worst drop of -11%;
since 2018 +25.9%/yr, worst drop -22%, worst year -3.9% (RESULTS.md §13). Past results, not a
promise: plan with 10-15% a year.

Avoid (every test lost money): fast trading, leverage, meme coins, prediction-market betting,
stat-arb pair trading, and anything promising a fixed daily profit.
