# Project memory (read first, every session)

This file keeps the project safe when a chat is compacted or a new session starts. The code, the
GitHub workflows and the results branches are the source of truth, not chat history. If chat memory
and this file disagree, trust this file and the repo, then update this file.

## Who this is for
- Owner: NaveenRaja-M75, India, beginner, limited English. Explain in short, simple sentences with numbers.
- Plan money: monthly SIP (my/holdings.json, `monthly_add` Rs 5,000, 10% yearly step-up). Crypto bot: $10-20 only.
- Be honest: no promises of daily profit. Never present an untested or fake result as real.

## Hard rules (never break)
1. Real-money trading is OFF by default (`LIVE_TRADING=false`, `BINANCE_TESTNET=true`). Real orders only after
   dry run -> testnet -> the owner's explicit "go", with `--execute --i-understand-i-can-lose-money`.
2. Binance API key: Spot trading only, withdrawals NEVER enabled, IP-restricted if possible.
   Keys live only in `.env` (git-ignored) or GitHub secrets. Never ask for keys or passwords in chat.
3. Do NOT run `npx vercel@latest ai-gateway setup` (owner said not to use it).
4. Safety code stays on: budget cap, loss-limit kill switch + latch file, exchange-side stop order,
   idempotent orders, data guards, 5% trade band, never sell coins the owner already had.
5. Do not weaken tests or guards to make a strategy look better. All tests must pass before a push (`python -m pytest -q`).
6. No AI model names/IDs in code, docs or commit messages (attribution trailers come from the session).

## What runs (GitHub Actions; this container cannot reach Binance/Yahoo, so data jobs run there)
| workflow | when | does |
|---|---|---|
| weekly.yml | daily 00:23 UTC | paper-trades `vote_voltgt` BTC $20, live dry-run rehearsal, alerts + market watch -> GitHub issues, saves `paper-state` |
| paper.yml | hourly | paper `chart_vote` (research only) |
| monthly-plan.yml | 1st of month | `python -m jevbot plan` -> MONTHLY_PLAN.md (split, $10/day goal, crore goal after inflation) |
| yearly-research.yml | 2 Jan | re-tests everything on fresh data |
| check.yml | on push | tests + backtests -> `backtest-results` branch |
| polymarket.yml ("research") | manual | runs any `research/*.py`, saves branch `results-<script>` |
Results are in branches (paper-state, backtest-results, results-*), because run logs are not readable from here.
Fetch with an explicit refspec, e.g. `git fetch origin paper-state:refs/remotes/origin/paper-state`.

## The decision (do not undo without new, out-of-sample evidence)
- Live crypto rule: `vote_voltgt` in jevbot/portfolio.py = share of 50/100/200-day averages price is above
  x min(1, 0.60 / 60-day vol). Checked daily 00:05 UTC (05:35 IST). BTC only while the account is under ~$60.
  Backtest BTC 2017+: +60.5%/yr, worst drop -47% (hold: -83%). About 3 orders a month at $20.
- Whole plan: 35% Nifty 50 + 35% gold (plain monthly SIP, held) + 30% crypto bot; April rebalance if >5 points off.
  Locked years +19.3%/yr, worst drop -11%. Plan with 10-15%/yr, 6% inflation.
- Taxes India: crypto 30% + 4% cess on each gain, losses not offset, 1% TDS. Nifty/gold ETF LTCG 12.5% after 12 months.

## Already tested and FAILED (see RESULTS.md; do not repeat unless the idea is genuinely new)
Short-term direction prediction is ~50% everywhere: hourly bots, journal learner (515k predictions),
Laya judge, gradient boosting, TimesFM, physics/Kalman/entropy, candlestick patterns (2/248 passed, both momentum),
ensembles, search loops (overfit: PBO high). Also failed after costs/Indian tax: stat-arb, Polymarket clock and
late-closer, crash guards, smart SIP (plain SIP won), intraday breakout (Yahoo BTC data faked +62%; Binance +27% pre-tax,
loses after per-trade tax), funding carry (+11.7%/yr pre-tax, killed by tax and $100 minimum).
Standards for a new idea: walk-forward, locked years, PBO, deflated Sharpe, random-entry control, |t| >= 3.5, exchange data only.

## Status (update when it changes)
- Last updated: 2026-11-01. 65 tests pass.
- Owner still to do: open demat + start Nifty/gold SIP; create Binance Spot-only key; follow docs/GO_LIVE.md.
- Jev: the Vercel AI Gateway lists `typesafe-ai/jev`, `jev-1.13.0`, `jev-1.14.0` (research/gateway_models.py).
  Test ready: research/jev_test.py (800 random moments, 10 coins, multi-timeframe description with no dates or price levels, graded 4h and 24h ahead), run via the "research" workflow with
  script=research/jev_test.py. Needs GitHub secret `AI_GATEWAY_API_KEY`, which the owner adds; Claude may not write secrets.
  Adopt Jev only if it beats a coin flip by more than 2 standard errors AND beats the shuffled-text control.
- Open offers: Binance/demat walkthrough.
- Dashboard: https://claude.ai/artifact/VfXXtQjnDQ6EEu1eSGkhsb

## Where to look
README.md (overview), RESULTS.md (every experiment), docs/GO_LIVE.md (steps), docs/PLAN.md (money plan),
docs/RUNBOOK.md (what to do in crashes / outages), jevbot/ (code), research/ (experiments), tests/.
