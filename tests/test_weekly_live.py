"""Weekly real-money path against a fake multi-coin Binance."""
import re

import pandas as pd
import pytest

from jevbot.broker import BinanceBroker
from jevbot.data import synthetic
from jevbot.weekly_live import WeeklyLive, plan_orders


class FakeExchange:
    def __init__(self, prices, usdt=100.0, owned=None):
        self.prices = prices
        self.bal = {"USDT": usdt, **{s[:-4]: 0.0 for s in prices}, **(owned or {})}
        self.orders = []
        self.book = {}          # orderId -> order, for stops and lookups
        self.by_client = {}
        self.next_id = 1
        self.fail_next_post = False   # simulate "order placed, but the reply was lost"
        self.fail_next_lookup = False # ... and the follow-up question to Binance failed too
        self.step = "0.00000100"
        self.clock_skew_ms = 0

    PLAIN = re.compile(r"^\d+(\.\d+)?$")   # the real Binance rejects numbers like 9e-05

    def check_numbers(self, params):
        for k in ("quantity", "quoteOrderQty", "stopPrice", "price"):
            if k in params:
                assert self.PLAIN.match(str(params[k])), f"{k}={params[k]!r} is not a plain decimal"

    def trigger_stops(self, sym, price):
        """Price fell through the stop while the bot was off."""
        for o in self.book.values():
            if o["symbol"] == sym and o["type"] == "STOP_LOSS_LIMIT" and o["status"] == "NEW" and price <= o["stopPrice"]:
                o.update(status="FILLED", executedQty=str(o["qty"]), cummulativeQuoteQty=str(o["qty"] * o["price"]))
                self.bal["USDT"] += o["qty"] * o["price"]

    def handler(self, sym):
        base = sym[:-4]

        def call(method, path, params=None, signed=False):
            params = params or {}
            if path == "/api/v3/exchangeInfo":
                return {"symbols": [{"baseAsset": base, "quoteAsset": "USDT", "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": self.step, "minQty": self.step},
                    {"filterType": "NOTIONAL", "minNotional": "5.00"}]}]}
            if path == "/api/v3/time":
                return {"serverTime": int(pd.Timestamp.now(tz="UTC").timestamp() * 1000) + self.clock_skew_ms}
            if path == "/api/v3/ticker/price":
                return {"price": str(self.prices[sym])}
            if path == "/api/v3/account":
                locked = {a: 0.0 for a in self.bal}
                for o in self.book.values():
                    if o.get("type") == "STOP_LOSS_LIMIT" and o["status"] == "NEW":
                        locked[o["symbol"][:-4]] += o["qty"]
                return {"balances": [{"asset": a, "free": str(v), "locked": str(locked[a])} for a, v in self.bal.items()]}
            if path == "/api/v3/order" and method == "GET":
                if self.fail_next_lookup and "origClientOrderId" in params:
                    self.fail_next_lookup = False
                    raise TimeoutError("connection lost")
                o = self.book.get(params.get("orderId")) or self.by_client.get(params.get("origClientOrderId"))
                if o is None:
                    raise RuntimeError("Binance 400: Order does not exist.")
                return o
            if path == "/api/v3/order" and method == "DELETE":
                o = self.book[params["orderId"]]
                if o["status"] == "NEW":
                    o["status"] = "CANCELED"
                    self.bal[base] += o["qty"]          # unlock
                return o
            price = self.prices[sym]
            self.check_numbers(params)
            if params.get("type") == "STOP_LOSS_LIMIT":
                qty = float(params["quantity"])
                assert qty <= self.bal[base] + 1e-12
                self.bal[base] -= qty                    # locked while the stop rests
                o = {"symbol": sym, "orderId": self.next_id, "type": "STOP_LOSS_LIMIT", "status": "NEW", "qty": qty,
                     "stopPrice": float(params["stopPrice"]), "price": float(params["price"]),
                     "executedQty": "0", "cummulativeQuoteQty": "0"}
                self.book[self.next_id] = o
                self.next_id += 1
                self.orders.append((sym, "STOP", params))
                return o
            self.orders.append((sym, params["side"], params))
            if params["side"] == "BUY":
                quote = float(params["quoteOrderQty"])
                assert quote <= self.bal["USDT"] + 1e-9
                qty = quote / price
                self.bal["USDT"] -= quote
                self.bal[base] += qty
                resp = {"status": "FILLED", "executedQty": str(qty), "cummulativeQuoteQty": str(quote), "fills": []}
                self.by_client[params.get("newClientOrderId")] = resp
                if self.fail_next_post:
                    self.fail_next_post = False
                    raise TimeoutError("reply lost after the order was placed")
                return resp
            qty = float(params["quantity"])
            assert qty <= self.bal[base] + 1e-12
            self.bal[base] -= qty
            self.bal["USDT"] += qty * price
            return {"executedQty": str(qty), "cummulativeQuoteQty": str(qty * price), "fills": []}
        return call


def market(n, ups):
    """Daily candles; coins in `ups` trend up, others down."""
    out = {}
    for i, s in enumerate(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]):
        d = synthetic(400, seed=80 + i, drift=0.01 if s in ups else -0.01, vol=0.005, start_price=100.0)
        d.index = pd.date_range(end=pd.Timestamp("2026-10-27", tz="UTC"), periods=400, freq="D")
        out[s] = d.iloc[:n] if n else d
    return out


@pytest.fixture
def setup(tmp_path, monkeypatch):
    def make(data, usdt=100.0, owned=None, budget=50, max_loss=10):
        prices = {s: float(d.close.iloc[-1]) for s, d in data.items()}
        ex = FakeExchange(prices, usdt, owned)

        def broker(sym):
            monkeypatch.setattr(BinanceBroker, "_request", lambda self, *a, **k: ex.handler(self.symbol)(*a, **k))
            return BinanceBroker(sym, "k", "s", True, tmp_path / f"b_{sym}.json")
        last = max(d.index[-1] for d in data.values())
        bot = WeeklyLive(list(data), "ts_momentum_14d", budget, max_loss, broker, tmp_path,
                         fetch=lambda s, i, n: data[s], clock=lambda: last + pd.Timedelta(days=1, hours=1))
        return bot, ex
    return make


def test_plan_sells_before_buys_and_skips_dust():
    orders = plan_orders({"A": 0.5, "B": 0.0, "C": 0.5}, {"A": 1, "B": 1, "C": 1},
                         {"A": 49, "B": 30, "C": 0}, 100)
    assert orders[0] == ("B", "SELL", 30.0)
    assert ("C", "BUY", 50.0) in orders and all(o[0] != "A" for o in orders)  # A off by 1 < 5 minimum


def test_dry_run_places_no_orders(setup):
    bot, ex = setup(market(0, {"BTCUSDT", "ETHUSDT"}))
    lines = bot.run(execute=False)
    assert ex.orders == [] and any("DRY RUN BUY" in line for line in lines)


def test_execute_buys_uptrends_within_budget_and_ignores_user_coins(setup):
    bot, ex = setup(market(0, {"BTCUSDT", "ETHUSDT"}), usdt=1000, owned={"SOL": 5.0})
    bot.run(execute=True)
    bought = {o[0] for o in ex.orders if o[1] == "BUY"}
    assert bought == {"BTCUSDT", "ETHUSDT"}
    spent = 1000 - ex.bal["USDT"]
    assert spent <= 50 + 1e-6  # budget, not the 1000 USDT on the account
    assert ex.bal["SOL"] == 5.0  # user's own coins untouched
    again = bot.run(execute=True)                      # same day: no second rebalance
    assert any("next rebalance" in line for line in again)
    assert not any(line.strip().startswith(("BTCUSDT: LIVE", "ETHUSDT: LIVE")) for line in again)


def test_kill_switch_sells_and_latches(setup, tmp_path):
    data = market(0, {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"})
    # positions must stay above Binance's 5 USDT minimum after the crash to be sellable
    bot, ex = setup(data, usdt=1000, budget=200, max_loss=10)
    bot.run(execute=True)
    crashed = {s: d.assign(close=d.close * 0.3) for s, d in data.items()}  # -70%
    bot.fetch = lambda s, i, n: crashed[s]
    ex.prices = {s: float(d.close.iloc[-1]) for s, d in crashed.items()}
    lines = bot.run(execute=True)
    assert any("LOSS LIMIT" in line for line in lines)
    assert all(ex.bal[s[:-4]] < 1e-5 for s in data)
    assert "KILL SWITCH" in bot.run(execute=True)[0]


def test_weekly_live_cli_refuses_without_switches(monkeypatch):
    from jevbot.__main__ import main
    monkeypatch.setenv("LIVE_TRADING", "false")
    with pytest.raises(SystemExit, match="LIVE_TRADING"):
        main(["weekly-live", "--execute"])
    monkeypatch.setenv("LIVE_TRADING", "true")
    with pytest.raises(SystemExit, match="i-understand"):
        main(["weekly-live", "--execute"])


def test_refuses_budget_too_thin_for_coin_count(setup):
    bot, ex = setup(market(0, {"BTCUSDT", "ETHUSDT"}), usdt=1000, budget=20)  # 4 coins x 5 USDT
    lines = bot.run(execute=True)
    assert any("REFUSED" in line for line in lines) and ex.orders == []



def test_exchange_stop_is_set_after_buying_and_replaced_next_run(setup):
    bot, ex = setup(market(0, {"BTCUSDT", "ETHUSDT"}), usdt=1000, budget=200, max_loss=20)
    bot.run(execute=True)
    stops = [o for o in ex.book.values() if o["status"] == "NEW"]
    assert len(stops) == 2 and all(o["stopPrice"] < ex.prices[o["symbol"]] for o in stops)
    bot.run(execute=True, force=True)
    assert sum(o["status"] == "CANCELED" for o in ex.book.values()) == 2      # old stops cancelled
    assert sum(o["status"] == "NEW" for o in ex.book.values()) == 2          # fresh ones set


def test_stop_that_fired_while_offline_is_recorded_and_stops_the_bot(setup):
    bot, ex = setup(market(0, {"BTCUSDT", "ETHUSDT"}), usdt=1000, budget=200, max_loss=20)
    bot.run(execute=True)
    for sym in list(ex.prices):
        ex.trigger_stops(sym, 0.0)                       # crash while the computer is off
    lines = bot.run(execute=True)
    assert any("exchange stop SOLD" in line for line in lines)
    assert "KILL SWITCH" in lines[-1] and "KILL SWITCH" in bot.run(execute=True)[0]


def test_lost_reply_does_not_double_buy(setup):
    bot, ex = setup(market(0, {"BTCUSDT"}), usdt=1000, budget=50)
    ex.fail_next_post = True
    bot.run(execute=True)
    assert sum(1 for o in ex.orders if o[1] == "BUY") == 1      # placed once, recovered by client id
    spent = 1000 - ex.bal["USDT"]
    assert 0 < spent <= 50 + 1e-6


def test_bad_or_stale_data_is_not_traded():
    from jevbot.weekly_live import data_problem
    d = market(0, {"BTCUSDT"})["BTCUSDT"]
    now = d.index[-1] + pd.Timedelta(days=1, hours=1)
    assert data_problem(d, now) == ""
    assert "old" in data_problem(d, now + pd.Timedelta(days=3))
    assert "days of data" in data_problem(d.iloc[-100:], now)
    spiky = d.copy()
    spiky.iloc[-1, spiky.columns.get_loc("close")] *= 3
    assert "bad data" in data_problem(spiky, now)


def test_no_micro_trades_on_a_large_account(setup):
    """A 1% drift on a big account must not trigger an order (band = 5% of equity)."""
    from jevbot.weekly_live import TRADE_BAND, plan_orders
    orders = plan_orders({"BTCUSDT": 0.5}, {"BTCUSDT": 100.0}, {"BTCUSDT": 1960.0}, 4000.0,
                         min_trade=max(5.0, TRADE_BAND * 4000))
    assert orders == []
    orders = plan_orders({"BTCUSDT": 0.5}, {"BTCUSDT": 100.0}, {"BTCUSDT": 1600.0}, 4000.0,
                         min_trade=max(5.0, TRADE_BAND * 4000))
    assert orders == [("BTCUSDT", "BUY", 400.0)]


# ---------------------------------------------------------------- final safety audit (real-money path)

def scaled(data, factor):
    """Same candles at a different price level (trend and signal unchanged)."""
    cols = ["open", "high", "low", "close"]
    return {s: d.assign(**{c: d[c] * factor for c in cols}) for s, d in data.items()}


def stops(ex):
    return [o for o in ex.book.values() if o["type"] == "STOP_LOSS_LIMIT" and o["status"] == "NEW"]


def test_small_bitcoin_amounts_are_written_as_plain_decimals(setup):
    """A few dollars of Bitcoin is below 0.0001 BTC (9 USDT at 120,000 = 0.00007).
    Python prints that as 9e-05, which Binance rejects; the fake exchange now rejects it too."""
    data = scaled(market(0, {"BTCUSDT"}), 120_000 / float(market(0, {"BTCUSDT"})["BTCUSDT"].close.iloc[-1]))
    data = {"BTCUSDT": data["BTCUSDT"]}
    bot, ex = setup(data, usdt=100, budget=9, max_loss=3)
    ex.step = "0.00001000"
    lines = bot.run(execute=True)
    assert not any(line.startswith("ERROR") for line in lines), lines
    qty = [o[2]["quantity"] for o in ex.orders if o[1] == "STOP"]
    assert qty and float(qty[0]) < 0.0001 and "e" not in qty[0]
    b = BinanceBroker("BTCUSDT", "k", "s", True, bot.dir / "fmt.json")
    assert b._fmt_qty(0.00009) == "0.00009" and b._floor_qty(0.00008) == 0.00008


def test_thin_account_can_still_sell_and_keeps_its_stop(setup):
    """Below 8 USDT per coin the bot must stop BUYING, never stop selling or protecting."""
    up = {"BTCUSDT": market(0, {"BTCUSDT"})["BTCUSDT"]}
    bot, ex = setup(up, usdt=100, budget=12, max_loss=6)
    bot.run(execute=True)
    assert len(stops(ex)) == 1
    fallen = scaled(up, 0.65)                                  # equity ~7.8: under the 8 USDT buying floor
    bot.fetch = lambda s, i, n: fallen[s]
    ex.prices = {"BTCUSDT": float(fallen["BTCUSDT"].close.iloc[-1])}
    lines = bot.run(execute=True, force=True)
    assert len(stops(ex)) == 1 and not any("LIVE BUY" in line for line in lines)
    down = market(0, set())["BTCUSDT"]                         # trend turns down at the same price level
    down = scaled({"BTCUSDT": down}, ex.prices["BTCUSDT"] / float(down.close.iloc[-1]))
    bot.fetch = lambda s, i, n: down[s]
    lines = bot.run(execute=True, force=True)
    assert any("LIVE SELL" in line for line in lines), lines
    assert ex.bal["BTC"] < 1e-5 and stops(ex) == []


def test_error_part_way_puts_the_stop_back(setup, monkeypatch):
    bot, ex = setup(market(0, {"BTCUSDT"}), usdt=1000, budget=200, max_loss=20)
    bot.run(execute=True)
    assert len(stops(ex)) == 1
    more = market(0, {"BTCUSDT", "ETHUSDT"})                   # a second coin to buy -> the buy will fail
    bot.fetch = lambda s, i, n: more[s]
    ex.prices = {s: float(d.close.iloc[-1]) for s, d in more.items()}

    def boom(self, *a, **k):
        raise RuntimeError("Binance 503: service unavailable")
    monkeypatch.setattr(BinanceBroker, "buy_usdt", boom)
    lines = bot.run(execute=True, force=True)
    assert any(line.startswith("ERROR") for line in lines)
    assert len(stops(ex)) == 1                                 # BTC is protected again


def test_order_sent_but_never_confirmed_is_recovered_next_run(setup):
    bot, ex = setup({"BTCUSDT": market(0, {"BTCUSDT"})["BTCUSDT"]}, usdt=1000, budget=50)
    ex.fail_next_post = ex.fail_next_lookup = True             # reply lost AND the follow-up question lost
    first = bot.run(execute=True)
    assert any(line.startswith("ERROR") for line in first)
    lines = bot.run(execute=True)
    assert any("recovered an unconfirmed BUY" in line for line in lines), lines
    assert sum(1 for o in ex.orders if o[1] == "BUY") == 1     # never bought twice
    assert 1000 - ex.bal["USDT"] <= 50 + 1e-6 and len(stops(ex)) == 1


def test_coins_sold_by_hand_are_noticed(setup):
    bot, ex = setup({"BTCUSDT": market(0, {"BTCUSDT"})["BTCUSDT"]}, usdt=1000, budget=50)
    bot.run(execute=True)
    o = stops(ex)[0]
    o["status"] = "CANCELED"                                   # the owner cancels the stop in the app ...
    ex.bal["BTC"] += o["qty"] / 2                              # ... and sells half the coins by hand
    lines = bot.run(execute=True)
    assert any("WARNING" in line and "outside the bot" in line for line in lines), lines
    assert not any(line.startswith("ERROR") for line in lines)


def test_stop_order_unknown_to_the_exchange_does_not_crash(setup):
    bot, ex = setup({"BTCUSDT": market(0, {"BTCUSDT"})["BTCUSDT"]}, usdt=1000, budget=50)
    bot.run(execute=True)
    o = stops(ex)[0]
    ex.bal["BTC"] += o["qty"]
    ex.book.clear()                                            # e.g. the monthly testnet wipe
    lines = bot.run(execute=True)
    assert any("no longer on the exchange" in line for line in lines)
    assert not any(line.startswith("ERROR") for line in lines) and len(stops(ex)) == 1


def test_live_price_is_used_and_a_wild_one_is_refused(setup):
    data = {"BTCUSDT": market(0, {"BTCUSDT"})["BTCUSDT"]}
    bot, ex = setup(data, usdt=1000, budget=50)
    ex.prices["BTCUSDT"] *= 0.9                                # market moved 10% since the daily close
    bot.run(execute=True)
    assert stops(ex)[0]["stopPrice"] < ex.prices["BTCUSDT"] * 0.96
    bot2, ex2 = setup(data, usdt=1000, budget=50)
    ex2.prices["BTCUSDT"] *= 0.3
    lines = bot2.run(execute=True)
    assert any("too far from the last close" in line for line in lines) and ex2.orders == []


def test_orders_are_signed_with_binance_time_not_the_pc_clock(setup):
    bot, ex = setup({"BTCUSDT": market(0, {"BTCUSDT"})["BTCUSDT"]})
    ex.clock_skew_ms = 90_000                                  # this PC is 90 seconds slow
    b = bot.make_broker("BTCUSDT")
    assert 85_000 < b.time_offset_ms < 95_000


def test_unconfirmed_buy_survives_the_stop_being_put_back(setup):
    """Holding coins, then an ADD order is lost: putting the stop back must not erase the note about it."""
    bot, ex = setup(market(0, {"BTCUSDT"}), usdt=1000, budget=200, max_loss=20)
    bot.run(execute=True)
    more = market(0, {"BTCUSDT", "ETHUSDT"})
    bot.fetch = lambda s, i, n: more[s]
    ex.prices = {s: float(d.close.iloc[-1]) for s, d in more.items()}
    ex.fail_next_post = ex.fail_next_lookup = True             # the ETH buy goes through but is never confirmed
    first = bot.run(execute=True, force=True)
    assert any(line.startswith("ERROR") for line in first) and len(stops(ex)) == 1
    lines = bot.run(execute=True, force=True)
    assert any("recovered an unconfirmed BUY" in line for line in lines), lines
    assert sum(1 for o in ex.orders if o[0] == "ETHUSDT" and o[1] == "BUY") == 1
    assert 1000 - ex.bal["USDT"] <= 200 + 1e-6
