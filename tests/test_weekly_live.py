"""Weekly real-money path against a fake multi-coin Binance."""
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
                    {"filterType": "LOT_SIZE", "stepSize": "0.00000100", "minQty": "0.00000100"},
                    {"filterType": "NOTIONAL", "minNotional": "5.00"}]}]}
            if path == "/api/v3/account":
                return {"balances": [{"asset": a, "free": str(v)} for a, v in self.bal.items()]}
            if path == "/api/v3/order" and method == "GET":
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
    assert "REFUSED" in lines[-1] and ex.orders == []



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
