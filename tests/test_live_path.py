"""The real-money path, exercised against a fake Binance so no network or money is involved."""
import pandas as pd
import pytest

from jevbot import runner
from jevbot.broker import BinanceBroker, PaperBroker
from jevbot.config import Settings
from jevbot.data import synthetic
from jevbot.policy import Policy, RiskGate
from jevbot.signals import Signal


class FakeBinance:
    """Minimal stand-in for the Binance spot REST API."""

    def __init__(self, usdt=100.0, btc=0.0, price=50_000.0):
        self.bal = {"USDT": usdt, "BTC": btc}
        self.price = price
        self.orders = []

    def __call__(self, method, path, params=None, signed=False):
        params = params or {}
        if path == "/api/v3/exchangeInfo":
            return {"symbols": [{"baseAsset": "BTC", "quoteAsset": "USDT", "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "0.00001000", "minQty": "0.00001000"},
                {"filterType": "NOTIONAL", "minNotional": "5.00"}]}]}
        if path == "/api/v3/account":
            assert signed
            return {"balances": [{"asset": a, "free": str(v)} for a, v in self.bal.items()]}
        if path == "/api/v3/order":
            assert signed and method == "POST"
            self.orders.append(params)
            if params["side"] == "BUY":
                quote = float(params["quoteOrderQty"])
                qty = quote / self.price
                self.bal["USDT"] -= quote
                self.bal["BTC"] += qty * 0.999  # fee taken in BTC
                return {"status": "FILLED", "executedQty": str(qty), "cummulativeQuoteQty": str(quote),
                        "fills": [{"commission": str(qty * 0.001), "commissionAsset": "BTC"}]}
            qty = float(params["quantity"])
            assert qty <= self.bal["BTC"] + 1e-12, "sold more BTC than the account holds"
            quote = qty * self.price
            self.bal["BTC"] -= qty
            self.bal["USDT"] += quote * 0.999
            return {"status": "FILLED", "executedQty": str(qty), "cummulativeQuoteQty": str(quote),
                    "fills": [{"commission": str(quote * 0.001), "commissionAsset": "USDT"}]}
        raise AssertionError(f"unexpected call {path}")


@pytest.fixture
def make_broker(tmp_path, monkeypatch):
    def make(fake):
        monkeypatch.setattr(BinanceBroker, "_request", lambda self, *a, **k: fake(*a, **k))
        return BinanceBroker("BTCUSDT", "key", "secret", testnet=True, state=tmp_path / "live.json")
    return make


def test_live_buy_then_sell_round_trip(make_broker):
    fake = FakeBinance(usdt=100)
    b = make_broker(fake)
    assert "LIVE BUY" in b.buy_usdt(50, 50_000)
    assert b.book.qty == pytest.approx(0.000999, rel=1e-6)
    assert "LIVE SELL" in b.sell_qty(b.book.qty, 50_000)
    assert b.book.qty < 1e-5  # only lot-size dust may remain
    assert b.book.realized_usdt < 0  # fees cost money on a flat price


def test_live_never_sells_users_existing_coins(make_broker):
    fake = FakeBinance(usdt=100, btc=1.0)  # user already owns 1 BTC
    b = make_broker(fake)
    assert b.sell_qty(1.0, 50_000).startswith("skip")  # bot bought nothing -> sells nothing
    b.buy_usdt(20, 50_000)
    b.sell_qty(10.0, 50_000)  # asks for far more than it bought
    assert fake.orders[-1]["side"] == "SELL"
    assert float(fake.orders[-1]["quantity"]) <= 20 / 50_000
    assert fake.bal["BTC"] > 0.999  # the user's own BTC is untouched


def test_live_respects_min_notional_and_balance(make_broker):
    b = make_broker(FakeBinance(usdt=3))
    assert b.buy_usdt(50, 50_000).startswith("skip")  # only 3 USDT < 5 minimum
    b = make_broker(FakeBinance(usdt=1000))
    b.buy_usdt(50, 50_000)
    assert float(b.book.cost_usdt) == pytest.approx(50)  # never spends more than asked


class Always(Signal):
    def __init__(self, v):
        self.v, self.name = v, f"always{v}"

    def scores(self, feats):
        return pd.Series(self.v, index=feats.index)


def fresh_bars(n=1000, price=50_000.0):
    df = synthetic(n, seed=7, start_price=price)
    end = pd.Timestamp.now(tz="UTC").floor("1min") - pd.Timedelta(minutes=1)
    df.index = pd.date_range(end=end, periods=n, freq="1min", tz="UTC")
    return df


def test_runner_buys_holds_and_sells(tmp_path, monkeypatch):
    s = Settings(max_position_usdt=50, daily_loss_limit_usdt=5)
    bars = fresh_bars()
    monkeypatch.setattr(runner, "fetch_recent", lambda *a, **k: bars)
    broker = PaperBroker(1000, 10, 2, tmp_path / "p.json")
    gate = RiskGate(50, 5)
    assert "PAPER BUY" in runner.step(s, Always(1.0), Policy(min_hold=0), broker, gate)
    assert "hold" in runner.step(s, Always(1.0), Policy(min_hold=0), broker, gate)
    assert broker.book.qty * bars["close"].iloc[-1] <= 50
    assert "PAPER SELL" in runner.step(s, Always(-1.0), Policy(min_hold=0), broker, gate)
    assert broker.book.qty == 0


def test_runner_flattens_and_stops_on_daily_loss(tmp_path, monkeypatch):
    s = Settings(max_position_usdt=50, daily_loss_limit_usdt=5)
    bars = fresh_bars()
    monkeypatch.setattr(runner, "fetch_recent", lambda *a, **k: bars)
    broker = PaperBroker(1000, 10, 2, tmp_path / "p.json")
    gate = RiskGate(50, 5)
    runner.step(s, Always(1.0), Policy(min_hold=0), broker, gate)
    crashed = bars.copy()
    crashed["close"] = crashed["close"] * 0.8  # 20% crash -> ~10 USDT loss on a 50 USDT position
    monkeypatch.setattr(runner, "fetch_recent", lambda *a, **k: crashed)
    out = runner.step(s, Always(1.0), Policy(min_hold=0), broker, gate)
    assert "daily loss" in out and "PAPER SELL" in out
    assert broker.book.qty == 0 and gate.tripped
    assert "HALT" in runner.step(s, Always(1.0), Policy(min_hold=0), broker, gate)  # no re-entry


def test_runner_refuses_stale_data(tmp_path, monkeypatch):
    s = Settings()
    old = fresh_bars()
    old.index = old.index - pd.Timedelta(hours=1)
    monkeypatch.setattr(runner, "fetch_recent", lambda *a, **k: old)
    broker = PaperBroker(1000, 10, 2, tmp_path / "p.json")
    out = runner.step(s, Always(1.0), Policy(min_hold=0), broker, RiskGate(50, 5))
    assert "stale" in out and broker.book.qty == 0


def test_live_command_refuses_without_both_switches(monkeypatch):
    from jevbot.__main__ import main
    monkeypatch.setenv("LIVE_TRADING", "false")
    with pytest.raises(SystemExit, match="LIVE_TRADING"):
        main(["live"])
    monkeypatch.setenv("LIVE_TRADING", "true")
    with pytest.raises(SystemExit, match="i-understand"):
        main(["live"])


def test_paper_once_runs_each_coin_and_writes_status(tmp_path, monkeypatch):
    from jevbot import __main__ as cli
    monkeypatch.chdir(tmp_path)
    bars = fresh_bars()
    monkeypatch.setattr(runner, "fetch_recent", lambda *a, **k: bars)
    monkeypatch.setattr("jevbot.data.fetch_recent", lambda *a, **k: bars)
    cli.main(["paper-once", "--symbols", "AAAUSDT,BBBUSDT", "--signal", "trend"])
    text = (tmp_path / "state" / "STATUS.md").read_text()
    assert "AAAUSDT" in text and "BBBUSDT" in text and "Total paper P&L" in text
    assert (tmp_path / "state" / "paper_AAAUSDT.json").exists()
