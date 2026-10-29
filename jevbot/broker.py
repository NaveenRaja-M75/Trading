"""Brokers. Both track only the coins THIS bot bought, so pre-existing holdings are never sold."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

LIVE_URL = "https://api.binance.com"
TESTNET_URL = "https://testnet.binance.vision"


@dataclass
class BotBook:
    """What the bot itself owns, persisted so restarts don't lose track."""
    qty: float = 0.0           # base asset bought by the bot and not yet sold
    cost_usdt: float = 0.0     # USDT spent on the current qty (incl. fees)
    realized_usdt: float = 0.0
    stop_order_id: int = 0     # protective stop resting on the exchange (0 = none)
    day: str = ""
    day_realized_usdt: float = 0.0
    fills: list = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "BotBook":
        return cls(**json.loads(path.read_text())) if path.exists() else cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fills = self.fills[-500:]
        path.write_text(json.dumps(self.__dict__, indent=1))

    def roll_day(self) -> None:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if self.day != today:
            self.day, self.day_realized_usdt = today, 0.0

    def on_buy(self, qty: float, usdt_spent: float, price: float) -> None:
        self.qty += qty
        self.cost_usdt += usdt_spent
        self.fills.append({"t": time.time(), "side": "BUY", "qty": qty, "price": price, "usdt": usdt_spent})

    def on_sell(self, qty: float, usdt_received: float, price: float) -> None:
        if self.qty <= 0:
            return
        frac = min(1.0, qty / self.qty)
        cost = self.cost_usdt * frac
        pnl = usdt_received - cost
        self.qty = max(0.0, self.qty - qty)
        self.cost_usdt -= cost
        self.realized_usdt += pnl
        self.day_realized_usdt += pnl
        self.fills.append({"t": time.time(), "side": "SELL", "qty": qty, "price": price,
                           "usdt": usdt_received, "pnl": pnl})

    def day_pnl(self, price: float) -> float:
        """Realized today + mark-to-market of the open position."""
        return self.day_realized_usdt + (self.qty * price - self.cost_usdt)


class PaperBroker:
    """Simulated fills at last price +/- slippage, charging the same fee as the backtest."""

    def __init__(self, start_usdt: float, fee_bps: float, slippage_bps: float, state: Path):
        self.state = state
        self.book = BotBook.load(state)
        self.fee, self.slip = fee_bps / 1e4, slippage_bps / 1e4
        self.usdt = start_usdt - self.book.cost_usdt + self.book.realized_usdt

    def buy_usdt(self, usdt: float, price: float) -> str:
        usdt = min(usdt, self.usdt)
        if usdt < 1.0:
            return "skip: not enough paper USDT"
        fill = price * (1 + self.slip)
        qty = usdt * (1 - self.fee) / fill
        self.usdt -= usdt
        self.book.on_buy(qty, usdt, fill)
        self.book.save(self.state)
        return f"PAPER BUY {qty:.8f} @ {fill:.2f} for {usdt:.2f} USDT"

    def sell_qty(self, qty: float, price: float) -> str:
        qty = min(qty, self.book.qty)
        if qty <= 0:
            return "skip: nothing to sell"
        fill = price * (1 - self.slip)
        got = qty * fill * (1 - self.fee)
        self.usdt += got
        self.book.on_sell(qty, got, fill)
        self.book.save(self.state)
        return f"PAPER SELL {qty:.8f} @ {fill:.2f} for {got:.2f} USDT"


class BinanceBroker:
    """Binance spot via signed REST. Testnet unless explicitly told otherwise."""

    def __init__(self, symbol: str, api_key: str, api_secret: str, testnet: bool, state: Path):
        if not api_key or not api_secret:
            raise ValueError("BINANCE_API_KEY / BINANCE_API_SECRET are required for live trading")
        self.symbol, self.key, self.secret = symbol, api_key, api_secret.encode()
        self.base = TESTNET_URL if testnet else LIVE_URL
        self.state = state
        self.book = BotBook.load(state)
        self._load_filters()

    # -- http -------------------------------------------------------------
    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False):
        params = dict(params or {})
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
            q = urllib.parse.urlencode(params)
            params_q = q + "&signature=" + hmac.new(self.secret, q.encode(), hashlib.sha256).hexdigest()
        else:
            params_q = urllib.parse.urlencode(params)
        url = f"{self.base}{path}" + (f"?{params_q}" if method == "GET" else "")
        data = None if method == "GET" else params_q.encode()
        req = urllib.request.Request(url, data=data, method=method, headers={"X-MBX-APIKEY": self.key})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Binance {e.code}: {e.read().decode(errors='replace')}") from None

    def _load_filters(self) -> None:
        info = self._request("GET", "/api/v3/exchangeInfo", {"symbol": self.symbol})["symbols"][0]
        self.base_asset, self.quote_asset = info["baseAsset"], info["quoteAsset"]
        f = {x["filterType"]: x for x in info["filters"]}
        self.step = float(f["LOT_SIZE"]["stepSize"])
        self.min_qty = float(f["LOT_SIZE"]["minQty"])
        notional = f.get("NOTIONAL") or f.get("MIN_NOTIONAL") or {}
        self.min_notional = float(notional.get("minNotional", 0.0))
        self.tick = float((f.get("PRICE_FILTER") or {}).get("tickSize", 0.01))

    def _floor_qty(self, qty: float) -> float:
        if self.step <= 0:
            return qty
        decimals = max(0, -int(math.floor(math.log10(self.step))))
        return round(math.floor(qty / self.step) * self.step, decimals)

    def _floor_price(self, p: float) -> float:
        decimals = max(0, -int(math.floor(math.log10(self.tick))))
        return round(math.floor(p / self.tick) * self.tick, decimals)

    def _order(self, params: dict) -> dict:
        """Place an order exactly once. Every order carries its own client id, so after a network
        error we ask Binance whether that order exists instead of blindly sending a second one."""
        params = {**params, "newClientOrderId": params.get("newClientOrderId")
                  or f"jev-{params['side'][0]}-{int(time.time() * 1000)}"}
        try:
            return self._request("POST", "/api/v3/order", params, signed=True)
        except Exception as first:
            try:
                found = self._request("GET", "/api/v3/order", {"symbol": self.symbol,
                                      "origClientOrderId": params["newClientOrderId"]}, signed=True)
            except Exception:
                raise first from None
            if found.get("status") in {"FILLED", "PARTIALLY_FILLED", "NEW"}:
                return found
            raise first from None

    def free_balance(self, asset: str) -> float:
        acct = self._request("GET", "/api/v3/account", signed=True)
        for b in acct["balances"]:
            if b["asset"] == asset:
                return float(b["free"])
        return 0.0

    @staticmethod
    def _fill_summary(resp: dict) -> tuple[float, float]:
        qty = float(resp.get("executedQty", 0.0))
        quote = float(resp.get("cummulativeQuoteQty", 0.0))
        return qty, quote

    # -- orders -----------------------------------------------------------
    def buy_usdt(self, usdt: float, price: float) -> str:
        usdt = min(usdt, self.free_balance(self.quote_asset))
        if usdt < max(self.min_notional, 1.0):
            return f"skip: {usdt:.2f} {self.quote_asset} below min notional {self.min_notional}"
        resp = self._order({"symbol": self.symbol, "side": "BUY", "type": "MARKET",
                            "quoteOrderQty": f"{usdt:.2f}", "newOrderRespType": "FULL"})
        qty, quote = self._fill_summary(resp)
        fee_base = sum(float(x["commission"]) for x in resp.get("fills", [])
                       if x.get("commissionAsset") == self.base_asset)
        net_qty = qty - fee_base
        if net_qty <= 0:
            return f"buy returned no fill: {resp.get('status')}"
        self.book.on_buy(net_qty, quote, quote / qty)
        self.book.save(self.state)
        return f"LIVE BUY {net_qty:.8f} {self.base_asset} for {quote:.2f} {self.quote_asset}"

    def sell_qty(self, qty: float, price: float) -> str:
        # never sell more than the bot bought, nor more than is actually free on the account
        qty = self._floor_qty(min(qty, self.book.qty, self.free_balance(self.base_asset)))
        if qty < self.min_qty or qty * price < self.min_notional:
            return f"skip: {qty} {self.base_asset} below exchange minimums"
        resp = self._order({"symbol": self.symbol, "side": "SELL", "type": "MARKET",
                            "quantity": f"{qty}", "newOrderRespType": "FULL"})
        sold, quote = self._fill_summary(resp)
        if sold <= 0:
            return f"sell returned no fill: {resp.get('status')}"
        fee_quote = sum(float(x["commission"]) for x in resp.get("fills", [])
                        if x.get("commissionAsset") == self.quote_asset)
        self.book.on_sell(sold, quote - fee_quote, quote / sold)
        self.book.save(self.state)
        return f"LIVE SELL {sold:.8f} {self.base_asset} for {quote - fee_quote:.2f} {self.quote_asset}"

    # -- protective stop resting on the exchange ---------------------------
    def sync_stop(self) -> tuple[float, str]:
        """Record a stop that fired while the bot was off, and cancel one still waiting.
        Returns (base qty the stop sold, message)."""
        if not self.book.stop_order_id:
            return 0.0, ""
        o = self._request("GET", "/api/v3/order", {"symbol": self.symbol, "orderId": self.book.stop_order_id},
                          signed=True)
        sold, quote = self._fill_summary(o)
        msg = ""
        if sold > 0:
            self.book.on_sell(sold, quote * 0.999, quote / sold)   # stop fills pay the normal ~0.1% fee
            msg = f"exchange stop SOLD {sold:.8f} {self.base_asset} for ~{quote:.2f} {self.quote_asset} while the bot was off"
        if o.get("status") in {"NEW", "PARTIALLY_FILLED"}:
            self._request("DELETE", "/api/v3/order", {"symbol": self.symbol, "orderId": self.book.stop_order_id},
                          signed=True)
        self.book.stop_order_id = 0
        self.book.save(self.state)
        return sold, msg

    def place_stop(self, stop_price: float) -> str:
        """Leave a stop-loss sell for the bot's coins on the exchange: it works with the computer off."""
        qty = self._floor_qty(min(self.book.qty, self.free_balance(self.base_asset)))
        stop = self._floor_price(stop_price)
        limit = self._floor_price(stop * 0.98)          # fill even if price falls through the stop fast
        if qty < self.min_qty or qty * limit < self.min_notional:
            return f"no exchange stop: {qty} {self.base_asset} is below Binance's minimum order"
        o = self._order({"symbol": self.symbol, "side": "SELL", "type": "STOP_LOSS_LIMIT", "timeInForce": "GTC",
                         "quantity": f"{qty}", "stopPrice": f"{stop}", "price": f"{limit}"})
        self.book.stop_order_id = int(o.get("orderId", 0))
        self.book.save(self.state)
        return f"exchange stop set: sell {qty} {self.base_asset} if price falls to {stop}"
