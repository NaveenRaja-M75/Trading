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
    pending: list = field(default_factory=list)   # orders sent but not yet confirmed (written BEFORE sending)

    @classmethod
    def load(cls, path: Path) -> "BotBook":
        return cls(**json.loads(path.read_text())) if path.exists() else cls()

    def confirm(self, client_id: str) -> None:
        self.pending = [p for p in self.pending if p.get("id") != client_id]

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
        self.time_offset_ms = 0
        self._sync_clock()
        self._load_filters()

    # -- http -------------------------------------------------------------
    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False):
        params = dict(params or {})
        if signed:
            params["timestamp"] = int(time.time() * 1000) + self.time_offset_ms
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

    def _sync_clock(self) -> None:
        """Binance rejects orders when this computer's clock is more than a few seconds off (common on
        home PCs), so sign with Binance's clock instead of ours."""
        try:
            server = int(self._request("GET", "/api/v3/time")["serverTime"])
            self.time_offset_ms = server - int(time.time() * 1000)
        except Exception:
            self.time_offset_ms = 0

    def _load_filters(self) -> None:
        info = self._request("GET", "/api/v3/exchangeInfo", {"symbol": self.symbol})["symbols"][0]
        self.base_asset, self.quote_asset = info["baseAsset"], info["quoteAsset"]
        f = {x["filterType"]: x for x in info["filters"]}
        self.step = float(f["LOT_SIZE"]["stepSize"])
        self.min_qty = float(f["LOT_SIZE"]["minQty"])
        notional = f.get("NOTIONAL") or f.get("MIN_NOTIONAL") or {}
        self.min_notional = float(notional.get("minNotional", 0.0))
        self.tick = float((f.get("PRICE_FILTER") or {}).get("tickSize", 0.01))

    @staticmethod
    def _decimals(step: float) -> int:
        return max(0, -int(math.floor(math.log10(step)))) if step > 0 else 8

    def _floor_qty(self, qty: float) -> float:
        if self.step <= 0:
            return qty
        return round(math.floor(qty / self.step + 1e-9) * self.step, self._decimals(self.step))

    def _floor_price(self, p: float) -> float:
        return round(math.floor(p / self.tick + 1e-9) * self.tick, self._decimals(self.tick))

    # Binance rejects numbers written like 9e-05, which is how Python prints small amounts
    # (0.00009 BTC is about 8 USDT), so amounts and prices are always written out in full.
    def _fmt_qty(self, qty: float) -> str:
        return f"{qty:.{self._decimals(self.step)}f}"

    def _fmt_price(self, p: float) -> str:
        return f"{p:.{self._decimals(self.tick)}f}"

    def _order(self, params: dict) -> dict:
        """Place an order exactly once. Every order carries its own client id, so after a network
        error we ask Binance whether that order exists instead of blindly sending a second one."""
        params = {**params, "newClientOrderId": params.get("newClientOrderId")
                  or f"jev-{params['side'][0]}-{int(time.time() * 1000)}"}
        # written down before sending: if this program dies right here, the next run asks Binance
        # what happened to this exact order instead of forgetting it (resolve_pending)
        self._cid = params["newClientOrderId"]
        self.book.pending.append({"id": self._cid, "side": params["side"], "type": params["type"]})
        self.book.save(self.state)
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
        return self._balance(asset)[0]

    def _balance(self, asset: str) -> tuple[float, float]:
        """(free, locked in open orders)."""
        acct = self._request("GET", "/api/v3/account", signed=True)
        for b in acct["balances"]:
            if b["asset"] == asset:
                return float(b["free"]), float(b.get("locked", 0.0))
        return 0.0, 0.0

    def last_price(self) -> float:
        return float(self._request("GET", "/api/v3/ticker/price", {"symbol": self.symbol})["price"])

    def _apply_buy(self, resp: dict) -> float:
        qty, quote = self._fill_summary(resp)
        if "fills" in resp:
            fee_base = sum(float(x["commission"]) for x in resp["fills"] if x.get("commissionAsset") == self.base_asset)
        else:                      # looked up after a lost reply: no fee detail, assume the normal 0.1%
            fee_base = qty * 0.001
        net_qty = qty - fee_base
        if net_qty > 0:
            self.book.on_buy(net_qty, quote, quote / qty)
        return net_qty

    def _apply_sell(self, resp: dict) -> tuple[float, float]:
        sold, quote = self._fill_summary(resp)
        if sold <= 0:
            return 0.0, 0.0
        if "fills" in resp:
            fee_quote = sum(float(x["commission"]) for x in resp["fills"] if x.get("commissionAsset") == self.quote_asset)
        else:
            fee_quote = quote * 0.001
        self.book.on_sell(sold, quote - fee_quote, quote / sold)
        return sold, quote - fee_quote

    def resolve_pending(self) -> str:
        """Orders that were written down but never confirmed (crash or lost connection mid-order).
        Ask Binance what became of each and bring the bot's own records in line."""
        msgs = []
        for pend in list(self.book.pending):
            try:
                o = self._request("GET", "/api/v3/order", {"symbol": self.symbol, "origClientOrderId": pend["id"]},
                                  signed=True)
            except RuntimeError as e:
                if "-2013" not in str(e) and "does not exist" not in str(e):
                    raise
                msgs.append(f"an unconfirmed {pend['side']} order from the last run never reached Binance (nothing to fix)")
            else:
                msg = f"recovered an unconfirmed {pend['side']} order from the last run"
                if pend.get("type") == "STOP_LOSS_LIMIT":
                    self.book.stop_order_id = int(o.get("orderId", 0))      # sync_stop deals with it next
                elif pend["side"] == "BUY":
                    msg += f": bought {self._apply_buy(o):.8f} {self.base_asset}"
                else:
                    msg += f": sold {self._apply_sell(o)[0]:.8f} {self.base_asset}"
                msgs.append(msg)
            self.book.confirm(pend["id"])
            self.book.save(self.state)
        return "; ".join(msgs)

    def reconcile(self, price: float) -> str:
        """The bot can only manage coins that are really in the account. If some were sold or moved by
        hand, shrink the bot's records to what is there (booked as sold at today's price)."""
        if self.book.qty <= 0:
            return ""
        free, locked = self._balance(self.base_asset)
        have = free + locked
        if have >= self.book.qty * 0.98:
            return ""
        missing = self.book.qty - have
        self.book.on_sell(missing, missing * price, price)
        self.book.save(self.state)
        return (f"WARNING: the bot bought {have + missing:.8f} {self.base_asset} but the account only holds {have:.8f}. "
                f"Someone sold or moved {missing:.8f} outside the bot; records corrected (counted as sold at {price:.2f}). "
                f"Please do not trade the bot's coins by hand.")

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
        net_qty = self._apply_buy(resp)
        self.book.confirm(self._cid)
        self.book.save(self.state)
        if net_qty <= 0:
            return f"buy returned no fill: {resp.get('status')}"
        return (f"LIVE BUY {net_qty:.8f} {self.base_asset} for "
                f"{float(resp.get('cummulativeQuoteQty', 0.0)):.2f} {self.quote_asset}")

    def sell_qty(self, qty: float, price: float) -> str:
        # never sell more than the bot bought, nor more than is actually free on the account
        qty = self._floor_qty(min(qty, self.book.qty, self.free_balance(self.base_asset)))
        if qty < self.min_qty or qty * price < self.min_notional:
            return f"skip: {self._fmt_qty(qty)} {self.base_asset} below exchange minimums"
        resp = self._order({"symbol": self.symbol, "side": "SELL", "type": "MARKET",
                            "quantity": self._fmt_qty(qty), "newOrderRespType": "FULL"})
        sold, got = self._apply_sell(resp)
        self.book.confirm(self._cid)
        self.book.save(self.state)
        if sold <= 0:
            return f"sell returned no fill: {resp.get('status')}"
        return f"LIVE SELL {sold:.8f} {self.base_asset} for {got:.2f} {self.quote_asset}"

    # -- protective stop resting on the exchange ---------------------------
    def sync_stop(self) -> tuple[float, str]:
        """Record a stop that fired while the bot was off, and cancel one still waiting.
        Returns (base qty the stop sold, message)."""
        if not self.book.stop_order_id:
            return 0.0, ""
        try:
            o = self._request("GET", "/api/v3/order", {"symbol": self.symbol, "orderId": self.book.stop_order_id},
                              signed=True)
        except RuntimeError as e:
            if "-2013" not in str(e) and "does not exist" not in str(e):
                raise
            # Binance no longer knows this order (the testnet is wiped about monthly). reconcile() then
            # checks the coins themselves, so a stop that did sell is still noticed.
            self.book.stop_order_id = 0
            self.book.save(self.state)
            return 0.0, "the old protective stop is no longer on the exchange; a new one will be placed"
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
            return ("no exchange stop possible: this position is too small for Binance's 5 USDT minimum order at the "
                    "stop price. The most it can lose is its own value.")
        o = self._order({"symbol": self.symbol, "side": "SELL", "type": "STOP_LOSS_LIMIT", "timeInForce": "GTC",
                         "quantity": self._fmt_qty(qty), "stopPrice": self._fmt_price(stop),
                         "price": self._fmt_price(limit)})
        self.book.stop_order_id = int(o.get("orderId", 0))
        self.book.confirm(self._cid)
        self.book.save(self.state)
        return f"exchange stop set: sell {self._fmt_qty(qty)} {self.base_asset} if price falls to {self._fmt_price(stop)}"
