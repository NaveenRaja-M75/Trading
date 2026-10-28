"""Settings loaded from environment / .env. Safe defaults: paper mode, testnet."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None or v == "" else v.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return default if v is None or v == "" else float(v)


@dataclass(frozen=True)
class Settings:
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    max_position_usdt: float = 50.0
    daily_loss_limit_usdt: float = 5.0
    paper_start_usdt: float = 1000.0
    live_trading: bool = False
    binance_testnet: bool = True
    binance_api_key: str = ""
    binance_api_secret: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    @property
    def cost_per_side(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 1e4

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        e = os.environ.get
        return cls(
            symbol=e("SYMBOL", "BTCUSDT").upper(),
            interval=e("INTERVAL", "1m"),
            fee_bps=_float("FEE_BPS", 10.0),
            slippage_bps=_float("SLIPPAGE_BPS", 2.0),
            max_position_usdt=_float("MAX_POSITION_USDT", 50.0),
            daily_loss_limit_usdt=_float("DAILY_LOSS_LIMIT_USDT", 5.0),
            paper_start_usdt=_float("PAPER_START_USDT", 1000.0),
            live_trading=_bool("LIVE_TRADING", False),
            binance_testnet=_bool("BINANCE_TESTNET", True),
            binance_api_key=e("BINANCE_API_KEY", ""),
            binance_api_secret=e("BINANCE_API_SECRET", ""),
            llm_base_url=e("LLM_BASE_URL", ""),
            llm_api_key=e("LLM_API_KEY", ""),
            llm_model=e("LLM_MODEL", ""),
        )
