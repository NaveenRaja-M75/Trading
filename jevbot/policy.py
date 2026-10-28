"""Policy: turns signal scores into target positions. Deterministic code decides; signals only advise."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Policy:
    enter: float = 0.5        # score needed to open a position
    exit: float = 0.0         # score at which an open long closes (hysteresis cuts churn)
    min_hold: int = 3         # bars to hold before allowing an exit
    allow_short: bool = False  # spot accounts can't short

    def targets(self, scores: pd.Series) -> pd.Series:
        s = scores.to_numpy(dtype=float)
        pos = np.zeros(len(s))
        cur, held = 0.0, 0
        for i, x in enumerate(s):
            held += 1
            if cur == 0:
                if x >= self.enter:
                    cur, held = 1.0, 0
                elif self.allow_short and x <= -self.enter:
                    cur, held = -1.0, 0
            elif held >= self.min_hold:
                if cur > 0 and x <= self.exit:
                    cur, held = 0.0, 0
                elif cur < 0 and x >= -self.exit:
                    cur, held = 0.0, 0
            pos[i] = cur
        return pd.Series(pos, index=scores.index)


@dataclass
class RiskGate:
    """Hard limits checked before every live/paper order. Trips stay tripped until restart."""
    max_position_usdt: float
    daily_loss_limit_usdt: float
    max_data_age_s: float = 180.0
    tripped: str = ""

    def check(self, day_pnl_usdt: float, data_age_s: float) -> str:
        if self.tripped:
            return self.tripped
        if day_pnl_usdt <= -abs(self.daily_loss_limit_usdt):
            self.tripped = f"daily loss limit hit ({day_pnl_usdt:.2f} USDT)"
            return self.tripped
        if data_age_s > self.max_data_age_s:
            return f"stale market data ({data_age_s:.0f}s old)"  # transient: don't latch
        return ""
