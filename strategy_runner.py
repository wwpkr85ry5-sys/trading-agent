from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    symbol: str
    start: str
    end: str
    initial_capital: float = 100_000.0
    fee_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 3.0
    max_leverage: float = 1.0
    vol_target: float = 0.10
    max_turnover: float = 0.50


class StrategyRunner:
    """Run strategy backtests on a set of strategy configurations."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def run(self, cfg: StrategyConfig):
        prices = self.data[cfg.symbol].dropna()
        signal = pd.Series(np.zeros(len(prices)), index=prices.index)
        return pd.DataFrame({"price": prices.values, "signal": signal.values}, index=prices.index)
