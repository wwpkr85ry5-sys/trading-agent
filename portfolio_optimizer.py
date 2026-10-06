from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class PortfolioWeights:
    weights: pd.Series
    rebalance_frequency: str = "monthly"


class PortfolioOptimizer:
    """Simple portfolio optimization utilities for mean-variance style allocations."""

    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate

    def _expected_returns(self, returns: pd.DataFrame) -> pd.Series:
        return returns.mean() * 252

    def _covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        return returns.cov() * 252

    def equal_weight(self, symbols: list[str]) -> pd.Series:
        weights = pd.Series(1.0 / len(symbols), index=symbols, dtype=float)
        return weights

    def mean_variance(self, returns: pd.DataFrame, target_vol: float | None = None) -> pd.Series:
        if returns.empty:
            raise ValueError("Returns DataFrame is empty.")

        mu = self._expected_returns(returns)
        cov = self._covariance(returns)

        n = len(mu)
        inv_cov = np.linalg.pinv(cov.to_numpy())
        ones = np.ones(n)

        if target_vol is None:
            weights = inv_cov @ mu.to_numpy()
        else:
            weights = (inv_cov @ ones) / (ones.T @ inv_cov @ ones)
            scale = target_vol / np.sqrt(weights.T @ cov.to_numpy() @ weights)
            weights = weights * scale

        weights = pd.Series(weights, index=returns.columns, dtype=float)
        total = weights.sum()
        if total == 0:
            return self.equal_weight(list(returns.columns))
        return weights / total

    def risk_parity(self, returns: pd.DataFrame) -> pd.Series:
        if returns.empty:
            raise ValueError("Returns DataFrame is empty.")

        vol = returns.std() * np.sqrt(252)
        inv_vol = 1.0 / vol.replace(0, np.nan).fillna(1.0)
        weights = inv_vol / inv_vol.sum()
        return weights.astype(float)
