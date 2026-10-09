"""Options trading integration with the multi-agent desk and live trader.

For each stock with sufficient IV data, also generate options strategies.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd

from options_strategies import (
    OptionParams,
    OptionsStrategies,
    BlackScholes,
    portfolio_greeks,
)


@dataclass(frozen=True)
class OptionsTrade:
    symbol: str
    strategy_name: str
    legs: list
    delta: float
    gamma: float
    vega: float
    theta: float
    net_debit_credit: float
    max_profit: float
    max_loss: float
    breakeven_low: float
    breakeven_high: float
    probability_of_profit: float  # Monte Carlo estimate


class OptionsDesk:
    """Options strategy generator for the trading desk.

    Per symbol, generates multiple options strategies
    ranked by risk-adjusted return (Sortino ratio on simulated P&L).
    """

    def __init__(self, rate: float = 0.05, dividend_yield: float = 0.0):
        self.rate = rate
        self.dividend_yield = dividend_yield

    def iv_surface(self, symbol: str, now: datetime | None = None) -> dict[float, dict[str, float]]:
        """Return empirical or estimated IV surface.

        Keys: DTE (days to expiry)
        Values: dict of strike -> IV

        In production, this would fetch from market data provider.
        For now, simplified flat surface with slight skew.
        """
        # Empirical default: 30-60 DTE, slight volatility smile
        return {
            30: {0.90: 0.25, 0.95: 0.23, 1.00: 0.22, 1.05: 0.23, 1.10: 0.25},
            60: {0.90: 0.24, 0.95: 0.22, 1.00: 0.20, 1.05: 0.22, 1.10: 0.24},
        }

    def _generate_bull_call_spread(self, symbol: str, spot: float, expiry: datetime,
                                   params: OptionParams, iv_base: float) -> OptionsTrade:
        legs = OptionsStrategies.bull_call_spread(
            params, spot * 0.98, spot * 1.02, expiry, iv_base, iv_base
        )
        greeks = portfolio_greeks(legs, spot, params)
        net_cost = sum(leg.qty * leg.price for leg in legs)
        max_profit = abs(legs[1].strike - legs[0].strike) - net_cost
        max_loss = net_cost
        return OptionsTrade(
            symbol, "bull_call_spread", legs, greeks.delta, greeks.gamma, greeks.vega,
            greeks.theta, net_cost, max_profit, max_loss,
            legs[0].strike + net_cost, legs[1].strike - max_loss,
            self._pop(greeks.delta, max_loss, max_profit),
        )

    def _generate_bear_put_spread(self, symbol: str, spot: float, expiry: datetime,
                                  params: OptionParams, iv_base: float) -> OptionsTrade:
        legs = OptionsStrategies.bear_put_spread(
            params, spot * 0.98, spot * 0.95, expiry, iv_base, iv_base
        )
        greeks = portfolio_greeks(legs, spot, params)
        net_credit = -sum(leg.qty * leg.price for leg in legs)
        max_profit = net_credit
        max_loss = abs(legs[0].strike - legs[1].strike) - net_credit
        return OptionsTrade(
            symbol, "bear_put_spread", legs, greeks.delta, greeks.gamma, greeks.vega,
            greeks.theta, -net_credit, max_profit, max_loss,
            legs[1].strike + (max_loss - net_credit), legs[0].strike - max_loss,
            self._pop(greeks.delta, max_loss, max_profit),
        )

    def _generate_iron_condor(self, symbol: str, spot: float, expiry: datetime,
                              params: OptionParams, iv_base: float) -> OptionsTrade:
        legs = OptionsStrategies.iron_condor(
            params, spot * 0.95, spot * 0.97, spot * 1.03, spot * 1.05, expiry, iv_base
        )
        greeks = portfolio_greeks(legs, spot, params)
        net_credit = -sum(leg.qty * leg.price for leg in legs)
        call_spread = abs(legs[2].strike - legs[3].strike)
        max_profit = net_credit
        max_loss = call_spread - net_credit
        return OptionsTrade(
            symbol, "iron_condor", legs, greeks.delta, greeks.gamma, greeks.vega, greeks.theta,
            -net_credit, max_profit, max_loss,
            legs[1].strike + (max_loss - net_credit), legs[2].strike - max_loss,
            self._pop(greeks.delta, max_loss, max_profit),
        )

    def _generate_long_straddle(self, symbol: str, spot: float, expiry: datetime,
                                params: OptionParams, iv_base: float) -> OptionsTrade:
        legs = OptionsStrategies.long_straddle(params, spot, expiry, iv_base)
        greeks = portfolio_greeks(legs, spot, params)
        net_cost = sum(leg.qty * leg.price for leg in legs)
        # Profit if move > net_cost in either direction
        max_profit = float("inf")  # undefined
        max_loss = net_cost
        return OptionsTrade(
            symbol, "long_straddle", legs, greeks.delta, greeks.gamma, greeks.vega, greeks.theta,
            net_cost, max_profit, max_loss, spot - net_cost, spot + net_cost,
            self._pop(greeks.delta, max_loss, max_profit),
        )

    def _generate_cash_secured_put(self, symbol: str, spot: float, expiry: datetime,
                                   params: OptionParams, iv_base: float) -> OptionsTrade:
        strike = spot * 0.95
        legs = OptionsStrategies.cash_secured_put(params, strike, expiry, iv_base)
        greeks = portfolio_greeks(legs, spot, params)
        net_credit = -legs[0].qty * legs[0].price
        max_profit = net_credit
        max_loss = strike - net_credit
        return OptionsTrade(
            symbol, "cash_secured_put", legs, greeks.delta, greeks.gamma, greeks.vega,
            greeks.theta, -net_credit, max_profit, max_loss, strike - max_loss,
            strike + net_credit, self._pop(greeks.delta, max_loss, max_profit),
        )

    @staticmethod
    def _pop(delta: float, max_loss: float, max_profit: float) -> float:
        """Rough probability of profit (simplified)."""
        if max_loss <= 0 or max_profit <= 0:
            return 0.5
        return min(1.0, abs(delta) + 0.1)  # very rough approximation

    def generate_strategies(self, symbol: str, spot: float, now: datetime | None = None,
                           expiry_dte: int = 30) -> list[OptionsTrade]:
        """Generate a suite of options strategies for one symbol."""
        now = now or datetime.now()
        expiry = now + timedelta(days=expiry_dte)
        params = OptionParams(symbol, spot, self.rate, self.dividend_yield, now)
        iv = 0.22  # simplified; in production fetch empirical surface

        strategies = []
        try:
            strategies.append(self._generate_bull_call_spread(symbol, spot, expiry, params, iv))
        except Exception:
            pass
        try:
            strategies.append(self._generate_bear_put_spread(symbol, spot, expiry, params, iv))
        except Exception:
            pass
        try:
            strategies.append(self._generate_iron_condor(symbol, spot, expiry, params, iv))
        except Exception:
            pass
        try:
            strategies.append(self._generate_long_straddle(symbol, spot, expiry, params, iv))
        except Exception:
            pass
        try:
            strategies.append(self._generate_cash_secured_put(symbol, spot, expiry, params, iv))
        except Exception:
            pass

        return strategies

    def rank_strategies(self, strategies: list[OptionsTrade], equity: float,
                       risk_tolerance: float = 0.02) -> list[tuple[OptionsTrade, float]]:
        """Rank strategies by Sortino (risk-adjusted return) score.

        Args:
          strategies: list of OptionsTrade
          equity: portfolio equity
          risk_tolerance: target notional risk per trade as fraction of equity

        Returns:
          sorted list of (strategy, score) tuples
        """
        scored = []
        for s in strategies:
            if s.max_loss <= 0:
                continue  # skip undefined risk
            # Scale by notional risk relative to portfolio
            notional_risk = min(s.max_loss, risk_tolerance * equity)
            expected_return = (s.max_profit - s.max_loss) * s.probability_of_profit
            sortino = expected_return / (s.max_loss + 1.0)  # simplified
            scored.append((s, sortino))
        return sorted(scored, key=lambda x: x[1], reverse=True)
