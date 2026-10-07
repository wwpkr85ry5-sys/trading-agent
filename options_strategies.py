"""Options strategies for the desk: spreads, straddles, strangles, collars, butterflies, etc.

Each strategy takes:
  - underlying price
  - implied volatility surface
  - risk parameters (delta, gamma, vega targets)

And returns:
  - leg list (strike, expiry, side, qty)
  - greeks (delta, gamma, vega, theta)
  - breakevens, max P&L, risk/reward

Much of this is analytic (Black-Scholes) but we also support
empirical IV surfaces from market data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class OptionParams:
    symbol: str
    spot: float
    rate: float = 0.05
    dividend_yield: float = 0.0
    now: datetime | None = None


@dataclass(frozen=True)
class OptionLeg:
    strike: float
    expiry: datetime
    side: str  # "call" or "put"
    qty: int  # positive = long, negative = short
    price: float  # premium per leg

    def payoff(self, spot: float) -> float:
        if self.side == "call":
            intrinsic = max(spot - self.strike, 0)
        else:
            intrinsic = max(self.strike - spot, 0)
        return self.qty * (intrinsic - self.price)


@dataclass(frozen=True)
class OptionGreeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class BlackScholes:
    """Analytic European option pricing and greeks."""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(d1: float, sigma: float, T: float) -> float:
        return d1 - sigma * np.sqrt(T)

    @staticmethod
    def call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        if T <= 0:
            return max(S - K, 0)
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        d2 = BlackScholes.d2(d1, sigma, T)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        if T <= 0:
            return max(K - S, 0)
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        d2 = BlackScholes.d2(d1, sigma, T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, side: str, q: float = 0.0) -> float:
        if T <= 0:
            return 1.0 if (side == "call" and S > K) or (side == "put" and S < K) else 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        if side == "call":
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return np.exp(-q * T) * (norm.cdf(d1) - 1.0)

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100.0  # per 1% IV change

    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, side: str, q: float = 0.0) -> float:
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        d2 = BlackScholes.d2(d1, sigma, T)
        if side == "call":
            return (
                -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
                + q * S * np.exp(-q * T) * norm.cdf(d1)
            ) / 365.0  # per day
        else:
            return (
                -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
                - q * S * np.exp(-q * T) * norm.cdf(-d1)
            ) / 365.0

    @staticmethod
    def greeks(S: float, K: float, T: float, r: float, sigma: float, side: str, q: float = 0.0) -> OptionGreeks:
        return OptionGreeks(
            delta=BlackScholes.delta(S, K, T, r, sigma, side, q),
            gamma=BlackScholes.gamma(S, K, T, r, sigma, q),
            vega=BlackScholes.vega(S, K, T, r, sigma, q),
            theta=BlackScholes.theta(S, K, T, r, sigma, side, q),
            rho=0.0,  # simplified
        )


class OptionsStrategies:
    """Pre-built option strategies based on market view."""

    @staticmethod
    def bull_call_spread(params: OptionParams, long_strike: float, short_strike: float,
                         expiry: datetime, iv_long: float, iv_short: float) -> list[OptionLeg]:
        """Long call at long_strike, short call at short_strike.
        Max profit: short_strike - long_strike - net_debit
        Max loss: net_debit
        Bullish, defined risk, lower premium.
        """
        long_price = BlackScholes.call(params.spot, long_strike, params._time_to_expiry(expiry),
                                       params.rate, iv_long, params.dividend_yield)
        short_price = BlackScholes.call(params.spot, short_strike, params._time_to_expiry(expiry),
                                        params.rate, iv_short, params.dividend_yield)
        return [
            OptionLeg(long_strike, expiry, "call", 1, long_price),
            OptionLeg(short_strike, expiry, "call", -1, short_price),
        ]

    @staticmethod
    def bear_put_spread(params: OptionParams, long_strike: float, short_strike: float,
                        expiry: datetime, iv_long: float, iv_short: float) -> list[OptionLeg]:
        """Short put at short_strike, long put at long_strike (further OTM).
        Max profit: net_credit
        Max loss: short_strike - long_strike - net_credit
        Bullish, defined risk, credit strategy.
        """
        short_price = BlackScholes.put(params.spot, short_strike, params._time_to_expiry(expiry),
                                       params.rate, iv_short, params.dividend_yield)
        long_price = BlackScholes.put(params.spot, long_strike, params._time_to_expiry(expiry),
                                      params.rate, iv_long, params.dividend_yield)
        return [
            OptionLeg(short_strike, expiry, "put", -1, short_price),
            OptionLeg(long_strike, expiry, "put", 1, long_price),
        ]

    @staticmethod
    def long_straddle(params: OptionParams, strike: float, expiry: datetime, iv: float) -> list[OptionLeg]:
        """Long call + long put at same strike.
        Profit from large move in either direction.
        High premium cost, high gamma (convex).
        Neutral to high-vol.
        """
        call_price = BlackScholes.call(params.spot, strike, params._time_to_expiry(expiry),
                                       params.rate, iv, params.dividend_yield)
        put_price = BlackScholes.put(params.spot, strike, params._time_to_expiry(expiry),
                                     params.rate, iv, params.dividend_yield)
        return [
            OptionLeg(strike, expiry, "call", 1, call_price),
            OptionLeg(strike, expiry, "put", 1, put_price),
        ]

    @staticmethod
    def long_strangle(params: OptionParams, call_strike: float, put_strike: float,
                      expiry: datetime, iv: float) -> list[OptionLeg]:
        """Long call (OTM) + long put (OTM).
        Cheaper than straddle, wider breakevens, less gamma.
        High vol play, undefined max profit.
        """
        call_price = BlackScholes.call(params.spot, call_strike, params._time_to_expiry(expiry),
                                       params.rate, iv, params.dividend_yield)
        put_price = BlackScholes.put(params.spot, put_strike, params._time_to_expiry(expiry),
                                     params.rate, iv, params.dividend_yield)
        return [
            OptionLeg(call_strike, expiry, "call", 1, call_price),
            OptionLeg(put_strike, expiry, "put", 1, put_price),
        ]

    @staticmethod
    def iron_condor(params: OptionParams, put_long: float, put_short: float,
                    call_short: float, call_long: float, expiry: datetime,
                    iv: float) -> list[OptionLeg]:
        """Bear put spread + bull call spread.
        Profit from stock staying in range.
        Defined risk, high probability if wide spreads.
        Net credit strategy.
        """
        return (
            OptionsStrategies.bear_put_spread(params, put_long, put_short, expiry, iv, iv)
            + OptionsStrategies.bull_call_spread(params, call_short, call_long, expiry, iv, iv)
        )

    @staticmethod
    def collar(params: OptionParams, long_put_strike: float, short_call_strike: float,
               expiry: datetime, iv: float) -> list[OptionLeg]:
        """Long stock + long put (protection) + short call (cap).
        Downside protected, upside capped, low/zero net cost.
        Common for hedging equity positions.
        """
        put_price = BlackScholes.put(params.spot, long_put_strike, params._time_to_expiry(expiry),
                                     params.rate, iv, params.dividend_yield)
        call_price = BlackScholes.call(params.spot, short_call_strike, params._time_to_expiry(expiry),
                                       params.rate, iv, params.dividend_yield)
        return [
            OptionLeg(long_put_strike, expiry, "put", 1, put_price),
            OptionLeg(short_call_strike, expiry, "call", -1, call_price),
        ]

    @staticmethod
    def butterfly_spread(params: OptionParams, lower: float, middle: float, upper: float,
                         expiry: datetime, iv: float, side: str = "call") -> list[OptionLeg]:
        """Buy 1 @ lower, sell 2 @ middle, buy 1 @ upper.
        Profit peak @ middle strike.
        Low debit, limited profit, limited loss.
        Neutral, low-vol play.
        """
        if side == "call":
            p_lower = BlackScholes.call(params.spot, lower, params._time_to_expiry(expiry),
                                        params.rate, iv, params.dividend_yield)
            p_middle = BlackScholes.call(params.spot, middle, params._time_to_expiry(expiry),
                                         params.rate, iv, params.dividend_yield)
            p_upper = BlackScholes.call(params.spot, upper, params._time_to_expiry(expiry),
                                        params.rate, iv, params.dividend_yield)
        else:
            p_lower = BlackScholes.put(params.spot, lower, params._time_to_expiry(expiry),
                                       params.rate, iv, params.dividend_yield)
            p_middle = BlackScholes.put(params.spot, middle, params._time_to_expiry(expiry),
                                        params.rate, iv, params.dividend_yield)
            p_upper = BlackScholes.put(params.spot, upper, params._time_to_expiry(expiry),
                                       params.rate, iv, params.dividend_yield)
        return [
            OptionLeg(lower, expiry, side, 1, p_lower),
            OptionLeg(middle, expiry, side, -2, p_middle),
            OptionLeg(upper, expiry, side, 1, p_upper),
        ]

    @staticmethod
    def calendar_spread(params: OptionParams, strike: float, short_expiry: datetime,
                        long_expiry: datetime, iv_short: float, iv_long: float,
                        side: str = "call") -> list[OptionLeg]:
        """Short nearer option, long farther option at same strike.
        Profit from theta decay (short leg) > theta bleed (long leg).
        Neutral, time-decay play.
        """
        if side == "call":
            p_short = BlackScholes.call(params.spot, strike, params._time_to_expiry(short_expiry),
                                        params.rate, iv_short, params.dividend_yield)
            p_long = BlackScholes.call(params.spot, strike, params._time_to_expiry(long_expiry),
                                       params.rate, iv_long, params.dividend_yield)
        else:
            p_short = BlackScholes.put(params.spot, strike, params._time_to_expiry(short_expiry),
                                       params.rate, iv_short, params.dividend_yield)
            p_long = BlackScholes.put(params.spot, strike, params._time_to_expiry(long_expiry),
                                      params.rate, iv_long, params.dividend_yield)
        return [
            OptionLeg(strike, short_expiry, side, -1, p_short),
            OptionLeg(strike, long_expiry, side, 1, p_long),
        ]

    @staticmethod
    def cash_secured_put(params: OptionParams, strike: float, expiry: datetime,
                         iv: float, qty: int = 1) -> list[OptionLeg]:
        """Short put with cash (or margin) reserved.
        Profit from theta, delta (if OTM).
        Defined risk: strike × qty × 100.
        Income strategy.
        """
        put_price = BlackScholes.put(params.spot, strike, params._time_to_expiry(expiry),
                                     params.rate, iv, params.dividend_yield)
        return [OptionLeg(strike, expiry, "put", -qty, put_price)]

    @staticmethod
    def covered_call(params: OptionParams, short_strike: float, expiry: datetime,
                     iv: float) -> list[OptionLeg]:
        """Long stock + short call.
        Profit from stock appreciation (up to strike), plus call premium.
        Income strategy, caps upside.
        """
        call_price = BlackScholes.call(params.spot, short_strike, params._time_to_expiry(expiry),
                                       params.rate, iv, params.dividend_yield)
        return [OptionLeg(short_strike, expiry, "call", -1, call_price)]


class OptionParams(OptionParams):
    def _time_to_expiry(self, expiry: datetime) -> float:
        now = self.now or datetime.now()
        days = (expiry - now).days
        return max(0.0, days / 365.0)


def portfolio_greeks(legs: list[OptionLeg], spot: float, params: OptionParams) -> OptionGreeks:
    """Aggregate greeks across all legs in a strategy."""
    total_delta = 0.0
    total_gamma = 0.0
    total_vega = 0.0
    total_theta = 0.0
    total_rho = 0.0

    for leg in legs:
        T = params._time_to_expiry(leg.expiry)
        greek = BlackScholes.greeks(spot, leg.strike, T, params.rate, 0.20, leg.side,
                                    params.dividend_yield)
        total_delta += leg.qty * greek.delta
        total_gamma += leg.qty * greek.gamma
        total_vega += leg.qty * greek.vega
        total_theta += leg.qty * greek.theta
        total_rho += leg.qty * greek.rho

    return OptionGreeks(total_delta, total_gamma, total_vega, total_theta, total_rho)
