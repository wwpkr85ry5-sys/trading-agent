"""Alpaca options trading integration.

Extends live_trader.py to also place options orders.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import requests

from options_desk import OptionsTrade, OptionsDesk


class OptionsTrader:
    """Execute options trades via Alpaca.

    Alpaca supports:
      - Simple options orders (buy/sell single leg)
      - Multi-leg orders (spreads, etc.)
    
    We prioritize spreads (defined risk) over naked calls/puts.
    """

    def __init__(self, base_url: str, session: requests.Session):
        self.base_url = base_url
        self.session = session

    def _req(self, method: str, path: str, **kw):
        r = self.session.request(method, self.base_url + path, timeout=15, **kw)
        r.raise_for_status()
        return r.json() if r.content else None

    def option_chain(self, symbol: str) -> dict:
        """Fetch option chain for a symbol."""
        return self._req("GET", f"/option_chains", params={"underlying_symbol": symbol})

    def place_options_order(self, symbol: str, strategy: OptionsTrade, qty: int = 1):
        """Place multi-leg options order.

        Args:
          symbol: underlying symbol
          strategy: OptionsTrade with legs
          qty: number of contracts (100 shares per contract)

        Returns:
          order response from Alpaca
        """
        legs = []
        for leg in strategy.legs:
            legs.append({
                "symbol": f"{symbol}{leg.expiry.strftime('%y%m%d')}{leg.side[0].upper()}{int(leg.strike * 1000)}",
                "side": "buy" if leg.qty > 0 else "sell",
                "quantity": int(abs(leg.qty) * qty),
                "type": "market",
                "time_in_force": "day",
            })
        return self._req("POST", "/orders", json={"legs": legs, "type": "multi_leg"})

    def close_options_position(self, symbol: str, expiry: datetime):
        """Close all options for a symbol/expiry combo."""
        pattern = f"{symbol}{expiry.strftime('%y%m%d')}*"
        positions = self._req("GET", "/positions", params={"symbol": pattern})
        for pos in positions or []:
            self._req("DELETE", f"/positions/{pos['symbol']}")


class HybridDesk:
    """Combines equity and options strategies from the multi-agent desk.

    For each symbol:
      1. Get equity signal (multi-agent desk)
      2. Get options signals (options desk)
      3. Decide: equity, options, or both
      4. Execute via live trader
    """

    def __init__(self, equity_weight: float = 0.6, options_weight: float = 0.4):
        self.eq_w = equity_weight
        self.opt_w = options_weight
        self.opts_desk = OptionsDesk()

    def hybrid_signal(self, symbol: str, spot: float, equity_signal: float,
                     equity: float, expiry_dte: int = 30) -> dict:
        """Combine equity + options signals.

        Returns:
          {"equity_position": float,  # [-1, 1]
           "options_strategy": OptionsTrade or None,
           "options_qty": int,
           "combined_delta": float}
        """
        # Equity component
        eq_pos = equity_signal * self.eq_w

        # Options component: only if high conviction
        opts_strat = None
        opts_qty = 0
        opts_delta = 0.0

        if abs(equity_signal) > 0.5:  # high conviction
            strategies = self.opts_desk.generate_strategies(symbol, spot)
            ranked = self.opts_desk.rank_strategies(strategies, equity)
            if ranked:
                best_strat, score = ranked[0]
                if score > 0.1:  # positive expected return
                    opts_strat = best_strat
                    # Size by notional risk
                    notional = min(best_strat.max_loss, equity * 0.02)
                    opts_qty = max(1, int(notional / (best_strat.max_loss / 100.0)))
                    opts_delta = best_strat.delta * opts_qty * self.opt_w

        return {
            "equity_position": eq_pos,
            "options_strategy": opts_strat,
            "options_qty": opts_qty,
            "combined_delta": eq_pos + opts_delta,
        }
