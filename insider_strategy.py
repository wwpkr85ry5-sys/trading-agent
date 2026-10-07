from __future__ import annotations

import pandas as pd
from datetime import datetime

from alternative_data import AlternativeDataPipeline
from trading_agent import Config, MLTradingAgent, backtest, metrics


class InsiderStrategyBacktester:
    """Backtest trading strategies using insider/institutional signals."""

    def __init__(self, alt_data_pipeline: AlternativeDataPipeline, agent: MLTradingAgent | None = None):
        self.pipeline = alt_data_pipeline
        self.agent = agent or MLTradingAgent(Config())

    def _generate_insider_signal(self, alt_data: dict, prices: pd.Series) -> pd.Series:
        """
        Generate trading signal based on insider/institutional activity.
        
        Returns a Series with values from -1 (bearish) to 1 (bullish).
        """
        summary = alt_data["summary"]
        signal = pd.Series(0.0, index=prices.index)

        # Insider signal component (weight: 0.3)
        insider_sentiment = {
            "bullish": 1.0,
            "neutral": 0.0,
            "bearish": -1.0,
        }
        insider_signal = insider_sentiment.get(summary["insider_sentiment"], 0.0)
        insider_weight = 0.3 * insider_signal

        # Institutional signal component (weight: 0.35)
        inst_buy = summary["institutional_buy_volume"]
        inst_sell = summary["institutional_sell_volume"]
        inst_total = inst_buy + inst_sell

        if inst_total > 0:
            inst_ratio = (inst_buy - inst_sell) / (inst_total + 1)
            institutional_weight = 0.35 * inst_ratio
        else:
            institutional_weight = 0.0

        # Smart money signal component (weight: 0.35)
        sm_buy = summary["smart_money_buy_volume"]
        sm_sell = summary["smart_money_sell_volume"]
        sm_total = sm_buy + sm_sell

        if sm_total > 0:
            sm_ratio = (sm_buy - sm_sell) / (sm_total + 1)
            smart_money_weight = 0.35 * sm_ratio
        else:
            smart_money_weight = 0.0

        combined_signal = insider_weight + institutional_weight + smart_money_weight
        signal[:] = combined_signal

        return signal

    def backtest_with_insider_signals(
        self,
        symbol: str,
        prices: pd.Series,
        lookback_days: int = 90,
        cfg: Config | None = None,
    ) -> dict:
        """
        Backtest a strategy using insider/institutional signals combined with ML signals.
        
        Returns a dict with backtest results and performance metrics.
        """
        cfg = cfg or Config()

        # Fetch alternative data
        alt_data = self.pipeline.get_combined_signals(symbol, lookback_days)

        # Generate insider-based signal
        insider_signal = self._generate_insider_signal(alt_data, prices)

        # Generate ML-based signal
        ml_signal = self.agent.generate_signal(prices)

        # Combine signals (simple average, can be weighted)
        combined_signal = (insider_signal * 0.4 + ml_signal * 0.6).fillna(0.0)

        # Run backtest
        result = backtest(prices, combined_signal, cfg)

        # Calculate metrics
        m = metrics(result["net"], cfg) if not result["net"].empty else {}

        return {
            "symbol": symbol,
            "backtest_result": result,
            "metrics": m,
            "insider_data": alt_data,
            "insider_signal": insider_signal,
            "ml_signal": ml_signal,
            "combined_signal": combined_signal,
        }

    def compare_strategies(
        self,
        symbol: str,
        prices: pd.Series,
        lookback_days: int = 90,
        cfg: Config | None = None,
    ) -> pd.DataFrame:
        """
        Compare three strategies:
        1. ML-only
        2. Insider/Institutional-only
        3. Combined (ML + Insider/Institutional)
        """
        cfg = cfg or Config()

        # Fetch alternative data
        alt_data = self.pipeline.get_combined_signals(symbol, lookback_days)

        # Strategy 1: ML-only
        ml_signal = self.agent.generate_signal(prices)
        ml_result = backtest(prices, ml_signal, cfg)
        ml_metrics = metrics(ml_result["net"], cfg) if not ml_result["net"].empty else {}

        # Strategy 2: Insider-only
        insider_signal = self._generate_insider_signal(alt_data, prices)
        insider_result = backtest(prices, insider_signal, cfg)
        insider_metrics = metrics(insider_result["net"], cfg) if not insider_result["net"].empty else {}

        # Strategy 3: Combined
        combined_signal = (insider_signal * 0.4 + ml_signal * 0.6).fillna(0.0)
        combined_result = backtest(prices, combined_signal, cfg)
        combined_metrics = metrics(combined_result["net"], cfg) if not combined_result["net"].empty else {}

        # Create comparison DataFrame
        comparison = pd.DataFrame(
            {
                "ML_Only": ml_metrics,
                "Insider_Only": insider_metrics,
                "Combined": combined_metrics,
            }
        ).fillna(0.0)

        return comparison
