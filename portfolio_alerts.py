from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from alternative_data import AlternativeDataPipeline


@dataclass
class InsiderAlert:
    symbol: str
    alert_type: str  # "insider_buy", "insider_sell", "institutional_buy", "smart_money_accumulation", etc.
    sentiment: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 to 1.0
    description: str
    data: dict
    timestamp: datetime


class PortfolioAlertManager:
    """Manages smart alerts based on insider, institutional, and smart money activity."""

    def __init__(self, alt_data_pipeline: AlternativeDataPipeline):
        self.pipeline = alt_data_pipeline
        self.alerts: list[InsiderAlert] = []
        self.portfolio_symbols: list[str] = []

    def add_portfolio_symbol(self, symbol: str) -> None:
        """Add a symbol to monitor."""
        if symbol not in self.portfolio_symbols:
            self.portfolio_symbols.append(symbol)

    def remove_portfolio_symbol(self, symbol: str) -> None:
        """Remove a symbol from monitoring."""
        if symbol in self.portfolio_symbols:
            self.portfolio_symbols.remove(symbol)

    def scan_portfolio(self, lookback_days: int = 30) -> list[InsiderAlert]:
        """Scan all portfolio symbols for insider/institutional activity."""
        self.alerts = []

        for symbol in self.portfolio_symbols:
            signals = self.pipeline.get_combined_signals(symbol, lookback_days)
            alerts = self._generate_alerts_from_signals(symbol, signals)
            self.alerts.extend(alerts)

        # Sort by confidence descending
        self.alerts.sort(key=lambda x: x.confidence, reverse=True)
        return self.alerts

    def _generate_alerts_from_signals(self, symbol: str, signals: dict) -> list[InsiderAlert]:
        """Generate alerts from insider/institutional signals."""
        alerts = []
        summary = signals["summary"]

        # Insider activity alerts
        if summary["insider_sentiment"] == "bullish":
            confidence = min(summary["insider_buy_count"] / max(1, summary["insider_count"]), 1.0)
            alerts.append(
                InsiderAlert(
                    symbol=symbol,
                    alert_type="insider_buy_accumulation",
                    sentiment="bullish",
                    confidence=confidence,
                    description=f"Insiders buying: {summary['insider_buy_count']} buys vs {summary['insider_sell_count']} sells",
                    data=summary,
                    timestamp=datetime.now(),
                )
            )

        elif summary["insider_sentiment"] == "bearish":
            confidence = min(summary["insider_sell_count"] / max(1, summary["insider_count"]), 1.0)
            alerts.append(
                InsiderAlert(
                    symbol=symbol,
                    alert_type="insider_sell_signal",
                    sentiment="bearish",
                    confidence=confidence,
                    description=f"Insiders selling: {summary['insider_sell_count']} sells vs {summary['insider_buy_count']} buys",
                    data=summary,
                    timestamp=datetime.now(),
                )
            )

        # Institutional activity alerts
        inst_buy = summary["institutional_buy_volume"]
        inst_sell = summary["institutional_sell_volume"]
        inst_total = inst_buy + inst_sell

        if inst_total > 0:
            if inst_buy > inst_sell * 1.5:  # 50% more buys
                confidence = min(inst_buy / (inst_total + 1), 1.0)
                alerts.append(
                    InsiderAlert(
                        symbol=symbol,
                        alert_type="institutional_accumulation",
                        sentiment="bullish",
                        confidence=confidence,
                        description=f"Institutional buying: ${inst_buy:,.0f} in buys vs ${inst_sell:,.0f} in sells",
                        data=summary,
                        timestamp=datetime.now(),
                    )
                )

            elif inst_sell > inst_buy * 1.5:  # 50% more sells
                confidence = min(inst_sell / (inst_total + 1), 1.0)
                alerts.append(
                    InsiderAlert(
                        symbol=symbol,
                        alert_type="institutional_distribution",
                        sentiment="bearish",
                        confidence=confidence,
                        description=f"Institutional selling: ${inst_sell:,.0f} in sells vs ${inst_buy:,.0f} in buys",
                        data=summary,
                        timestamp=datetime.now(),
                    )
                )

        # Smart money activity alerts
        sm_buy = summary["smart_money_buy_volume"]
        sm_sell = summary["smart_money_sell_volume"]
        sm_total = sm_buy + sm_sell

        if sm_total > 0:
            if sm_buy > sm_sell * 2.0:  # 2x more buys
                confidence = min(sm_buy / (sm_total + 1), 1.0)
                alerts.append(
                    InsiderAlert(
                        symbol=symbol,
                        alert_type="smart_money_accumulation",
                        sentiment="bullish",
                        confidence=confidence,
                        description=f"Smart money (hedge funds, whales) accumulating: ${sm_buy:,.0f} in positions",
                        data=summary,
                        timestamp=datetime.now(),
                    )
                )

            elif sm_sell > sm_buy * 2.0:  # 2x more sells
                confidence = min(sm_sell / (sm_total + 1), 1.0)
                alerts.append(
                    InsiderAlert(
                        symbol=symbol,
                        alert_type="smart_money_distribution",
                        sentiment="bearish",
                        confidence=confidence,
                        description=f"Smart money liquidating positions: ${sm_sell:,.0f} in sales",
                        data=summary,
                        timestamp=datetime.now(),
                    )
                )

        return alerts

    def get_bullish_alerts(self) -> list[InsiderAlert]:
        """Get all bullish alerts."""
        return [a for a in self.alerts if a.sentiment == "bullish"]

    def get_bearish_alerts(self) -> list[InsiderAlert]:
        """Get all bearish alerts."""
        return [a for a in self.alerts if a.sentiment == "bearish"]

    def get_high_confidence_alerts(self, min_confidence: float = 0.7) -> list[InsiderAlert]:
        """Get alerts above confidence threshold."""
        return [a for a in self.alerts if a.confidence >= min_confidence]

    def export_alerts_to_csv(self, filename: str = "portfolio_alerts.csv") -> None:
        """Export alerts to CSV."""
        data = [
            {
                "timestamp": a.timestamp.isoformat(),
                "symbol": a.symbol,
                "alert_type": a.alert_type,
                "sentiment": a.sentiment,
                "confidence": a.confidence,
                "description": a.description,
            }
            for a in self.alerts
        ]
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
