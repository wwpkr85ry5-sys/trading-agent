from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class AlternativeDataSource(ABC):
    """Abstract base class for alternative data sources (insider, institutional, etc.)."""

    @abstractmethod
    def fetch_insider_trades(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """Fetch insider trading activity."""
        pass

    @abstractmethod
    def fetch_institutional_flows(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """Fetch institutional buying/selling pressure."""
        pass

    @abstractmethod
    def fetch_smart_money_moves(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """Fetch smart money (hedge fund, whale) activity."""
        pass


class QuiverQuantDataSource(AlternativeDataSource):
    """Fetch data from QuiverQuant API (insider trades, government trades, lobbying, etc.)."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or ""
        self.base_url = "https://api.quiverquant.com/beta"
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def fetch_insider_trades(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """Fetch insider trading data from QuiverQuant."""
        logger.info(f"Fetching insider trades for {symbol} from QuiverQuant")
        try:
            # Endpoint: /insider/trades
            endpoint = f"{self.base_url}/insider/trades"
            params = {
                "ticker": symbol,
                "days": lookback_days,
            }
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch insider trades for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_institutional_flows(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """Fetch institutional investor activity from QuiverQuant."""
        logger.info(f"Fetching institutional flows for {symbol} from QuiverQuant")
        try:
            # Endpoint: /institutional/flows (if available)
            endpoint = f"{self.base_url}/institutional/flows"
            params = {
                "ticker": symbol,
                "days": lookback_days,
            }
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch institutional flows for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_smart_money_moves(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """Fetch hedge fund and smart money activity from QuiverQuant."""
        logger.info(f"Fetching smart money moves for {symbol} from QuiverQuant")
        try:
            # Endpoint: /hedge-fund/trades or similar
            endpoint = f"{self.base_url}/hedge-fund/trades"
            params = {
                "ticker": symbol,
                "days": lookback_days,
            }
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch smart money moves for {symbol}: {e}")
            return pd.DataFrame()


class LiquidTradeDataSource(AlternativeDataSource):
    """Fetch data from LiquidTrade API (institutional, smart money, whale activity)."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or ""
        self.base_url = "https://api.liquidtrade.io"
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def fetch_insider_trades(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """Fetch insider trading data from LiquidTrade."""
        logger.info(f"Fetching insider trades for {symbol} from LiquidTrade")
        try:
            endpoint = f"{self.base_url}/insider-trades"
            params = {
                "symbol": symbol,
                "days": lookback_days,
            }
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or "trades" not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data["trades"])
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch insider trades for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_institutional_flows(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """Fetch institutional buying/selling from LiquidTrade."""
        logger.info(f"Fetching institutional flows for {symbol} from LiquidTrade")
        try:
            endpoint = f"{self.base_url}/institutional-flows"
            params = {
                "symbol": symbol,
                "days": lookback_days,
            }
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or "flows" not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data["flows"])
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch institutional flows for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_smart_money_moves(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """Fetch smart money and whale activity from LiquidTrade."""
        logger.info(f"Fetching smart money moves for {symbol} from LiquidTrade")
        try:
            endpoint = f"{self.base_url}/smart-money"
            params = {
                "symbol": symbol,
                "days": lookback_days,
            }
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or "activity" not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data["activity"])
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch smart money moves for {symbol}: {e}")
            return pd.DataFrame()


class AlternativeDataPipeline:
    """Pipeline for fetching and processing alternative data (insider, institutional, smart money)."""

    def __init__(self, source: AlternativeDataSource | None = None):
        self.source = source

    def set_source(self, source: AlternativeDataSource) -> None:
        """Set the alternative data source."""
        self.source = source

    def get_insider_activity(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """Get insider trading activity."""
        if self.source is None:
            logger.warning("No alternative data source configured")
            return pd.DataFrame()
        return self.source.fetch_insider_trades(symbol, lookback_days)

    def get_institutional_flows(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """Get institutional buying/selling activity."""
        if self.source is None:
            logger.warning("No alternative data source configured")
            return pd.DataFrame()
        return self.source.fetch_institutional_flows(symbol, lookback_days)

    def get_smart_money_activity(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """Get smart money and whale activity."""
        if self.source is None:
            logger.warning("No alternative data source configured")
            return pd.DataFrame()
        return self.source.fetch_smart_money_moves(symbol, lookback_days)

    def get_combined_signals(self, symbol: str, lookback_days: int = 90) -> dict:
        """Get combined insider + institutional + smart money signals."""
        insider = self.get_insider_activity(symbol, lookback_days)
        institutional = self.get_institutional_flows(symbol, lookback_days)
        smart_money = self.get_smart_money_activity(symbol, lookback_days)

        return {
            "insider_trades": insider,
            "institutional_flows": institutional,
            "smart_money_activity": smart_money,
            "summary": self._generate_summary(insider, institutional, smart_money),
        }

    def _generate_summary(self, insider: pd.DataFrame, institutional: pd.DataFrame, smart_money: pd.DataFrame) -> dict:
        """Generate summary statistics from all data sources."""
        summary = {
            "insider_count": len(insider),
            "insider_buy_count": 0,
            "insider_sell_count": 0,
            "institutional_count": len(institutional),
            "institutional_buy_volume": 0.0,
            "institutional_sell_volume": 0.0,
            "smart_money_count": len(smart_money),
            "smart_money_buy_volume": 0.0,
            "smart_money_sell_volume": 0.0,
            "insider_sentiment": "neutral",
            "institutional_sentiment": "neutral",
            "smart_money_sentiment": "neutral",
        }

        # Insider sentiment
        if not insider.empty:
            if "transaction_type" in insider.columns:
                summary["insider_buy_count"] = len(insider[insider["transaction_type"] == "BUY"])
                summary["insider_sell_count"] = len(insider[insider["transaction_type"] == "SELL"])
                if summary["insider_buy_count"] > summary["insider_sell_count"]:
                    summary["insider_sentiment"] = "bullish"
                elif summary["insider_sell_count"] > summary["insider_buy_count"]:
                    summary["insider_sentiment"] = "bearish"

        # Institutional sentiment
        if not institutional.empty:
            if "flow_type" in institutional.columns and "volume" in institutional.columns:
                buy_vol = institutional[institutional["flow_type"] == "BUY"]["volume"].sum()
                sell_vol = institutional[institutional["flow_type"] == "SELL"]["volume"].sum()
                summary["institutional_buy_volume"] = float(buy_vol)
                summary["institutional_sell_volume"] = float(sell_vol)
                if buy_vol > sell_vol:
                    summary["institutional_sentiment"] = "bullish"
                elif sell_vol > buy_vol:
                    summary["institutional_sentiment"] = "bearish"

        # Smart money sentiment
        if not smart_money.empty:
            if "trade_type" in smart_money.columns and "size" in smart_money.columns:
                buy_size = smart_money[smart_money["trade_type"] == "BUY"]["size"].sum()
                sell_size = smart_money[smart_money["trade_type"] == "SELL"]["size"].sum()
                summary["smart_money_buy_volume"] = float(buy_size)
                summary["smart_money_sell_volume"] = float(sell_size)
                if buy_size > sell_size:
                    summary["smart_money_sentiment"] = "bullish"
                elif sell_size > buy_size:
                    summary["smart_money_sentiment"] = "bearish"

        return summary


def create_alternative_pipeline(source_type: str = "quiverquant", api_key: str | None = None) -> AlternativeDataPipeline:
    """Factory function to create an alternative data pipeline."""
    if source_type == "quiverquant":
        source = QuiverQuantDataSource(api_key=api_key)
    elif source_type == "liquidtrade":
        source = LiquidTradeDataSource(api_key=api_key)
    else:
        raise ValueError(f"Unknown alternative data source: {source_type}")

    return AlternativeDataPipeline(source)
