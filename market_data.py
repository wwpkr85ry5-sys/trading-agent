from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for market data sources."""

    @abstractmethod
    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch market data for a symbol between start and end dates."""
        pass


class YFinanceDataSource(DataSource):
    """Fetch market data from Yahoo Finance."""

    def __init__(self, interval: str = "1d"):
        self.interval = interval

    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        logger.info(f"Fetching {symbol} from {start.date()} to {end.date()} ({self.interval})")
        data = yf.download(symbol, start=start, end=end, interval=self.interval, progress=False)
        if data.empty:
            raise ValueError(f"No data returned for {symbol}")
        return data


class CSVDataSource(DataSource):
    """Load market data from a CSV file."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)

    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        logger.info(f"Loading {symbol} from {self.csv_path}")
        df = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
        df = df[(df.index >= start) & (df.index <= end)]
        if df.empty:
            raise ValueError(f"No data in CSV for date range {start.date()} to {end.date()}")
        return df


class MarketDataPipeline:
    """Pipeline for fetching and processing market data."""

    def __init__(self, source: DataSource, cache_dir: str | Path | None = None):
        self.source = source
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, symbol: str, start: datetime, end: datetime) -> Path:
        """Generate cache file path for a symbol and date range."""
        filename = f"{symbol}_{start.date()}_{end.date()}.csv"
        return self.cache_dir / filename

    def _load_from_cache(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame | None:
        """Load data from cache if available."""
        if not self.cache_dir:
            return None
        cache_path = self._get_cache_path(symbol, start, end)
        if cache_path.exists():
            logger.info(f"Loading {symbol} from cache: {cache_path}")
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return None

    def _save_to_cache(self, data: pd.DataFrame, symbol: str, start: datetime, end: datetime) -> None:
        """Save data to cache."""
        if not self.cache_dir:
            return
        cache_path = self._get_cache_path(symbol, start, end)
        data.to_csv(cache_path)
        logger.info(f"Cached {symbol} to {cache_path}")

    def get_prices(self, symbol: str, start: datetime | None = None, end: datetime | None = None) -> pd.Series:
        """Fetch adjusted close prices for a symbol."""
        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=365)

        # Try cache first
        cached = self._load_from_cache(symbol, start, end)
        if cached is not None:
            return cached["Adj Close"].rename(symbol)

        # Fetch from source
        data = self.source.fetch(symbol, start, end)
        self._save_to_cache(data, symbol, start, end)

        if "Adj Close" in data.columns:
            return data["Adj Close"].rename(symbol)
        elif "Close" in data.columns:
            return data["Close"].rename(symbol)
        else:
            raise ValueError(f"No price column found in data for {symbol}")

    def get_ohlcv(self, symbol: str, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol."""
        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=365)

        # Try cache first
        cached = self._load_from_cache(symbol, start, end)
        if cached is not None:
            return cached

        # Fetch from source
        data = self.source.fetch(symbol, start, end)
        self._save_to_cache(data, symbol, start, end)
        return data

    def get_multiple(self, symbols: list[str], start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        """Fetch prices for multiple symbols and combine into a DataFrame."""
        prices = {}
        for symbol in symbols:
            try:
                prices[symbol] = self.get_prices(symbol, start, end)
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame(prices)


def create_pipeline(source_type: str = "yfinance", **kwargs) -> MarketDataPipeline:
    """Factory function to create a data pipeline."""
    if source_type == "yfinance":
        source = YFinanceDataSource(interval=kwargs.get("interval", "1d"))
    elif source_type == "csv":
        source = CSVDataSource(kwargs.get("csv_path"))
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    return MarketDataPipeline(source, cache_dir=kwargs.get("cache_dir"))
