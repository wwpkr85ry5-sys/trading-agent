import logging
from datetime import datetime, timedelta

import pandas as pd

from market_data import create_pipeline
from trading_agent import Config, MLTradingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    # Create data pipeline with caching
    pipeline = create_pipeline(source_type="yfinance", cache_dir=".cache")

    # Fetch data for multiple stocks
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=252)  # 1 year of trading days

    logger.info(f"Fetching data for {len(symbols)} symbols")
    prices_df = pipeline.get_multiple(symbols, start_date, end_date)

    if prices_df.empty:
        logger.error("No data available")
        return

    logger.info(f"Fetched {len(prices_df)} rows of data")

    # Generate positions for each symbol
    cfg = Config()
    positions = {}

    for symbol in symbols:
        try:
            if symbol in prices_df.columns and not prices_df[symbol].empty:
                agent = MLTradingAgent(cfg=cfg)
                position = agent.generate_position(prices_df[symbol].dropna())
                positions[symbol] = position
                logger.info(f"Generated position for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to generate position for {symbol}: {e}")

    # Combine results
    result = pd.DataFrame(positions)
    result.to_csv("multi_symbol_positions.csv")
    logger.info(f"Saved positions to multi_symbol_positions.csv")


if __name__ == "__main__":
    main()
