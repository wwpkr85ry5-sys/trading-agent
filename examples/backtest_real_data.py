from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

from market_data import create_pipeline
from trading_agent import Config, MLTradingAgent, backtest, metrics, walk_forward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    # Create data pipeline
    pipeline = create_pipeline(source_type="yfinance", cache_dir=".cache")

    # Fetch data for a stock
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    logger.info(f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}")
    prices = pipeline.get_prices(symbol, start_date, end_date)

    if prices.empty:
        logger.error(f"No data available for {symbol}")
        return

    logger.info(f"Fetched {len(prices)} price points for {symbol}")

    # Generate trading signals and positions
    cfg = Config()
    agent = MLTradingAgent(cfg=cfg)
    position = agent.generate_position(prices)

    logger.info(f"Generated {len(position)} positions")

    # Run backtest
    signal = agent.generate_signal(prices)
    result = backtest(prices, signal, cfg)

    logger.info(f"Backtest complete. Final equity: {result['equity'].iloc[-1]:.2f}")

    # Compute metrics
    m = metrics(result["net"], cfg)
    logger.info(f"Metrics: Sharpe={m['sharpe']:.4f}, Return={m['ann_return']:.2%}, MaxDD={m['max_drawdown']:.2%}")

    # Save results
    output = pd.DataFrame(
        {
            "price": prices.values,
            "signal": signal.values,
            "position": position.values,
            "equity": result["equity"].values,
        },
        index=prices.index,
    )

    output_file = f"{symbol}_backtest_results.csv"
    output.to_csv(output_file)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
