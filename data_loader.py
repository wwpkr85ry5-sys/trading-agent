from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_loader import load_prices
from trading_agent import Config, MLTradingAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate positions from a price series using the trading-agent package.")
    parser.add_argument("--csv", type=str, default=None, help="Path to a CSV file containing prices.")
    parser.add_argument("--output", type=str, default="positions.csv", help="Output path for the generated positions CSV.")
    args = parser.parse_args()

    if args.csv is None:
        import numpy as np

        prices = pd.Series(
            100.0 * np.exp(np.cumsum(np.random.default_rng(42).normal(0.0005, 0.01, 200))),
            index=pd.date_range("2020-01-01", periods=200, freq="D"),
        )
    else:
        prices = load_prices(Path(args.csv))

    agent = MLTradingAgent(Config())
    position = agent.generate_position(prices)

    out = pd.DataFrame({"price": prices.values, "position": position.values})
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
