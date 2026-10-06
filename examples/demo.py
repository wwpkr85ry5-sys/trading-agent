import numpy as np
import pandas as pd

from trading_agent import Config, MLTradingAgent


def main() -> None:
    prices = pd.Series(
        100.0 * np.exp(np.cumsum(np.random.default_rng(42).normal(0.0005, 0.01, 250))),
        index=pd.date_range("2020-01-01", periods=250, freq="D"),
    )

    agent = MLTradingAgent(Config())
    position = agent.generate_position(prices)

    result = pd.DataFrame({"price": prices.values, "position": position.values})
    print(result.head())


if __name__ == "__main__":
    main()
