# trading-agent

A compact ML-based trading package with:
- price feature engineering
- gradient-boosted signal model
- risk management and turnover controls
- backtesting and walk-forward validation

## Quick start

```python
import pandas as pd
from trading_agent import Config, MLTradingAgent

prices = pd.Series([...], index=pd.date_range("2020-01-01", periods=500, freq="D"))
agent = MLTradingAgent(Config())
position = agent.generate_position(prices)
print(position.head())
```

## Features

- `Config`: strategy and risk settings
- `SignalModel`: short-horizon predictive model
- `RiskController`: volatility scaling, drawdown protections, caps
- `backtest`: transaction-cost-aware backtest engine
- `metrics`: Sharpe, return, drawdown metrics
- `walk_forward`: time-series validation loop

## Notes

This project is a minimal, educational trading package intended for experimentation and testing.
