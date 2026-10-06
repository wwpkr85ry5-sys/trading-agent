# trading-agent

A compact ML-powered trading package for generating signals, enforcing risk limits, and running backtests.

[![CI](https://github.com/wwpkr85ry5-sys/trading-agent/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/wwpkr85ry5-sys/trading-agent/actions/workflows/python-package-conda.yml)

## Features

- ML signal generation using a gradient-boosted regressor
- Volatility scaling and drawdown protection
- Turnover caps and leverage enforcement
- Transaction-cost-aware backtesting
- Walk-forward validation
- Simple CLI for generating position output from CSV data

## Installation

```bash
python -m pip install -e .
```

or with pip requirements:

```bash
python -m pip install -r requirements.txt
```

## Quick start

```python
import pandas as pd
from trading_agent import Config, MLTradingAgent

prices = pd.Series(
    [100.0, 101.2, 102.5, 101.8, 103.1],
    index=pd.date_range("2020-01-01", periods=5, freq="D"),
)

agent = MLTradingAgent(Config())
position = agent.generate_position(prices)
print(position)
```

## Backtesting

```python
import pandas as pd
from trading_agent import Config, backtest

prices = pd.Series([100, 101, 102, 101, 103], index=pd.date_range("2020-01-01", periods=5, freq="D"))
signal = pd.Series([0.0, 1.0, 1.0, 0.0, 0.5], index=prices.index)

result = backtest(prices, signal, Config())
print(result.tail())
```

## Walk-forward validation

```python
import pandas as pd
from trading_agent import Config, walk_forward

prices = pd.Series(
    [100.0 + i * 0.25 for i in range(250)],
    index=pd.date_range("2020-01-01", periods=250, freq="D"),
)

def fit_fn(train):
    return lambda x: pd.Series(0.5, index=x.index, dtype=float)

output = walk_forward(prices, fit_fn, Config(), train_days=30, test_days=10, embargo=1)
print(output["oos_metrics"])
```

## CLI usage

You can run the package from the command line using a CSV file containing a price series:

```bash
python -m trading_agent --csv data/prices.csv --output positions.csv
```

The CSV can contain either:
- a single `price` column, or
- a single numeric column with no header

Example:

```csv
price
100.0
101.2
102.5
101.8
103.1
```

## Example script

See `examples/demo.py` for a full example using synthetic prices and a generated position curve.

## Project status

- CI: passing
- Python: 3.10+
- Dependencies: NumPy, pandas, scikit-learn, pytest

## License

This project is distributed as open source for research and experimentation.
