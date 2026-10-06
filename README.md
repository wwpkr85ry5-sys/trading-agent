# trading-agent

A complete ML-powered trading system with real market data integration, signal generation, risk management, backtesting, and portfolio tooling.

[![CI](https://github.com/wwpkr85ry5-sys/trading-agent/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/wwpkr85ry5-sys/trading-agent/actions/workflows/python-package-conda.yml)

## Features

- **Real market data pipeline** with Yahoo Finance and CSV support
- **Local caching** for repeated fetches
- **ML signal generation** using a gradient-boosted regressor
- **Risk controls** for leverage, turnover, and drawdowns
- **Backtest engine** with transaction costs
- **Portfolio optimization utilities**
- **Parameter search utilities**
- **Reporting and strategy runner modules**

## Installation

```bash
python -m pip install -e .
```

## Quick Start

```python
from datetime import datetime, timedelta
from trading_agent import MLTradingAgent, Config, create_pipeline

pipeline = create_pipeline(source_type="yfinance", cache_dir=".cache")
end = datetime.now()
start = end - timedelta(days=365)

prices = pipeline.get_prices("AAPL", start, end)

agent = MLTradingAgent(Config())
position = agent.generate_position(prices)
print(position.head())
```

## Portfolio Optimization

```python
from trading_agent import PortfolioOptimizer

opt = PortfolioOptimizer()
weights = opt.equal_weight(["AAPL", "MSFT", "GOOG"])
print(weights)
```

## Parameter Search

```python
from trading_agent import ParameterGrid, ParameterSearch

grid = ParameterGrid({
    "max_leverage": [0.5, 1.0, 2.0],
    "vol_target": [0.05, 0.10, 0.15],
})

search = ParameterSearch(grid)

def evaluator(**kwargs):
    return kwargs["max_leverage"] + kwargs["vol_target"]

results = search.search(evaluator)
print(results)
```

## Reporting

```python
from trading_agent import PerformanceReport
import pandas as pd

report = PerformanceReport(output_dir="reports")
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
report.export(df, "portfolio_report.csv")
```

## CLI

```bash
python -m trading_agent --csv my_prices.csv --output positions.csv
```

## Example scripts

- `examples/demo.py`
- `examples/backtest_real_data.py`
- `examples/multi_symbol_portfolio.py`

## Development

```bash
make install-dev
make test
make lint
```

## License

Open source for research and experimentation.
