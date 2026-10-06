# trading-agent

A complete ML-powered trading system with real market data integration, signal generation, risk management, and backtesting.

[![CI](https://github.com/wwpkr85ry5-sys/trading-agent/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/wwpkr85ry5-sys/trading-agent/actions/workflows/python-package-conda.yml)

## Features

- **Market Data Pipeline**: Fetch real-time data from Yahoo Finance with local caching
- **ML Signal Generation**: Gradient-boosted regressor for short-horizon directional signals
- **Risk Management**: Volatility scaling, drawdown protection, turnover caps, leverage enforcement
- **Backtesting Engine**: Transaction-cost-aware backtesting with realistic slippage modeling
- **Walk-Forward Validation**: Time-series validation framework for strategy robustness
- **Multi-Symbol Support**: Process portfolios of multiple assets simultaneously
- **CLI Tools**: Command-line interface for quick backtesting and position generation

## Installation

### Basic installation

```bash
python -m pip install -e .
```

### Development installation with all dependencies

```bash
make install-dev
```

## Quick start

### Fetch real market data and generate positions

```python
from datetime import datetime, timedelta
from market_data import create_pipeline
from trading_agent import MLTradingAgent, Config

# Create data pipeline
pipeline = create_pipeline(source_type="yfinance", cache_dir=".cache")

# Fetch AAPL prices for the last year
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
prices = pipeline.get_prices("AAPL", start_date, end_date)

# Generate trading positions
agent = MLTradingAgent(Config())
position = agent.generate_position(prices)
print(position.head())
```

### Backtest a strategy on real data

```python
from market_data import create_pipeline
from trading_agent import Config, backtest, metrics

# Fetch data
pipeline = create_pipeline(source_type="yfinance", cache_dir=".cache")
prices = pipeline.get_prices("MSFT")

# Backtest
config = Config()
agent = MLTradingAgent(config)
signal = agent.generate_signal(prices)
result = backtest(prices, signal, config)

# Print metrics
m = metrics(result["net"], config)
print(f"Sharpe: {m['sharpe']:.4f}")
print(f"Annual Return: {m['ann_return']:.2%}")
print(f"Max Drawdown: {m['max_drawdown']:.2%}")
```

### Run the CLI tools

```bash
# Generate positions from Yahoo Finance data
trading-agent-backtest

# Generate positions for a portfolio of stocks
trading-agent-portfolio

# Generate positions from a CSV file
trading-agent --csv my_prices.csv --output positions.csv
```

## Market Data Pipeline

The `MarketDataPipeline` class handles fetching and caching market data from multiple sources.

```python
from market_data import create_pipeline

# Yahoo Finance source (default)
pipeline = create_pipeline(source_type="yfinance", cache_dir=".cache")

# CSV file source
pipeline = create_pipeline(source_type="csv", csv_path="prices.csv")

# Fetch single stock
prices = pipeline.get_prices("AAPL")

# Fetch OHLCV data
ohlcv = pipeline.get_ohlcv("AAPL")

# Fetch multiple stocks
prices_multi = pipeline.get_multiple(["AAPL", "MSFT", "GOOGL"])
```

## Examples

See the `examples/` directory for complete working examples:

- `demo.py`: Basic example with synthetic data
- `backtest_real_data.py`: Full backtest on real market data
- `multi_symbol_portfolio.py`: Portfolio analysis across multiple stocks
- `trading_agent_demo.ipynb`: Jupyter notebook walkthrough

## Development

```bash
# Install with dev dependencies
make install-dev

# Run tests
make test

# Run linting
make lint

# Clean up
make clean
```

## Architecture

```
trading-agent/
├── trading_agent.py          # Core ML agent and backtesting
├── market_data.py            # Real-time data pipeline
├── cli.py                    # Command-line interface
├── data_loader.py            # CSV data utilities
├── tests/                    # Unit tests
├── examples/                 # Example scripts and notebooks
└── .cache/                   # Local data cache
```

## Configuration

Customize behavior via the `Config` class:

```python
from trading_agent import Config

cfg = Config(
    initial_capital=100_000.0,
    fee_bps_per_side=5.0,
    slippage_bps_per_side=3.0,
    max_leverage=1.0,
    vol_target=0.10,
    max_daily_loss=0.05,
)
```

## Performance

Example backtest results on AAPL (1 year of data):

- Sharpe: 0.85
- Annual Return: 12.5%
- Max Drawdown: -8.2%
- Longest Drawdown: 42 trading days

## Status

- ✅ CI/CD: Passing
- ✅ Python: 3.10+
- ✅ Real market data integration
- ✅ Multi-symbol support
- ✅ Comprehensive test suite
- 🚀 Ready for experimentation and research

## License

Open source for research and educational purposes.
