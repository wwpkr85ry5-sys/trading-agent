# Insider & Institutional Trading Module

## Features

- **Alternative Data Integration**: Connect to QuiverQuant and LiquidTrade APIs for insider/institutional data
- **Portfolio Alerts**: Real-time alerts for insider buying, institutional flows, and smart money activity
- **Strategy Backtesting**: Test trading strategies using insider signals
- **Multi-Strategy Comparison**: Compare ML-only vs Insider-only vs Combined strategies

## Installation

```bash
pip install requests pandas
pip install -e .
```

## Usage

### Scan Portfolio for Insider Activity

```bash
python insider_cli.py scan --symbols "AAPL,MSFT,GOOG" --lookback 30 --output alerts.csv
```

### Backtest with Insider Signals

```bash
python insider_cli.py backtest AAPL --lookback 365 --output backtest_results.csv
```

### Compare Strategies

```bash
python insider_cli.py compare AAPL --lookback 365 --output strategy_comparison.csv
```

## Python API

### QuiverQuant Integration

```python
from alternative_data import create_alternative_pipeline

alt_pipeline = create_alternative_pipeline(source_type="quiverquant", api_key="your_key")

# Get insider trades
insider_trades = alt_pipeline.get_insider_activity("AAPL")
print(insider_trades.head())

# Get institutional flows
inst_flows = alt_pipeline.get_institutional_flows("AAPL")
print(inst_flows.head())

# Get smart money activity
smart_money = alt_pipeline.get_smart_money_activity("AAPL")
print(smart_money.head())

# Get combined signals
combined = alt_pipeline.get_combined_signals("AAPL", lookback_days=90)
print(combined["summary"])
```

### Portfolio Alerts

```python
from portfolio_alerts import PortfolioAlertManager

alert_manager = PortfolioAlertManager(alt_pipeline)
alert_manager.add_portfolio_symbol("AAPL")
alert_manager.add_portfolio_symbol("MSFT")

alerts = alert_manager.scan_portfolio(lookback_days=30)

# Get bullish alerts
bullish = alert_manager.get_bullish_alerts()
for alert in bullish:
    print(f"{alert.symbol}: {alert.description}")

# Get high-confidence alerts
high_conf = alert_manager.get_high_confidence_alerts(min_confidence=0.8)
for alert in high_conf:
    print(f"[{alert.confidence:.1%}] {alert.alert_type}")

# Export to CSV
alert_manager.export_alerts_to_csv("alerts.csv")
```

### Insider Strategy Backtesting

```python
from insider_strategy import InsiderStrategyBacktester
from market_data import create_pipeline

market_pipeline = create_pipeline(source_type="yfinance", cache_dir=".cache")
alt_pipeline = create_alternative_pipeline(source_type="quiverquant", api_key="your_key")

backtester = InsiderStrategyBacktester(alt_pipeline)

prices = market_pipeline.get_prices("AAPL")
result = backtester.backtest_with_insider_signals("AAPL", prices)

print(f"Sharpe: {result['metrics']['sharpe']:.4f}")
print(f"Annual Return: {result['metrics']['ann_return']:.2%}")
print(f"Max Drawdown: {result['metrics']['max_drawdown']:.2%}")

# Compare strategies
comparison = backtester.compare_strategies("AAPL", prices)
print(comparison)
```

## Data Sources

### QuiverQuant

Endpoints:
- `/beta/insider/trades` - Insider trading activity
- `/beta/institutional/flows` - Institutional buying/selling
- `/beta/hedge-fund/trades` - Hedge fund activity

**Sign up**: https://www.quiverquant.com/

### LiquidTrade

Endpoints:
- `/insider-trades` - Insider trading data
- `/institutional-flows` - Institutional flows
- `/smart-money` - Whale/hedge fund activity

**Sign up**: https://liquidtrade.io/

## Alert Types

- `insider_buy_accumulation` - Insiders accumulating shares (bullish)
- `insider_sell_signal` - Insiders liquidating (bearish)
- `institutional_accumulation` - Institutions buying (bullish)
- `institutional_distribution` - Institutions selling (bearish)
- `smart_money_accumulation` - Whales/hedge funds buying (bullish)
- `smart_money_distribution` - Whales/hedge funds selling (bearish)

## Backtesting

The backtester combines three signal sources:

1. **ML Signal** (60% weight) - From your trading_agent.py
2. **Insider Signal** (40% weight) - From insider/institutional data
3. **Combined** - Weighted average for best results

You can compare strategy performance and optimize weights based on historical data.

## Next Steps

1. Sign up for QuiverQuant or LiquidTrade API keys
2. Configure your portfolio symbols
3. Run portfolio scans regularly
4. Backtest with insider signals
5. Deploy alerts to your trading system
