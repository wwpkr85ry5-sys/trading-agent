#!/usr/bin/env python
"""CLI for insider/institutional trading signals and portfolio alerts."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta

import pandas as pd

from alternative_data import create_alternative_pipeline
from market_data import create_pipeline
from portfolio_alerts import PortfolioAlertManager
from insider_strategy import InsiderStrategyBacktester
from trading_agent import MLTradingAgent, Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scan_portfolio(args):
    """Scan portfolio for insider/institutional activity."""
    alt_pipeline = create_alternative_pipeline(source_type=args.data_source, api_key=args.api_key)
    alert_manager = PortfolioAlertManager(alt_pipeline)

    symbols = args.symbols.split(",")
    for symbol in symbols:
        alert_manager.add_portfolio_symbol(symbol.strip())

    logger.info(f"Scanning {len(symbols)} symbols for insider/institutional activity...")
    alerts = alert_manager.scan_portfolio(lookback_days=args.lookback)

    # Display alerts
    if alerts:
        print(f"\n{'='*80}")
        print(f"Portfolio Alerts ({len(alerts)} total)")
        print(f"{'='*80}\n")

        for alert in alerts[:10]:  # Show top 10
            print(f"[{alert.sentiment.upper()}] {alert.symbol} - {alert.alert_type}")
            print(f"  Confidence: {alert.confidence:.1%}")
            print(f"  {alert.description}")
            print()
    else:
        print("No significant insider/institutional activity detected.")

    # Export to CSV if requested
    if args.output:
        alert_manager.export_alerts_to_csv(args.output)
        logger.info(f"Alerts exported to {args.output}")


def backtest_with_insiders(args):
    """Backtest strategy using insider signals."""
    # Setup
    market_pipeline = create_pipeline(source_type="yfinance", cache_dir=".cache")
    alt_pipeline = create_alternative_pipeline(source_type=args.data_source, api_key=args.api_key)
    agent = MLTradingAgent(Config())
    backtester = InsiderStrategyBacktester(alt_pipeline, agent)

    # Fetch price data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.lookback)
    prices = market_pipeline.get_prices(args.symbol, start_date, end_date)

    if prices.empty:
        logger.error(f"No price data for {args.symbol}")
        return

    logger.info(f"Backtesting {args.symbol} with insider signals...")
    result = backtester.backtest_with_insider_signals(args.symbol, prices, args.lookback)

    # Display results
    print(f"\n{'='*80}")
    print(f"Backtest Results: {args.symbol}")
    print(f"{'='*80}\n")

    metrics_result = result["metrics"]
    print(f"Sharpe Ratio:     {metrics_result.get('sharpe', 0):.4f}")
    print(f"Annual Return:    {metrics_result.get('ann_return', 0):.2%}")
    print(f"Max Drawdown:     {metrics_result.get('max_drawdown', 0):.2%}")
    print(f"Calmar Ratio:     {metrics_result.get('calmar', 0):.4f}")

    # Export results
    if args.output:
        result["backtest_result"].to_csv(args.output)
        logger.info(f"Backtest results exported to {args.output}")


def compare_strategies(args):
    """Compare ML vs Insider vs Combined strategies."""
    # Setup
    market_pipeline = create_pipeline(source_type="yfinance", cache_dir=".cache")
    alt_pipeline = create_alternative_pipeline(source_type=args.data_source, api_key=args.api_key)
    agent = MLTradingAgent(Config())
    backtester = InsiderStrategyBacktester(alt_pipeline, agent)

    # Fetch price data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.lookback)
    prices = market_pipeline.get_prices(args.symbol, start_date, end_date)

    if prices.empty:
        logger.error(f"No price data for {args.symbol}")
        return

    logger.info(f"Comparing strategies for {args.symbol}...")
    comparison = backtester.compare_strategies(args.symbol, prices, args.lookback)

    # Display comparison
    print(f"\n{'='*80}")
    print(f"Strategy Comparison: {args.symbol}")
    print(f"{'='*80}\n")
    print(comparison.to_string())
    print()

    # Export comparison
    if args.output:
        comparison.to_csv(args.output)
        logger.info(f"Strategy comparison exported to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Insider & Institutional Trading Signals")
    subparsers = parser.add_subparsers(dest="command")

    # Scan portfolio command
    scan_parser = subparsers.add_parser("scan", help="Scan portfolio for insider/institutional activity")
    scan_parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOG", help="Comma-separated symbols")
    scan_parser.add_argument("--lookback", type=int, default=30, help="Lookback days")
    scan_parser.add_argument("--data-source", type=str, default="quiverquant", help="Data source (quiverquant or liquidtrade)")
    scan_parser.add_argument("--api-key", type=str, default=None, help="API key for data source")
    scan_parser.add_argument("--output", type=str, default=None, help="Output CSV file")
    scan_parser.set_defaults(func=scan_portfolio)

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Backtest strategy with insider signals")
    backtest_parser.add_argument("symbol", type=str, help="Stock symbol")
    backtest_parser.add_argument("--lookback", type=int, default=365, help="Lookback days")
    backtest_parser.add_argument("--data-source", type=str, default="quiverquant", help="Data source")
    backtest_parser.add_argument("--api-key", type=str, default=None, help="API key")
    backtest_parser.add_argument("--output", type=str, default=None, help="Output CSV file")
    backtest_parser.set_defaults(func=backtest_with_insiders)

    # Compare strategies command
    compare_parser = subparsers.add_parser("compare", help="Compare ML vs Insider vs Combined strategies")
    compare_parser.add_argument("symbol", type=str, help="Stock symbol")
    compare_parser.add_argument("--lookback", type=int, default=365, help="Lookback days")
    compare_parser.add_argument("--data-source", type=str, default="quiverquant", help="Data source")
    compare_parser.add_argument("--api-key", type=str, default=None, help="API key")
    compare_parser.add_argument("--output", type=str, default=None, help="Output CSV file")
    compare_parser.set_defaults(func=compare_strategies)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
