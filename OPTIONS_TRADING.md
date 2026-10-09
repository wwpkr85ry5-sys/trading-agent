# Options Trading Module

## Features

- **Black-Scholes Pricing** - Analytic European option pricing and greeks (delta, gamma, vega, theta)
- **Pre-built Strategies** - Bull call spread, bear put spread, iron condor, straddle, strangle, collar, butterfly, calendar spread, cash-secured put, covered call
- **Portfolio Greeks** - Aggregate delta, gamma, vega, theta across multi-leg strategies
- **IV Surface** - Empirical volatility surface with smile/skew (placeholder for real market data)
- **Strategy Ranking** - Sortino ratio (risk-adjusted return) scoring
- **Alpaca Integration** - Place multi-leg options orders via Alpaca API
- **Hybrid Desk** - Combine equity + options signals for optimal allocation

## Installation

```bash
pip install numpy scipy requests
```

## Quick Start

### Generate Options Strategies

```python
from options_desk import OptionsDesk
from datetime import datetime

desk = OptionsDesk(rate=0.05, dividend_yield=0.0)

# Generate suite of strategies for AAPL
strategies = desk.generate_strategies(
    symbol="AAPL",
    spot=180.0,
    expiry_dte=30  # 30 days to expiry
)

for s in strategies:
    print(f"{s.strategy_name}:")
    print(f"  Delta: {s.delta:.3f}")
    print(f"  Max Profit: ${s.max_profit:.2f}")
    print(f"  Max Loss: ${s.max_loss:.2f}")
    print(f"  P(Profit): {s.probability_of_profit:.1%}")
    print()

# Rank by Sortino ratio
ranked = desk.rank_strategies(strategies, equity=100_000)
for strategy, score in ranked:
    print(f"{strategy.strategy_name}: {score:.3f}")
```

### Greeks & Pricing

```python
from options_strategies import BlackScholes, OptionParams, OptionsStrategies
from datetime import datetime, timedelta

# Single option pricing
spot, strike, expiry_dte = 100.0, 105.0, 30
T = expiry_dte / 365.0
iv = 0.25

call_price = BlackScholes.call(spot, strike, T, 0.05, iv)
put_price = BlackScholes.put(spot, strike, T, 0.05, iv)

print(f"Call: ${call_price:.2f}")
print(f"Put: ${put_price:.2f}")

# Greeks
greeks = BlackScholes.greeks(spot, strike, T, 0.05, iv, "call")
print(f"Delta: {greeks.delta:.3f}")
print(f"Gamma: {greeks.gamma:.5f}")
print(f"Vega: {greeks.vega:.3f}")
print(f"Theta: {greeks.theta:.3f}")
```

### Pre-built Strategies

```python
from options_strategies import OptionParams, OptionsStrategies
from datetime import datetime, timedelta

now = datetime.now()
params = OptionParams(
    symbol="AAPL",
    spot=180.0,
    rate=0.05,
    dividend_yield=0.0,
    now=now
)

expiry = now + timedelta(days=30)

# Bull call spread
legs = OptionsStrategies.bull_call_spread(
    params,
    long_strike=178.0,
    short_strike=182.0,
    expiry=expiry,
    iv_long=0.20,
    iv_short=0.20
)

for leg in legs:
    print(f"Strike: {leg.strike}, Side: {leg.side}, Qty: {leg.qty}, Price: ${leg.price:.2f}")

# Iron condor
legs = OptionsStrategies.iron_condor(
    params,
    put_long=174.0,
    put_short=176.0,
    call_short=184.0,
    call_long=186.0,
    expiry=expiry,
    iv=0.20
)

net_credit = -sum(leg.qty * leg.price for leg in legs)
print(f"Net Credit: ${net_credit:.2f}")
```

### Hybrid Desk (Equity + Options)

```python
from options_trader import HybridDesk

desk = HybridDesk(equity_weight=0.6, options_weight=0.4)

# Equity signal from multi-agent desk
equity_signal = 0.8  # strong bullish

# Compute hybrid (equity + options)
result = desk.hybrid_signal(
    symbol="AAPL",
    spot=180.0,
    equity_signal=equity_signal,
    equity=100_000,
    expiry_dte=30
)

print(f"Equity Position: {result['equity_position']:.3f}")
if result['options_strategy']:
    print(f"Options Strategy: {result['options_strategy'].strategy_name}")
    print(f"Options Qty: {result['options_qty']}")
print(f"Combined Delta: {result['combined_delta']:.3f}")
```

### Live Alpaca Execution

```python
from options_trader import OptionsTrader
import requests

base_url = "https://paper-api.alpaca.markets/v2"
session = requests.Session()
session.headers.update({
    "APCA-API-KEY-ID": os.environ["ALPACA_KEY_ID"],
    "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"],
})

trader = OptionsTrader(base_url, session)

# Get option chain
chain = trader.option_chain("AAPL")
print(chain)

# Place multi-leg options order
result = trader.place_options_order(
    symbol="AAPL",
    strategy=best_strategy,
    qty=1  # 1 contract = 100 shares
)
print(result)
```

## Strategy Guide

### Bullish Strategies

- **Bull Call Spread**: Long call + short call (lower strike). Max profit: spread width - debit. Defined risk.
- **Bear Put Spread**: Short put + long put (lower strike). Max profit: credit. Defined risk, high probability.
- **Cash-Secured Put**: Short put with capital reserved. Income play, defined risk.
- **Long Call**: Buy call. Unlimited upside, limited loss (premium).

### Bearish Strategies

- **Bear Call Spread**: Long call + short call (higher strike). Defined risk.
- **Bull Put Spread**: Short put + long put (higher strike). Income, defined risk.
- **Long Put**: Buy put. Unlimited downside profit, limited loss.

### Neutral/Volatility Strategies

- **Iron Condor**: Bear put spread + bull call spread. High probability, defined risk, range-bound.
- **Long Straddle**: Long call + long put at same strike. Profit from large move. High premium.
- **Long Strangle**: Long call (OTM) + long put (OTM). Cheaper straddle, wider breakevens.
- **Butterfly Spread**: Buy 1 lower, sell 2 middle, buy 1 upper. Peak profit at middle. Low debit.
- **Calendar Spread**: Short near, long far (same strike). Profit from theta decay. Neutral.

### Income/Hedge Strategies

- **Covered Call**: Long stock + short call. Capped upside, premium income.
- **Collar**: Long stock + long put + short call. Protect downside, cap upside, low cost.

## Integration with Multi-Agent Desk

```python
from multi_agent import MultiAgentDesk, load_ohlc
from options_desk import OptionsDesk
from options_trader import HybridDesk

# Get equity signal
desk = MultiAgentDesk()
analysis = desk.analyze("AAPL", "stock", load_ohlc("AAPL"))
eq_signal = analysis["position"]
eq_action = analysis["action"]

# Get options signal
opts_desk = OptionsDesk()
strategies = opts_desk.generate_strategies("AAPL", spot=180.0)
ranked = opts_desk.rank_strategies(strategies, equity=100_000)

# Combine
hybrid_desk = HybridDesk(equity_weight=0.6, options_weight=0.4)
result = hybrid_desk.hybrid_signal("AAPL", 180.0, eq_signal, 100_000)

# Execute
if result["options_strategy"]:
    # trade options strategy
    pass
else:
    # trade equity only
    pass
```

## Greeks & Risk Management

- **Delta**: Directional exposure. +1 = long stock equivalent, -1 = short stock equivalent.
- **Gamma**: Delta acceleration. High gamma = convex, profits from moves either direction.
- **Vega**: IV exposure. +Vega benefits from IV increases, -vega benefits from decreases.
- **Theta**: Time decay (per day). +Theta benefits sellers, -theta hurts sellers.

**Portfolio management:**
- Target delta to market directional view
- Target gamma for convexity
- Target vega based on IV forecast
- Theta is free profit if range-bound

## Important Notes

- This is simplified Black-Scholes. Real market prices differ (bid-ask, dividends, early exercise).
- IV surface is placeholder. Integrate real market data (Bloomberg, market data vendor).
- No transaction costs included in pricing. Add in production.
- Alpaca options support is real; test in paper mode first.
- Options require approval from broker. Not all accounts eligible.
- You can lose more than your initial investment if shorting naked calls/puts.
- Always use spreads (defined risk) until expertise grows.

## References

- Black-Scholes model: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
- Greeks: https://www.investopedia.com/terms/g/greeks.asp
- Options strategies: https://www.investopedia.com/articles/trading/092115/most-common-options-strategies.asp
