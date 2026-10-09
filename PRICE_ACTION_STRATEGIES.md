# Price Action & FVG Strategies

## Fair Value Gap (FVG) Strategy

Identifies and trades market inefficiencies where prices gap without overlap, creating potential reversal points.

### Concept

**Fair Value Gap** occurs when:
- **Bullish FVG**: Current candle low > Previous candle high (gap up)
- **Bearish FVG**: Current candle high < Previous candle low (gap down)

These gaps represent areas of "unfair value" that price often returns to fill.

### Signals

- **Bullish Signal**: Gap low becomes support → Price bounces
- **Bearish Signal**: Gap high becomes resistance → Price rejects
- **Confirmation**: Volume > Average + RSI validation

### Parameters

- `FVG Lookback Period`: Number of bars to analyze (default: 5)
- `Minimum Gap Size`: Minimum gap as % of price (default: 0.2%)
- `Confirmation Bars`: Bars until signal expires (default: 2)
- `Use RSI Confirmation`: Filter with RSI overbought/oversold (default: true)

### Trading

1. Enter on gap signal confirmation
2. Exit when gap is filled or reversed
3. Stop loss: Beyond opposite FVG level
4. Target: Gap level + 1-2% extension

---

## Imbalance & Candle Close (ICC) Indicator

Detects market structure breaks and liquidity sweeps using imbalance ratios.

### Concept

**ICC** combines:
1. **Imbalance**: Body size vs Wick size ratio
2. **Structure**: Higher Highs (HH) and Higher Lows (HL) breaks
3. **Sweeps**: Liquidity grab (wick penetration) then reversal

### Signals

- **Bullish ICC**: Low sweep + Bullish candle + Extreme imbalance
- **Bearish ICC**: High sweep + Bearish candle + Extreme imbalance
- **Structure Break**: Upstroke (HH/HL) or Downstroke (LL/LH)

### Interpretation

- **Green Zone**: Bullish imbalance (potential reversal up)
- **Red Zone**: Bearish imbalance (potential reversal down)
- **Diamonds**: Liquidity sweeps (institutional activity)
- **Triangles**: Structure breaks (trend confirmation)

### Trading

1. Wait for structure break (upstroke/downstroke)
2. Confirm with liquidity sweep
3. Enter when imbalance normalizes
4. Exit at opposite structure level

---

## Price Action Strategy

Trades based on support/resistance, candle patterns, and trend confirmation.

### Candle Patterns

**Bullish Patterns:**
- **Pin Bar**: Long lower wick + small body + close near high
- **Bullish Engulfing**: Previous bearish candle completely engulfed
- **Hammer**: Long lower wick + small body at high

**Bearish Patterns:**
- **Inverse Pin Bar**: Long upper wick + small body + close near low
- **Bearish Engulfing**: Previous bullish candle completely engulfed

**Neutral Patterns:**
- **Inside Bar**: High < Previous high, Low > Previous low (consolidation)

### Support & Resistance

- **Pivot Levels**: Classic pivot points (R2, R1, S1, S2)
- **Local Extremes**: Recent highs and lows
- **Key Levels**: Highest high and lowest low over lookback period

### Trend Confirmation

- **Fast MA (9-period)** vs **Slow MA (21-period)**
- **Uptrend**: Fast > Slow
- **Downtrend**: Fast < Slow
- **Volume Confirmation**: Volume > Average × 1.2

### Trading Rules

**Long Entry:**
1. Bullish candle pattern (pin bar, engulfing, hammer)
2. Price at support level
3. Uptrend confirmed (MA cross)
4. Volume confirmation (if enabled)

**Short Entry:**
1. Bearish candle pattern
2. Price at resistance level
3. Downtrend confirmed (MA cross)
4. Volume confirmation (if enabled)

**Exit:**
- Stop Loss: Recent support/resistance ± ATR
- Take Profit: Next resistance/support level
- Time Exit: 20 bars maximum

---

## Combined Strategy Usage

### On TradingView

```
1. Add all three strategies to chart
2. Set timeframe: 4H or Daily (FVG works best on higher TF)
3. Enable alerts for each strategy
4. Adjust parameters for your symbol
```

### Combining Signals

**Strong Long Setup:**
- FVG: Bullish gap forming support ✓
- ICC: Bullish sweep + structure break ✓
- Price Action: Pin bar at support + uptrend ✓

**Strong Short Setup:**
- FVG: Bearish gap forming resistance ✓
- ICC: Bearish sweep + structure break ✓
- Price Action: Engulfing at resistance + downtrend ✓

### Settings by Timeframe

**1H (Scalping)**
- FVG Lookback: 3, Min Gap: 0.1%
- ICC Lookback: 10
- Price Action: Fast MA: 5, Slow MA: 13

**4H (Swing)**
- FVG Lookback: 5, Min Gap: 0.2%
- ICC Lookback: 20
- Price Action: Fast MA: 9, Slow MA: 21

**Daily (Position)**
- FVG Lookback: 7, Min Gap: 0.3%
- ICC Lookback: 30
- Price Action: Fast MA: 13, Slow MA: 34

---

## Integration with trading-agent

### Python Backtesting

```python
from price_action_strategy import PriceActionSignalGenerator
from fvg_strategy import FVGSignalGenerator
from icc_strategy import ICCSignalGenerator
from market_data import create_pipeline

pipeline = create_pipeline(source_type="yfinance", cache_dir=".cache")
prices = pipeline.get_prices("AAPL")

# Generate signals
fvg_gen = FVGSignalGenerator(lookback=5)
pa_gen = PriceActionSignalGenerator(lookback=20)
icc_gen = ICCSignalGenerator(lookback=20)

fvg_signals = fvg_gen.generate_signals(prices)
pa_signals = pa_gen.generate_signals(prices)
icc_signals = icc_gen.generate_signals(prices)

# Combine signals (simple voting)
combined = (fvg_signals + pa_signals + icc_signals) / 3

# Backtest
from trading_agent import backtest, metrics, Config
result = backtest(prices, combined, Config())
print(metrics(result["net"], Config()))
```

---

## Performance Tips

1. **Use on higher timeframes** (4H+) for better reliability
2. **Combine with volume** for confirmation
3. **Wait for confluence** (2+ strategies agreeing)
4. **Size appropriately** with ATR-based stops
5. **Backtest on YOUR symbols** - parameters vary by asset
6. **Use alerts** instead of staring at chart all day
7. **Track winners/losers** to optimize parameters

---

## Resources

- FVG Theory: https://en.wikipedia.org/wiki/Market_microstructure
- Price Action: https://www.investopedia.com/terms/p/price-action.asp
- Smart Money Concepts: https://www.investopedia.com/terms/s/smart-money.asp
