from live_trader import order_for, pos_key, target_notional, to_alpaca


def test_long_only_clamps_shorts():
    assert target_notional(-0.8, 10_000, 4, 0.25, allow_short=False) == 0.0


def test_symbol_cap_and_equal_weight():
    # 2 symbols -> 50% each, but capped at 25%
    assert target_notional(1.0, 10_000, 2, 0.25, False) == 2500.0
    # 10 symbols -> 10% each, below cap
    assert target_notional(1.0, 10_000, 10, 0.25, False) == 1000.0


def test_short_allowed():
    assert target_notional(-1.0, 10_000, 4, 0.25, True) == -2500.0


def test_min_trade_filter():
    assert order_for(1000, 990, 25) is None
    assert order_for(1000, 0, 25) == ("buy", 1000.0)
    assert order_for(0, 500, 25) == ("sell", 500.0)


def test_symbol_mapping():
    assert to_alpaca("BTC-USD", "crypto") == "BTC/USD"
    assert to_alpaca("AAPL", "stock") == "AAPL"
    assert pos_key("BTC/USD") == "BTCUSD"
