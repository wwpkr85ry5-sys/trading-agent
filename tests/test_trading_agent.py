import numpy as np
import pandas as pd
import pytest

from trading_agent import Config, MLTradingAgent, RiskController, backtest, metrics, walk_forward


def make_prices(n=250, drift=0.0005, vol=0.01, seed=42):
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, vol, n)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, index=pd.date_range("2020-01-01", periods=n, freq="D"))


def test_config_clamps_positions():
    cfg = Config(max_leverage=1.5)
    assert cfg.clamp_position(2.0) == 1.5
    assert cfg.clamp_position(-2.0) == -1.5
    assert cfg.clamp_position(0.4) == 0.4


def test_backtest_rejects_invalid_inputs():
    prices = make_prices(40)
    signal = pd.Series(np.zeros(len(prices)))
    cfg = Config()

    with pytest.raises(TypeError):
        backtest(prices.to_numpy(), signal, cfg)

    with pytest.raises(TypeError):
        backtest(prices, signal.to_numpy(), cfg)


def test_backtest_produces_expected_columns_and_shapes():
    prices = make_prices(60)
    signal = pd.Series(np.linspace(-1.0, 1.0, len(prices)), index=prices.index)

    result = backtest(prices, signal, Config())

    assert list(result.columns) == ["position", "gross", "costs", "net", "equity"]
    assert len(result) == len(prices)
    assert result["equity"].iloc[-1] >= 0.0


def test_metrics_requires_minimum_observations():
    cfg = Config(min_obs=5)
    net = pd.Series([0.01, -0.02, 0.03], dtype=float)

    with pytest.raises(ValueError):
        metrics(net, cfg)


def test_metrics_returns_expected_keys():
    cfg = Config(min_obs=3)
    net = pd.Series([0.01, -0.02, 0.03, 0.04, -0.01], dtype=float)

    out = metrics(net, cfg)
    assert set(out.keys()) == {"sharpe", "ann_return", "max_drawdown", "longest_dd_bars", "calmar", "n_obs"}
    assert out["n_obs"] == len(net)


def test_risk_controller_caps_turnover():
    cfg = Config(max_turnover=0.5, max_leverage=1.0)
    rc = RiskController(cfg)

    position = pd.Series([0.2, 1.0], dtype=float)
    adjusted = rc.cap_turnover(position)

    assert adjusted.iloc[1] == pytest.approx(0.2)
    assert adjusted.iloc[0] == pytest.approx(0.2)


def test_risk_controller_apply_does_not_produce_inf_or_nan():
    prices = make_prices(100)
    signal = pd.Series(np.linspace(-1.0, 1.0, len(prices)), index=prices.index)
    cfg = Config(max_leverage=1.0, warmup_window=10)
    rc = RiskController(cfg)

    pos = rc.apply(signal, prices, pd.Series(100.0 + np.arange(len(prices)), index=prices.index))
    assert np.isfinite(pos.to_numpy()).all()
    assert (pos.abs() <= cfg.max_leverage).all()


def test_walk_forward_returns_expected_structure():
    prices = make_prices(200)
    cfg = Config(min_obs=10)

    def fit_fn(train):
        return lambda x: pd.Series(0.5, index=x.index, dtype=float)

    out = walk_forward(prices=prices, fit_fn=fit_fn, cfg=cfg, train_days=30, test_days=10, embargo=1)

    assert "folds" in out
    assert "positive_folds" in out
    assert "oos_metrics" in out
    assert "oos_returns" in out
    assert isinstance(out["folds"], pd.DataFrame)
    assert isinstance(out["oos_returns"], pd.Series)


def test_walk_forward_raises_when_not_enough_data():
    prices = make_prices(30)
    cfg = Config()

    def fit_fn(train):
        return lambda x: pd.Series(0.0, index=x.index, dtype=float)

    with pytest.raises(ValueError):
        walk_forward(prices, fit_fn, cfg, train_days=30, test_days=10, embargo=1)


def test_ml_trading_agent_generates_position():
    prices = make_prices(140)
    cfg = Config(max_leverage=1.0, min_signal_strength=0.0)
    agent = MLTradingAgent(cfg=cfg)

    position = agent.generate_position(prices)
    assert isinstance(position, pd.Series)
    assert len(position) == len(prices)
    assert np.isfinite(position.to_numpy()).all()
    assert (position.abs() <= cfg.max_leverage).all()


def test_ml_trading_agent_fit_and_generate_is_deterministic():
    prices = make_prices(140)
    agent = MLTradingAgent(cfg=Config(max_leverage=1.0, min_signal_strength=0.0))

    pos1 = agent.fit_and_generate(prices)
    pos2 = agent.fit_and_generate(prices)

    pd.testing.assert_series_equal(pos1, pos2, check_names=False)
