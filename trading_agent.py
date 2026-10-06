from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass(frozen=True)
class Config:
    """Configuration for strategy execution and risk controls."""

    initial_capital: float = 100_000.0
    fee_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 3.0
    max_leverage: float = 1.0
    periods_per_year: int = 365
    vol_target: float = 0.10
    max_position_weight: float = 1.0
    max_daily_loss: float = 0.05
    min_signal_strength: float = 0.05
    min_obs: int = 30
    max_turnover: float = 0.50
    warmup_window: int = 20
    max_drawdown: float = 0.20
    signal_clip: float = 1.0
    zscore_clip: float = 4.0

    def clamp_position(self, value: float) -> float:
        value = float(value)
        return max(-self.max_leverage, min(self.max_leverage, value))


class FeatureEngineer:
    """Construct predictive features from a price series."""

    @staticmethod
    def build_features(prices: pd.Series) -> pd.DataFrame:
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")

        prices = prices.sort_index().astype(float)
        if prices.empty:
            return pd.DataFrame(index=prices.index)

        ret = np.log(prices / prices.shift(1)).fillna(0.0)

        feat = pd.DataFrame(index=prices.index)
        feat["ret"] = ret
        feat["ret_2"] = ret.shift(1)
        feat["ret_5"] = ret.shift(1).rolling(5).mean()
        feat["ret_10"] = ret.shift(1).rolling(10).mean()
        feat["vol_5"] = ret.shift(1).rolling(5).std().fillna(0.0)
        feat["vol_10"] = ret.shift(1).rolling(10).std().fillna(0.0)
        feat["mom_5"] = prices.pct_change(5)
        feat["mom_10"] = prices.pct_change(10)
        feat["zscore_10"] = (
            (prices - prices.rolling(10).mean()) / prices.rolling(10).std()
        ).fillna(0.0)
        feat["trend_10"] = prices.rolling(10).mean().pct_change().fillna(0.0)
        feat["future_ret_5"] = ret.shift(-5)
        return feat


class SignalModel:
    """Gradient-boosted regressor for forecasting short-horizon directional returns."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.model = HistGradientBoostingRegressor(random_state=42, max_depth=6)
        self.features_: list[str] = []

    def fit(self, prices: pd.Series) -> "SignalModel":
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")

        features = FeatureEngineer.build_features(prices)
        target = features["future_ret_5"].fillna(0.0)
        model_features = [c for c in features.columns if c != "future_ret_5"]

        self.features_ = model_features
        self.model.fit(features[model_features], target)
        return self

    def predict(self, prices: pd.Series) -> pd.Series:
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model has not been initialized")
        if not hasattr(self, "features_") or not self.features_:
            raise ValueError("Model has not been fitted yet")

        features = FeatureEngineer.build_features(prices)
        raw = pd.Series(
            self.model.predict(features[self.features_]),
            index=features.index,
            dtype=float,
        )
        signal = raw.clip(-2.0, 2.0)

        sigma = signal.std(ddof=0)
        if sigma != 0:
            signal = signal / sigma

        return signal.fillna(0.0)


class RiskController:
    """Apply volatility scaling, drawdown protection, turnover caps, and leverage limits."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _vol_target_scale(self, signal: pd.Series, volatility: pd.Series) -> pd.Series:
        vol = volatility.replace(0, np.nan).clip(lower=1e-9)
        scale = self.cfg.vol_target / vol
        return signal * scale.fillna(0.0)

    def apply(
        self,
        signal: pd.Series,
        prices: pd.Series,
        equity_curve: pd.Series | None = None,
    ) -> pd.Series:
        if signal.empty:
            return signal.copy()

        signal = signal.copy().astype(float).fillna(0.0)
        ret = np.log(prices / prices.shift(1)).fillna(0.0)
        vol = ret.ewm(span=max(10, self.cfg.warmup_window), adjust=False).std().fillna(0.0)

        position = self._vol_target_scale(signal, vol)
        position = position.clip(-self.cfg.max_position_weight, self.cfg.max_position_weight)

        if equity_curve is not None and not equity_curve.empty:
            drawdown = equity_curve / equity_curve.cummax() - 1.0
            emergency = drawdown < -self.cfg.max_daily_loss
            if emergency.any():
                position = position.mask(emergency, 0.0)

        position = position.replace([np.inf, -np.inf], 0.0)
        position = position.fillna(0.0)
        position = position.clip(-self.cfg.max_leverage, self.cfg.max_leverage)
        return position

    def cap_turnover(self, position: pd.Series) -> pd.Series:
        if position.empty:
            return position.copy()

        adjusted = position.copy().astype(float).fillna(0.0)
        prev = adjusted.shift(1).fillna(0.0)
        delta = (adjusted - prev).abs()

        turnover_cap = prev.abs() + self.cfg.max_turnover
        mask = delta > turnover_cap

        adjusted.loc[mask] = prev.loc[mask]
        adjusted = adjusted.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return adjusted.clip(-self.cfg.max_leverage, self.cfg.max_leverage)

    def enforce_limits(
        self,
        signal: pd.Series,
        prices: pd.Series,
        equity_curve: pd.Series | None = None,
    ) -> pd.Series:
        position = self.apply(signal, prices, equity_curve)
        return self.cap_turnover(position)


class MLTradingAgent:
    """High-level strategy wrapper combining ML signals and risk management."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.model = SignalModel(cfg=self.cfg)
        self.risk = RiskController(self.cfg)
        self._trained = False

    def fit(self, prices: pd.Series) -> "MLTradingAgent":
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")
        self.model.fit(prices)
        self._trained = True
        return self

    def generate_signal(self, prices: pd.Series) -> pd.Series:
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")

        if not self._trained:
            self.fit(prices)

        raw_signal = self.model.predict(prices)
        signal = raw_signal.copy()
        signal = signal.where(abs(signal) > self.cfg.min_signal_strength, 0.0)
        return signal.fillna(0.0)

    def generate_position(self, prices: pd.Series, equity_curve: pd.Series | None = None) -> pd.Series:
        signal = self.generate_signal(prices)
        return self.risk.enforce_limits(signal, prices, equity_curve)

    def fit_and_generate(self, prices: pd.Series) -> pd.Series:
        return self.generate_position(prices)


def backtest(
    prices: pd.Series,
    signal: pd.Series,
    cfg: Config,
    opens: pd.Series | None = None,
) -> pd.DataFrame:
    """Run a realistic backtest with transaction costs and slippage."""
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series")
    if opens is not None and not isinstance(opens, pd.Series):
        raise TypeError("opens must be a pandas Series or None")

    prices = prices.sort_index().astype(float)
    signal = signal.reindex(prices.index).fillna(0.0).astype(float)

    if len(prices) == 0:
        return pd.DataFrame(
            {
                "position": pd.Series(dtype=float, index=prices.index),
                "gross": pd.Series(dtype=float, index=prices.index),
                "costs": pd.Series(dtype=float, index=prices.index),
                "net": pd.Series(dtype=float, index=prices.index),
                "equity": pd.Series(dtype=float, index=prices.index),
            },
            index=prices.index,
        )

    position = signal.shift(1).fillna(0.0)
    position = position.clip(-cfg.max_leverage, cfg.max_leverage).astype(float)

    if opens is not None:
        opens = opens.reindex(prices.index).ffill().astype(float)
        returns = np.log(opens.shift(-1) / opens).fillna(0.0)
    else:
        returns = np.log(prices / prices.shift(1)).fillna(0.0)

    turnover = position.diff().abs().fillna(0.0)
    turnover.iloc[0] = abs(position.iloc[0])
    costs = turnover * (cfg.fee_bps_per_side + cfg.slippage_bps_per_side) / 1e4

    net = position * returns - costs
    equity = cfg.initial_capital * np.exp(net.cumsum())

    return pd.DataFrame(
        {
            "position": position,
            "gross": position * returns,
            "costs": costs,
            "net": net,
            "equity": equity,
        },
        index=prices.index,
    )


def metrics(net: pd.Series, cfg: Config, min_obs: int | None = None) -> dict:
    """Compute annualized return, Sharpe, and drawdown metrics."""
    r = pd.Series(net).dropna().astype(float)
    if r.empty:
        raise ValueError("need at least one observation")
    min_obs = cfg.min_obs if min_obs is None else min_obs
    if len(r) < min_obs:
        raise ValueError(f"need >= {min_obs} observations, got {len(r)}")

    ann_ret = r.mean() * cfg.periods_per_year
    ann_vol = r.std() * np.sqrt(cfg.periods_per_year)
    equity = np.exp(np.cumsum(r))
    drawdown = equity / np.maximum.accumulate(equity) - 1.0

    under = (drawdown < 0).astype(int)
    if len(under) == 0:
        longest = 0
    else:
        longest = under.groupby((under != under.shift()).cumsum()).sum().max()
        longest = int(longest if pd.notna(longest) else 0)

    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    return {
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0.0,
        "ann_return": ann_ret,
        "max_drawdown": max_dd,
        "longest_dd_bars": int(longest),
        "calmar": ann_ret / abs(max_dd) if max_dd < 0 else 0.0,
        "n_obs": len(r),
    }


def walk_forward(
    prices: pd.Series,
    fit_fn: Callable[[pd.Series], Callable[[pd.Series], pd.Series]],
    cfg: Config,
    train_days: int = 180,
    test_days: int = 60,
    embargo: int = 1,
    opens: pd.Series | None = None,
):
    """Run time-series walk-forward validation."""
    prices = pd.Series(prices).sort_index().astype(float)
    if opens is not None:
        opens = pd.Series(opens).sort_index().astype(float)

    span = train_days + embargo + test_days
    if len(prices) < span:
        raise ValueError(f"Not enough observations: need at least {span}, got {len(prices)}")

    folds, oos = [], []

    for i in range(0, len(prices) - span + 1, test_days):
        window = prices.iloc[i : i + span]
        window_opens = opens.iloc[i : i + span] if opens is not None else None

        signal_fn = fit_fn(window.iloc[:train_days])
        signal = signal_fn(window)
        bt_result = backtest(window, signal, cfg, window_opens).iloc[train_days + embargo :]

        if bt_result.empty:
            continue

        oos.append(bt_result["net"])
        folds.append({"start": bt_result.index[0], "fold_return": bt_result["net"].sum()})

    df = pd.DataFrame(folds)
    stitched = pd.concat(oos) if oos else pd.Series(dtype=float)

    return {
        "folds": df,
        "positive_folds": f"{(df['fold_return'] > 0).sum()}/{len(df)}",
        "oos_metrics": metrics(stitched, cfg) if not stitched.empty else {
            "sharpe": 0.0,
            "ann_return": 0.0,
            "max_drawdown": 0.0,
            "longest_dd_bars": 0,
            "calmar": 0.0,
            "n_obs": 0,
        },
        "oos_returns": stitched,
    }


__all__ = [
    "Config",
    "FeatureEngineer",
    "SignalModel",
    "RiskController",
    "MLTradingAgent",
    "backtest",
    "metrics",
    "walk_forward",
]
