"""Multi-agent trading desk.

For each symbol:
  1. Rank 7 strategy families on the trailing year (net of costs) and pick the best.
  2. Run 7 agents, each a different parameterisation of that winning strategy.
  3. A head agent (risk-managed portfolio-manager logic) combines the 7 agents into one position.

This module only produces signals / paper positions. It does not place orders.
Past performance does not guarantee future results.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from trading_agent import Config, FeatureEngineer, backtest

# ----------------------------------------------------------------------------
# Universe and per-asset-class assumptions
# ----------------------------------------------------------------------------
UNIVERSE = {
    "stock": ["AAPL", "MSFT", "NVDA", "SPY"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "currency": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"],
    "futures": ["ES=F", "NQ=F", "GC=F", "CL=F"],
}
PERIODS = {"stock": 252, "crypto": 365, "currency": 260, "futures": 252}
# (fee_bps_per_side, slippage_bps_per_side)
COSTS = {"stock": (1.0, 2.0), "crypto": (10.0, 5.0), "currency": (1.0, 1.0), "futures": (1.0, 2.0)}


# ----------------------------------------------------------------------------
# Strategy families. Each returns a signal in [-1, 1] indexed like ohlc.
# Signals are computed on bar t; backtest() applies them from bar t+1.
# ----------------------------------------------------------------------------
def _hold(raw: pd.Series, limit: int | None = None) -> pd.Series:
    return raw.ffill(limit=limit).fillna(0.0)


def trend(o: pd.DataFrame, p: dict, es: int) -> pd.Series:
    c = o["close"]
    return np.sign(c.rolling(p["fast"]).mean() - c.rolling(p["slow"]).mean()).fillna(0.0)


def mean_reversion(o: pd.DataFrame, p: dict, es: int) -> pd.Series:
    c = o["close"]
    z = (c - c.rolling(p["n"]).mean()) / c.rolling(p["n"]).std()
    return (-z / p["z"]).clip(-1, 1).fillna(0.0)


def breakout(o: pd.DataFrame, p: dict, es: int) -> pd.Series:
    hi = o["high"].rolling(p["n"]).max().shift(1)
    lo = o["low"].rolling(p["n"]).min().shift(1)
    raw = pd.Series(np.where(o["close"] > hi, 1.0, np.where(o["close"] < lo, -1.0, np.nan)), index=o.index)
    return _hold(raw)


def fvg(o: pd.DataFrame, p: dict, es: int) -> pd.Series:
    """3-candle fair value gap: candle-1 high < candle-3 low (bull) or the inverse (bear)."""
    h2, l2 = o["high"].shift(2), o["low"].shift(2)
    bull = (o["low"] > h2) & ((o["low"] - h2) / o["close"] > p["min_gap"])
    bear = (o["high"] < l2) & ((l2 - o["high"]) / o["close"] > p["min_gap"])
    raw = pd.Series(np.where(bull, 1.0, np.where(bear, -1.0, np.nan)), index=o.index)
    return _hold(raw, p["hold"])


def supply_demand(o: pd.DataFrame, p: dict, es: int) -> pd.Series:
    lo = o["low"].rolling(p["n"]).min().shift(1)
    hi = o["high"].rolling(p["n"]).max().shift(1)
    demand = (o["low"] <= lo * (1 + p["zone"])) & (o["close"] > o["open"])
    supply = (o["high"] >= hi * (1 - p["zone"])) & (o["close"] < o["open"])
    raw = pd.Series(np.where(demand, 1.0, np.where(supply, -1.0, np.nan)), index=o.index)
    return _hold(raw, p["hold"])


def price_action(o: pd.DataFrame, p: dict, es: int) -> pd.Series:
    body = (o["close"] - o["open"]).abs()
    lower = np.minimum(o["open"], o["close"]) - o["low"]
    upper = o["high"] - np.maximum(o["open"], o["close"])
    bull_pin = (lower > 2 * body) & (lower > upper)
    bear_pin = (upper > 2 * body) & (upper > lower)
    po, pc = o["open"].shift(1), o["close"].shift(1)
    bull_eng = (pc < po) & (o["close"] > o["open"]) & (o["close"] >= po) & (o["open"] <= pc)
    bear_eng = (pc > po) & (o["close"] < o["open"]) & (o["close"] <= po) & (o["open"] >= pc)
    up = o["close"] > o["close"].rolling(p["n"]).mean()
    long_ = (bull_pin | bull_eng) & up
    short_ = (bear_pin | bear_eng) & ~up
    raw = pd.Series(np.where(long_, 1.0, np.where(short_, -1.0, np.nan)), index=o.index)
    return _hold(raw, p["hold"])


def ml(o: pd.DataFrame, p: dict, es: int) -> pd.Series:
    """Gradient-boosted forecaster trained ONLY on data before the evaluation window."""
    c = o["close"]
    feats = FeatureEngineer.build_features(c)
    cols = [x for x in feats.columns if x != "future_ret_5"]
    train = feats.iloc[: max(es - 5, 0)]  # -5 so labels never overlap the eval window
    if len(train) < 60:
        return pd.Series(0.0, index=o.index)
    model = HistGradientBoostingRegressor(
        max_depth=p["max_depth"], learning_rate=p["lr"], random_state=p["seed"]
    ).fit(train[cols], train["future_ret_5"].fillna(0.0))
    pred = pd.Series(model.predict(feats[cols]), index=o.index)
    scale = pred.iloc[:es].std() or 1.0
    return np.tanh(pred / scale).fillna(0.0)


STRATEGIES = {
    "trend": trend,
    "mean_reversion": mean_reversion,
    "breakout": breakout,
    "fvg": fvg,
    "supply_demand": supply_demand,
    "price_action": price_action,
    "ml": ml,
}

_HOLDS = [3, 5, 5, 8, 10, 10, 15]
VARIANTS: dict[str, list[dict]] = {
    "trend": [{"fast": f, "slow": s} for f, s in [(5, 20), (8, 21), (10, 30), (20, 50), (30, 80), (50, 100), (50, 200)]],
    "mean_reversion": [{"n": n, "z": 2.0} for n in [10, 15, 20, 30, 40, 50, 60]],
    "breakout": [{"n": n} for n in [10, 15, 20, 30, 40, 55, 80]],
    "fvg": [{"min_gap": g, "hold": h} for g, h in zip([0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0075], _HOLDS)],
    "supply_demand": [{"n": n, "zone": 0.005, "hold": h} for n, h in zip([10, 15, 20, 30, 40, 50, 60], _HOLDS)],
    "price_action": [{"n": n, "hold": h} for n, h in zip([10, 20, 30, 50, 100, 150, 200], _HOLDS)],
    "ml": [
        {"max_depth": d, "lr": lr, "seed": s}
        for s, (d, lr) in enumerate([(2, 0.05), (3, 0.05), (4, 0.05), (5, 0.05), (6, 0.05), (3, 0.10), (4, 0.10)])
    ],
}


def _sharpe(net: pd.Series, ppy: int) -> float:
    r = net.dropna()
    s = r.std()
    if len(r) < 10 or not s or np.isnan(s):
        return 0.0
    return float(r.mean() / s * np.sqrt(ppy))


# ----------------------------------------------------------------------------
# Agents
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyAgent:
    name: str
    strategy: str
    params: dict

    def signal(self, ohlc: pd.DataFrame, eval_start: int) -> pd.Series:
        return STRATEGIES[self.strategy](ohlc, self.params, eval_start).clip(-1, 1)


class HeadAgent:
    """Combines the 7 agents: performance-weighted consensus, agreement gate,
    volatility targeting, leverage cap and a drawdown circuit-breaker."""

    def __init__(self, lookback: int = 63, min_agree: int = 4, target_vol: float = 0.10, max_dd: float = 0.15):
        self.lookback = lookback
        self.min_agree = min_agree
        self.target_vol = target_vol
        self.max_dd = max_dd

    def decide(self, signals: pd.DataFrame, close: pd.Series, cfg: Config) -> dict:
        nets = pd.DataFrame({c: backtest(close, signals[c], cfg)["net"] for c in signals.columns})
        roll = nets.rolling(self.lookback).mean() / nets.rolling(self.lookback).std().replace(0, np.nan)
        w = (roll.shift(1).clip(lower=0.0) + 0.1).fillna(1.0)  # lagged: no look-ahead
        w = w.div(w.sum(axis=1), axis=0)

        consensus = (signals * w).sum(axis=1)
        agree = (np.sign(signals).mul(np.sign(consensus), axis=0) > 0).sum(axis=1)
        raw = consensus.where(agree >= self.min_agree, 0.0)

        ret = np.log(close / close.shift(1))
        vol = ret.rolling(20).std() * np.sqrt(cfg.periods_per_year)
        scale = (self.target_vol / vol.replace(0, np.nan)).clip(upper=cfg.max_leverage).fillna(0.0)
        pos = (raw * scale).clip(-cfg.max_leverage, cfg.max_leverage)

        eq = backtest(close, pos, cfg)["equity"]
        dd = eq / eq.cummax() - 1.0
        pos = pos.mask(dd.shift(1).fillna(0.0) < -self.max_dd, 0.0)
        return {"position": pos, "consensus": consensus, "agree": agree, "weights": w}


# ----------------------------------------------------------------------------
# Desk
# ----------------------------------------------------------------------------
class MultiAgentDesk:
    def __init__(self, holdout: int = 63, head: HeadAgent | None = None):
        self.holdout = holdout
        self.head = head or HeadAgent()

    @staticmethod
    def _cfg(asset_class: str) -> Config:
        fee, slip = COSTS[asset_class]
        return replace(Config(), fee_bps_per_side=fee, slippage_bps_per_side=slip,
                       periods_per_year=PERIODS[asset_class], max_leverage=1.0)

    def select_strategy(self, ohlc: pd.DataFrame, cfg: Config, eval_start: int) -> tuple[str, dict]:
        """Rank strategies on the trailing year EXCLUDING the final holdout bars."""
        scores = {}
        end = len(ohlc) - self.holdout
        for name, fn in STRATEGIES.items():
            sig = fn(ohlc, VARIANTS[name][3], eval_start)
            net = backtest(ohlc["close"], sig, cfg)["net"].iloc[eval_start:end]
            scores[name] = _sharpe(net, cfg.periods_per_year)
        return max(scores, key=scores.get), scores

    def analyze(self, symbol: str, asset_class: str, ohlc: pd.DataFrame) -> dict:
        cfg = self._cfg(asset_class)
        ppy = PERIODS[asset_class]
        es = len(ohlc) - ppy
        if es < 80:
            raise ValueError(f"{symbol}: need at least {ppy + 80} bars, got {len(ohlc)}")

        best, scores = self.select_strategy(ohlc, cfg, es)
        agents = [StrategyAgent(f"{best}-{i + 1}", best, p) for i, p in enumerate(VARIANTS[best])]
        signals = pd.DataFrame({a.name: a.signal(ohlc, es) for a in agents})
        out = self.head.decide(signals, ohlc["close"], cfg)

        pos = out["position"]
        net = backtest(ohlc["close"], pos, cfg)["net"]
        last = float(pos.iloc[-1])
        return {
            "symbol": symbol,
            "asset_class": asset_class,
            "best_strategy": best,
            "select_sharpe": round(scores[best], 3),
            "holdout_sharpe": round(_sharpe(net.iloc[-self.holdout:], ppy), 3),
            "agents_agree": int(out["agree"].iloc[-1]),
            "consensus": round(float(out["consensus"].iloc[-1]), 3),
            "position": round(last, 3),
            "action": "LONG" if last > 0.05 else "SHORT" if last < -0.05 else "FLAT",
            "all_scores": {k: round(v, 3) for k, v in scores.items()},
        }

    def run(self, universe: dict[str, list[str]] | None = None) -> pd.DataFrame:
        rows = []
        for asset_class, symbols in (universe or UNIVERSE).items():
            for sym in symbols:
                try:
                    rows.append(self.analyze(sym, asset_class, load_ohlc(sym)))
                except Exception as exc:  # keep the desk running if one symbol fails
                    rows.append({"symbol": sym, "asset_class": asset_class, "action": f"ERROR: {exc}"})
        return pd.DataFrame(rows)


def load_ohlc(symbol: str, period: str = "2y") -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="7-agent desk + head agent (signals only, no order execution)")
    ap.add_argument("--classes", default=",".join(UNIVERSE), help="comma list: stock,crypto,currency,futures")
    ap.add_argument("--out", default="desk_decisions.csv")
    args = ap.parse_args()
    uni = {k: UNIVERSE[k] for k in args.classes.split(",")}
    df = MultiAgentDesk().run(uni)
    df.to_csv(args.out, index=False)
    print(df.drop(columns=["all_scores"], errors="ignore").to_string(index=False))


if __name__ == "__main__":
    main()
