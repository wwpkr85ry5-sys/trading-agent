import numpy as np
import pandas as pd

from multi_agent import STRATEGIES, VARIANTS, MultiAgentDesk


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx)


def test_seven_variants_per_strategy():
    assert all(len(v) == 7 for v in VARIANTS.values())
    assert set(VARIANTS) == set(STRATEGIES)


def test_desk_runs_on_synthetic_data():
    row = MultiAgentDesk().analyze("TEST", "stock", synthetic())
    assert row["best_strategy"] in STRATEGIES
    assert -1.0 <= row["position"] <= 1.0
    assert 0 <= row["agents_agree"] <= 7
