from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ParameterGrid:
    params: dict[str, list[Any]] = field(default_factory=dict)

    def iterate(self):
        keys = list(self.params.keys())
        values = [self.params[k] for k in keys]
        for item in product(*values):
            yield dict(zip(keys, item))


class ParameterSearch:
    """Simple parameter search for tuning trading configs."""

    def __init__(self, grid: ParameterGrid):
        self.grid = grid

    def search(self, evaluator):
        results = []
        for params in self.grid.iterate():
            score = evaluator(**params)
            results.append({**params, "score": score})
        return pd.DataFrame(results)
