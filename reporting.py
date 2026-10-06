from __future__ import annotations

from pathlib import Path

import pandas as pd


class PerformanceReport:
    """Generate portfolio summary tables for performance metrics."""

    def __init__(self, output_dir: str | Path = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def summarize(self, results: pd.DataFrame, file_name: str = "performance_summary.csv") -> pd.DataFrame:
        summary = results.describe().T
        summary.to_csv(self.output_dir / file_name)
        return summary

    def export(self, data: pd.DataFrame, file_name: str = "portfolio_report.csv") -> pd.DataFrame:
        data.to_csv(self.output_dir / file_name)
        return data
