from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_prices(csv_path: str | Path) -> pd.Series:
    df = pd.read_csv(csv_path)

    if "price" in df.columns:
        series = df["price"].astype(float)
    elif len(df.columns) == 1:
        series = df.iloc[:, 0].astype(float)
    else:
        raise ValueError("CSV must contain a 'price' column or a single numeric price column.")

    return pd.Series(series.to_numpy(), index=pd.RangeIndex(len(series)))
