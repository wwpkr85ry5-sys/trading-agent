from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


class CacheStore:
    """Simple SQLite-backed cache for market data and results."""

    def __init__(self, db_path: str | Path = ".cache/market_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT,
                    start TEXT,
                    end TEXT,
                    data BLOB,
                    PRIMARY KEY (symbol, start, end)
                )
                """
            )

    def get(self, symbol: str, start: str, end: str) -> pd.DataFrame | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT data FROM market_data WHERE symbol=? AND start=? AND end=?",
                (symbol, start, end),
            ).fetchone()
        if row is None:
            return None
        return pd.read_pickle(row[0])

    def set(self, symbol: str, start: str, end: str, data: pd.DataFrame) -> None:
        payload = data.to_pickle()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO market_data(symbol, start, end, data) VALUES (?, ?, ?, ?)",
                (symbol, start, end, payload),
            )

    def clear(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM market_data")
