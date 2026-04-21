"""Thin DuckDB accessor. Stores computed indices + intermediate aggregates for reuse across notebooks, pipeline and dashboard."""
from pathlib import Path

import duckdb

from virality.config import DUCKDB_PATH


def get_connection(db_path: Path = DUCKDB_PATH) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))
