"""Run the full ETL pipeline: raw → interim → processed → DuckDB.

Usage:
    python scripts/run_pipeline.py [--source youtube]

Steps:
    1. Load raw CSV via loader.load_youtube_trending()
    2. Clean via cleaner.clean_youtube_raw()        → data/interim/
    3. Build features via cleaner.build_youtube_processed() → data/processed/
    4. Write processed table into DuckDB             → data/db/virality.duckdb
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from virality.config import DATA_INTERIM, DATA_PROCESSED
from virality.data.cleaner import build_youtube_processed, clean_youtube_raw
from virality.data.loader import load_youtube_trending
from virality.db.duckdb_client import get_connection

INTERIM_PATH = DATA_INTERIM / "youtube_trending_clean.parquet"
PROCESSED_PATH = DATA_PROCESSED / "youtube_trending.parquet"
DUCKDB_TABLE = "youtube_trending"


def _log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def run_youtube() -> None:
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    _log("Loading raw CSV …")
    t0 = time.perf_counter()
    raw = load_youtube_trending()
    _log(f"  {len(raw):,} rows loaded in {time.perf_counter() - t0:.1f}s")

    # ── Step 2: Clean (raw → interim) ────────────────────────────────────────
    _log("Cleaning (raw → interim) …")
    t0 = time.perf_counter()
    interim = clean_youtube_raw(raw)
    _log(f"  {len(interim):,} rows after dedup in {time.perf_counter() - t0:.1f}s")

    _log(f"  Writing → {INTERIM_PATH}")
    interim.to_parquet(INTERIM_PATH, index=False)

    # ── Step 3: Feature build (interim → processed) ───────────────────────────
    _log("Building features (interim → processed) …")
    t0 = time.perf_counter()
    processed = build_youtube_processed(interim)
    _log(f"  {len(processed):,} rows, {len(processed.columns)} columns in {time.perf_counter() - t0:.1f}s")

    _log(f"  Writing → {PROCESSED_PATH}")
    processed.to_parquet(PROCESSED_PATH, index=False)

    # ── Step 4: DuckDB ────────────────────────────────────────────────────────
    _log(f"Registering in DuckDB table '{DUCKDB_TABLE}' …")
    con = get_connection()
    con.execute(f"DROP TABLE IF EXISTS {DUCKDB_TABLE}")
    con.execute(
        f"CREATE TABLE {DUCKDB_TABLE} AS SELECT * FROM read_parquet('{PROCESSED_PATH.as_posix()}')"
    )
    row_count = con.execute(f"SELECT COUNT(*) FROM {DUCKDB_TABLE}").fetchone()[0]
    _log(f"  DuckDB table '{DUCKDB_TABLE}': {row_count:,} rows")

    _log("Pipeline complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run virality ETL pipeline")
    parser.add_argument(
        "--source",
        choices=["youtube"],
        default="youtube",
        help="Dataset to process (default: youtube)",
    )
    args = parser.parse_args()

    if args.source == "youtube":
        run_youtube()


if __name__ == "__main__":
    main()
