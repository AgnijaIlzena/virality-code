"""Centralised paths and coefficients. Imported by pipeline, dashboard and DB client."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXTERNAL = ROOT / "data" / "external"
DATA_DB = ROOT / "data" / "db"

DUCKDB_PATH = DATA_DB / "virality.duckdb"

# EWI starting coefficients — cf. BRIEF.md §2.2. To be re-calibrated via regression.
EWI_COEFFICIENTS = {
    "sentiment": 0.4,
    "outrage": 0.3,
    "humor": 0.2,
    "emoji_density": 0.1,
}
