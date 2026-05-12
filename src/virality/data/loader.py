"""Load raw datasets from data/raw/ into DataFrames.

No transformation here — callers get the raw bytes faithfully reproduced.
Cleaning logic lives in cleaner.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from virality.config import DATA_RAW

# Columns with known mixed-type issues in the Kaggle dump — force str so
# pandas does not silently coerce them during chunked reads.
_YT_DTYPE_OVERRIDES: dict[str, str] = {
    "video_id": "str",
    "video_category_id": "str",
    "video_licensed_content": "str",
    "channel_have_hidden_subscribers": "str",
}


def load_youtube_trending(path: Path | None = None) -> pd.DataFrame:
    """Return the raw YouTube trending CSV as a DataFrame.

    Args:
        path: override the default location (data/raw/youtube_trending/).

    Returns:
        DataFrame with 28 columns, all object/float as stored in the file.
        No type casting or null handling is applied here.
    """
    csv_path = path or DATA_RAW / "youtube_trending" / "youtube_trending_videos_global.csv"
    return pd.read_csv(
        csv_path,
        dtype=_YT_DTYPE_OVERRIDES,
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",
    )
