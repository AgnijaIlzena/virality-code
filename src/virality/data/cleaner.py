"""Two-step cleaning pipeline: raw → interim → processed.

Step 1 — clean_youtube_raw(df)
    Input : raw DataFrame from loader.load_youtube_trending()
    Output: cleaned DataFrame saved to data/interim/youtube_trending_clean.parquet

Step 2 — build_youtube_processed(df)
    Input : interim DataFrame (output of step 1)
    Output: feature-ready DataFrame saved to data/processed/youtube_trending.parquet

Both functions are pure transforms — no I/O.  The run_pipeline script handles
reading and writing.
"""
from __future__ import annotations

import re

import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ISO8601_PATTERN = re.compile(
    r"PT(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?"
)


def _parse_iso8601_duration(series: pd.Series) -> pd.Series:
    """Convert ISO 8601 duration strings (PT2M28S) to total seconds (int)."""
    def _to_seconds(val: str) -> float | None:
        if pd.isna(val):
            return None
        m = _ISO8601_PATTERN.match(str(val))
        if not m:
            return None
        h = int(m.group("hours") or 0)
        mins = int(m.group("minutes") or 0)
        s = int(m.group("seconds") or 0)
        return h * 3600 + mins * 60 + s

    return series.map(_to_seconds)


def _safe_bool(series: pd.Series) -> pd.Series:
    """Coerce mixed-type boolean-ish column to nullable boolean."""
    mapping = {"true": True, "false": False, "1": True, "0": False}
    return (
        series.astype(str).str.strip().str.lower()
        .map(mapping)
        .astype("boolean")
    )


# ---------------------------------------------------------------------------
# Step 1 — raw → interim
# ---------------------------------------------------------------------------

def clean_youtube_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Apply structural cleaning to the raw YouTube trending DataFrame.

    Operations (in order):
    1. Drop rows with no video_id (22 rows — no usable key).
    2. Parse datetime columns to UTC-aware timestamps.
    3. Parse ISO 8601 video_duration to integer seconds.
    4. Coerce boolean columns that were loaded as mixed strings.
    5. Cast engagement counts to Int64 (nullable int), filling disabled-
       comment / hidden-like rows with 0 rather than dropping them.
    6. Fill low-signal nulls with sentinels rather than dropping rows.
    7. Deduplicate to one row per (video_id, video_trending_country),
       keeping the *earliest* trending date — this is the event we care
       about for TTV proxy computation.

    Returns a cleaned DataFrame; does not write to disk.
    """
    out = df.copy()

    # 1. Drop unkeyed rows
    out = out.dropna(subset=["video_id"]).reset_index(drop=True)
    out["video_id"] = out["video_id"].str.strip()
    out = out[out["video_id"] != "nan"].reset_index(drop=True)

    # 2. Parse datetimes (UTC)
    for col in ("video_published_at", "channel_published_at"):
        out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")

    # video_trending__date is date-only (YYYY-MM-DD) — keep as date
    out["video_trending__date"] = pd.to_datetime(
        out["video_trending__date"], errors="coerce"
    ).dt.date

    # 3. Parse ISO 8601 duration → seconds
    out["video_duration_seconds"] = _parse_iso8601_duration(out["video_duration"])
    out = out.drop(columns=["video_duration"])

    # 4. Fix boolean columns
    for col in ("video_licensed_content", "channel_have_hidden_subscribers"):
        out[col] = _safe_bool(out[col])

    # 5. Engagement counts: 0 for null (disabled comments/likes are real zeros)
    for col in ("video_view_count", "video_like_count", "video_comment_count",
                "channel_view_count", "channel_subscriber_count", "channel_video_count"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("Int64")

    # 6. Fill low-signal string nulls with sentinels
    out["video_description"] = out["video_description"].fillna("")
    out["video_tags"] = out["video_tags"].fillna("")
    out["video_category_id"] = out["video_category_id"].fillna("Unknown")
    out["channel_country"] = out["channel_country"].fillna("Unknown")
    out["video_trending_country"] = out["video_trending_country"].fillna("Unknown")

    # 7. Dedup: one row per video × country, keeping earliest trending date
    out = (
        out
        .sort_values("video_trending__date", ascending=True, na_position="last")
        .drop_duplicates(subset=["video_id", "video_trending_country"], keep="first")
        .reset_index(drop=True)
    )

    return out


# ---------------------------------------------------------------------------
# Step 2 — interim → processed
# ---------------------------------------------------------------------------

def build_youtube_processed(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature-ready table from the cleaned interim DataFrame.

    Adds:
    - hours_to_trend   : float — time from publish to first trending (TTV proxy).
                         Null when either timestamp is missing.
    - TTV_available    : bool  — False for this dataset (no engagement time-series).
                         Signals downstream that hours_to_trend is a proxy, not true TTV.
    - engagement_rate  : float — (likes + comments) / views, clipped to [0, 1].
                         Proxy for content resonance independent of raw view count.
    - video_age_days   : int   — days between channel creation and video publish.

    Selects the subset of columns needed for index computation and dashboard,
    dropping channel metadata that is not used by any index.

    Returns a DataFrame ready for DuckDB ingestion; does not write to disk.
    """
    out = df.copy()

    # TTV proxy: hours from publication to first appearance in trending
    published = pd.to_datetime(out["video_published_at"], utc=True, errors="coerce")
    trending = pd.to_datetime(out["video_trending__date"].astype(str), utc=True, errors="coerce")
    delta_hours = (trending - published).dt.total_seconds() / 3600
    out["hours_to_trend"] = delta_hours.where(delta_hours >= 0)  # negative = data error
    out["TTV_available"] = False  # no per-hour engagement series in this dataset

    # Engagement rate
    total_engagement = out["video_like_count"].astype(float) + out["video_comment_count"].astype(float)
    views = out["video_view_count"].astype(float).replace(0, pd.NA)
    out["engagement_rate"] = (total_engagement / views).clip(0, 1)

    # Video age at publish time relative to channel creation
    channel_birth = pd.to_datetime(out["channel_published_at"], utc=True, errors="coerce")
    age_delta = published - channel_birth
    out["video_age_days"] = (age_delta.dt.total_seconds() / 86400).clip(lower=0)

    # Final column selection for index pipeline
    keep = [
        # Identity
        "video_id",
        "video_trending_country",
        "video_trending__date",
        # Content
        "video_title",
        "video_description",
        "video_tags",
        "video_category_id",
        "video_duration_seconds",
        # Timestamps
        "video_published_at",
        # Engagement raw
        "video_view_count",
        "video_like_count",
        "video_comment_count",
        # Derived features
        "hours_to_trend",
        "TTV_available",
        "engagement_rate",
        "video_age_days",
        # Channel context (useful for normalisation)
        "channel_id",
        "channel_title",
        "channel_subscriber_count",
        "channel_video_count",
        "channel_country",
    ]
    return out[keep].reset_index(drop=True)
