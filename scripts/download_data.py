"""Download datasets from Kaggle into data/raw/.

Usage:
    python scripts/download_data.py

Requires kaggle.json at C:/Users/<you>/.kaggle/kaggle.json
Get it at: kaggle.com → Account → Create New API Token
"""
import shutil
from pathlib import Path

import kagglehub

from virality.config import DATA_RAW

# Notebooks (kaggle.com/code/...) — use notebook_output_download
NOTEBOOKS = {
    "social_media_engagement": "nigarali/social-media-engagement",
}

# Datasets (kaggle.com/datasets/...) — use dataset_download
DATASETS = {
    "youtube_trending": "canerkonuk/youtube-trending-videos-global",
    # "twitter_sentiment140": "kazanova/sentiment140",
    # "reddit_top_posts":     "datasets/reddit",
}


def _copy_to(cache_path: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for f in Path(cache_path).iterdir():
        if f.is_dir():
            shutil.copytree(f, dest / f.name, dirs_exist_ok=True)
        else:
            shutil.copy(f, dest / f.name)


def download_all() -> None:
    for name, slug in NOTEBOOKS.items():
        print(f"Downloading notebook output {slug} ...")
        cache_path = kagglehub.notebook_output_download(slug)
        _copy_to(cache_path, DATA_RAW / name)
        print(f"  → {DATA_RAW / name}")

    for name, slug in DATASETS.items():
        print(f"Downloading dataset {slug} ...")
        cache_path = kagglehub.dataset_download(slug)
        _copy_to(cache_path, DATA_RAW / name)
        print(f"  → {DATA_RAW / name}")

    print("Done.")


if __name__ == "__main__":
    download_all()
