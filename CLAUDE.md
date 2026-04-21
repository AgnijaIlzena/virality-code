# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

The directory skeleton is in place (see **Architecture** below). `requirements.txt` lists the runtime deps; there is no test runner, no linter, no CI yet — don't fabricate invocations for tooling that hasn't been introduced, scaffold it explicitly with the user first.

`BRIEF.md` is the source of truth for scope and is written in French. Read it before making architectural decisions — it encodes non-obvious priority calls (e.g. which indices are MVP vs. stretch, fallback strategies when data is missing).

## How to run things

```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

DuckDB is lazily opened on first call to `virality.db.duckdb_client.get_connection()`; the file lives at `data/db/virality.duckdb` and is git-ignored.

## Target stack

Python 3.11+ · Pandas · Scikit-learn · Streamlit · FastAPI · DuckDB. When introducing new tooling, stay within this stack unless there's a concrete reason to deviate.

## Architecture

```
virality-code/
├── BRIEF.md                 # scope & priorities (FR) — source of truth
├── CLAUDE.md                # this file
├── README.md                # quickstart
├── requirements.txt
├── .gitignore
│
├── data/                    # git-ignored contents; only .gitkeep is tracked
│   ├── raw/                 # immutable originals (Kaggle dumps, etc.) — never edited in place
│   ├── interim/             # intermediate artefacts from cleaning steps
│   ├── processed/           # final feature tables ready for indices / modelling
│   ├── external/            # third-party lookups: lexiques d'indignation, stopwords, emoji maps
│   └── db/                  # DuckDB file (virality.duckdb) — computed indices cache
│
├── notebooks/               # numbered, commented: 01_exploration, 02_cleaning, 03_ewi_calibration, …
│                            # notebooks import from src/virality, they do not redefine logic
│
├── src/
│   ├── virality/            # core package — importable as `virality.*`
│   │   ├── config.py        # centralised paths + EWI coefficients (α, β, γ, δ)
│   │   ├── data/            # loading (loader.py) and cleaning (cleaner.py) — raw → interim → processed
│   │   ├── indices/         # TTV (ttv.py), EWI (ewi.py), secondary indices (vqs.py, cer.py, hsi.py)
│   │   ├── nlp/             # language detection, sentiment (VADER / CamemBERT), outrage, humor, emoji density
│   │   └── db/              # DuckDB client — read/write cached indices & intermediate aggregates
│   │
│   └── dashboard/           # Streamlit MVP
│       └── app.py           # entry point: `streamlit run src/dashboard/app.py`
│
├── api/                     # FastAPI scoring service (stretch — do not scaffold until MVP ships)
├── scripts/                 # CLI entry points (e.g. `python scripts/run_pipeline.py`) — thin wrappers over src/virality
└── tests/                   # pytest tests for indices & data modules
```

### Where things belong

- **Adding a new dataset** → drop the original in `data/raw/`, add a loader in `src/virality/data/loader.py`, never mutate `data/raw/`.
- **Writing cleaning logic** → `src/virality/data/cleaner.py` producing files in `data/interim/` (step-by-step) and final outputs in `data/processed/`.
- **Implementing an index** → one module per index under `src/virality/indices/`. Pure column-level transforms taking a DataFrame and returning a DataFrame with the new column. No I/O inside index functions.
- **Calibrating / exploring** → notebooks under `notebooks/`, prefixed with a two-digit order. Notebooks must import from `src.virality.*` — never duplicate pipeline logic inline.
- **Persisting computed indices** → write through `virality.db.duckdb_client.get_connection()`. DuckDB is the shared cache between the pipeline, notebooks and the Streamlit dashboard.
- **Dashboard views** → `src/dashboard/app.py` (or components under `src/dashboard/`). The dashboard reads from DuckDB, it does not recompute indices on the fly.
- **NLP model choice** → `src/virality/nlp/` keeps language detection in one place so sentiment/humor modules can dispatch to VADER (EN) vs. CamemBERT (FR) without callers needing to know.

## Product shape & delivery order

The brief prescribes a specific build order — respect it, since later deliverables depend on artifacts from earlier ones:

1. **Reproducible indices pipeline** (Pandas-based transforms on the raw dataset → index columns)
2. **Exploratory notebooks** (commented, used to calibrate coefficients)
3. **Streamlit MVP** — visualisation/exploration of indices (this is the primary deliverable)
4. *Stretch*: FastAPI scoring service ("pre-publication" prediction)
5. *Stretch*: HTML/markdown data-story "anatomie d'un post viral par plateforme"

Do not jump ahead to the FastAPI or data-story without explicit user direction — they are gated behind the MVP.

## The two MVP indices

Both indices are the core IP of this project. Their formulas are fixed by the brief:

- **TTV (Temps jusqu'à Viralité)** — time in hours for cumulative engagement to cross 50% of total. Requires a timestamped engagement series. **Fallback is required**: when the series is absent, either use a proxy (e.g. `engagement_24h / engagement_total`) or mark `TTV_available = False`. Do not silently drop rows that lack the series.
- **EWI (Emotional Weight Index)** — weighted combo: `α·|sentiment| + β·outrage + γ·humor + δ·emoji_density`, initial coefficients `(0.4, 0.3, 0.2, 0.1)`. These are **starting values** meant to be re-calibrated via regression once labels exist. Output is normalised to `[0, 1]`.
  - Sentiment: VADER for EN, CamemBERT-sentiment for FR — dataset language determines the choice, so detect language before picking the model.
  - Humor/outrage: HuggingFace models or lexical heuristics — pick based on what the dataset actually supports.

Secondary indices (VQS, CER, HSI) are explicitly *nice-to-have* and should not block MVP work.

## Data

Primary dataset: [Kaggle — Social Media Engagement](https://www.kaggle.com/code/nigarali/social-media-engagement/input). Before writing any index code, verify the four unknowns from the brief: column availability, volumetry, presence of an engagement time-series (critical for TTV), and content language (drives NLP model choice). These checks should be the first notebook.

## Architectural guidance for future Claude sessions

- **Index computation should be pure, column-level transforms** — the Streamlit app and the eventual FastAPI service will both consume the same pipeline output, so keep indices in a reusable module, not embedded in notebook cells or app callbacks.
- **Coefficients (α, β, γ, δ for EWI) must be configurable**, not hardcoded inside the compute function. They will be re-calibrated.
- **Language-dependent NLP choices** (VADER vs. CamemBERT) need a detection step upstream; don't hardcode one.
