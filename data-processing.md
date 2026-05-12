# Data Processing Pipeline

---

## English Version

### Overview

Once raw data is dropped into `data/raw/`, it flows through a structured pipeline before reaching the Streamlit dashboard or notebooks. The architecture enforces a strict separation between reading, cleaning, transforming, and persisting.

---

### Step 1 — `data/raw/` (source, immutable)

Drop the original Kaggle file here and **never edit it in place**. It is the immutable source of truth.

Current file: `social_media_engagement_data.xlsx`

---

### Step 2 — `src/virality/data/loader.py` (load raw → DataFrame)

Reads from `data/raw/` and returns a raw, untransformed DataFrame. No cleaning logic here.

```python
# src/virality/data/loader.py
from virality.config import DATA_RAW
import pandas as pd

def load_social_media() -> pd.DataFrame:
    return pd.read_excel(DATA_RAW / "social_media_engagement_data.xlsx")
```

> **Status**: missing — needs to be created.

---

### Step 3 — `src/virality/data/cleaner.py` (clean → interim + processed)

Takes the raw DataFrame and applies all cleaning logic as importable functions. Writes output to two locations:

- `data/interim/` — one file per cleaning step (e.g. `01_cleaned_types.parquet`, `02_deduped.parquet`). Useful for debugging and auditing intermediate state.
- `data/processed/` — the final, fully cleaned DataFrame, ready for index computation.

Cleaning steps (based on `nettoyage_donnees.py`):

1. Standardise column names (strip whitespace, replace spaces with `_`)
2. Parse `Post_Timestamp` to datetime
3. Cast numeric columns (`Likes`, `Comments`, `Shares`, `Impressions`, `Reach`, `Engagement_Rate`)
4. Normalise `Post_Content` (lowercase, strip)
5. Drop rows missing `Post_Content` or `Engagement_Rate`
6. Fill known nullable columns (`Campaign_ID`, `Influencer_ID` → `"Unknown"`)
7. Deduplicate on `Post_ID`

> **Status**: `nettoyage_donnees.py` exists but is a flat script with a hardcoded path. It needs to be refactored into `cleaner.py` as proper functions using `config.DATA_RAW`, and must save output files to `data/interim/` and `data/processed/`.

---

### Step 4 — `src/virality/indices/` (pure transforms → index columns)

Each index module receives the processed DataFrame and **adds a column**, returning a new DataFrame. No file I/O inside these functions.

| File | Output columns |
|------|---------------|
| `ttv.py` | `TTV`, `TTV_available` |
| `ewi.py` | `sentiment_score`, `outrage_score`, `humor_score`, `emoji_density`, `EWI` |
| `vqs.py` | `VQS` *(nice-to-have)* |
| `cer.py` | `CER` *(nice-to-have)* |
| `hsi.py` | `HSI` *(nice-to-have)* |

EWI coefficients (α, β, γ, δ) are never hardcoded inside the function — they are read from `config.EWI_COEFFICIENTS` and designed to be re-calibrated via regression.

---

### Step 5 — `src/virality/db/duckdb_client.py` (persist to DuckDB)

After indices are computed, the enriched DataFrame is written into `data/db/virality.duckdb`. The Streamlit dashboard and notebooks **read exclusively from DuckDB** — they never recompute indices on the fly.

---

### Full Flow Diagram

```
data/raw/
    │
    └── loader.py ──────────────────▶  raw DataFrame
                                            │
                                       cleaner.py
                                            │
                           ┌────────────────┴────────────────┐
                      data/interim/                    data/processed/
                      (step artifacts)               (final clean table)
                                                            │
                                                  indices/*.py
                                                  (pure transforms)
                                                            │
                                                  DataFrame + index columns
                                                            │
                                                   duckdb_client.py
                                                            │
                                                 data/db/virality.duckdb
                                                            │
                                         ┌──────────────────┴──────────────┐
                                    dashboard/app.py                  notebooks/
```

---

---

## Version Française

### Vue d'ensemble

Une fois les données brutes déposées dans `data/raw/`, elles traversent un pipeline structuré avant d'atteindre le dashboard Streamlit ou les notebooks. L'architecture impose une séparation stricte entre la lecture, le nettoyage, la transformation et la persistance.

---

### Étape 1 — `data/raw/` (source, immuable)

Déposer le fichier Kaggle d'origine ici et **ne jamais le modifier en place**. C'est la source de vérité immuable du projet.

Fichier actuel : `social_media_engagement_data.xlsx`

---

### Étape 2 — `src/virality/data/loader.py` (lecture brute → DataFrame)

Lit depuis `data/raw/` et retourne un DataFrame brut, sans aucune transformation. Aucune logique de nettoyage ici.

```python
# src/virality/data/loader.py
from virality.config import DATA_RAW
import pandas as pd

def load_social_media() -> pd.DataFrame:
    return pd.read_excel(DATA_RAW / "social_media_engagement_data.xlsx")
```

> **Statut** : fichier manquant — à créer.

---

### Étape 3 — `src/virality/data/cleaner.py` (nettoyage → interim + processed)

Prend le DataFrame brut et applique toute la logique de nettoyage sous forme de fonctions importables. Écrit les résultats dans deux emplacements :

- `data/interim/` — un fichier par étape de nettoyage (ex. `01_cleaned_types.parquet`, `02_deduped.parquet`). Utile pour déboguer et auditer les états intermédiaires.
- `data/processed/` — le DataFrame final entièrement nettoyé, prêt pour le calcul des indices.

Étapes de nettoyage (basées sur `nettoyage_donnees.py`) :

1. Standardisation des noms de colonnes (suppression des espaces, remplacement par `_`)
2. Parsing de `Post_Timestamp` en datetime
3. Conversion des colonnes numériques (`Likes`, `Comments`, `Shares`, `Impressions`, `Reach`, `Engagement_Rate`)
4. Normalisation de `Post_Content` (minuscules, strip)
5. Suppression des lignes sans `Post_Content` ou `Engagement_Rate`
6. Remplissage des colonnes nullables connues (`Campaign_ID`, `Influencer_ID` → `"Unknown"`)
7. Déduplication sur `Post_ID`

> **Statut** : `nettoyage_donnees.py` existe mais est un script plat avec un chemin codé en dur. Il doit être refactorisé en `cleaner.py` sous forme de fonctions utilisant `config.DATA_RAW`, et doit sauvegarder les fichiers de sortie dans `data/interim/` et `data/processed/`.

---

### Étape 4 — `src/virality/indices/` (transformations pures → colonnes d'indices)

Chaque module d'indice reçoit le DataFrame traité et **ajoute une colonne**, retournant un nouveau DataFrame. Aucun I/O fichier à l'intérieur de ces fonctions.

| Fichier | Colonnes produites |
|---------|-------------------|
| `ttv.py` | `TTV`, `TTV_available` |
| `ewi.py` | `sentiment_score`, `outrage_score`, `humor_score`, `emoji_density`, `EWI` |
| `vqs.py` | `VQS` *(nice-to-have)* |
| `cer.py` | `CER` *(nice-to-have)* |
| `hsi.py` | `HSI` *(nice-to-have)* |

Les coefficients EWI (α, β, γ, δ) ne sont jamais codés en dur dans la fonction — ils sont lus depuis `config.EWI_COEFFICIENTS` et conçus pour être recalibrés par régression.

---

### Étape 5 — `src/virality/db/duckdb_client.py` (persistance dans DuckDB)

Une fois les indices calculés, le DataFrame enrichi est écrit dans `data/db/virality.duckdb`. Le dashboard Streamlit et les notebooks **lisent exclusivement depuis DuckDB** — ils ne recalculent jamais les indices à la volée.

---

### Schéma du flux complet

```
data/raw/
    │
    └── loader.py ──────────────────▶  DataFrame brut
                                            │
                                       cleaner.py
                                            │
                           ┌────────────────┴────────────────┐
                      data/interim/                    data/processed/
                      (artefacts par étape)          (table finale nettoyée)
                                                            │
                                                  indices/*.py
                                                  (transformations pures)
                                                            │
                                                  DataFrame + colonnes d'indices
                                                            │
                                                   duckdb_client.py
                                                            │
                                                 data/db/virality.duckdb
                                                            │
                                         ┌──────────────────┴──────────────┐
                                    dashboard/app.py                  notebooks/
```
