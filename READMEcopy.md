# Virality Code

Décoder, mesurer et prédire la viralité d'un post social à partir de ses caractéristiques intrinsèques — rythme de diffusion, charge émotionnelle, format, timing — indépendamment de la taille de l'audience initiale. Le MVP calcule deux indices reproductibles (**TTV** — Temps jusqu'à Viralité, **EWI** — Poids Émotionnel) et les expose dans un dashboard Streamlit adossé à une base DuckDB. Voir `BRIEF.md` pour le cadrage complet et `CLAUDE.md` pour l'architecture détaillée.

## Démarrer le projet

```bash
# 1. Créer et activer un environnement virtuel (Python 3.11+)
python -m venv .venv
# .venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
source .venv/Scripts/activate # bash

# 2. Installer les dépendances
python -m pip install -r requirements.txt

# 3. Installer le package local en mode éditable (requis pour les imports virality.*)
python -m pip install -e .

# 4. Configurer l'API Kaggle
#    Kaggle → Account → Settings → Create New API Token → sauvegarder kaggle.json dans :
#    Windows : C:\Users\<vous>\.kaggle\kaggle.json
#    ou ajouter dans .env

# 5. Télécharger les datasets
python scripts/download_data.py
#    → copie les fichiers dans data/raw/social_media_engagement/ et data/raw/youtube_trending/
#    Pour ajouter d'autres datasets, décommenter les slugs dans DATASETS dans scripts/download_data.py

# 6. Lancer le pipeline ETL (nettoyage + features + DuckDB)
python scripts/run_pipeline.py
#    → data/interim/youtube_trending_clean.parquet   (données nettoyées, dédupliquées)
#    → data/processed/youtube_trending.parquet        (features prêtes pour les indices)
#    → data/db/virality.duckdb  table : youtube_trending
#    Durée estimée : 5–10 min sur le dataset complet (9M lignes)
#    Option : --source youtube (seul dataset disponible pour l'instant)

# 7. Vérifier le résultat dans DuckDB (optionnel)
python -c "
from virality.db.duckdb_client import get_connection
con = get_connection()
print(con.execute('SELECT COUNT(*), ROUND(AVG(hours_to_trend),1) FROM youtube_trending').fetchdf())
"

# 8. Lancer le dashboard Streamlit
streamlit run src/dashboard/app.py
```

La base DuckDB est créée automatiquement au premier appel de `virality.db.duckdb_client.get_connection()` dans `data/db/virality.duckdb`.

## Fichiers produits par le pipeline

| Fichier | Contenu | Lignes approx. |
|---|---|---|
| `data/interim/youtube_trending_clean.parquet` | CSV nettoyé, types corrigés, dédupliqué par vidéo × pays | ~1 M |
| `data/processed/youtube_trending.parquet` | 21 colonnes features (TTV proxy, engagement_rate, …) | ~1 M |
| `data/db/virality.duckdb` table `youtube_trending` | Même contenu que processed, requêtable via SQL | ~1 M |

Ces fichiers sont dans `.gitignore` — chaque membre de l'équipe les régénère avec les étapes 5 et 6.

## Organisation du dépôt

Voir la section **Architecture** de `CLAUDE.md` pour savoir où placer les datasets, le code de nettoyage, les indices et les notebooks.
