# Virality Code

Décoder, mesurer et prédire la viralité d'un post social à partir de ses caractéristiques intrinsèques — rythme de diffusion, charge émotionnelle, format, timing — indépendamment de la taille de l'audience initiale. Le MVP calcule deux indices reproductibles (**TTV** — Temps jusqu'à Viralité, **EWI** — Poids Émotionnel) et les expose dans un dashboard Streamlit adossé à une base DuckDB. Voir `BRIEF.md` pour le cadrage complet et `CLAUDE.md` pour l'architecture détaillée.

## Démarrer le projet

```bash
# 1. Créer et activer un environnement virtuel (Python 3.11+)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Déposer le dataset Kaggle brut dans data/raw/
#    (cf. BRIEF.md §3 pour la source)

# 4. Lancer le dashboard Streamlit
streamlit run src/dashboard/app.py
```

La base DuckDB est créée automatiquement au premier appel de `virality.db.duckdb_client.get_connection()` dans `data/db/virality.duckdb`.

## Organisation du dépôt

Voir la section **Architecture** de `CLAUDE.md` pour savoir où placer les datasets, le code de nettoyage, les indices et les notebooks.
