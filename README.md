# ⚡ Virality Code

> Décoder, mesurer et prédire la viralité d'un post social à partir de ses
> caractéristiques intrinsèques — charge émotionnelle, format, timing, plateforme —
> indépendamment de la taille de l'audience initiale.

---

## 🎯 Problématique

La viralité sur les réseaux sociaux semble relever de la chance.
Pourtant, derrière chaque post qui explose se cachent des **patterns mesurables**.

> **Peut-on prédire si un post va devenir viral, uniquement à partir
> de ses caractéristiques — avant même de le publier ?**

---

## 📊 Ce que fait l'application

| Page | Description |
|---|---|
| 🏠 Vue d'ensemble | KPIs globaux, distribution du Virality Score, répartition par tier |
| 📊 Indices TTV & EWI | Visualisation du Poids Émotionnel et du Temps jusqu'à Viralité |
| 🔬 Analyse par plateforme | Comparaison Facebook / Instagram / Twitter / LinkedIn |
| 🤖 Modèle ML | Performance du Random Forest, importance des variables |
| 🎯 Prédire mon post | Saisir les caractéristiques d'un post et obtenir une prédiction |
| 💡 Explorateur | Table triable de tous les posts avec leurs indices |
| ⚙️ Config | Informations sur le dataset, export CSV |

---

## 🧠 Les indices calculés

### Virality Score
```
(Engagement / Impressions) × 0.6 + (Engagement / Reach) × 0.4
```
Mesure l'efficacité d'un post — combien de personnes qui l'ont vu ont interagi.

### EWI — Emotional Weight Index
```
EWI = α×|sentiment| + β×outrage + γ×humour + δ×emoji_density
Coefficients par défaut : α=0.4, β=0.3, γ=0.2, δ=0.1
```
Mesure la charge émotionnelle du post. Ajustable en temps réel dans la sidebar.

### TTV — Temps jusqu'à Viralité
```
TTV_proxy = (1 − Virality_Score_norm) × 72h
```
Estime la vitesse d'explosion d'un post. Un TTV court = post qui prend vite.

---

## 🚀 Installation et lancement

### Prérequis
- **Python 3.11 ou supérieur** — [télécharger ici](https://www.python.org/downloads/)
- **Git** — [télécharger ici](https://git-scm.com/)

---

### Étape 1 — Cloner le projet

```bash
git clone https://github.com/.....
cd virality-code
```

---

### Étape 2 — Créer et activer l'environnement virtuel

```bash
# Créer le venv
python -m venv .venv

# Activer sur Windows
.venv\Scripts\activate

# Activer sur macOS / Linux
source .venv/bin/activate
```

> ✅ Tu dois voir `(.venv)` apparaître au début de ta ligne de commande.

---

### Étape 3 — Installer les dépendances

```bash
pip install -r requirements.txt
```

Cela installe automatiquement :

| Package | Rôle |
|---|---|
| `pandas` | Manipulation des données |
| `scikit-learn` | Modèle Random Forest |
| `streamlit` | Dashboard interactif |
| `plotly` | Graphiques interactifs |
| `duckdb` | Base de données légère |
| `openpyxl` | Lecture du fichier Excel |
| `jupyter` | Notebooks d'exploration |

---

### Étape 4 — Ajouter le dataset

Dépose le fichier de données dans le dossier `data/raw/` :

```
virality-code/
└── data/
    └── raw/
        └── social_media_engagement_data.xlsx  ← ici
```

> 📥 Le dataset est disponible sur
> [Kaggle — Social Media Engagement](https://www.kaggle.com/code/nigarali/social-media-engagement/input)
> Le fichier est git-ignoré pour des raisons de taille — il faut le télécharger manuellement.

---

### Étape 5 — Lancer le dashboard

```bash
streamlit run src/dashboard/app.py
```

L'application s'ouvre automatiquement dans ton navigateur à l'adresse :
```
http://localhost:8501
```

---

## 📁 Structure du projet

```
virality-code/
├── data/
│   ├── raw/          # Dataset original Excel (non versionné)
│   ├── interim/      # Données nettoyées intermédiaires
│   ├── processed/    # Features finales prêtes pour le modèle
│   └── db/           # Base DuckDB (cache des indices)
│
├── src/
│   ├── dashboard/
│   │   └── app.py    # ← Point d'entrée Streamlit
│   └── virality/
│       ├── config.py         # Chemins et coefficients EWI
│       ├── data/             # Chargement et nettoyage
│       ├── indices/          # Calcul TTV, EWI, Virality Score
│       ├── nlp/              # Sentiment, humour, outrage
│       └── db/               # Client DuckDB
│
├── notebooks/        # Exploration et calibration des indices
├── requirements.txt  # Dépendances Python
└── README.md
```

---

## ⚙️ Commandes utiles

```bash
# Relancer le dashboard
streamlit run src/dashboard/app.py

# Mettre à jour les dépendances après un git pull
pip install -r requirements.txt

# Lancer un notebook Jupyter
jupyter notebook notebooks/
```

---

## 👥 Équipe

Projet réalisé dans le cadre du **Projet Fil Rouge** — Ynov 2026.

---

## 🛠️ Stack technique

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)
![Plotly](https://img.shields.io/badge/Plotly-latest-purple)
![DuckDB](https://img.shields.io/badge/DuckDB-0.10+-yellow)