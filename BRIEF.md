# 🚀 Virality Code — Project BRIEF

> Brief destiné à **Claude Code** pour scaffolder, développer et itérer sur le projet.
> Stack cible : **Python 3.11+ · Pandas · Scikit-learn · Streamlit · FastAPI**
> Auteur : Agnia Ilzena ·  Avril 2026

---

## 1. 🎯 Problématique

La viralité sur les réseaux sociaux semble relever de la magie ou de la chance.
Pourtant, derrière chaque post qui explose se cachent des **patterns mesurables** :
rythme de diffusion, charge émotionnelle, format, timing, profil de l'auteur.

**Question centrale :**
> Peut-on **décoder, mesurer et prédire** la viralité d'un post à partir de ses
> caractéristiques intrinsèques, indépendamment de la taille de l'audience initiale ?

**Objectifs business :**
- Fournir aux community managers, marques et créateurs un **outil d'aide à la
  décision avant publication**.
- Transformer l'intuition créative en **science reproductible**.
- Produire une **data-story publique** ("anatomie d'un post viral") à fort potentiel presse.

**Livrables finaux (dans l'ordre) :**
1. Pipeline reproductible de calcul des indices
2. Analyses exploratoires + notebooks commentés
3. **Application Streamlit** (MVP) — visualisation et exploration des indices
4. *(Stretch goal)* **API FastAPI** de scoring prédictif *"avant publication"*
5. *(Stretch goal)* Data-story HTML/markdown : *"Anatomie d'un post viral par plateforme"*

---

## 2. 📐 Indices à construire (focus MVP)

Deux indices sont **prioritaires** pour cette première itération :

### 2.1 TTV — Temps jusqu'à Viralité
```
TTV = t( cumulative_engagement(t) ≥ 0.5 × total_engagement )
```
- **Unité** : heures depuis la publication.
- **Interprétation** : vitesse d'explosion. Un TTV court = post qui "prend" vite.
- **Prérequis data** : horodatage de publication + série temporelle
  d'engagement (ou au minimum engagement à plusieurs timestamps).
- **Fallback** si la série temporelle est absente : estimer TTV via un proxy
  (ex. ratio engagement_24h / engagement_total si disponible, sinon flag `TTV_available = False`).

### 2.2 EWI — Poids Émotionnel (Emotional Weight Index)
```
EWI = α × |sentiment| + β × outrage_score + γ × humor_score + δ × emoji_density
```
- **Coefficients initiaux** : α=0.4, β=0.3, γ=0.2, δ=0.1 (à calibrer ensuite par régression).
- **Calcul des composants** :
  - `sentiment` → VADER (anglais) ou CamemBERT-sentiment (FR)
  - `outrage_score` → lexique d'indignation (mots-clés + modèle fine-tuné si besoin)
  - `humor_score` → modèle HuggingFace `humor-detection` ou heuristique (ponctuation, !, LOL, 😂)
  - `emoji_density` → `count(emojis) / count(chars)`
- **Output** : score normalisé `[0, 1]`.

### 2.3 Indices secondaires (nice-to-have, non prioritaires MVP)
- **VQS** — Quotient de Viralité : `(likes + 2×shares + 3×comments) / followers`
- **CER** — Entropie du contenu (Shannon sur vocabulaire)
- **HSI** — Force du hook (importance SHAP des 10 premiers mots)

---

## 3. 📦 Données

### 3.1 Dataset principal identifié
**[Kaggle — Social Media Engagement](https://www.kaggle.com/code/nigarali/social-media-engagement/input)**

À inspecter dès l'étape 1 pour vérifier :
- [ ] Colonnes disponibles (likes, shares, comments, timestamp, content text, platform, follower_count ?)
- [ ] Volumétrie (nb de posts, nb de plateformes)
- [ ] Présence d'une série temporelle d'engagement (critique pour TTV)
- [ ] Langue(s) des contenus (impacte le modèle NLP à utiliser)

### 3.2 Datasets candidats complémentaires (à explorer)
- Kaggle — *Twitter/X Sentiment140* (1.6M tweets)
- Kaggle — *YouTube Trending Videos (multi-country)*
- Kaggle — *Reddit Top Posts*
- Kaggle — *TikTok Trending*
- data.gouv.fr — *Pratiques en ligne des personnes (INSEE)* pour con