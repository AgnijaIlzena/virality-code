"""
Streamlit entry point — Virality Code Dashboard
Run with: streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

# ── resolve imports whether launched from repo root or src/ ──────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Virality Code",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .metric-card {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    border: 1px solid #2d2d4e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
  }
  .metric-card h2 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    margin: 0;
    color: #a78bfa;
  }
  .metric-card p { margin: 0; color: #94a3b8; font-size: 0.85rem; }

  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
  }
  .badge-viral { background: #4ade8022; color: #4ade80; border: 1px solid #4ade8044; }
  .badge-mid   { background: #fbbf2422; color: #fbbf24; border: 1px solid #fbbf2444; }
  .badge-low   { background: #f8717122; color: #f87171; border: 1px solid #f8717144; }

  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #a78bfa;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    border-bottom: 1px solid #2d2d4e;
    padding-bottom: 0.5rem;
  }
</style>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────

PLOTLY_TEMPLATE = "plotly_dark"
PALETTE = px.colors.qualitative.Pastel

@st.cache_data(show_spinner="Chargement des données…")
def load_raw() -> pd.DataFrame:
    """Load raw Excel — tries multiple plausible paths."""
    candidates = [
        ROOT / "data" / "raw" / "social_media_engagement_data.xlsx",
        ROOT / "src" / "virality" / "data" / "social_media_engagement_data.xlsx",
        Path("data/raw/social_media_engagement_data.xlsx"),
        Path("social_media_engagement_data.xlsx"),
    ]
    for p in candidates:
        if p.exists():
            return pd.read_excel(p)
    raise FileNotFoundError(
        "Fichier introuvable. Place social_media_engagement_data.xlsx dans data/raw/"
    )


@st.cache_data(show_spinner="Calcul des indices…")
def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # ── normalise column names ────────────────────────────────────────────────
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # ── timestamp ────────────────────────────────────────────────────────────
    if "Post_Timestamp" in df.columns:
        df["Post_Timestamp"] = pd.to_datetime(df["Post_Timestamp"], errors="coerce")
    elif "Date" in df.columns and "Time" in df.columns:
        df["Post_Timestamp"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce"
        )

    # ── numerics ─────────────────────────────────────────────────────────────
    for col in ["Likes", "Comments", "Shares", "Impressions", "Reach", "Engagement_Rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Likes", "Comments", "Shares"])

    # ── engagement totaux ─────────────────────────────────────────────────────
    df["Total_Engagement"] = df["Likes"] + df["Comments"] + df["Shares"]

    imp = df["Impressions"] if "Impressions" in df.columns else df["Total_Engagement"] + 1
    reach = df["Reach"] if "Reach" in df.columns else df["Total_Engagement"] + 1

    df["Engagement_per_Impression"] = df["Total_Engagement"] / (imp + 1)
    df["Engagement_per_Reach"]      = df["Total_Engagement"] / (reach + 1)

    # ── Virality Score ────────────────────────────────────────────────────────
    df["Virality_Score"] = (
        df["Engagement_per_Impression"] * 0.6 +
        df["Engagement_per_Reach"] * 0.4
    )
    v_max = df["Virality_Score"].max()
    if v_max > 0:
        df["Virality_Score_norm"] = df["Virality_Score"] / v_max
    else:
        df["Virality_Score_norm"] = 0.0

    # ── EWI ───────────────────────────────────────────────────────────────────
    # sentiment component
    sentiment_map = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0, "Mixed": 0.5}
    if "Sentiment" in df.columns:
        df["Sentiment_Score"] = df["Sentiment"].map(sentiment_map).fillna(0)
    else:
        df["Sentiment_Score"] = 0.0

    # emoji density (simple heuristic on Post_Content)
    def _emoji_density(text: str) -> float:
        if not isinstance(text, str) or len(text) == 0:
            return 0.0
        import unicodedata
        emoji_chars = [c for c in text if unicodedata.category(c) == "So"]
        return len(emoji_chars) / max(len(text), 1)

    if "Post_Content" in df.columns:
        df["Post_Content_str"] = df["Post_Content"].astype(str)
        df["Emoji_Density"]    = df["Post_Content_str"].apply(_emoji_density)
        df["Word_Count"]       = df["Post_Content_str"].str.split().str.len().fillna(0)
        df["Text_Length"]      = df["Post_Content_str"].apply(lambda x: len(x))
    else:
        df["Emoji_Density"] = 0.0
        df["Word_Count"]    = 0
        df["Text_Length"]   = 0

    # humor heuristic: exclamation marks + 😂 LOL
    def _humor_score(text: str) -> float:
        if not isinstance(text, str):
            return 0.0
        score = 0.0
        score += text.count("!") * 0.1
        score += text.lower().count("lol") * 0.3
        score += text.count("😂") * 0.3
        score += text.count("😆") * 0.2
        return min(score, 1.0)

    if "Post_Content" in df.columns:
        df["Humor_Score"] = df["Post_Content_str"].apply(_humor_score)
    else:
        df["Humor_Score"] = 0.0

    # outrage heuristic: CAPS ratio + strong neg words
    OUTRAGE_WORDS = {"scandal", "shocking", "outrage", "disgrace", "horrible",
                     "unacceptable", "wtf", "insane", "disgusting", "fraud"}

    def _outrage_score(text: str) -> float:
        if not isinstance(text, str) or len(text) == 0:
            return 0.0
        caps = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        words = set(text.lower().split())
        lexical = len(words & OUTRAGE_WORDS) / max(len(words), 1)
        return min(caps * 0.5 + lexical * 5, 1.0)

    if "Post_Content" in df.columns:
        df["Outrage_Score"] = df["Post_Content_str"].apply(_outrage_score)
    else:
        df["Outrage_Score"] = 0.0

    # EWI formula (BRIEF §2.2)
    α, β, γ, δ = 0.4, 0.3, 0.2, 0.1
    df["EWI"] = (
        α * df["Sentiment_Score"].abs() +
        β * df["Outrage_Score"] +
        γ * df["Humor_Score"] +
        δ * df["Emoji_Density"] * 100   # scale to same order of magnitude
    ).clip(0, 1)

    # ── TTV proxy ────────────────────────────────────────────────────────────
    # Real TTV needs a time-series; without it we proxy: faster posts have more
    # engagement per impression (small TTV) — we invert & normalise.
    df["TTV_available"] = False
    df["TTV_proxy"] = (
        1 - df["Virality_Score_norm"]          # high virality → low TTV (hours)
    ) * 72                                      # map to a [0, 72h] range

    # ── Hook Score ────────────────────────────────────────────────────────────
    df["Hook_Score"] = (
        df["Likes"] * 0.4 +
        df["Shares"] * 0.4 +
        (imp) * 0.1 +
        (1 / (df["Word_Count"] + 1)) * 100
    )

    # ── Time features ────────────────────────────────────────────────────────
    if "Time" in df.columns:
        try:
            df["Hour"] = df["Time"].astype(str).str.split(":").str[0].astype(int)
        except Exception:
            df["Hour"] = 12
    elif "Post_Timestamp" in df.columns:
        df["Hour"] = df["Post_Timestamp"].dt.hour
    else:
        df["Hour"] = 12

    df["Time_Slot"] = pd.cut(
        df["Hour"],
        bins=[-1, 4, 11, 17, 23],
        labels=["🌙 Nuit", "🌅 Matin", "☀️ Après-midi", "🌆 Soirée"],
    )

    # ── VQS (nice-to-have) ───────────────────────────────────────────────────
    followers_col = next(
        (c for c in df.columns if "follower" in c.lower()), None
    )
    if followers_col:
        df["VQS"] = (df["Likes"] + 2 * df["Shares"] + 3 * df["Comments"]) / (
            df[followers_col].replace(0, np.nan)
        )
    else:
        df["VQS"] = np.nan

    # ── Virality tier ────────────────────────────────────────────────────────
    q33 = df["Virality_Score_norm"].quantile(0.33)
    q66 = df["Virality_Score_norm"].quantile(0.66)

    def _tier(v):
        if v >= q66:
            return "🔥 Viral"
        elif v >= q33:
            return "📈 Moyen"
        return "💤 Faible"

    df["Virality_Tier"] = df["Virality_Score_norm"].apply(_tier)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚡ Virality Code")
        st.markdown("*Décoder la viralité sociale*")
        st.divider()

        page = st.radio(
            "Navigation",
            ["🏠 Vue d'ensemble", "📊 Indices TTV & EWI", "🔬 Analyse par plateforme",
             "💡 Explorateur de posts", "⚙️ Configuration"],
            label_visibility="collapsed",
        )
        st.divider()

        # EWI coefficient sliders
        st.markdown("**Coefficients EWI**")
        alpha = st.slider("α — Sentiment",    0.0, 1.0, 0.4, 0.05)
        beta  = st.slider("β — Outrage",      0.0, 1.0, 0.3, 0.05)
        gamma = st.slider("γ — Humour",       0.0, 1.0, 0.2, 0.05)
        delta = st.slider("δ — Emoji density",0.0, 1.0, 0.1, 0.05)
        st.caption(f"Somme = {alpha+beta+gamma+delta:.2f} (idéal = 1.0)")

    # ── LOAD DATA ─────────────────────────────────────────────────────────────
    try:
        raw = load_raw()
        df  = build_features(raw)
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "📂 Dépose `social_media_engagement_data.xlsx` dans `data/raw/` "
            "puis recharge la page."
        )
        st.stop()

    # recompute EWI with sidebar coefficients
    df["EWI"] = (
        alpha * df["Sentiment_Score"].abs() +
        beta  * df["Outrage_Score"] +
        gamma * df["Humor_Score"] +
        delta * df.get("Emoji_Density", pd.Series(0, index=df.index)) * 100
    ).clip(0, 1)

    platform_col = next(
        (c for c in df.columns if "platform" in c.lower()), None
    )
    platforms = ["Toutes"] + (sorted(df[platform_col].dropna().unique().tolist())
                              if platform_col else [])

    # ── SIDEBAR FILTERS ───────────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown("**Filtres**")
        selected_platform = st.selectbox("Plateforme", platforms)
        virality_range    = st.slider(
            "Virality Score (normalisé)",
            0.0, 1.0, (0.0, 1.0), 0.01,
        )

    # apply filters
    mask = (
        df["Virality_Score_norm"].between(*virality_range)
    )
    if selected_platform != "Toutes" and platform_col:
        mask &= df[platform_col] == selected_platform
    filtered = df[mask]

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1 — Vue d'ensemble
    # ─────────────────────────────────────────────────────────────────────────
    if page == "🏠 Vue d'ensemble":
        st.markdown("# ⚡ Virality Code")
        st.markdown("### Tableau de bord — Analyse de viralité sociale")
        st.divider()

        # KPIs
        c1, c2, c3, c4, c5 = st.columns(5)
        kpis = [
            (c1, "Posts analysés",    f"{len(filtered):,}",             "dataset filtré"),
            (c2, "Virality Score moy",f"{filtered['Virality_Score_norm'].mean():.3f}", "0 → 1"),
            (c3, "EWI moyen",         f"{filtered['EWI'].mean():.3f}",  "Poids émotionnel"),
            (c4, "TTV proxy moy",     f"{filtered['TTV_proxy'].mean():.1f}h", "estimé"),
            (c5, "Engagement total",  f"{filtered['Total_Engagement'].sum():,.0f}", "L+C+S"),
        ]
        for col, label, value, sub in kpis:
            with col:
                st.markdown(
                    f'<div class="metric-card"><h2>{value}</h2>'
                    f'<p>{label}</p><p style="color:#64748b;font-size:0.75rem">{sub}</p></div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.markdown('<p class="section-title">Distribution du Virality Score</p>',
                        unsafe_allow_html=True)
            fig = px.histogram(
                filtered, x="Virality_Score_norm", nbins=60,
                color_discrete_sequence=["#a78bfa"],
                labels={"Virality_Score_norm": "Virality Score (normalisé)"},
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(showlegend=False, height=300,
                              margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown('<p class="section-title">Répartition par tier</p>',
                        unsafe_allow_html=True)
            tier_counts = filtered["Virality_Tier"].value_counts().reset_index()
            tier_counts.columns = ["Tier", "Count"]
            fig2 = px.pie(
                tier_counts, names="Tier", values="Count",
                color_discrete_sequence=["#4ade80", "#fbbf24", "#f87171"],
                template=PLOTLY_TEMPLATE,
                hole=0.5,
            )
            fig2.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig2, use_container_width=True)

        if platform_col:
            st.markdown('<p class="section-title">Virality Score moyen par plateforme</p>',
                        unsafe_allow_html=True)
            plat_df = (
                filtered.groupby(platform_col)["Virality_Score_norm"]
                .mean().sort_values(ascending=False).reset_index()
            )
            fig3 = px.bar(
                plat_df, x=platform_col, y="Virality_Score_norm",
                color="Virality_Score_norm",
                color_continuous_scale="Purples",
                template=PLOTLY_TEMPLATE,
                labels={"Virality_Score_norm": "Score moyen"},
            )
            fig3.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10),
                               coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2 — Indices TTV & EWI
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "📊 Indices TTV & EWI":
        st.markdown("## 📊 Indices principaux : TTV & EWI")
        st.divider()

        tab_ewi, tab_ttv = st.tabs(["🧠 EWI — Poids Émotionnel", "⏱️ TTV — Temps jusqu'à Viralité"])

        with tab_ewi:
            st.markdown("""
**Formule :** `EWI = α·|sentiment| + β·outrage + γ·humour + δ·emoji_density`

*Coefficients ajustables dans la barre latérale.*
""")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<p class="section-title">Distribution EWI</p>',
                            unsafe_allow_html=True)
                fig = px.histogram(
                    filtered, x="EWI", nbins=50,
                    color_discrete_sequence=["#f59e0b"],
                    template=PLOTLY_TEMPLATE,
                )
                fig.update_layout(height=280, margin=dict(t=5,b=5,l=5,r=5))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown('<p class="section-title">EWI vs Virality Score</p>',
                            unsafe_allow_html=True)
                fig2 = px.scatter(
                    filtered.sample(min(2000, len(filtered))),
                    x="EWI", y="Virality_Score_norm",
                    color="Virality_Tier" if "Virality_Tier" in filtered.columns else None,
                    color_discrete_sequence=["#4ade80", "#fbbf24", "#f87171"],
                    opacity=0.6,
                    template=PLOTLY_TEMPLATE,
                    labels={"Virality_Score_norm": "Virality Score"},
                    trendline="ols",
                )
                fig2.update_layout(height=280, margin=dict(t=5,b=5,l=5,r=5))
                st.plotly_chart(fig2, use_container_width=True)

            # composants breakdown
            st.markdown('<p class="section-title">Composants EWI moyens</p>',
                        unsafe_allow_html=True)
            comp_data = {
                "Composant": ["|Sentiment|", "Outrage", "Humour", "Emoji density×100"],
                "Moyenne": [
                    filtered["Sentiment_Score"].abs().mean(),
                    filtered["Outrage_Score"].mean(),
                    filtered["Humor_Score"].mean(),
                    filtered.get("Emoji_Density", pd.Series(0)).mean() * 100,
                ],
                "Coefficient": [alpha, beta, gamma, delta],
            }
            comp_df = pd.DataFrame(comp_data)
            comp_df["Contribution"] = comp_df["Moyenne"] * comp_df["Coefficient"]
            fig3 = px.bar(
                comp_df, x="Composant", y="Contribution",
                color="Composant",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template=PLOTLY_TEMPLATE,
            )
            fig3.update_layout(height=260, showlegend=False,
                               margin=dict(t=5,b=5,l=5,r=5))
            st.plotly_chart(fig3, use_container_width=True)

        with tab_ttv:
            st.markdown("""
**TTV (Temps jusqu'à Viralité)** = temps (h) pour atteindre 50 % de l'engagement total.

⚠️ *Le dataset ne contient pas de série temporelle d'engagement — TTV est estimé par proxy :*
`TTV_proxy = (1 - Virality_Score_norm) × 72h`
Un post très viral a un TTV court (explose vite). `TTV_available = False` pour tous les posts.
""")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<p class="section-title">Distribution TTV proxy (heures)</p>',
                            unsafe_allow_html=True)
                fig = px.histogram(
                    filtered, x="TTV_proxy", nbins=50,
                    color_discrete_sequence=["#38bdf8"],
                    template=PLOTLY_TEMPLATE,
                    labels={"TTV_proxy": "TTV estimé (heures)"},
                )
                fig.update_layout(height=280, margin=dict(t=5,b=5,l=5,r=5))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown('<p class="section-title">TTV proxy vs EWI</p>',
                            unsafe_allow_html=True)
                fig2 = px.scatter(
                    filtered.sample(min(2000, len(filtered))),
                    x="EWI", y="TTV_proxy",
                    color="Virality_Tier" if "Virality_Tier" in filtered.columns else None,
                    color_discrete_sequence=["#4ade80", "#fbbf24", "#f87171"],
                    opacity=0.5,
                    template=PLOTLY_TEMPLATE,
                    labels={"TTV_proxy": "TTV estimé (h)"},
                )
                fig2.update_layout(height=280, margin=dict(t=5,b=5,l=5,r=5))
                st.plotly_chart(fig2, use_container_width=True)

            if platform_col:
                st.markdown('<p class="section-title">TTV moyen par plateforme</p>',
                            unsafe_allow_html=True)
                ttv_plat = (
                    filtered.groupby(platform_col)["TTV_proxy"]
                    .mean().sort_values().reset_index()
                )
                fig3 = px.bar(
                    ttv_plat, x="TTV_proxy", y=platform_col, orientation="h",
                    color="TTV_proxy",
                    color_continuous_scale="Blues_r",
                    template=PLOTLY_TEMPLATE,
                    labels={"TTV_proxy": "TTV moyen (h)"},
                )
                fig3.update_layout(height=max(200, len(ttv_plat)*45),
                                   margin=dict(t=5,b=5,l=5,r=5),
                                   coloraxis_showscale=False)
                st.plotly_chart(fig3, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 3 — Analyse par plateforme
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "🔬 Analyse par plateforme":
        st.markdown("## 🔬 Analyse par plateforme")
        st.divider()

        if not platform_col:
            st.warning("Colonne 'Platform' non trouvée dans le dataset.")
            st.stop()

        # summary table
        grp = filtered.groupby(platform_col).agg(
            Posts=("Total_Engagement", "count"),
            Virality_moy=("Virality_Score_norm", "mean"),
            EWI_moy=("EWI", "mean"),
            TTV_moy=("TTV_proxy", "mean"),
            Engagement_total=("Total_Engagement", "sum"),
            Likes_moy=("Likes", "mean"),
            Shares_moy=("Shares", "mean"),
            Comments_moy=("Comments", "mean"),
        ).round(4).reset_index()

        st.dataframe(
            grp.style.background_gradient(subset=["Virality_moy", "EWI_moy"],
                                           cmap="Purples"),
            use_container_width=True,
            height=260,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-title">Virality Score par plateforme</p>',
                        unsafe_allow_html=True)
            fig = px.box(
                filtered, x=platform_col, y="Virality_Score_norm",
                color=platform_col,
                color_discrete_sequence=PALETTE,
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(height=320, showlegend=False,
                              margin=dict(t=5,b=5,l=5,r=5))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<p class="section-title">EWI par plateforme</p>',
                        unsafe_allow_html=True)
            fig2 = px.box(
                filtered, x=platform_col, y="EWI",
                color=platform_col,
                color_discrete_sequence=PALETTE,
                template=PLOTLY_TEMPLATE,
            )
            fig2.update_layout(height=320, showlegend=False,
                               margin=dict(t=5,b=5,l=5,r=5))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<p class="section-title">Impact du moment de publication</p>',
                    unsafe_allow_html=True)
        if "Time_Slot" in filtered.columns:
            time_grp = (
                filtered.groupby(["Time_Slot" if platform_col else "Time_Slot",
                                   platform_col])["Virality_Score_norm"]
                .mean().reset_index()
            )
            fig3 = px.bar(
                time_grp, x="Time_Slot", y="Virality_Score_norm",
                color=platform_col,
                barmode="group",
                color_discrete_sequence=PALETTE,
                template=PLOTLY_TEMPLATE,
                labels={"Virality_Score_norm": "Score moyen", "Time_Slot": "Créneau horaire"},
            )
            fig3.update_layout(height=320, margin=dict(t=5,b=5,l=5,r=5))
            st.plotly_chart(fig3, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 4 — Explorateur de posts
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "💡 Explorateur de posts":
        st.markdown("## 💡 Explorateur de posts")
        st.divider()

        sort_col = st.selectbox(
            "Trier par",
            ["Virality_Score_norm", "EWI", "TTV_proxy", "Total_Engagement",
             "Hook_Score", "Likes", "Shares"],
        )
        ascending = st.checkbox("Ordre croissant", False)
        n_rows    = st.slider("Nombre de posts à afficher", 10, 200, 50)

        show_cols = ["Virality_Score_norm", "EWI", "TTV_proxy",
                     "Total_Engagement", "Likes", "Comments", "Shares",
                     "Virality_Tier"]
        if platform_col:
            show_cols = [platform_col] + show_cols
        if "Post_Content" in filtered.columns:
            show_cols = ["Post_Content"] + show_cols
        if "Sentiment" in filtered.columns:
            show_cols = show_cols + ["Sentiment"]

        show_cols = [c for c in show_cols if c in filtered.columns]

        display = (
            filtered[show_cols]
            .sort_values(sort_col, ascending=ascending)
            .head(n_rows)
            .reset_index(drop=True)
        )

        def _style_tier(val):
            colors = {"🔥 Viral": "#4ade8033", "📈 Moyen": "#fbbf2433", "💤 Faible": "#f8717133"}
            return f"background-color: {colors.get(val, '')}"

        if "Virality_Tier" in display.columns:
            styled = display.style.applymap(_style_tier, subset=["Virality_Tier"])
        else:
            styled = display

        st.dataframe(styled, use_container_width=True, height=500)

        st.divider()
        st.markdown('<p class="section-title">Corrélations entre indices</p>',
                    unsafe_allow_html=True)
        corr_cols = [c for c in ["Virality_Score_norm", "EWI", "TTV_proxy",
                                  "Hook_Score", "Total_Engagement",
                                  "Sentiment_Score", "Outrage_Score",
                                  "Humor_Score", "Emoji_Density"]
                     if c in filtered.columns]
        corr = filtered[corr_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            template=PLOTLY_TEMPLATE,
            aspect="auto",
        )
        fig.update_layout(height=420, margin=dict(t=5,b=5,l=5,r=5))
        st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 5 — Configuration
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "⚙️ Configuration":
        st.markdown("## ⚙️ Informations & Configuration")
        st.divider()

        st.markdown("### Dataset chargé")
        st.json({
            "Lignes brutes":    len(raw),
            "Lignes filtrées":  len(filtered),
            "Colonnes":         list(df.columns),
            "Plateforme col":   platform_col,
        })

        st.markdown("### Coefficients EWI actifs")
        st.json({"α (sentiment)": alpha, "β (outrage)": beta,
                 "γ (humour)": gamma, "δ (emoji)": delta,
                 "somme": round(alpha+beta+gamma+delta, 3)})

        st.markdown("### Aperçu des données brutes")
        st.dataframe(raw.head(20), use_container_width=True)

        st.markdown("### Aperçu des features calculées")
        feat_cols = ["Virality_Score_norm", "EWI", "TTV_proxy",
                     "Hook_Score", "Sentiment_Score",
                     "Outrage_Score", "Humor_Score", "Emoji_Density",
                     "Time_Slot", "Virality_Tier"]
        show = [c for c in feat_cols if c in df.columns]
        st.dataframe(df[show].head(20), use_container_width=True)

        st.markdown("### 📥 Export CSV")
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger les données filtrées (CSV)",
            csv, "virality_filtered.csv", "text/csv",
        )


if __name__ == "__main__":
    main()