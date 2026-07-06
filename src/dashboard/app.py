"""
Streamlit entry point — Virality Code Dashboard
Run with: streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import cv2
from PIL import Image
import torch
import clip
import librosa
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import unicodedata

st.set_page_config(
    page_title="Virality Code",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #07070f; color: #e2e8f0; }
  h1, h2, h3 { font-family: 'Syne', sans-serif; }
  section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d0d1f 0%, #0a0a18 100%); border-right: 1px solid #1e1e3a; }
  .metric-card { background: linear-gradient(135deg, #0d0d20 0%, #12122a 100%); border: 1px solid #252550; border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 0.5rem; position: relative; overflow: hidden; }
  .metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #7c3aed, #06b6d4); }
  .metric-card h2 { font-family: 'IBM Plex Mono', monospace; font-size: 1.9rem; margin: 0 0 4px 0; color: #c4b5fd; letter-spacing: -0.02em; }
  .metric-card p { margin: 0; color: #64748b; font-size: 0.82rem; }
  .metric-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 4px; }
  .section-title { font-family: 'Syne', sans-serif; font-size: 0.72rem; color: #7c3aed; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 1rem; display: flex; align-items: center; gap: 8px; }
  .section-title::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, #252550, transparent); }
  .info-box { background: #0d1b2a; border: 1px solid #1e3a5f; border-left: 3px solid #06b6d4; border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.85rem; color: #94a3b8; margin-bottom: 1rem; }
  .warn-box { background: #1a1505; border: 1px solid #3b2f05; border-left: 3px solid #fbbf24; border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.85rem; color: #d4a017; margin-bottom: 1rem; }
  .limit-box { background: #1a0d1a; border: 1px solid #3b1a3b; border-left: 3px solid #c084fc; border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.85rem; color: #d8b4fe; margin-bottom: 1rem; }
  .pred-card-viral { background: linear-gradient(135deg, #052e16 0%, #0a3d20 100%); border: 2px solid #4ade80; border-radius: 20px; padding: 2rem; text-align: center; }
  .pred-card-mid { background: linear-gradient(135deg, #3b2700 0%, #4a3200 100%); border: 2px solid #fbbf24; border-radius: 20px; padding: 2rem; text-align: center; }
  .pred-card-low { background: linear-gradient(135deg, #300 0%, #400 100%); border: 2px solid #f87171; border-radius: 20px; padding: 2rem; text-align: center; }
  .pred-score { font-family: 'IBM Plex Mono', monospace; font-size: 3.5rem; font-weight: 700; margin: 0; }
  .pred-label { font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700; margin: 0.3rem 0; }
  .reco-viral { background: #052e1633; border-left: 3px solid #4ade80; border-radius: 0 8px 8px 0; padding: 0.6rem 1rem; margin: 4px 0; font-size: 0.85rem; color: #86efac; }
  .reco-warn  { background: #3b270633; border-left: 3px solid #fbbf24; border-radius: 0 8px 8px 0; padding: 0.6rem 1rem; margin: 4px 0; font-size: 0.85rem; color: #fde68a; }
  .reco-tip   { background: #0d1b2a33; border-left: 3px solid #38bdf8; border-radius: 0 8px 8px 0; padding: 0.6rem 1rem; margin: 4px 0; font-size: 0.85rem; color: #7dd3fc; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
PALETTE = ["#a78bfa","#34d399","#fb923c","#60a5fa","#f472b6","#facc15","#38bdf8"]
EWI_OUTRAGE_WORDS = {
    "scandal","shocking","outrage","disgrace","horrible","unacceptable",
    "wtf","insane","disgusting","fraud","lie","cheat","corrupt","expose",
    "terrible","awful","worst","disaster","catastrophe","shame","criminal",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="📂 Chargement des données…")
def load_data():
    candidates = [
        ROOT / "data" / "raw" / "social_media_engagement_data.xlsx",
        ROOT / "src" / "virality" / "data" / "social_media_engagement_data.xlsx",
        Path("data/raw/social_media_engagement_data.xlsx"),
        Path("social_media_engagement_data.xlsx"),
    ]
    path = None
    for p in candidates:
        if p.exists():
            path = p
            break
    if path is None:
        raise FileNotFoundError("Fichier introuvable. Place `social_media_engagement_data.xlsx` dans `data/raw/`")
    xl = pd.ExcelFile(path)
    sheet_names_lower = {s.lower().strip(): s for s in xl.sheet_names}
    target = sheet_names_lower.get("working file") or xl.sheet_names[0]
    df = pd.read_excel(path, sheet_name=target)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _emoji_density(text: str) -> float:
    if not isinstance(text, str) or len(text) == 0: return 0.0
    return len([c for c in text if unicodedata.category(c) == "So"]) / max(len(text), 1)


def _humor_score(text: str) -> float:
    if not isinstance(text, str): return 0.0
    s = 0.0
    s += text.count("!") * 0.1
    s += text.lower().count("lol") * 0.3
    s += text.lower().count("haha") * 0.2
    s += text.count("😂") * 0.35 + text.count("😆") * 0.25 + text.count("🤣") * 0.35
    return min(s, 1.0)


def _outrage_score(text: str) -> float:
    if not isinstance(text, str) or len(text) == 0: return 0.0
    caps = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    words = set(text.lower().split())
    lexical = len(words & EWI_OUTRAGE_WORDS) / max(len(words), 1)
    return min(caps * 0.5 + lexical * 5, 1.0)


# ============================================================
# MULTIMODAL ANALYSIS (CLIP pour image/vidéo, librosa + reconnaissance
# vocale optionnelle pour l'audio)
# ============================================================

# Labels envoyés à CLIP en anglais (CLIP est entraîné majoritairement en anglais,
# ça donne une bien meilleure précision), avec leur traduction pour l'affichage.
SCENE_LABELS_EN = [
    "a photo of a person", "food", "a car", "nature landscape", "an animal",
    "technology or gadgets", "a sport or workout scene", "fashion or clothing",
    "a screenshot with text", "a selfie", "an outdoor landscape", "a product photo",
    "a company logo or graphic design", "an office workspace", "a classroom or school",
    "a gym or sports hall", "a beach", "a city street", "a restaurant or cafe",
    "a concert or stage performance", "a mountain or outdoor adventure",
    "a kitchen", "a store or shop", "a pet", "a party or celebration",
]
LABEL_FR = {
    "a photo of a person": "personne",
    "food": "nourriture",
    "a car": "voiture",
    "nature landscape": "nature",
    "an animal": "animal",
    "technology or gadgets": "technologie",
    "a sport or workout scene": "sport",
    "fashion or clothing": "mode",
    "a screenshot with text": "capture d'écran / texte",
    "a selfie": "selfie",
    "an outdoor landscape": "paysage extérieur",
    "a product photo": "produit",
    "a company logo or graphic design": "logo / design graphique",
    "an office workspace": "bureau / entreprise",
    "a classroom or school": "école / salle de classe",
    "a gym or sports hall": "salle de sport",
    "a beach": "plage",
    "a city street": "rue en ville",
    "a restaurant or cafe": "restaurant / café",
    "a concert or stage performance": "concert / scène",
    "a mountain or outdoor adventure": "montagne / aventure",
    "a kitchen": "cuisine",
    "a store or shop": "magasin",
    "a pet": "animal de compagnie",
    "a party or celebration": "fête / célébration",
}

# Seuils indicatifs de similarité cosinus CLIP (échelle typique ~0.15 à ~0.35)
COHERENCE_LOW = 0.19
COHERENCE_MID = 0.25
COHERENCE_FLOOR = 0.15   # borne basse approx. pour la normalisation 0-1
COHERENCE_CEIL = 0.34    # borne haute approx. pour la normalisation 0-1


@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def _encode_image_feat(image):
    """Encode une image PIL en vecteur CLIP normalisé."""
    model, preprocess, device = load_clip()
    with torch.no_grad():
        feat = model.encode_image(preprocess(image).unsqueeze(0).to(device))
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def _encode_text_feat(text):
    """Encode un texte libre en vecteur CLIP normalisé (tronqué à 300 caractères)."""
    if not text or not text.strip():
        return None
    model, preprocess, device = load_clip()
    with torch.no_grad():
        tokens = clip.tokenize([text[:300]], truncate=True).to(device)
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def _classify_feat(feat, labels_en=SCENE_LABELS_EN, top_k=3):
    """Classifie un vecteur image CLIP contre une liste de labels texte (zero-shot).
    Retourne une liste de (label_français, score) triée par score décroissant."""
    model, preprocess, device = load_clip()
    with torch.no_grad():
        text_feat = model.encode_text(clip.tokenize(labels_en).to(device))
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        sims = (feat @ text_feat.T).softmax(dim=-1)[0]
    order = torch.argsort(sims, descending=True)
    return [(LABEL_FR.get(labels_en[i], labels_en[i]), sims[i].item()) for i in order[:top_k]]


def compute_coherence(media_feat, text):
    """Similarité cosinus CLIP entre un média (image/vidéo, ou texte transcrit) et
    le texte du post. Retourne None si l'un des deux manque."""
    text_feat = _encode_text_feat(text)
    if media_feat is None or text_feat is None:
        return None
    return (media_feat @ text_feat.T).item()


def text_text_similarity(text_a, text_b):
    """Similarité sémantique approximative entre deux textes via les embeddings
    texte de CLIP (utile pour comparer une transcription audio à la légende)."""
    feat_a = _encode_text_feat(text_a)
    feat_b = _encode_text_feat(text_b)
    if feat_a is None or feat_b is None:
        return None
    return (feat_a @ feat_b.T).item()


def coherence_verdict(sim):
    """Traduit un score de similarité CLIP en verdict qualitatif."""
    if sim is None:
        return None
    if sim < COHERENCE_LOW:
        return "faible"
    elif sim < COHERENCE_MID:
        return "moyenne"
    return "bonne"


def normalize_score(sim, floor=COHERENCE_FLOOR, ceil=COHERENCE_CEIL):
    """Ramène une similarité CLIP (~0.15-0.35) sur une échelle 0-1 exploitable."""
    if sim is None:
        return None
    return max(0.0, min(1.0, (sim - floor) / (ceil - floor)))


def timing_score(hour):
    """Score 0-1 de qualité du créneau horaire de publication."""
    if 18 <= hour <= 21:
        return 1.0
    if (12 <= hour <= 14) or (5 <= hour <= 8):
        return 0.65
    if hour >= 22 or hour <= 4:
        return 0.20
    return 0.45


def analyze_image(image):
    """Analyse réelle du contenu d'une image via CLIP (zero-shot, ~24 scènes/lieux).
    Retourne (top3_labels_scores, feature_vector)."""
    feat = _encode_image_feat(image)
    topk = _classify_feat(feat, SCENE_LABELS_EN, top_k=3)
    return topk, feat


def extract_video_frames(video_path, every_n=15, max_frames=12):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            if len(frames) >= max_frames:
                break
        count += 1
    cap.release()
    return frames


def analyze_video(video_path):
    """Analyse réelle d'une vidéo : échantillonne des frames sur toute sa durée,
    moyenne leurs embeddings CLIP, puis classifie ce vecteur moyen (zero-shot).
    Retourne (top3_labels_scores, feature_vector) ou (None, None) si aucune frame."""
    frames = extract_video_frames(video_path)
    if not frames:
        return None, None

    model, preprocess, device = load_clip()
    with torch.no_grad():
        imgs = torch.stack([preprocess(f) for f in frames]).to(device)
        feats = model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        avg_feat = feats.mean(dim=0, keepdim=True)
        avg_feat = avg_feat / avg_feat.norm(dim=-1, keepdim=True)

    topk = _classify_feat(avg_feat, SCENE_LABELS_EN, top_k=3)
    return topk, avg_feat


def analyze_audio(audio_path):
    y, sr = librosa.load(audio_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = float((y ** 2).mean())
    return {
        "tempo": float(tempo) if np.isscalar(tempo) else float(np.asarray(tempo).flatten()[0]),
        "energy": energy,
    }


def audio_text_mismatch(audio_info, sentiment_score, humor_score, outrage_score):
    """Heuristique simple : compare l'énergie/tempo de l'audio à la charge
    émotionnelle du texte pour repérer un décalage évident. Retourne
    'calme_vs_intense', 'rapide_vs_neutre' ou None."""
    if not audio_info:
        return None
    tempo = audio_info.get("tempo", 0) or 0
    energy = audio_info.get("energy", 0) or 0
    fast_energetic = tempo > 120 and energy > 0.01
    calm = tempo < 90 and energy < 0.005
    text_is_intense = outrage_score > 0.3 or humor_score > 0.3 or abs(sentiment_score) > 0.7
    text_is_neutral = abs(sentiment_score) < 0.2 and outrage_score < 0.1 and humor_score < 0.1

    if calm and text_is_intense:
        return "calme_vs_intense"
    if fast_energetic and text_is_neutral:
        return "rapide_vs_neutre"
    return None


# ── Transcription vocale (optionnelle — dépend de librairies tierces) ──────────
# Ces fonctions ne font JAMAIS planter l'app : si `SpeechRecognition` et/ou
# `moviepy` ne sont pas installés, elles renvoient simplement (None, message).
# Pour activer cette fonctionnalité :  pip install SpeechRecognition moviepy pydub

def transcribe_audio_file(wav_path):
    """Transcrit un fichier .wav en texte (français) via reconnaissance vocale."""
    try:
        import speech_recognition as sr
    except ImportError:
        return None, "Transcription vocale désactivée (pip install SpeechRecognition pour l'activer)."
    r = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data, language="fr-FR")
        return text, None
    except sr.UnknownValueError:
        return None, "Aucune parole compréhensible détectée dans l'audio."
    except sr.RequestError:
        return None, "Service de reconnaissance vocale indisponible (connexion internet requise)."
    except Exception as e:
        return None, f"Transcription impossible ({e})."


def convert_to_wav(src_path, dst_path="temp_audio_conv.wav"):
    """Convertit mp3 (ou autre) vers wav via pydub, si disponible."""
    try:
        from pydub import AudioSegment
    except ImportError:
        return None
    try:
        audio = AudioSegment.from_file(src_path)
        audio.export(dst_path, format="wav")
        return dst_path
    except Exception:
        return None


def extract_audio_from_video(video_path, dst_path="temp_video_audio.wav"):
    """Extrait la piste audio d'une vidéo en .wav via moviepy, si disponible."""
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        return None, "Analyse de la piste audio vidéo désactivée (pip install moviepy pour l'activer)."
    try:
        clip_ = VideoFileClip(video_path)
        if clip_.audio is None:
            clip_.close()
            return None, "Cette vidéo ne contient pas de piste audio."
        clip_.audio.write_audiofile(dst_path, logger=None)
        clip_.close()
        return dst_path, None
    except Exception as e:
        return None, f"Extraction audio impossible ({e})."


def compute_final_viral_score(p_viral, sel_hour, visual_coherence_sim=None, speech_coherence_sim=None):
    """
    Combine plusieurs signaux en un seul score de potentiel viral (0-100%) :
      - p_viral               : probabilité 'Viral' donnée par le modèle ML (0-1)
      - timing                : qualité du créneau horaire choisi (0-1)
      - cohérence visuelle    : accord entre le texte et l'image/vidéo (0-1, si présent)
      - cohérence vocale      : accord entre le texte et la parole détectée dans l'audio (0-1, si présent)
    Les poids se redistribuent automatiquement selon les signaux disponibles,
    pour rester équitable même sans média importé.
    Retourne (score_final_0_100, détail_des_contributions).
    """
    components = [("Score ML (modèle)", p_viral, 0.55), ("Créneau horaire", timing_score(sel_hour), 0.20)]

    v_norm = normalize_score(visual_coherence_sim)
    if v_norm is not None:
        components.append(("Cohérence texte / visuel", v_norm, 0.25))

    s_norm = normalize_score(speech_coherence_sim)
    if s_norm is not None:
        components.append(("Cohérence texte / audio parlé", s_norm, 0.20))

    total_w = sum(w for _, _, w in components)
    final = sum(v * w for _, v, w in components) / total_w
    detail = [(name, val, w / total_w) for name, val, w in components]
    return final * 100, detail


@st.cache_data(show_spinner="⚙️ Calcul des indices…")
def build_features(df_raw: pd.DataFrame, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1):
    df = df_raw.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(r"[^\w]", "_", regex=True)

    for ts_col in ["Post_Timestamp", "Date"]:
        if ts_col in df.columns:
            df["Post_Timestamp"] = pd.to_datetime(df[ts_col], errors="coerce", dayfirst=True)
            break

    for col in ["Likes","Comments","Shares","Impressions","Reach","Engagement_Rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Likes","Comments","Shares"])
    df["Total_Engagement"] = df["Likes"] + df["Comments"] + df["Shares"]

    imp   = df["Impressions"] if "Impressions" in df.columns else df["Total_Engagement"] + 1
    reach = df["Reach"]       if "Reach"       in df.columns else df["Total_Engagement"] + 1

    df["Engagement_per_Impression"] = df["Total_Engagement"] / (imp + 1)
    df["Engagement_per_Reach"]      = df["Total_Engagement"] / (reach + 1)
    df["Virality_Score"] = df["Engagement_per_Impression"] * 0.6 + df["Engagement_per_Reach"] * 0.4
    v_max = df["Virality_Score"].max()
    df["Virality_Score_norm"] = df["Virality_Score"] / v_max if v_max > 0 else 0.0

    sentiment_map = {"Positive":1.0,"Neutral":0.0,"Negative":-1.0,"Mixed":0.5}
    df["Sentiment_Score"] = df["Sentiment"].map(sentiment_map).fillna(0) if "Sentiment" in df.columns else 0.0

    if "Post_Content" in df.columns:
        df["Post_Content_str"] = df["Post_Content"].astype(str)
    else:
        df["Post_Content_str"] = ""

    df["Emoji_Density"] = df["Post_Content_str"].apply(_emoji_density)
    df["Humor_Score"]   = df["Post_Content_str"].apply(_humor_score)
    df["Outrage_Score"] = df["Post_Content_str"].apply(_outrage_score)
    df["Word_Count"]    = df["Post_Content_str"].str.split().str.len().fillna(0)
    df["Text_Length"]   = df["Post_Content_str"].apply(len)

    df["EWI"] = (
        alpha * df["Sentiment_Score"].abs() +
        beta  * df["Outrage_Score"] +
        gamma * df["Humor_Score"] +
        delta * df["Emoji_Density"] * 100
    ).clip(0, 1)

    df["TTV_available"] = False
    df["TTV_proxy"] = (1 - df["Virality_Score_norm"]) * 72

    df["Hook_Score"] = (df["Likes"]*0.4 + df["Shares"]*0.4 + imp*0.1 + (1/(df["Word_Count"]+1))*100)
    hs_max = df["Hook_Score"].max()
    df["Hook_Score_norm"] = df["Hook_Score"] / hs_max if hs_max > 0 else 0.0

    if "Time" in df.columns:
        try:    df["Hour"] = df["Time"].astype(str).str.extract(r"(\d{1,2}):").astype(float)
        except: df["Hour"] = 12.0
    elif "Post_Timestamp" in df.columns:
        df["Hour"] = df["Post_Timestamp"].dt.hour.astype(float)
    else:
        df["Hour"] = 12.0
    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce").fillna(12).astype(int)

    if "Time_Periods" in df.columns:
        df["Time_Slot"] = df["Time_Periods"].fillna("Unknown")
    else:
        df["Time_Slot"] = pd.cut(df["Hour"], bins=[-1,4,11,17,23],
            labels=["🌙 Nuit (0–4h)","🌅 Matin (5–11h)","☀️ Après-midi (12–17h)","🌆 Soirée (18–23h)"]).astype(str)

    if "Weekday_Type" in df.columns:
        df["Day_Type"] = df["Weekday_Type"].fillna("Unknown")
    elif "Post_Timestamp" in df.columns:
        df["Day_Type"] = df["Post_Timestamp"].dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")
    else:
        df["Day_Type"] = "Unknown"

    q33 = df["Virality_Score_norm"].quantile(0.33)
    q66 = df["Virality_Score_norm"].quantile(0.66)

    def _tier(v):
        if v >= q66:   return "🔥 Viral"
        elif v >= q33: return "📈 Moyen"
        return "💤 Faible"
    df["Virality_Tier"] = df["Virality_Score_norm"].apply(_tier)

    # ------------------------------------------------------------------
    # Colonnes multimodales (valeurs par défaut pour le dataset historique)
    # ------------------------------------------------------------------
    df["Image_Category"] = "unknown"
    df["Image_Score"] = 0.0
    df["Video_Category"] = "unknown"
    df["Audio_Energy"] = 0.0
    df["Audio_Tempo"] = 0.0

    return df, q33, q66


# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def train_model(df: pd.DataFrame):
    df = df.copy()
    feature_cols = []
    encoders = {}

    # =========================
    # Encodage catégories multimodales (image / vidéo)
    # =========================
    for col in ["Image_Category", "Video_Category"]:
        if col in df.columns and f"{col}_enc" not in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            feature_cols.append(f"{col}_enc")

    # =========================
    # Encodage plateforme / type de post
    # =========================
    for col in ["Platform", "Post_Type"]:
        actual_col = next((c for c in df.columns if c.lower() == col.lower()), None)
        if actual_col:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[actual_col].astype(str))
            encoders[col] = le
            feature_cols.append(f"{col}_enc")

    if "Hour" in df.columns:
        feature_cols.append("Hour")

    time_slot_col = next((c for c in df.columns if c.lower() in ["time_slot","time_periods"]), None)
    if time_slot_col:
        le = LabelEncoder()
        df["Time_Slot_enc"] = le.fit_transform(df[time_slot_col].astype(str))
        encoders["Time_Slot"] = le
        feature_cols.append("Time_Slot_enc")

    day_col = next((c for c in df.columns if "weekday" in c.lower() or c == "Day_Type"), None)
    if day_col:
        le = LabelEncoder()
        df["Day_Type_enc"] = le.fit_transform(df[day_col].astype(str))
        encoders["Day_Type"] = le
        feature_cols.append("Day_Type_enc")

    # =========================
    # Colonnes numériques textuelles + multimodales (chacune ajoutée UNE seule fois)
    # =========================
    for col in ["EWI","Humor_Score","Outrage_Score","Emoji_Density","Word_Count","Sentiment_Score"]:
        if col in df.columns:
            feature_cols.append(col)

    for col in ["Image_Score", "Audio_Energy", "Audio_Tempo"]:
        if col in df.columns:
            feature_cols.append(col)

    # Sécurité : on retire d'éventuels doublons tout en gardant l'ordre
    feature_cols = list(dict.fromkeys(feature_cols))

    if not feature_cols:
        return None, None, None, None, None, None

    target_map = {"🔥 Viral": 2, "📈 Moyen": 1, "💤 Faible": 0}
    df["target"] = df["Virality_Tier"].map(target_map)
    valid = df[feature_cols + ["target"]].dropna()
    X = valid[feature_cols].fillna(0)
    y = valid["target"]

    if len(X) < 50:
        return None, None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
    model.fit(X_train_s, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_s))
    feat_imp = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)

    return model, scaler, encoders, feature_cols, acc, feat_imp


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMANDATIONS DYNAMIQUES
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(pred_class, pred_proba, platform, post_type,
                              sel_hour, sel_sentiment, user_ewi,
                              user_humor, user_word_count, user_emoji_density,
                              user_outrage, df_context=None,
                              media_kind=None, media_label=None, media_confidence=None,
                              coherence_score=None, type_mismatch=None, audio_mismatch=None):
    """
    Génère des recommandations dynamiques selon le contexte complet du post.
    Retourne une liste de (type, message) où type = 'viral' | 'warn' | 'tip'

    Paramètres multimodaux (optionnels) :
      media_kind        : "Image" | "Vidéo" | "Audio" | None
      media_label       : catégorie détectée par CLIP (ex: "sport", "nourriture"...)
      media_confidence  : confiance du modèle CLIP sur ce label (0-1)
      coherence_score   : similarité CLIP texte/média (image ou vidéo), None si pas de texte
      type_mismatch     : "Image"/"Video"/"Audio" détecté si différent du type de post choisi
      audio_mismatch    : "calme_vs_intense" | "rapide_vs_neutre" | None (audio uniquement)
    """
    recos = []

    # ── 0. Cohérence type de post déclaré vs fichier réellement importé ────────
    if type_mismatch:
        recos.append(("warn", f"🧩 Vous avez choisi le type de post « {post_type} » mais le fichier importé est en réalité un(e) {type_mismatch}. Alignez le type sélectionné avec votre média pour une prédiction fiable."))

    # ── 0bis. Cohérence texte / contenu visuel réel (image ou vidéo) ───────────
    if media_kind in ("Image", "Vidéo") and media_label:
        conf_txt = f" ({media_confidence:.0%} de confiance)" if media_confidence is not None else ""
        if coherence_score is None:
            recos.append(("tip", f"📝 {media_kind} détectée : « {media_label} »{conf_txt}. Ajoutez un texte d'accompagnement pour que l'app vérifie que votre légende correspond bien à ce que montre le média."))
        else:
            verdict = coherence_verdict(coherence_score)
            if verdict == "faible":
                recos.append(("warn", f"🔀 Incohérence probable : votre {media_kind.lower()} montre plutôt « {media_label} »{conf_txt}, mais votre texte semble parler d'autre chose. Un décalage entre légende et visuel dilue le message et fait fuir l'algorithme — réalignez les deux."))
            elif verdict == "moyenne":
                recos.append(("tip", f"🔀 Cohérence moyenne entre le texte et le {media_kind.lower()} détecté (« {media_label} »{conf_txt}). Relisez votre légende pour vérifier qu'elle colle bien à l'image/vidéo."))
            else:
                recos.append(("viral", f"✅ Bonne cohérence entre votre texte et le {media_kind.lower()} détecté (« {media_label} »{conf_txt}) — le message est clair et aligné."))

    # ── 0ter. Cohérence audio / charge émotionnelle du texte ────────────────────
    if media_kind == "Audio" and audio_mismatch:
        if audio_mismatch == "calme_vs_intense":
            recos.append(("warn", "🎧 Le son importé est plutôt calme et lent, alors que votre texte est très chargé émotionnellement (indignation, humour marqué ou sentiment fort). Ce décalage peut nuire à la crédibilité perçue du post."))
        elif audio_mismatch == "rapide_vs_neutre":
            recos.append(("tip", "🎧 Le son importé est énergique et rapide, alors que votre texte reste neutre. Un texte plus dynamique ou expressif accompagnerait mieux cette bande-son."))
    tier_labels = {2: "🔥 Viral", 1: "📈 Moyen", 0: "💤 Faible"}
    tier = tier_labels[pred_class]
    confidence = pred_proba[pred_class]

    # ── 1. Résultat global ────────────────────────────────────────────────────
    if pred_class == 2:
        recos.append(("viral", f"✅ Excellent potentiel viral ({confidence:.0%} de confiance) ! Publiez maintenant et boostez le post dans les 2 premières heures pour maximiser l'effet de vague."))
    elif pred_class == 1:
        recos.append(("warn", f"📈 Potentiel moyen ({confidence:.0%}). Quelques ajustements peuvent faire basculer votre post vers le tier Viral."))
    else:
        recos.append(("warn", f"💤 Faible potentiel ({confidence:.0%}). Plusieurs optimisations sont nécessaires avant publication."))

    # ── 2. Heure de publication ───────────────────────────────────────────────
    if sel_hour >= 0 and sel_hour <= 4:
        recos.append(("warn", f"🌙 Heure de publication risquée ({sel_hour}h du matin). Publiez plutôt entre 18h et 21h — l'audience est 3× plus active en soirée."))
    elif sel_hour >= 5 and sel_hour <= 8:
        recos.append(("tip", f"🌅 Publication matinale ({sel_hour}h). Correct pour LinkedIn et Twitter (professionnels actifs le matin), mais sous-optimal pour Instagram et Facebook."))
    elif sel_hour >= 12 and sel_hour <= 14:
        recos.append(("tip", "☀️ Créneau pause déjeuner — bon pour les contenus courts et visuels (Instagram, TikTok)."))
    elif sel_hour >= 18 and sel_hour <= 21:
        recos.append(("viral", "🌆 Excellent créneau ! La soirée (18h–21h) est le pic d'activité sur toutes les plateformes."))
    elif sel_hour >= 22:
        recos.append(("warn", f"🌙 Publication tardive ({sel_hour}h). L'audience diminue fortement après 22h — programmez pour le lendemain soir."))

    # ── 3. Plateforme + sentiment ─────────────────────────────────────────────
    if platform == "Twitter" and sel_sentiment == "Negative":
        recos.append(("tip", "🐦 Sur Twitter, le contenu négatif/controversé génère plus de retweets — mais attention au ratio commentaires négatifs qui peut nuire à votre image."))
    elif platform == "LinkedIn" and sel_sentiment == "Negative":
        recos.append(("warn", "💼 Sur LinkedIn, le sentiment négatif pénalise fortement l'engagement. Reformulez avec un angle constructif ou une leçon apprise."))
    elif platform == "Instagram" and sel_sentiment == "Neutral":
        recos.append(("warn", "📸 Sur Instagram, le contenu neutre est peu partagé. Ajoutez une touche émotionnelle (inspiration, humour, surprise) pour booster l'EWI."))
    elif platform == "Facebook" and user_ewi < 0.15:
        recos.append(("warn", "👥 Sur Facebook, l'algorithme favorise les posts qui génèrent des réactions. Ajoutez une question en fin de post pour stimuler les commentaires."))

    # ── 4. EWI ────────────────────────────────────────────────────────────────
    if user_ewi < 0.1:
        recos.append(("warn", f"🧠 EWI très faible ({user_ewi:.3f}). Votre post manque de charge émotionnelle. Ajoutez du sentiment fort, de l'humour ou des emojis expressifs."))
    elif user_ewi > 0.7:
        recos.append(("tip", f"🧠 EWI élevé ({user_ewi:.3f}). Attention à ne pas paraître trop agressif ou clickbait — l'authenticité reste la clé."))

    # ── 5. Humour ─────────────────────────────────────────────────────────────
    if user_humor < 0.05 and platform in ["Instagram", "Twitter", "Facebook"]:
        recos.append(("tip", "😂 Ajoutez une touche d'humour (!, 😂, 🤣, LOL) — les posts humoristiques sont 2× plus partagés sur les réseaux sociaux grand public."))

    # ── 6. Longueur du texte ──────────────────────────────────────────────────
    if user_word_count > 80:
        recos.append(("warn", f"📝 Texte trop long ({user_word_count} mots). Les posts courts (< 50 mots) ont un meilleur Hook Score. Réduisez et mettez l'essentiel en premier."))
    elif user_word_count < 5 and user_word_count > 0:
        recos.append(("tip", "📝 Texte très court. Pour certaines plateformes (Instagram), c'est OK avec une bonne image. Sur Facebook/LinkedIn, développez légèrement votre message."))

    # ── 7. Type de contenu ────────────────────────────────────────────────────
    if post_type in ["Image", "Video", "Audio"]:
        recos.append(("tip", f"📎 Type '{post_type}' sélectionné — notre modèle analyse uniquement le texte accompagnant votre {post_type.lower()}. Le contenu visuel/audio réel n'est pas pris en compte dans ce scoring (voir Limitations)."))

    if post_type == "Video" and platform == "Instagram":
        recos.append(("viral", "🎬 Vidéo + Instagram = combo fort. Les Reels ont 22% de portée en plus que les posts statiques selon les données 2024."))
    elif post_type == "Image" and platform == "LinkedIn":
        recos.append(("tip", "🖼️ Sur LinkedIn, les infographies et carrousels d'images performent mieux que les images simples."))
    elif post_type == "Text" and platform == "Instagram":
        recos.append(("warn", "⚠️ Un post uniquement textuel sur Instagram est fortement désavantagé par l'algorithme. Ajoutez une image ou une vidéo."))

    # ── 8. Emojis ────────────────────────────────────────────────────────────
    if user_emoji_density == 0 and platform in ["Instagram", "Twitter"]:
        recos.append(("tip", "😊 Aucun emoji détecté. Sur Instagram et Twitter, 1 à 3 emojis pertinents augmentent le taux de clic de ~25%."))

    # ── 9. Outrage ────────────────────────────────────────────────────────────
    if user_outrage > 0.5:
        recos.append(("warn", "⚡ Score d'indignation élevé. Ce type de contenu génère beaucoup de réactions mais peut nuire à votre image sur le long terme. Utilisez avec précaution."))

    # ── 10. Instagram hashtags ────────────────────────────────────────────────
    if platform == "Instagram" and pred_class <= 1:
        recos.append(("tip", "# Sur Instagram, ajoutez 5 à 10 hashtags pertinents dans les commentaires pour étendre votre portée sans alourdir la légende."))

    return recos


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(col, label, value, sub=""):
    with col:
        st.markdown(
            f'<div class="metric-card"><p class="label">{label}</p>'
            f'<h2>{value}</h2><p>{sub}</p></div>',
            unsafe_allow_html=True,
        )


def section(title):
    st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)


def info(txt):
    st.markdown(f'<div class="info-box">{txt}</div>', unsafe_allow_html=True)


def warn_box(txt):
    st.markdown(f'<div class="warn-box">{txt}</div>', unsafe_allow_html=True)


def limit_box(txt):
    st.markdown(f'<div class="limit-box">{txt}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        raw = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("📂 Dépose `social_media_engagement_data.xlsx` dans `data/raw/` puis recharge.")
        st.stop()

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0;">
          <div style="font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:800;
                      background:linear-gradient(90deg,#a78bfa,#38bdf8);
                      -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            ⚡ VIRALITY<br>CODE
          </div>
          <div style="font-size:0.7rem; color:#475569; letter-spacing:0.1em; margin-top:4px;">
            DÉCODER · MESURER · PRÉDIRE
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        page = st.radio("Navigation", [
            "🏠 Vue d'ensemble",
            "📊 Indices TTV & EWI",
            "🔬 Analyse approfondie",
            "💡 Explorateur de posts",
            "🤖 Prédiction viralité",
            "⚙️ Config & Export",
        ], label_visibility="collapsed")
        st.divider()

        st.markdown("**🎛️ Coefficients EWI**")
        alpha = st.slider("α — Sentiment",     0.0, 1.0, 0.4, 0.05)
        beta  = st.slider("β — Indignation",   0.0, 1.0, 0.3, 0.05)
        gamma = st.slider("γ — Humour",        0.0, 1.0, 0.2, 0.05)
        delta = st.slider("δ — Emoji density", 0.0, 1.0, 0.1, 0.05)
        coef_sum = alpha + beta + gamma + delta
        color = "#4ade80" if abs(coef_sum - 1.0) < 0.05 else "#f87171"
        st.markdown(f'<p style="font-size:0.8rem;color:{color};">Σ = {coef_sum:.2f} {"✓" if abs(coef_sum-1.0)<0.05 else "(idéal = 1.0)"}</p>', unsafe_allow_html=True)
        st.divider()
        st.markdown("**🔍 Filtres globaux**")

    # ── Build features ────────────────────────────────────────────────────────
    df, q33, q66 = build_features(raw, alpha, beta, gamma, delta)
    df = df.copy()
    platform_col = next((c for c in df.columns if "platform" in c.lower()), None)
    posttype_col = next((c for c in df.columns if "post_type" in c.lower() or "posttype" in c.lower()), None)
    content_col  = next((c for c in df.columns if "post_content" in c.lower()), None)

    with st.sidebar:
        platforms = ["Toutes"] + (sorted(df[platform_col].dropna().unique().tolist()) if platform_col else [])
        selected_platform = st.selectbox("Plateforme", platforms)
        post_types = ["Tous"] + (sorted(df[posttype_col].dropna().unique().tolist()) if posttype_col else [])
        selected_type = st.selectbox("Type de post", post_types)
        virality_range = st.slider("Virality Score (min–max)", 0.0, 1.0, (0.0, 1.0), 0.01)

    mask = df["Virality_Score_norm"].between(*virality_range)
    if selected_platform != "Toutes" and platform_col:
        mask &= df[platform_col] == selected_platform
    if selected_type != "Tous" and posttype_col:
        mask &= df[posttype_col] == selected_type
    filtered = df[mask]

    n_filtered = len(filtered)
    if n_filtered == 0:
        st.warning("Aucune donnée pour ces filtres.")
        st.stop()

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1 — Vue d'ensemble
    # ─────────────────────────────────────────────────────────────────────────
    if page == "🏠 Vue d'ensemble":
        st.markdown('<h1 style="font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;margin-bottom:0;">⚡ Virality Code</h1><p style="color:#64748b;font-size:0.9rem;margin-top:4px;font-family:IBM Plex Mono,monospace;">DÉCODER · MESURER · PRÉDIRE LA VIRALITÉ SOCIALE</p>', unsafe_allow_html=True)
        st.divider()

        c1,c2,c3,c4,c5 = st.columns(5)
        metric_card(c1, "Posts analysés",     f"{n_filtered:,}",                             "dataset filtré")
        metric_card(c2, "Virality Score moy", f"{filtered['Virality_Score_norm'].mean():.3f}","normalisé [0→1]")
        metric_card(c3, "EWI moyen",          f"{filtered['EWI'].mean():.3f}",               "poids émotionnel")
        metric_card(c4, "TTV proxy moyen",    f"{filtered['TTV_proxy'].mean():.1f}h",        "temps estimé")
        metric_card(c5, "Engagement total",   f"{int(filtered['Total_Engagement'].sum()):,}", "L + C + S")
        st.divider()

        col_l, col_r = st.columns([3,2])
        with col_l:
            section("Distribution du Virality Score")
            fig = px.histogram(filtered, x="Virality_Score_norm", nbins=60,
                               color_discrete_sequence=["#7c3aed"], template=PLOTLY_TEMPLATE,
                               labels={"Virality_Score_norm": "Virality Score (normalisé)"})
            fig.add_vline(x=q33, line_dash="dash", line_color="#fbbf24", annotation_text="Q33", annotation_font_color="#fbbf24")
            fig.add_vline(x=q66, line_dash="dash", line_color="#4ade80", annotation_text="Q66", annotation_font_color="#4ade80")
            fig.update_layout(showlegend=False, height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            section("Répartition par tier de viralité")
            tc = filtered["Virality_Tier"].value_counts().reset_index()
            tc.columns = ["Tier","Count"]
            fig2 = px.pie(tc, names="Tier", values="Count", hole=0.55,
                          color_discrete_sequence=["#4ade80","#fbbf24","#f87171"], template=PLOTLY_TEMPLATE)
            fig2.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig2, use_container_width=True)

        if platform_col:
            col_a, col_b = st.columns(2)
            with col_a:
                section("Virality Score moyen par plateforme")
                plat_df = filtered.groupby(platform_col)["Virality_Score_norm"].mean().sort_values(ascending=False).reset_index()
                fig3 = px.bar(plat_df, x=platform_col, y="Virality_Score_norm",
                              color="Virality_Score_norm",
                              color_continuous_scale=[[0,"#1e1e3a"],[0.5,"#7c3aed"],[1,"#c4b5fd"]],
                              template=PLOTLY_TEMPLATE, text_auto=".3f")
                fig3.update_layout(height=320, coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=5,b=5,l=5,r=5))
                st.plotly_chart(fig3, use_container_width=True)
            with col_b:
                section("EWI moyen par plateforme")
                ewi_df = filtered.groupby(platform_col)["EWI"].mean().sort_values(ascending=False).reset_index()
                fig4 = px.bar(ewi_df, x=platform_col, y="EWI",
                              color="EWI",
                              color_continuous_scale=[[0,"#1a1a0a"],[0.5,"#d97706"],[1,"#fbbf24"]],
                              template=PLOTLY_TEMPLATE, text_auto=".3f")
                fig4.update_layout(height=320, coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=5,b=5,l=5,r=5))
                st.plotly_chart(fig4, use_container_width=True)

        section("🏆 Top 5 posts les plus viraux")
        top_cols = [c for c in [content_col, platform_col, posttype_col, "Virality_Score_norm","EWI","TTV_proxy","Total_Engagement","Virality_Tier"] if c and c in filtered.columns]
        st.dataframe(filtered[top_cols].nlargest(5,"Virality_Score_norm").reset_index(drop=True), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2 — Indices TTV & EWI
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "📊 Indices TTV & EWI":
        st.markdown('<h2 style="font-family:Syne,sans-serif;">📊 Indices principaux : TTV & EWI</h2>', unsafe_allow_html=True)
        st.divider()
        tab_ewi, tab_ttv, tab_vs = st.tabs(["🧠 EWI — Poids Émotionnel","⏱️ TTV — Temps jusqu'à Viralité","📈 Virality Score"])

        with tab_ewi:
            info("<b>Formule EWI :</b> α·|sentiment| + β·indignation + γ·humour + δ·emoji_density<br>Les coefficients sont ajustables dans la barre latérale. Score normalisé [0, 1].")
            c1,c2 = st.columns(2)
            with c1:
                section("Distribution EWI")
                fig = px.histogram(filtered, x="EWI", nbins=50, color_discrete_sequence=["#d97706"], template=PLOTLY_TEMPLATE)
                fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=5,b=5,l=5,r=5))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                section("EWI vs Virality Score")
                sample = filtered.sample(min(2000,n_filtered), random_state=42)
                fig2 = px.scatter(sample, x="EWI", y="Virality_Score_norm", color="Virality_Tier",
                                  color_discrete_map={"🔥 Viral":"#4ade80","📈 Moyen":"#fbbf24","💤 Faible":"#f87171"},
                                  opacity=0.55, template=PLOTLY_TEMPLATE)
                fig2.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=5,b=5,l=5,r=5))
                st.plotly_chart(fig2, use_container_width=True)

            section("Contribution des composants EWI")
            comp_df = pd.DataFrame({
                "Composant": ["|Sentiment|","Indignation","Humour","Emoji×100"],
                "Valeur moy": [filtered["Sentiment_Score"].abs().mean(), filtered["Outrage_Score"].mean(), filtered["Humor_Score"].mean(), filtered["Emoji_Density"].mean()*100],
                "Coeff": [alpha,beta,gamma,delta],
            })
            comp_df["Contribution"] = comp_df["Valeur moy"] * comp_df["Coeff"]
            c1,c2 = st.columns([2,1])
            with c1:
                fig3 = px.bar(comp_df, x="Composant", y="Contribution", color="Composant",
                              color_discrete_sequence=["#a78bfa","#f87171","#4ade80","#fbbf24"],
                              template=PLOTLY_TEMPLATE, text_auto=".4f")
                fig3.update_layout(height=260, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=5,b=5,l=5,r=5))
                st.plotly_chart(fig3, use_container_width=True)
            with c2:
                st.dataframe(comp_df.style.format({"Valeur moy":"{:.4f}","Contribution":"{:.4f}","Coeff":"{:.2f}"}), use_container_width=True)

        with tab_ttv:
            warn_box("⚠️ <b>TTV — Mode proxy :</b> le dataset ne contient pas de série temporelle d'engagement. TTV est estimé par : <code>TTV_proxy = (1 − Virality_Score_norm) × 72h</code>. Un post très viral explose vite → TTV court.")
            c1,c2 = st.columns(2)
            with c1:
                section("Distribution TTV proxy (heures)")
                fig = px.histogram(filtered, x="TTV_proxy", nbins=50, color_discrete_sequence=["#38bdf8"], template=PLOTLY_TEMPLATE)
                fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=5,b=5,l=5,r=5))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                section("TTV proxy vs EWI")
                sample = filtered.sample(min(2000,n_filtered), random_state=42)
                fig2 = px.scatter(sample, x="EWI", y="TTV_proxy", color="Virality_Tier",
                                  color_discrete_map={"🔥 Viral":"#4ade80","📈 Moyen":"#fbbf24","💤 Faible":"#f87171"},
                                  opacity=0.5, template=PLOTLY_TEMPLATE)
                fig2.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=5,b=5,l=5,r=5))
                st.plotly_chart(fig2, use_container_width=True)

        with tab_vs:
            info("<b>Virality Score :</b> (Engagement/Impression × 0.6) + (Engagement/Reach × 0.4), normalisé [0,1].")
            c1,c2 = st.columns(2)
            with c1:
                section("Hook Score vs Virality Score")
                sample = filtered.sample(min(2000,n_filtered), random_state=42)
                fig = px.scatter(sample, x="Hook_Score_norm", y="Virality_Score_norm", color="EWI",
                                 color_continuous_scale="Plasma", opacity=0.6, template=PLOTLY_TEMPLATE)
                fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=5,b=5,l=5,r=5))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                if "Sentiment" in filtered.columns:
                    section("Virality Score par sentiment")
                    fig2 = px.box(filtered, x="Sentiment", y="Virality_Score_norm", color="Sentiment",
                                  color_discrete_sequence=["#4ade80","#60a5fa","#f87171","#fbbf24"], template=PLOTLY_TEMPLATE)
                    fig2.update_layout(height=300, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=5,b=5,l=5,r=5))
                    st.plotly_chart(fig2, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 3 — Analyse approfondie
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "🔬 Analyse approfondie":
        st.markdown('<h2 style="font-family:Syne,sans-serif;">🔬 Analyse approfondie</h2>', unsafe_allow_html=True)
        st.divider()
        tab1, tab2, tab3 = st.tabs(["🌍 Par plateforme","🕐 Par horaire & jour","📝 Par format"])

        with tab1:
            if not platform_col:
                st.warning("Colonne 'Platform' introuvable.")
            else:
                grp = filtered.groupby(platform_col).agg(
                    Posts=("Total_Engagement","count"),
                    Virality_moy=("Virality_Score_norm","mean"),
                    EWI_moy=("EWI","mean"),
                    TTV_moy=("TTV_proxy","mean"),
                    Engagement_total=("Total_Engagement","sum"),
                    Likes_moy=("Likes","mean"),
                    Shares_moy=("Shares","mean"),
                    Comments_moy=("Comments","mean"),
                ).round(4).reset_index()
                section("Tableau récapitulatif")
                st.dataframe(grp, use_container_width=True)
                c1,c2 = st.columns(2)
                with c1:
                    section("Box Virality Score par plateforme")
                    fig = px.box(filtered, x=platform_col, y="Virality_Score_norm", color=platform_col, color_discrete_sequence=PALETTE, template=PLOTLY_TEMPLATE)
                    fig.update_layout(height=340, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=5,b=5,l=5,r=5))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    section("Box EWI par plateforme")
                    fig2 = px.box(filtered, x=platform_col, y="EWI", color=platform_col, color_discrete_sequence=PALETTE, template=PLOTLY_TEMPLATE)
                    fig2.update_layout(height=340, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=5,b=5,l=5,r=5))
                    st.plotly_chart(fig2, use_container_width=True)
                best_plat = grp.loc[grp["Virality_moy"].idxmax(), platform_col]
                st.success(f"✅ **Insight :** La plateforme **{best_plat}** obtient le meilleur Virality Score moyen.")

        with tab2:
            section("Virality Score moyen par créneau horaire")
            time_grp = filtered.groupby("Time_Slot")["Virality_Score_norm"].agg(["mean","count"]).reset_index().rename(columns={"mean":"Virality_moy","count":"N_posts"}).sort_values("Virality_moy",ascending=False)
            fig = px.bar(time_grp, x="Time_Slot", y="Virality_moy",
                         color="Virality_moy",
                         color_continuous_scale=[[0,"#1e1e3a"],[0.5,"#7c3aed"],[1,"#c4b5fd"]],
                         template=PLOTLY_TEMPLATE, text_auto=".4f")
            fig.update_layout(height=320, coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=5,b=5,l=5,r=5))
            st.plotly_chart(fig, use_container_width=True)

            if platform_col and "Hour" in filtered.columns:
                section("Heatmap : heure × plateforme")
                heatmap_df = filtered.groupby([platform_col,"Hour"])["Virality_Score_norm"].mean().reset_index().pivot(index=platform_col, columns="Hour", values="Virality_Score_norm")
                fig4 = px.imshow(heatmap_df, color_continuous_scale="Purples", template=PLOTLY_TEMPLATE, aspect="auto")
                fig4.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=5,b=5,l=5,r=5))
                st.plotly_chart(fig4, use_container_width=True)

            best_time = time_grp.iloc[0]
            st.success(f"✅ **Insight :** Meilleur créneau : **{best_time['Time_Slot']}** (score moy {best_time['Virality_moy']:.4f})")

        with tab3:
            if not posttype_col:
                st.warning("Colonne 'Post_Type' introuvable.")
            else:
                section("Virality Score moyen par type de post")
                type_grp = filtered.groupby(posttype_col)["Virality_Score_norm"].agg(["mean","count"]).reset_index().rename(columns={"mean":"Virality_moy","count":"N_posts"}).sort_values("Virality_moy",ascending=False)
                fig = px.bar(type_grp, x=posttype_col, y="Virality_moy",
                             color="Virality_moy",
                             color_continuous_scale=[[0,"#1e0a3a"],[0.5,"#7c3aed"],[1,"#c4b5fd"]],
                             template=PLOTLY_TEMPLATE, text_auto=".4f")
                fig.update_layout(height=320, coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=5,b=5,l=5,r=5))
                st.plotly_chart(fig, use_container_width=True)
                best_type = type_grp.iloc[0]
                st.success(f"✅ **Insight :** Format le plus viral : **{best_type[posttype_col]}** (score moy {best_type['Virality_moy']:.4f})")

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 4 — Explorateur de posts
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "💡 Explorateur de posts":
        st.markdown('<h2 style="font-family:Syne,sans-serif;">💡 Explorateur de posts</h2>', unsafe_allow_html=True)
        st.divider()
        col_sort, col_order, col_n = st.columns([2,1,1])
        with col_sort:
            sort_col = st.selectbox("Trier par", ["Virality_Score_norm","EWI","TTV_proxy","Total_Engagement","Hook_Score_norm","Likes","Shares","Comments"])
        with col_order:
            ascending = st.checkbox("↑ Croissant", False)
        with col_n:
            n_rows = st.slider("Nb de posts", 10, 300, 50)

        show_cols = [c for c in [content_col, platform_col, posttype_col, "Virality_Score_norm","EWI","TTV_proxy","Total_Engagement","Likes","Comments","Shares","Virality_Tier","Sentiment","Time_Slot","Day_Type"] if c and c in filtered.columns]
        display = filtered[show_cols].sort_values(sort_col, ascending=ascending).head(n_rows).reset_index(drop=True)

        def _style_tier(val):
            colors = {
                "🔥 Viral": "background-color:#052e16;color:white;",
                "📈 Moyen": "background-color:#3b2700;color:white;",
                "💤 Faible": "background-color:#300;color:white;",
            }
            return colors.get(val, "")

        styler_obj = display.style
        if "Virality_Tier" in display.columns:
            try:
                styled = styler_obj.map(_style_tier, subset=["Virality_Tier"])
            except AttributeError:
                styled = styler_obj.applymap(_style_tier, subset=["Virality_Tier"])
        else:
            styled = display
        st.dataframe(styled, use_container_width=True, height=500)

        st.divider()
        section("Matrice de corrélations entre indices")
        corr_cols = [c for c in ["Virality_Score_norm","EWI","TTV_proxy","Hook_Score_norm","Total_Engagement","Sentiment_Score","Outrage_Score","Humor_Score","Emoji_Density","Word_Count"] if c in filtered.columns]
        fig = px.imshow(filtered[corr_cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", template=PLOTLY_TEMPLATE, aspect="auto", zmin=-1, zmax=1)
        fig.update_layout(height=460, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=5,b=5,l=5,r=5))
        st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 5 — Prédiction de viralité
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "🤖 Prédiction viralité":
        st.markdown('<h2 style="font-family:Syne,sans-serif;">🤖 Prédicteur de Viralité</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color:#64748b;">Entrez les caractéristiques de votre post — le modèle prédit ses chances de devenir viral.</p>', unsafe_allow_html=True)
        st.divider()

        # ── Avertissement sur les limitations ─────────────────────────────────
        limit_box("""
        <b>⚠️ Comment lire ce prédicteur</b><br><br>
        <b>1. Le classifieur (Gradient Boosting) apprend sur des données historiques sans image/vidéo réelle.</b>
        Le dataset d'entraînement ne contient pas d'images ou de vidéos associées aux anciens posts : les colonnes
        Image_Category, Video_Category, Audio_Energy, etc. y sont donc constantes. Un modèle ne peut pas apprendre
        de relation à partir d'une variable qui ne varie jamais — c'est pour ça que le <b>score de viralité en %</b>
        ci-dessous bouge très peu, voire pas du tout, selon le fichier que vous importez.<br><br>
        <b>2. En revanche, l'app analyse réellement votre média au moment de la prédiction</b> grâce à CLIP
        (vision par ordinateur) : détection du contenu de l'image/vidéo, et surtout <b>vérification de cohérence</b>
        entre ce que montre le média et ce que raconte votre texte. C'est cette analyse — affichée ci-dessous et
        dans les recommandations — qui vous dira si, par exemple, votre image de sport ne correspond pas à un texte
        parlant d'école et d'entreprise. Cette cohérence est indicative (seuils approximatifs) et n'influence pas
        directement le % de viralité du modèle, mais c'est le signal le plus utile pour corriger votre post.
        """)

        with st.spinner("🧠 Entraînement du modèle…"):
            result = train_model(df)

        if result[0] is None:
            st.error("Données insuffisantes pour entraîner le modèle.")
            st.stop()

        model, scaler, encoders, feature_cols, acc, feat_imp = result

        acc_color = "#4ade80" if acc > 0.65 else "#fbbf24" if acc > 0.5 else "#f87171"
        st.markdown(
            f'<div style="background:#0d0d20;border:1px solid #252550;border-radius:12px;padding:0.8rem 1.2rem;margin-bottom:1rem;display:flex;gap:2rem;align-items:center;">'
            f'<div><span style="font-family:IBM Plex Mono,monospace;font-size:1.5rem;color:{acc_color};">{acc:.1%}</span><br><span style="font-size:0.75rem;color:#64748b;">Accuracy (test set)</span></div>'
            f'<div><span style="font-family:IBM Plex Mono,monospace;font-size:1.5rem;color:#94a3b8;">{len(df):,}</span><br><span style="font-size:0.75rem;color:#64748b;">Posts d\'entraînement</span></div>'
            f'<div><span style="font-family:IBM Plex Mono,monospace;font-size:1.5rem;color:#94a3b8;">GBM</span><br><span style="font-size:0.75rem;color:#64748b;">Gradient Boosting</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_form, col_result = st.columns([1,1])

        with col_form:
            section("Caractéristiques de votre post")

            platforms_available = sorted(df[platform_col].dropna().unique().tolist()) if platform_col else ["Instagram","Facebook","Twitter","LinkedIn"]
            sel_platform = st.selectbox("📱 Plateforme", platforms_available)

            types_available = sorted(df[posttype_col].dropna().unique().tolist()) if posttype_col else ["Video","Image","Text","Link"]
            sel_type = st.selectbox("📝 Type de post", types_available)

            # Avertissement dynamique sur le type de contenu
            if sel_type in ["Video","Image","Audio"]:
                st.markdown(
                    f'<div style="background:#1a1a05;border-left:3px solid #fbbf24;border-radius:0 6px 6px 0;padding:0.5rem 0.8rem;font-size:0.8rem;color:#fde68a;margin-bottom:8px;">'
                    f'ℹ️ Type <b>{sel_type}</b> sélectionné. Seul le texte d\'accompagnement sera analysé — '
                    f'le contenu {sel_type.lower()} réel n\'est pas pris en compte par le modèle.'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            sel_hour = st.slider("🕐 Heure de publication", 0, 23, 18)
            time_period = "Night" if sel_hour <= 4 else "Morning" if sel_hour <= 11 else "Afternoon" if sel_hour <= 17 else "Evening"
            sel_day = st.selectbox("📅 Jour de publication", ["Weekday","Weekend"])

            st.markdown("---")
            st.markdown("**Contenu du post**")

            post_text = st.text_area("Texte du post (optionnel — améliore la prédiction EWI)",
                                     placeholder="Écris ton post ici pour une analyse plus précise…", height=100)
            uploaded_file = st.file_uploader(
                "Ajouter une image, vidéo ou audio",
                type=["jpg","jpeg","png","mp4","wav","mp3"]
            )

            st.markdown("---")
            st.markdown("**Sentiment attendu**")

            sel_sentiment = st.select_slider("Sentiment du message",
                                             options=["Negative","Mixed","Neutral","Positive"], value="Positive")

        with col_result:
            section("Résultat de la prédiction")

            # Compute NLP features
            user_emoji_density = _emoji_density(post_text) if post_text else 0.0
            user_humor         = _humor_score(post_text)   if post_text else 0.0
            user_outrage       = _outrage_score(post_text) if post_text else 0.0
            user_word_count    = len(post_text.split())    if post_text else 0
            sentiment_map_pred = {"Negative":-1.0,"Mixed":0.5,"Neutral":0.0,"Positive":1.0}
            user_sentiment     = sentiment_map_pred.get(sel_sentiment, 0.0)

            # Build feature row (valeurs par défaut pour tous les champs multimodaux)
            row = {"Image_Score": 0.0, "Audio_Energy": 0.0, "Audio_Tempo": 0.0}

            if "Platform" in encoders:
                try:    row["Platform_enc"] = encoders["Platform"].transform([sel_platform])[0]
                except: row["Platform_enc"] = 0
            if "Post_Type" in encoders:
                try:    row["Post_Type_enc"] = encoders["Post_Type"].transform([sel_type])[0]
                except: row["Post_Type_enc"] = 0
            if "Hour" in feature_cols:
                row["Hour"] = sel_hour
            if "Time_Slot_enc" in feature_cols and "Time_Slot" in encoders:
                try:    row["Time_Slot_enc"] = encoders["Time_Slot"].transform([time_period])[0]
                except: row["Time_Slot_enc"] = 0
            if "Day_Type_enc" in feature_cols and "Day_Type" in encoders:
                try:    row["Day_Type_enc"] = encoders["Day_Type"].transform([sel_day])[0]
                except: row["Day_Type_enc"] = 0

            user_ewi = (alpha*abs(user_sentiment) + beta*user_outrage + gamma*user_humor + delta*user_emoji_density*100)
            row["EWI"]           = user_ewi
            row["Humor_Score"]   = user_humor
            row["Outrage_Score"] = user_outrage
            row["Emoji_Density"] = user_emoji_density
            row["Word_Count"]    = user_word_count
            row["Sentiment_Score"] = user_sentiment

            # ─────────────────────────────
            # Analyse réelle du média importé (une seule lecture du fichier, un seul traitement)
            # ─────────────────────────────
            media_kind = None          # "Image" | "Vidéo" | "Audio" | None
            media_label = None
            media_confidence = None
            media_topk = None
            coherence_score = None
            type_mismatch = None
            audio_mismatch = None
            audio_info = None

            SUFFIX_TO_KIND = {"jpg": "Image", "jpeg": "Image", "png": "Image",
                               "mp4": "Vidéo", "wav": "Audio", "mp3": "Audio"}

            if uploaded_file is not None:
                suffix = uploaded_file.name.split(".")[-1].lower()
                file_bytes = uploaded_file.read()  # une seule lecture du flux
                detected_kind = SUFFIX_TO_KIND.get(suffix)

                # Le type de post choisi doit correspondre au fichier réellement importé
                if detected_kind and sel_type not in ("", None):
                    sel_type_norm = sel_type.strip().lower()
                    kind_norm_map = {"Image": "image", "Vidéo": "video", "Audio": "audio"}
                    if kind_norm_map.get(detected_kind) not in sel_type_norm and detected_kind.lower() not in sel_type_norm:
                        type_mismatch = detected_kind

                if suffix in ["jpg", "jpeg", "png"]:
                    media_kind = "Image"
                    image = Image.open(BytesIO(file_bytes))
                    media_topk, image_feat = analyze_image(image)
                    media_label, media_confidence = media_topk[0]
                    row["Image_Score"] = media_confidence
                    if "Image_Category" in encoders:
                        try:    row["Image_Category_enc"] = encoders["Image_Category"].transform([media_label])[0]
                        except: row["Image_Category_enc"] = 0
                    coherence_score = compute_coherence(image_feat, post_text)
                    st.image(image, use_container_width=True)
                    st.success(f"Image détectée : **{media_label}** ({media_confidence:.0%})")

                elif suffix == "mp4":
                    media_kind = "Vidéo"
                    temp_path = "temp_upload.mp4"
                    with open(temp_path, "wb") as f:
                        f.write(file_bytes)
                    media_topk, video_feat = analyze_video(temp_path)
                    if media_topk:
                        media_label, media_confidence = media_topk[0]
                        row["Image_Score"] = media_confidence
                        if "Video_Category" in encoders:
                            try:    row["Video_Category_enc"] = encoders["Video_Category"].transform([media_label])[0]
                            except: row["Video_Category_enc"] = 0
                        coherence_score = compute_coherence(video_feat, post_text)
                        st.success(f"Vidéo détectée : **{media_label}** ({media_confidence:.0%}, moyenne sur plusieurs frames)")
                    else:
                        st.warning("Impossible d'extraire des frames de cette vidéo.")

                elif suffix in ["wav", "mp3"]:
                    media_kind = "Audio"
                    temp_path = f"temp_upload.{suffix}"
                    with open(temp_path, "wb") as f:
                        f.write(file_bytes)
                    audio_info = analyze_audio(temp_path)
                    row["Audio_Energy"] = audio_info["energy"]
                    row["Audio_Tempo"] = audio_info["tempo"]
                    audio_mismatch = audio_text_mismatch(audio_info, user_sentiment, user_humor, user_outrage)
                    st.success(f"Audio analysé — tempo : {audio_info['tempo']:.0f} BPM, énergie : {audio_info['energy']:.4f}")

                # ── Affichage du top-3 des catégories détectées (image/vidéo) ──
                if media_topk:
                    st.markdown("**Catégories détectées (CLIP) :**")
                    for lbl, sc in media_topk:
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
                            f'<span style="width:100px;font-size:0.8rem;">{lbl}</span>'
                            f'<div style="flex:1;background:#1e1e3a;border-radius:4px;height:10px;">'
                            f'<div style="width:{sc*100:.1f}%;background:#a78bfa;height:100%;border-radius:4px;"></div></div>'
                            f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#c4b5fd;">{sc:.0%}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # ── Affichage du score de cohérence texte / média (image ou vidéo) ──
                if media_kind in ("Image", "Vidéo"):
                    if coherence_score is None:
                        st.info("ℹ️ Ajoutez un texte d'accompagnement pour vérifier la cohérence texte/média.")
                    else:
                        verdict = coherence_verdict(coherence_score)
                        verdict_color = {"faible": "#f87171", "moyenne": "#fbbf24", "bonne": "#4ade80"}[verdict]
                        verdict_txt = {"faible": "⚠️ Cohérence faible — texte et visuel semblent parler de choses différentes",
                                       "moyenne": "🔶 Cohérence moyenne",
                                       "bonne": "✅ Bonne cohérence texte / visuel"}[verdict]
                        st.markdown(
                            f'<div style="background:#0d0d20;border:1px solid #252550;border-left:3px solid {verdict_color};border-radius:0 8px 8px 0;padding:0.6rem 1rem;margin-top:6px;">'
                            f'<span style="color:{verdict_color};font-weight:600;">{verdict_txt}</span>'
                            f'<br><span style="font-size:0.75rem;color:#64748b;">Similarité CLIP texte↔média : {coherence_score:.3f} (indicatif)</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                if type_mismatch:
                    st.warning(f"🧩 Type de post choisi « {sel_type} » ≠ fichier importé détecté comme **{type_mismatch}**.")

            X_user = pd.DataFrame([{col: row.get(col, 0) for col in feature_cols}])
            X_user_s = scaler.transform(X_user.fillna(0))

            pred_class = model.predict(X_user_s)[0]
            pred_proba = model.predict_proba(X_user_s)[0]

            tier_labels  = {2:"🔥 Viral", 1:"📈 Moyen", 0:"💤 Faible"}
            tier_label   = tier_labels[pred_class]
            tier_proba   = pred_proba[pred_class]
            card_class   = {2:"pred-card-viral", 1:"pred-card-mid", 0:"pred-card-low"}[pred_class]
            score_color  = {2:"#4ade80", 1:"#fbbf24", 0:"#f87171"}[pred_class]

            st.markdown(
                f'<div class="{card_class}">'
                f'<p class="pred-score" style="color:{score_color};">{tier_proba:.0%}</p>'
                f'<p class="pred-label" style="color:{score_color};">{tier_label}</p>'
                f'<p style="color:#94a3b8;font-size:0.85rem;margin-top:0.5rem;">Confiance du modèle</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("**Probabilités détaillées**")
            for i, (label, prob) in enumerate(zip(["💤 Faible","📈 Moyen","🔥 Viral"], pred_proba)):
                c = ["#f87171","#fbbf24","#4ade80"][i]
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                    f'<span style="width:80px;font-size:0.8rem;">{label}</span>'
                    f'<div style="flex:1;background:#1e1e3a;border-radius:4px;height:14px;">'
                    f'<div style="width:{prob*100:.1f}%;background:{c};height:100%;border-radius:4px;"></div></div>'
                    f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:{c};">{prob:.1%}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div style="background:#0d0d20;border:1px solid #252550;border-radius:8px;padding:0.8rem;margin-top:1rem;">'
                f'<p style="margin:0;font-size:0.75rem;color:#64748b;">EWI calculé de votre post</p>'
                f'<p style="margin:0;font-family:IBM Plex Mono,monospace;font-size:1.2rem;color:#fbbf24;">{user_ewi:.3f}</p></div>',
                unsafe_allow_html=True,
            )

        # ── Recommandations dynamiques ─────────────────────────────────────────
        st.divider()
        section("💡 Recommandations personnalisées")

        recos = generate_recommendations(
            pred_class=pred_class,
            pred_proba=pred_proba,
            platform=sel_platform,
            post_type=sel_type,
            sel_hour=sel_hour,
            sel_sentiment=sel_sentiment,
            user_ewi=user_ewi,
            user_humor=user_humor,
            user_word_count=user_word_count,
            user_emoji_density=user_emoji_density,
            user_outrage=user_outrage,
            media_kind=media_kind,
            media_label=media_label,
            media_confidence=media_confidence,
            coherence_score=coherence_score,
            type_mismatch=type_mismatch,
            audio_mismatch=audio_mismatch,
        )

        css_class = {"viral":"reco-viral","warn":"reco-warn","tip":"reco-tip"}
        for reco_type, reco_text in recos:
            st.markdown(f'<div class="{css_class[reco_type]}">{reco_text}</div>', unsafe_allow_html=True)

        # ── Feature importance ─────────────────────────────────────────────────
        st.divider()
        section("Importance des variables dans le modèle")
        c1,c2 = st.columns([2,1])
        with c1:
            fig_fi = px.bar(feat_imp.head(10), y="Feature", x="Importance", orientation="h",
                            color="Importance",
                            color_continuous_scale=[[0,"#1e1e3a"],[0.5,"#7c3aed"],[1,"#c4b5fd"]],
                            template=PLOTLY_TEMPLATE)
            fig_fi.update_layout(height=340, coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=5,b=5,l=5,r=5))
            st.plotly_chart(fig_fi, use_container_width=True)
        with c2:
            st.dataframe(feat_imp.head(10).style.format({"Importance":"{:.4f}"}), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 6 — Config & Export
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "⚙️ Config & Export":
        st.markdown('<h2 style="font-family:Syne,sans-serif;">⚙️ Configuration & Export</h2>', unsafe_allow_html=True)
        st.divider()

        section("Dataset chargé")
        col_a, col_b = st.columns(2)
        with col_a:
            st.json({"Lignes brutes": len(raw), "Lignes filtrées": n_filtered, "Colonnes": list(raw.columns.tolist()), "Plateforme col": platform_col, "PostType col": posttype_col})
        with col_b:
            section("Coefficients EWI actifs")
            st.json({"α (sentiment)": alpha, "β (outrage)": beta, "γ (humour)": gamma, "δ (emoji)": delta, "Σ": round(alpha+beta+gamma+delta,3)})

        section("Aperçu données brutes")
        st.dataframe(raw.head(20), use_container_width=True)

        section("Aperçu features calculées")
        feat_preview = [c for c in ["Virality_Score_norm","EWI","TTV_proxy","Hook_Score_norm","Sentiment_Score","Outrage_Score","Humor_Score","Emoji_Density","Word_Count","Time_Slot","Day_Type","Virality_Tier"] if c in df.columns]
        st.dataframe(df[feat_preview].head(20), use_container_width=True)

        st.divider()
        section("📥 Export")
        c1,c2 = st.columns(2)
        with c1:
            csv = filtered.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Données filtrées (CSV)", csv, "virality_filtered.csv", "text/csv", use_container_width=True)
        with c2:
            feat_preview2 = [c for c in feat_preview if c in filtered.columns]
            stats = filtered[feat_preview2].describe().round(4)
            csv2 = stats.to_csv().encode("utf-8")
            st.download_button("⬇️ Statistiques descriptives (CSV)", csv2, "virality_stats.csv", "text/csv", use_container_width=True)


if __name__ == "__main__":
    main()