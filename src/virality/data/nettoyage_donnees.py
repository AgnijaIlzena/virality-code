#IMPORT + LECTURE DU FICHIER EXCEL
import pandas as pd
import numpy as np

file_path = "social_media_engagement_data.xlsx"

df = pd.read_excel(file_path)

print(df.head())
print(df.info())
#NETTOYAGE DES DONNÉES
#Standardisation des colonnes
df.columns = df.columns.str.strip().str.replace(" ", "_")
#Conversion des types
df["Post_Timestamp"] = pd.to_datetime(df["Post_Timestamp"], errors="coerce")

numeric_cols = [
    "Likes", "Comments", "Shares",
    "Impressions", "Reach", "Engagement_Rate"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
#Nettoyage texte
df["Post_Content"] = df["Post_Content"].astype(str).str.lower().str.strip()
#Valeurs manquantes
df.isnull().sum()

df = df.dropna(subset=["Post_Content", "Engagement_Rate"])
df = df.fillna({
    "Campaign_ID": "Unknown",
    "Influencer_ID": "Unknown"
})
#Déduplication
df = df.drop_duplicates(subset=["Post_ID"])