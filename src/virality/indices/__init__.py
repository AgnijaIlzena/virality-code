import numpy as np

def compute_indices(df):

    df = df.copy()

    # ENGAGEMENT TOTAL
    df["Total_Engagement"] = df["Likes"] + df["Comments"] + df["Shares"]

    # VIRALITÉ
    df["Engagement_per_Impression"] = df["Total_Engagement"] / (df["Impressions"] + 1)
    df["Engagement_per_Reach"] = df["Total_Engagement"] / (df["Reach"] + 1)

    df["Virality_Score"] = (
        df["Engagement_per_Impression"] * 0.6 +
        df["Engagement_per_Reach"] * 0.4
    )

    # EMOTION
    sentiment_map = {
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1,
        "Mixed": 0.5
    }

    df["Sentiment_Score"] = df["Sentiment"].map(sentiment_map).fillna(0)

    df["Emotion_Weight"] = (
        df["Sentiment_Score"] * df["Likes"] +
        0.5 * df["Shares"] -
        0.2 * df["Comments"]
    )

    # TEXT FEATURES
    df["Text_Length"] = df["Post_Content"].apply(lambda x: len(str(x).split()))
    df["Word_Count"] = df["Post_Content"].str.split().str.len()

    df["Hook_Score"] = (
        df["Likes"] * 0.4 +
        df["Shares"] * 0.4 +
        df["Impressions"] * 0.1 +
        (1 / (df["Word_Count"] + 1)) * 100
    )

    # TIME FEATURES
    df["Hour"] = df["Time"].str.split(":").str[0].astype(int)

    df["Is_Morning"] = df["Hour"].between(5, 11)
    df["Is_Afternoon"] = df["Hour"].between(12, 17)
    df["Is_Evening"] = df["Hour"].between(18, 23)
    df["Is_Night"] = df["Hour"].between(0, 4)

    return df