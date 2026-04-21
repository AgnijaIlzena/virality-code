import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Import pandas to create a dummy dataframe

def plot_all(df):

    # 1. Engagement par plateforme
    plt.figure()
    sns.barplot(data=df, x="Platform", y="Virality_Score")
    plt.title("Virality par plateforme")
    plt.xticks(rotation=45)
    plt.show()

    # 2. Moment de la journée
    plt.figure()
    sns.barplot(data=df, x="Is_Evening", y="Virality_Score")
    plt.title("Impact soirée sur viralité")
    plt.show()

    # 3. Sentiment impact
    plt.figure()
    sns.boxplot(data=df, x="Sentiment", y="Emotion_Weight")
    plt.title("Impact émotionnel")
    plt.show()

    # 4. Hook Score vs Engagement
    plt.figure()
    sns.scatterplot(data=df, x="Hook_Score", y="Virality_Score")
    plt.title("Hook vs Viralité")
    plt.show()

# Dummy DataFrame to resolve NameError for demonstration
# In a real scenario, 'df' should come from the data loading and processing pipeline.
dummy_data = {
    'Platform': ['Facebook', 'Instagram', 'Twitter', 'Facebook'],
    'Virality_Score': [0.5, 0.7, 0.4, 0.6],
    'Is_Evening': [True, False, True, False],
    'Sentiment': ['Positive', 'Neutral', 'Negative', 'Positive'],
    'Emotion_Weight': [10, 5, -3, 8],
    'Hook_Score': [70, 85, 60, 75]
}
df = pd.DataFrame(dummy_data)

plot_all(df)