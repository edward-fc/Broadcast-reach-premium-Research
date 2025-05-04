import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.model_selection import train_test_split

# Load full dataset
df = pd.read_csv('pilot_study/top_stocks_3000_with_sentiment.csv')

# (Re)extract tickers
ticker_re = re.compile(r"\$[A-Z]{1,5}\b")
df['tickers'] = df['text'].astype(str).apply(lambda t: ticker_re.findall(t))

# Sample 500 tweets
gold, _ = train_test_split(df, train_size=500, random_state=42)

# Prepare annotation file with only model_pred and tickers
anot_df = gold[['id', 'text', 'finbert_sentiment', 'tickers']].copy()
anot_df.rename(columns={'finbert_sentiment': 'model_pred'}, inplace=True)

# Save to CSV
output_path = 'pilot_study/500_tweet_annotation_template.csv'
anot_df.to_csv(output_path, index=False)
print(f"Exported annotation template without gold fields: {output_path}")
