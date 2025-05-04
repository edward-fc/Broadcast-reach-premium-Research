import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sys

# Load FinBERT model
tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Sentiment thresholds\ nNEG_THRESHOLD = 0.2  # includes tweets with bearish ≥20%
POS_THRESHOLD = 0.2  # includes tweets with bullish ≥20%
NEG_THRESHOLD = 0.2  # includes tweets with bullish ≥20%

def load_stock_patterns(stocks_csv):
    """Load regex patterns for company names and optional product context."""
    stocks = pd.read_csv(stocks_csv)
    patterns = []
    for _, row in stocks.iterrows():
        company = re.escape(row['Company'])
        try:
            products = eval(row.get('Context', '[]'))
            # filter empty products
            items = products if products else []
        except Exception:
            items = []
        context = [company] + items
        pattern = re.compile(r"\b(?:" + r"|".join(re.escape(x) for x in context) + r")\b", re.IGNORECASE)
        patterns.append((row['Ticker'], pattern))
    return patterns


def annotate_and_filter(input_csv, output_csv, stocks_csv):
    patterns = load_stock_patterns(stocks_csv)
    df = pd.read_csv(input_csv, parse_dates=['timestamp'])
    records = []

    for index_row, row in df.iterrows():
        # Print progress as a percentage
        percentage = (index_row / len(df)) * 100
        print(f"Processing: {percentage:.2f}% ({index_row}/{len(df)})", end='\r')
        text = row.get('text', '')
        matched = []
        for ticker, pat in patterns:
            if pat.search(text):
                matched.append(ticker)
        if not matched:
            continue

        # Sentiment scoring
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        logits = model(**inputs).logits
        scores = torch.softmax(logits, dim=1)[0].tolist()
        sent_bear, sent_neut, sent_bull = scores

        # Filter out too-neutral tweets\ nif sent_neut >= 0.8:
        if sent_bear < NEG_THRESHOLD and sent_bull < POS_THRESHOLD:
            continue

        records.append({
            **row.to_dict(),
            'sent_bear': sent_bear,
            'sent_neut': sent_neut,
            'sent_bull': sent_bull,
            'matched_tickers': ','.join(set(matched))
        })

    out = pd.DataFrame(records)
    out.to_csv(output_csv, index=False)
    print(f"Filtered and annotated {len(out)} tweets with clear positive/negative sentiment to {output_csv}")

if __name__ == '__main__':
    if len(sys.argv)!=4:
        print('Usage: python annotate_stocks_filtered.py <input_csv> <output_csv> <stocks_csv>')
        sys.exit(1)
    annotate_and_filter(sys.argv[1], sys.argv[2], sys.argv[3])
