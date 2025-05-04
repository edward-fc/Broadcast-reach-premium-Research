import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load FinBERT model
tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Regex pattern for Tesla mentions
tesla_pattern = re.compile(r"\b[Tt]esla\b|\$TSLA\b")

# Regex for any ticker mentions
ticker_pattern = re.compile(r"\$[A-Z]{1,5}\b")


def annotate_and_filter(input_csv, output_csv):
    df = pd.read_csv(input_csv, parse_dates=['createdAt'])
    records = []
    for _, row in df.iterrows():
        text = row.get('fullText', row.get('text', ''))
        # filter for Tesla mentions only
        if not tesla_pattern.search(text):
            continue
        # sentiment scoring
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        sent_bear, sent_neut, sent_bull = scores.tolist()
        # extract tickers
        tickers = ticker_pattern.findall(text)
        records.append({
            **row.to_dict(),
            'sent_bear': sent_bear,
            'sent_neut': sent_neut,
            'sent_bull': sent_bull,
            'tickers': tickers
        })
    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    print(f"Annotated and filtered {len(out_df)} Tesla-related tweets to {output_csv}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: python annotate_tesla.py <input_csv> <output_csv>')
        sys.exit(1)
    annotate_and_filter(sys.argv[1], sys.argv[2])
