import pandas as pd

# Load the full dataset
df = pd.read_csv('pilot_study/stockerbot-export1.csv')

# Count mentions per symbol and sort descending
symbol_counts = df['symbols'].value_counts().reset_index()
symbol_counts.columns = ['Symbol', 'Count']
symbol_counts = symbol_counts.sort_values('Count', ascending=False)

# Initialize container for selected tweets
selected = []
limit = 3000

def collect_for_symbol(symbols_df, symbol, needed, df):
    """
    Collect up to `needed` tweets for a given symbol.
    Returns a DataFrame of collected tweets.
    """
    subset = df[df['symbols'] == symbol].head(needed)
    return subset

# Iterate over symbols until 3000 tweets collected
total_collected = 0
for sym in symbol_counts['Symbol']:
    if total_collected >= limit:
        break
    remaining = limit - total_collected
    batch = collect_for_symbol(symbol_counts, sym, remaining, df)
    selected.append(batch)
    total_collected += len(batch)

# Concatenate all batches into one DataFrame
result_df = pd.concat(selected, ignore_index=True)

# Verify count
print(f"Total tweets collected: {len(result_df)}")

# Optionally save to CSV
result_df.to_csv('pilot_study/top_stocks_3000_tweets.csv', index=False)
