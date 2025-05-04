import pandas as pd
import yfinance as yf
from datetime import timedelta

# Load Biden tweets dataset
tweets = pd.read_csv('annotate_congress.csv', parse_dates=['date'])
window_days=3
threshold=0.02
# Function to download and compute abnormal move

def compute_move(ticker, ts):
    if pd.isna(ticker) or ticker == '':
        return None

    # Split the ticker string into individual tickers
    tickers = ticker.split(',')

    # Initialize variables to store the results
    moves = []

    for single_ticker in tickers:
        try:
            # Fetch price data for the individual ticker
            df = yf.download(
                single_ticker,
                start=(ts - timedelta(days=1)).strftime('%Y-%m-%d'),
                end=(ts + timedelta(days=window_days + 1)).strftime('%Y-%m-%d'),
                progress=False
            )
        except Exception:
            continue

        if df.empty:
            continue

        # Convert index to naive dates for comparison
        df.index = df.index.tz_localize(None)

        # Compute the move for the individual ticker
        try:
            day0 = float(df['Close'].iloc[0])  # First day's close price
            dayN = float(df['Close'].iloc[-1])  # Last day's close price
            ret = (dayN - day0) / day0  # Calculate return

            # Compare the return with the threshold
            if ret > threshold:
                moves.append('up')
            elif ret < -threshold:
                moves.append('down')
            else:
                moves.append('neutral')
        except Exception:
            continue

    # Return the result for the first ticker or aggregate results
    return moves[0] if moves else None

# Initialize price_move column
tweets['price_move'] = None

print("Fetching price data and computing 3-day moves (using for-loop)...")
# Iterate over rows with a for-loop
for idx, row in tweets.iterrows():
    move = compute_move(row['matched_tickers'], row['date'])
    tweets.at[idx, 'price_move'] = move

# Save result
tweets.to_csv('annotate_congress_with_price_moves.csv', index=False)
print("Saved enriched file: annotate_joe_with_price_moves.csv")
