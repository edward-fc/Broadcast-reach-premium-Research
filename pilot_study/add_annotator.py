import pandas as pd

# Load the CSV file
input_file = "pilot_study/stockerbot-export1.csv"
output_file = "pilot_study/top_stocks_verified.csv"

# Read the CSV file
df = pd.read_csv(input_file)

# Normalize the 'verified' column to lowercase strings
df['verified'] = df['verified'].astype(str).str.upper()

# Filter for verified tweets
verified_tweets = df[df['verified'] == 'TRUE']

# Save the filtered tweets to a new CSV file
verified_tweets.to_csv(output_file, index=False)

print(f"Filtered {len(verified_tweets)} verified tweets into {output_file}")
