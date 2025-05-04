import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to fetch stock data, calculate returns, and CAR
def calculate_CAR(tweet_date, ticker='TSLA'):
    tweet_datetime = datetime.strptime(tweet_date[:10], '%Y-%m-%d')
    start_date = tweet_datetime - timedelta(days=30)
    end_date = tweet_datetime + timedelta(days=5)

    # Download stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Daily Return'] = stock_data['Close'].pct_change()

    # Estimating expected return using market model approximation (mean of returns)
    estimation_window = stock_data.loc[start_date:tweet_datetime - timedelta(days=1)]
    expected_daily_return = estimation_window['Daily Return'].mean()

    # Calculate Abnormal Returns (AR)
    event_window = stock_data.loc[tweet_datetime + timedelta(days=1):tweet_datetime + timedelta(days=3)].copy()
    event_window['Expected Return'] = expected_daily_return
    event_window['Abnormal Return'] = event_window['Daily Return'] - event_window['Expected Return']

    # Calculate CAR
    CAR = event_window['Abnormal Return'].sum()
    return CAR

# Load CSV and process tweets
tweets_df = pd.read_csv('musk_annotate.csv')
results = []

for index, row in tweets_df.iterrows():
    tweet_id = row['id']
    tweet_date = row['createdAt']
    tweet_text = row['fullText']
    CAR = calculate_CAR(tweet_date)
    results.append({'id': tweet_id, 'date': tweet_date, 'text': tweet_text, 'CAR': CAR})

# Output results
results_df = pd.DataFrame(results)
results_df.to_csv('tweets_CAR_results.csv', index=False)

# Statistical Analysis
mean_CAR = results_df['CAR'].mean()
median_CAR = results_df['CAR'].median()
max_CAR = results_df.loc[results_df['CAR'].idxmax()]
min_CAR = results_df.loc[results_df['CAR'].idxmin()]

print(f"Mean CAR: {mean_CAR:.4%}")
print(f"Median CAR: {median_CAR:.4%}")
print(f"Max CAR: {max_CAR['CAR']:.4%} on {max_CAR['date']} (Tweet ID: {max_CAR['id']})")
print(f"Min CAR: {min_CAR['CAR']:.4%} on {min_CAR['date']} (Tweet ID: {min_CAR['id']})")

# Visualization of CAR distribution
plt.figure(figsize=(10,6))
results_df['CAR'].hist(bins=30, edgecolor='black')
plt.title('Distribution of CAR')
plt.xlabel('CAR')
plt.ylabel('Frequency')
plt.axvline(mean_CAR, color='red', linestyle='dashed', linewidth=2, label=f'Mean CAR ({mean_CAR:.2%})')
plt.legend()
plt.grid(False)
plt.savefig('CAR_distribution.png')

# Find significant tweets
significant_tweets = results_df[(results_df['CAR'] >= 0.10) | (results_df['CAR'] <= -0.10)]
significant_tweets.to_csv('significant_tweets.csv', index=False)

# Plot Tesla stock price with significant tweet markers
full_start_date = datetime.strptime(tweets_df['createdAt'].min()[:10], '%Y-%m-%d') - timedelta(days=30)
full_end_date = datetime.strptime(tweets_df['createdAt'].max()[:10], '%Y-%m-%d') + timedelta(days=5)
full_stock_data = yf.download('TSLA', start=full_start_date, end=full_end_date)

plt.figure(figsize=(14, 7))
plt.plot(full_stock_data['Close'], label='Tesla Stock Price')

for _, tweet in significant_tweets.iterrows():
    tweet_date = datetime.strptime(tweet['date'][:10], '%Y-%m-%d')
    if tweet_date in full_stock_data.index:
        plt.axvline(tweet_date, color='red', linestyle='--', linewidth=1)

plt.title('Tesla Stock Price with Significant Tweets')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid(True)
plt.savefig('Tesla_stock_significant_tweets.png')

print("\nSignificant tweets graph saved as Tesla_stock_significant_tweets.png")
