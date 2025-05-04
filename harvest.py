import os
import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import tweepy
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

# Helper to ensure environment variables are set
def get_env_var(name):
    val = os.getenv(name)
    if not val:
        sys.stderr.write(f"Error: environment variable {name} is not set.\n")
        sys.stderr.write("Please configure credentials in .env or export them.\n")
        sys.exit(1)
    return val

# Initialize Tweepy Client (v2) with rate-limit backoff
def init_client():
    bearer_token = get_env_var('TW_BEARER_TOKEN')
    return tweepy.Client(
        bearer_token=bearer_token,
        wait_on_rate_limit=False  # we'll handle backoff manually
    )
client = init_client()

# Rate-limit backoff helper
def backoff_sleep(response):
    headers = response.headers
    reset = int(headers.get('x-rate-limit-reset', time.time() + 60))
    wait = max(reset - int(time.time()), 60)
    sys.stderr.write(f"Rate limit hit. Sleeping for {wait} seconds...\n")
    time.sleep(wait)

# Get user ID for a handle, with backoff
def get_user_id(handle):
    while True:
        try:
            resp = client.get_user(username=handle.lstrip('@'))
            if resp.data:
                return resp.data.id
            else:
                raise RuntimeError(f"Unable to fetch user ID for {handle}")
        except tweepy.errors.TooManyRequests as e:
            backoff_sleep(e.response)

# Fetch tweets via v2 endpoint and filter by date, with basic rate-limit backoff
def fetch_tweets(usernames, start, end, output_csv):
    records = []
    start_iso = start.replace(tzinfo=timezone.utc).isoformat()
    end_iso = end.replace(tzinfo=timezone.utc).isoformat()
    for handle in usernames:
        uid = get_user_id(handle)
        paginator = tweepy.Paginator(
            client.get_users_tweets,
            uid,
            start_time=start_iso,
            end_time=end_iso,
            tweet_fields=['created_at','text'],
            max_results=100
        )
        for page in paginator:
            if page.data:
                for tweet in page.data:
                    records.append({
                        'user': handle,
                        'created_at': tweet.created_at,
                        'text': tweet.text
                    })
    pd.DataFrame(records).to_csv(output_csv, index=False)

# Fetch historical price bars via yfinance
def fetch_price_bars(ticker, start, end, interval, output_csv):
    df = yf.download(
        ticker,
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        interval=interval
    )
    df.to_csv(output_csv)

# Main entry point
def main():
    parser = argparse.ArgumentParser(description="Harvest tweets and price bars.")
    parser.add_argument('--users', nargs='+', required=True,
                        help="List of Twitter handles (e.g. @POTUS)")
    parser.add_argument('--ticker', required=True,
                        help="Stock ticker (e.g. AAPL)")
    parser.add_argument('--date', required=True,
                        help="Date in YYYY-MM-DD format")
    args = parser.parse_args()

    # Parse target day
    try:
        day = datetime.fromisoformat(args.date)
    except ValueError:
        sys.stderr.write("Error: --date must be YYYY-MM-DD\n")
        sys.exit(1)
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    # Fetch and save
    fetch_tweets(args.users, start, end, 'sample_tweets.csv')
    fetch_price_bars(args.ticker, start, end, '1m', 'sample_bars.csv')

if __name__ == '__main__':
    main()
