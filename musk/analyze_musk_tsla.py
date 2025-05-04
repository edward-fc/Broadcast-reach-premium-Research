import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import argparse


def main(tweets_file, year):
    tweets = pd.read_csv(tweets_file, parse_dates=['createdAt'])
    first_date = tweets['createdAt'].dt.date.min()
    last_date  = tweets['createdAt'].dt.date.max()
    start_date = pd.to_datetime(first_date) - pd.Timedelta(days=3)
    end_date   = pd.to_datetime(last_date) + pd.Timedelta(days=3)
    tsla_daily = yf.download(
        'TSLA',
        start=start_date.strftime('%Y-%m-%d'),
        end=(end_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        interval='1d'
    )
    plt.figure(figsize=(10, 6))
    plt.plot(tsla_daily.index, tsla_daily['Close'], label='TSLA Close')
    for ts in tweets['createdAt']:
        plt.axvline(ts, color='red', linestyle='--', alpha=0.5)
    plt.title(f'TSLA Daily Close vs Tesla-related Musk Tweets ({year})')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'daily_{year}_tesla_tweets.png')
    print(f"Saved daily_{year}_tesla_tweets.png")

    tsla_daily['Return'] = tsla_daily['Close'].pct_change()
    tweet_days = tweets['createdAt'].dt.floor('D')
    impact = {day: tsla_daily.loc[day, 'Return']
              for day in tweet_days.unique() if day in tsla_daily.index}
    if impact:
        impact_date = max(impact, key=lambda d: abs(impact[d]))
        print(f'Most impactful Tesla tweet date: {impact_date.date()} with return {impact[impact_date]:.2%}')
        intraday = yf.download(
            'TSLA',
            start=impact_date.strftime('%Y-%m-%d'),
            end=(impact_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
            interval='5m'
        )
        plt.figure(figsize=(10, 6))
        plt.plot(intraday.index, intraday['Close'], label='TSLA 5-min')
        plt.axvline(impact_date + pd.Timedelta(hours=16, minutes=48),
                    color='red', linestyle='--', alpha=0.7)
        plt.title(f'TSLA Intraday on {impact_date.date()} (Tesla-related tweet)')
        plt.xlabel('Time')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'intraday_{impact_date.date()}_tesla.png')
        print(f"Saved intraday_{impact_date.date()}_tesla.png")
    else:
        print('No matching trading days for Tesla-related tweets.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Annotated Tesla tweets CSV')
    parser.add_argument('--year', type=int, required=True)
    args = parser.parse_args()
    main(args.file, args.year)
