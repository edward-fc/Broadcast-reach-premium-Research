import pandas as pd

def main():
    # Load full Elon Musk tweet dataset
    df = pd.read_csv('all_musk_posts.csv', parse_dates=['createdAt'])

    # Filter to August 2018 (one month)
    start = pd.to_datetime('2024-08-01')
    end   = pd.to_datetime('2024-08-31')
    mask = (df['createdAt'].dt.date >= start.date()) & \
           (df['createdAt'].dt.date <= end.date())
    month_df = df.loc[mask]

    # Save filtered tweets
    month_df.to_csv('musk_aug2018_posts.csv', index=False)
    print(f"Filtered {len(month_df)} tweets to musk_aug2018_posts.csv")

if __name__ == '__main__':
    main()
