import pandas as pd
import argparse

def main(year):
    # Load full Elon Musk tweet dataset
    df = pd.read_csv('musk/all_musk_posts.csv', parse_dates=['createdAt'])

    # Filter to the specified calendar year
    start = pd.to_datetime(f'{year}-01-01')
    end   = pd.to_datetime(f'{year}-12-31')
    mask = (df['createdAt'].dt.date >= start.date()) & \
           (df['createdAt'].dt.date <= end.date())
    year_df = df.loc[mask]

    # Save filtered tweets
    output_file = f'musk_{year}_posts.csv'
    year_df.to_csv(output_file, index=False)
    print(f"Filtered {len(year_df)} tweets to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter Musk tweets by year')
    parser.add_argument('--year', type=int, required=True,
                        help='Calendar year to filter (e.g., 2018)')
    args = parser.parse_args()
    main(args.year)
