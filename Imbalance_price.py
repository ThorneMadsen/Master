import pandas as pd

def calculate_afrr_costs(filepath):
    """Calculates total and average monthly aFRR reserve cost in DKK for DK2 from 2023 onwards."""
    df = pd.read_csv(filepath)

    # Convert relevant columns to numeric, filling errors with 0
    numeric_cols = ['aFRR_DownPurchased', 'aFRR_UpPurchased', 'aFRR_DownCapPriceDKK', 'aFRR_UpCapPriceDKK']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Parse timestamp and filter
    df['timestamp'] = pd.to_datetime(df['HourUTC'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp']) # Ensure timestamp is valid
    
    # Perform filtering in one step
    df_filtered = df[(df['timestamp'].dt.year >= 2023) & (df['PriceArea'] == 'DK2')].copy()

    if df_filtered.empty:
        print("Warning: No relevant data found after filtering.")
        return 0, 0 # Return 0 for both total and average

    # Calculate hourly cost
    df_filtered['hourly_cost_dkk'] = (
        (df_filtered['aFRR_UpPurchased'] * df_filtered['aFRR_UpCapPriceDKK']) + \
        (df_filtered['aFRR_DownPurchased'] * df_filtered['aFRR_DownCapPriceDKK'])
    )
    
    # Calculate total cost
    total_cost = df_filtered['hourly_cost_dkk'].sum()
    
    # Calculate average monthly cost
    # Ensure timestamp is the index for resampling or grouping
    df_filtered_indexed = df_filtered.set_index('timestamp')
    # Group by month start ('MS') and sum hourly costs
    monthly_costs = df_filtered_indexed['hourly_cost_dkk'].resample('MS').sum()
    # Calculate the average of non-zero monthly costs (if any)
    average_monthly_cost = monthly_costs[monthly_costs > 0].mean() if not monthly_costs[monthly_costs > 0].empty else 0

    return total_cost, average_monthly_cost

def analyze_daily_afrr_purchases(filepath):
    """Analyzes daily aFRR purchases for correlations, minimums, and modes."""
    df = pd.read_csv(filepath)

    # Convert relevant columns to numeric, handling errors
    numeric_cols = ['aFRR_DownPurchased', 'aFRR_UpPurchased']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Parse timestamp and filter
    df['timestamp'] = pd.to_datetime(df['HourUTC'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp']) # Ensure timestamp is valid
    
    # Perform filtering in one step
    df_filtered = df[(df['timestamp'].dt.year >= 2023) & (df['PriceArea'] == 'DK2')].copy()

    if df_filtered.empty:
        print("Warning: No relevant data found for daily analysis after filtering.")
        return

    # Set timestamp as index for resampling
    df_filtered_indexed = df_filtered.set_index('timestamp')
    
    # Resample daily and sum up/down purchases
    daily_sums = df_filtered_indexed[['aFRR_UpPurchased', 'aFRR_DownPurchased']].resample('D').sum()
    
    # Calculate total daily purchases
    daily_sums['total_daily_afrr'] = daily_sums['aFRR_UpPurchased'] + daily_sums['aFRR_DownPurchased']

    # Calculate correlation between daily up and down purchases
    correlation = daily_sums['aFRR_UpPurchased'].corr(daily_sums['aFRR_DownPurchased'])

    # Find the minimum daily total sum
    min_sum = daily_sums['total_daily_afrr'].min()
    min_sum_days = daily_sums[daily_sums['total_daily_afrr'] == min_sum].index.strftime('%Y-%m-%d').tolist()

    # Find the most frequent daily total sum (mode)
    mode_sum = daily_sums['total_daily_afrr'].mode()
    most_frequent_sum = mode_sum[0] if not mode_sum.empty else "N/A" # Handle cases with no unique mode or empty data

    print(f"\n--- Daily aFRR Purchase Analysis (DK2, 2023+) ---")
    print(f"Correlation between daily Up and Down purchases: {correlation:.2f}")
    print(f"Lowest total daily purchase: {min_sum:,.2f}")
    print(f"Date(s) with the lowest sum ({len(min_sum_days)} out of {len(daily_sums)} total days): {', '.join(min_sum_days)}")
    print(f"Most frequent total daily purchase sum: {most_frequent_sum:,.2f}")
    print(f"-------------------------------------------------")

if __name__ == "__main__":
    csv_filepath = "data/afrr_reserves_dk2_2023_today.csv"
    total_dkk_cost, avg_monthly_cost = calculate_afrr_costs(csv_filepath)

    # Output cost result
    print(f"\n--- aFRR Reserve Costs (DK2, 2023+) ---")
    print(f"Total Cost:      {total_dkk_cost:,.2f} DKK")
    print(f"Avg Monthly Cost: {avg_monthly_cost:,.2f} DKK")
    print(f"----------------------------------------")

    # Perform and output daily analysis
    analyze_daily_afrr_purchases(csv_filepath)
