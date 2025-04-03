import numpy as np
import pandas as pd

def load_and_process_data():
    """Load and process aFRR and DMI data, returning aligned feature arrays."""
    # Load data
    afrr_df = pd.read_csv("data/afrr_activated_dk2_2023_today.csv")
    dmi_df = pd.read_csv("data/DMI_data.csv")
    
    # Convert timestamps and filter 2022 data
    afrr_df['timestamp'] = pd.to_datetime(afrr_df['HourUTC'], utc=True)
    afrr_df = afrr_df[afrr_df['timestamp'].dt.year >= 2023]
    
    # Calculate net aFRR (up - down)
    afrr_df['aFRR_Net_Quantity_MW'] = afrr_df['aFRR_UpActivated'] - afrr_df['aFRR_DownActivated']
    
    # Convert DMI timestamp to UTC
    dmi_df['timestamp'] = pd.to_datetime(dmi_df.iloc[:, 0], utc=True)
    
    # Merge data
    merged_df = pd.merge(
        afrr_df[['timestamp', 'aFRR_Net_Quantity_MW']],
        dmi_df[['timestamp', 'mean_temp', 'mean_wind_speed']],
        on='timestamp',
        how='inner'
    )
    
    # Sort 
    merged_df = merged_df.sort_values('timestamp')

    # --- Fill Gaps ---
    if not merged_df.empty:
        print(f"\nData before gap filling - Min: {merged_df['timestamp'].min()}, Max: {merged_df['timestamp'].max()}, Length: {len(merged_df)}")
        # Create a full hourly range
        full_range = pd.date_range(start=merged_df['timestamp'].min(), end=merged_df['timestamp'].max(), freq='H', tz='UTC')
        # Set timestamp as index for reindexing
        merged_df = merged_df.set_index('timestamp')
        # Reindex to fill gaps (introduces NaNs)
        merged_df = merged_df.reindex(full_range)
        # Forward fill NaNs, then backward fill any remaining at the start
        merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
        # Reset index to get timestamp column back
        merged_df = merged_df.reset_index().rename(columns={'index': 'timestamp'})
        print(f"Data after gap filling - Min: {merged_df['timestamp'].min()}, Max: {merged_df['timestamp'].max()}, Length: {len(merged_df)}")
    # --- End Gap Filling ---

    # Ensure data starts exactly at midnight UTC
    first_midnight_index = merged_df[merged_df['timestamp'].dt.hour == 0].index.min()
    if pd.isna(first_midnight_index):
        print("Error: No midnight timestamp found in merged data. Cannot proceed.")
        # Return empty arrays to prevent errors later
        return np.array([]), np.array([]), np.array([]), pd.Series([], dtype='datetime64[ns, UTC]')
    
    merged_df = merged_df.loc[first_midnight_index:].reset_index(drop=True)
    print(f"Filtered merged data to start from first midnight ({merged_df.iloc[0]['timestamp']}). New shape: {merged_df.shape}")

    # --- Truncate to full days ---
    num_full_days = len(merged_df) // 24
    rows_to_keep = num_full_days * 24
    if rows_to_keep < len(merged_df):
        merged_df = merged_df.iloc[:rows_to_keep]
        print(f"Truncated data to {rows_to_keep} rows ({num_full_days} full days). Last timestamp: {merged_df.iloc[-1]['timestamp']}")
    # --- End Truncation ---

    # Extract arrays
    return (
        merged_df['aFRR_Net_Quantity_MW'].values,
        merged_df['mean_temp'].values,
        merged_df['mean_wind_speed'].values,
        merged_df['timestamp']
    )

def create_sequences(afrr, temp, wind, timestamps, seq_length=24):
    """Create 24-hour sequences with features plus day sin/cos."""
    X = []
    total_length = len(afrr)
    
    for i in range(0, total_length - seq_length + 1, seq_length):
        # Get first timestamp of sequence for day sin/cos
        first_timestamp = timestamps.iloc[i]
        day_of_year = first_timestamp.dayofyear
        days_in_year = 366.0 if first_timestamp.is_leap_year else 365.0
        day_angle = (day_of_year / days_in_year) * 2 * np.pi
        
        # Create sequence
        X_seq = np.stack([
            afrr[i:i+seq_length],
            temp[i:i+seq_length],
            wind[i:i+seq_length],
            np.full(seq_length, np.sin(day_angle)),
            np.full(seq_length, np.cos(day_angle))
        ], axis=0)
        X.append(X_seq)
    
    return np.array(X)

if __name__ == "__main__":
    # Load and process data
    afrr, temp, wind, timestamps = load_and_process_data()

    # --- Diagnostics ---
    if timestamps.empty:
        print("Processing resulted in empty data. Cannot create sequences.")
    else:
        print("\n--- Timestamp Diagnostics (after midnight filter) ---")
        print(f"Total hours available: {len(timestamps)}")
        print(f"Is total hours divisible by 24? {'Yes' if len(timestamps) % 24 == 0 else 'No'}")
        print(f"First 5 timestamps:\n{timestamps.head()}")
        print(f"Last 5 timestamps:\n{timestamps.tail()}")

        # Check for gaps (difference between consecutive timestamps should be 1 hour)
        time_diffs = timestamps.diff().iloc[1:]
        expected_diff = pd.Timedelta(hours=1)
        gaps = time_diffs[time_diffs != expected_diff]
        if not gaps.empty:
            print(f"\nWarning: Found {len(gaps)} timestamp gaps (expected 1 hour difference):")
            print(gaps)
        else:
            print("\nNo timestamp gaps found.")
        # --- End Diagnostics ---

        # Create sequences
        X = create_sequences(afrr, temp, wind, timestamps)

        # Save to file
        if X.size > 0:
            np.save("data/X.npy", X)
            print(f"\nSaved {len(X)} sequences to data/X.npy")
        else:
             print("\nNo sequences were created.")
