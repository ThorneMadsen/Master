import numpy as np
import pandas as pd

def combine_mfrr_data(file_path_2021_2023: str, file_path_2023_2025: str) -> pd.DataFrame:
    """Combines two mFRR reserve datasets, prioritizing the newer data for overlaps."""
    # Define columns to use and renaming for the second dataset
    cols_2021 = ['HourUTC', 'mFRR_DownPurchased', 'mFRR_UpPurchased']
    cols_2023_select = ['HourUTC', 'mFRRDownActBal', 'mFRRUpActBal']
    cols_2023_rename = {
        'mFRRDownActBal': 'mFRR_DownPurchased',
        'mFRRUpActBal': 'mFRR_UpPurchased'
    }

    # Load the first dataset
    try:
        df1 = pd.read_csv(file_path_2021_2023, usecols=cols_2021, parse_dates=['HourUTC'])
    except FileNotFoundError:
        print(f"Error: File not found at {file_path_2021_2023}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"Error reading {file_path_2021_2023}: {e}")
        # Attempt to load without parsing dates initially if specific error occurs
        try:
            df1 = pd.read_csv(file_path_2021_2023, usecols=cols_2021)
            df1['HourUTC'] = pd.to_datetime(df1['HourUTC'], errors='coerce')
        except Exception as inner_e:
             print(f"Failed to load and parse dates for {file_path_2021_2023}: {inner_e}")
             return pd.DataFrame()


    # Load the second dataset
    try:
        df2 = pd.read_csv(file_path_2023_2025, usecols=cols_2023_select, parse_dates=['HourUTC'])
    except FileNotFoundError:
        print(f"Error: File not found at {file_path_2023_2025}")
        return pd.DataFrame() # Or handle appropriately
    except ValueError as e:
        print(f"Error reading {file_path_2023_2025}: {e}")
        # Attempt to load without parsing dates initially if specific error occurs
        try:
            df2 = pd.read_csv(file_path_2023_2025, usecols=cols_2023_select)
            df2['HourUTC'] = pd.to_datetime(df2['HourUTC'], errors='coerce')
        except Exception as inner_e:
             print(f"Failed to load and parse dates for {file_path_2023_2025}: {inner_e}")
             return pd.DataFrame() # Or handle appropriately

    # Rename columns in the second dataset
    df2 = df2.rename(columns=cols_2023_rename)

    # Concatenate the dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Sort by HourUTC to ensure correct order before dropping duplicates
    combined_df = combined_df.sort_values(by='HourUTC', ascending=True)

    # Drop duplicates, keeping the last (most recent dataset's entry)
    combined_df = combined_df.drop_duplicates(subset=['HourUTC'], keep='last')

    # Optional: Reset index after dropping duplicates
    combined_df = combined_df.reset_index(drop=True)

    # Drop rows where HourUTC parsing failed (if any)
    combined_df = combined_df.dropna(subset=['HourUTC'])


    return combined_df


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

def load_and_process_imbalance_data():
    """Load and process Imbalance and DMI data, returning aligned feature arrays."""
    # Load data
    imbalance_df = pd.read_csv("data/imbalance_dk2_nov2022_march2025.csv")
    dmi_df = pd.read_csv("data/DMI_data.csv")

    # Convert timestamps and filter 2023 data
    imbalance_df['timestamp'] = pd.to_datetime(imbalance_df['HourUTC'], utc=True)
    imbalance_df = imbalance_df[imbalance_df['timestamp'].dt.year >= 2023]

    # Select relevant columns (no calculation needed for imbalance)
    imbalance_df = imbalance_df[['timestamp', 'ImbalanceMWh']]

    # Convert DMI timestamp to UTC
    dmi_df['timestamp'] = pd.to_datetime(dmi_df.iloc[:, 0], utc=True)

    # Merge data
    merged_df = pd.merge(
        imbalance_df,
        dmi_df[['timestamp', 'mean_temp', 'mean_wind_speed']],
        on='timestamp',
        how='inner'
    )

    # Sort
    merged_df = merged_df.sort_values('timestamp')

    # --- Fill Gaps ---
    if not merged_df.empty:
        print(f"\nImbalance Data before gap filling - Min: {merged_df['timestamp'].min()}, Max: {merged_df['timestamp'].max()}, Length: {len(merged_df)}")
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
        print(f"Imbalance Data after gap filling - Min: {merged_df['timestamp'].min()}, Max: {merged_df['timestamp'].max()}, Length: {len(merged_df)}")
    # --- End Gap Filling ---

    # Ensure data starts exactly at midnight UTC
    first_midnight_index = merged_df[merged_df['timestamp'].dt.hour == 0].index.min()
    if pd.isna(first_midnight_index):
        print("Error: No midnight timestamp found in merged imbalance data. Cannot proceed.")
        # Return empty arrays to prevent errors later
        return np.array([]), np.array([]), np.array([]), pd.Series([], dtype='datetime64[ns, UTC]')

    merged_df = merged_df.loc[first_midnight_index:].reset_index(drop=True)
    print(f"Filtered merged imbalance data to start from first midnight ({merged_df.iloc[0]['timestamp']}). New shape: {merged_df.shape}")

    # --- Truncate to full days ---
    num_full_days = len(merged_df) // 24
    rows_to_keep = num_full_days * 24
    if rows_to_keep < len(merged_df):
        merged_df = merged_df.iloc[:rows_to_keep]
        print(f"Truncated imbalance data to {rows_to_keep} rows ({num_full_days} full days). Last timestamp: {merged_df.iloc[-1]['timestamp']}")
    # --- End Truncation ---

    # Extract arrays
    return (
        merged_df['ImbalanceMWh'].values,
        merged_df['mean_temp'].values,
        merged_df['mean_wind_speed'].values,
        merged_df['timestamp']
    )

def create_sequences(feature1, temp, wind, timestamps, seq_length=24):
    """Create 24-hour sequences with features plus day sin/cos."""
    X = []
    total_length = len(feature1)
    
    for i in range(0, total_length - seq_length + 1, seq_length):
        # Get first timestamp of sequence for day sin/cos
        first_timestamp = timestamps.iloc[i]
        day_of_year = first_timestamp.dayofyear
        days_in_year = 366.0 if first_timestamp.is_leap_year else 365.0
        day_angle = (day_of_year / days_in_year) * 2 * np.pi
        
        # Create sequence
        X_seq = np.stack([
            feature1[i:i+seq_length],
            temp[i:i+seq_length],
            wind[i:i+seq_length],
            np.full(seq_length, np.sin(day_angle)),
            np.full(seq_length, np.cos(day_angle))
        ], axis=0)
        X.append(X_seq)
    
    return np.array(X)

if __name__ == "__main__":
    # --- Combine mFRR data ---
    mfrr_path1 = 'data/mfrr_reserves_dk2_2021_june2023.csv'
    mfrr_path2 = 'data/mfrr_reserves_dk2_june2023_march2025.csv'
    mfrr_combined_df = combine_mfrr_data(mfrr_path1, mfrr_path2)
    if not mfrr_combined_df.empty:
        print("\n--- Combined mFRR Data ---")
        print(mfrr_combined_df.head())
        print(f"Total mFRR rows: {len(mfrr_combined_df)}")
        # Save the combined data
        output_path = 'data/mfrr_reserves_dk2_combined.csv'
        try:
            mfrr_combined_df.to_csv(output_path, index=False)
            print(f"Combined mFRR data saved to {output_path}")
        except Exception as e:
            print(f"Error saving combined mFRR data: {e}")
    # -------------------------

    # Load and process aFRR/DMI data
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
        np.save("data/X.npy", X)
        print(f"\nSaved {len(X)} sequences to data/X.npy")


    # ---- Process Imbalance Data for X2 ----
    print("\n\n--- Processing Imbalance Data for X2 ---")
    imbalance, imb_temp, imb_wind, imb_timestamps = load_and_process_imbalance_data()

    # --- Imbalance Diagnostics ---
    if imb_timestamps.empty:
        print("Imbalance Processing resulted in empty data. Cannot create sequences.")
    else:
        print("\n--- Imbalance Timestamp Diagnostics (after midnight filter) ---")
        print(f"Total hours available: {len(imb_timestamps)}")
        print(f"Is total hours divisible by 24? {'Yes' if len(imb_timestamps) % 24 == 0 else 'No'}")
        print(f"First 5 timestamps:\n{imb_timestamps.head()}")
        print(f"Last 5 timestamps:\n{imb_timestamps.tail()}")

        # Check for gaps
        imb_time_diffs = imb_timestamps.diff().iloc[1:]
        expected_diff = pd.Timedelta(hours=1)
        imb_gaps = imb_time_diffs[imb_time_diffs != expected_diff]
        if not imb_gaps.empty:
            print(f"\nWarning: Found {len(imb_gaps)} timestamp gaps in imbalance data (expected 1 hour difference):")
            print(imb_gaps)
        else:
            print("\nNo timestamp gaps found in imbalance data.")
            
        X2 = create_sequences(imbalance, imb_temp, imb_wind, imb_timestamps)
        
        # Convert imbalance feature (index 0) from MW to GW
        if X2.size > 0: # Ensure X2 is not empty before modifying
            X2[:, 0, :] /= 1000.0
            print("Converted imbalance feature in X2 from MW to GW.")

        # Check for NaNs in X2 before saving
        if np.isnan(X2).any():
            print("\nWarning: NaNs found in X2 sequences. Check the input data and processing steps.")
            # Optionally handle the NaN case, e.g., skip saving or attempt filling
        else:
            np.save("data/X2.npy", X2)
            print(f"\nSaved {len(X2)} sequences to data/X2.npy")
