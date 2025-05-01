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
    
    # Convert timestamps
    afrr_df['timestamp'] = pd.to_datetime(afrr_df['HourUTC'], utc=True)
    
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
    imbalance_df = pd.read_csv("data/imbalance_dk2_nov2022_march2025.csv")
    dmi_df = pd.read_csv("data/DMI_data.csv")

    imbalance_df['timestamp'] = pd.to_datetime(imbalance_df['HourUTC'], utc=True)
    imbalance_df = imbalance_df[['timestamp', 'ImbalanceMWh']]

    dmi_df['timestamp'] = pd.to_datetime(dmi_df.iloc[:, 0], utc=True)

    required_dmi_cols = ['timestamp', 'mean_temp', 'mean_wind_speed', 'wind_dir_sin', 'wind_dir_cos']
    if not all(col in dmi_df.columns for col in required_dmi_cols):
        missing_cols = [col for col in required_dmi_cols if col not in dmi_df.columns]
        print(f"Error: Missing required DMI columns: {missing_cols}. Cannot proceed.")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), pd.Series([], dtype='datetime64[ns, UTC]')

    merged_df = pd.merge(
        imbalance_df,
        dmi_df[required_dmi_cols],
        on='timestamp',
        how='inner'
    )

    if merged_df.empty:
        print("Warning: Merged imbalance/DMI dataframe is empty after inner join.")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), pd.Series([], dtype='datetime64[ns, UTC]')

    merged_df = merged_df.sort_values('timestamp')

    # --- Fill Gaps ---
    full_range = pd.date_range(start=merged_df['timestamp'].min(), end=merged_df['timestamp'].max(), freq='H', tz='UTC')
    merged_df = merged_df.set_index('timestamp')
    merged_df = merged_df.reindex(full_range)
    cols_to_fill = ['ImbalanceMWh', 'mean_temp', 'mean_wind_speed', 'wind_dir_sin', 'wind_dir_cos']
    merged_df[cols_to_fill] = merged_df[cols_to_fill].fillna(method='ffill').fillna(method='bfill')
    merged_df = merged_df.reset_index().rename(columns={'index': 'timestamp'})
    # --- End Gap Filling ---

    first_midnight_index = merged_df[merged_df['timestamp'].dt.hour == 0].index.min()
    if pd.isna(first_midnight_index):
        print("Error: No midnight timestamp found in merged imbalance data after processing. Cannot proceed.")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), pd.Series([], dtype='datetime64[ns, UTC]')

    merged_df = merged_df.loc[first_midnight_index:].reset_index(drop=True)

    # --- Truncate to full days ---
    num_full_days = len(merged_df) // 24
    rows_to_keep = num_full_days * 24
    if rows_to_keep == 0:
         print("Warning: Not enough data for a full 24-hour sequence after filtering.")
         return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), pd.Series([], dtype='datetime64[ns, UTC]')
    if rows_to_keep < len(merged_df):
        merged_df = merged_df.iloc[:rows_to_keep]
    # --- End Truncation ---

    return (
        merged_df['ImbalanceMWh'].values,
        merged_df['mean_temp'].values,
        merged_df['mean_wind_speed'].values,
        merged_df['wind_dir_sin'].values,
        merged_df['wind_dir_cos'].values,
        merged_df['timestamp']
    )

def create_sequences(feature1, temp, wind_speed, wind_dir_sin, wind_dir_cos, timestamps, seq_length=24):
    """Create 24-hour sequences with features plus day sin/cos and week sin/cos."""
    X = []
    total_length = len(feature1)

    if total_length < seq_length:
        return np.array(X)

    for i in range(0, total_length - seq_length + 1, seq_length):
        first_timestamp = timestamps.iloc[i]
        
        # Day of year encoding
        day_of_year = first_timestamp.dayofyear
        days_in_year = 366.0 if first_timestamp.is_leap_year else 365.0
        day_angle = (day_of_year / days_in_year) * 2 * np.pi
        
        # Day of week encoding (Monday=0, Sunday=6)
        day_of_week = first_timestamp.dayofweek 
        week_angle = (day_of_week / 7.0) * 2 * np.pi

        X_seq = np.stack([
            feature1[i:i+seq_length],
            temp[i:i+seq_length],
            wind_speed[i:i+seq_length],
            wind_dir_sin[i:i+seq_length],
            wind_dir_cos[i:i+seq_length],
            np.full(seq_length, np.sin(day_angle)),
            np.full(seq_length, np.cos(day_angle)),
            np.full(seq_length, np.sin(week_angle)), # Add day of week sin
            np.full(seq_length, np.cos(week_angle))  # Add day of week cos
        ], axis=0)
        X.append(X_seq)

    return np.array(X)

if __name__ == "__main__":
    # --- Combine mFRR data (Optional Section) ---
    # mfrr_path1 = 'data/mfrr_reserves_dk2_2021_june2023.csv'
    # mfrr_path2 = 'data/mfrr_reserves_dk2_june2023_march2025.csv'
    # mfrr_combined_df = combine_mfrr_data(mfrr_path1, mfrr_path2)
    # if not mfrr_combined_df.empty:
    #     output_path = 'data/mfrr_reserves_dk2_combined.csv'
    #     try:
    #         mfrr_combined_df.to_csv(output_path, index=False)
    #         print(f"Combined mFRR data saved to {output_path}")
    #     except Exception as e:
    #         print(f"Error saving combined mFRR data: {e}")
    # --------------------------------------------

    # --- Process aFRR Data for X (Assuming no wind dir needed here yet) ---
    # print("\n--- Processing aFRR Data for X ---")
    # afrr, temp, wind, timestamps = load_and_process_data()
    # if timestamps.size > 0:
    #     if len(timestamps) % 24 == 0:
    #         # X = create_sequences(afrr, temp, wind, ???, timestamps)
    #         # np.save("data/X.npy", X)
    #         # print(f"\nSaved {len(X)} sequences to data/X.npy. Shape: {X.shape}")
    #         pass
    #     else:
    #         print(f"Warning: aFRR data length ({len(timestamps)}) not divisible by 24. Skipping X creation.")
    # else:
    #     print("aFRR processing resulted in empty data. Skipping X creation.")
    # ----------------------------------------------------------------------

    # ---- Process Imbalance Data for X2 ----
    print("\n--- Processing Imbalance Data for X2 ---")
    imbalance, imb_temp, imb_wind_speed, imb_wind_dir_sin, imb_wind_dir_cos, imb_timestamps = load_and_process_imbalance_data()

    if imb_timestamps.size == 0:
        print("Imbalance processing resulted in empty data. Cannot create sequences for X2.")
    else:
        if len(imb_timestamps) % 24 != 0:
             print(f"Warning: Imbalance data length ({len(imb_timestamps)}) not divisible by 24 after processing. Cannot create sequences correctly.")
        else:
            print(f"Processed {len(imb_timestamps)} hours of imbalance data.")
            X2 = create_sequences(imbalance, imb_temp, imb_wind_speed, imb_wind_dir_sin, imb_wind_dir_cos, imb_timestamps)

            if X2.size == 0:
                print("Warning: create_sequences returned an empty array for X2.")
            else:
                print("\nTail of Imbalance feature (X2[:, 0, :]) before GW conversion:")
                print(X2[-3:, 0, :]) # Print imbalance from the last 3 sequences

                X2[:, 0, :] /= 1000.0 # Convert imbalance feature from MW to GW

                if np.isnan(X2).any():
                    print("\nWarning: NaNs found in final X2 sequences. Check input data and gap filling.")
                else:
                    np.save("data/X2.npy", X2)
                    print(f"\nSaved {len(X2)} sequences to data/X2.npy. Shape: {X2.shape}") # Shape should be (num_sequences, 9, 24)
