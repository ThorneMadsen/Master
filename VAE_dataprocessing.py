import numpy as np
import pandas as pd

def load_and_trim_data(df, dmi_df, imbalance_df):
    """
    Loads and trims the datasets to ensure consistent lengths.
    Converts total generation from MW to GW and imbalance from MWh to GWh.
    """
    numerical_columns = df.columns[2:]
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col].replace('n/e', 0), errors='coerce').fillna(0)

    # Sum all numerical columns to get total generation in MW
    df['Total Generation [MW]'] = df[numerical_columns].sum(axis=1)
    
    # Convert production from MW to GW (1 MW = 0.001 GW)
    df['Total Generation [GW]'] = df['Total Generation [MW]'] * 0.001

    # Process imbalance data: convert 'n/e' to 0 and make numeric
    imbalance_df['Total Imbalance [MWh] - IBA|DK1'] = pd.to_numeric(
        imbalance_df['Total Imbalance [MWh] - IBA|DK1'].replace('n/e', 0), errors='coerce'
    ).fillna(0)
    
    # Adjust sign based on "Deficit" or "Surplus"
    imbalance_df['Total Imbalance [MWh] - IBA|DK1'] = imbalance_df.apply(
        lambda row: -row['Total Imbalance [MWh] - IBA|DK1'] if row['Situation'] == 'Deficit' else row['Total Imbalance [MWh] - IBA|DK1'],
        axis=1
    )
    
    # Convert imbalance from MWh to GWh (1 MWh = 0.001 GWh)
    imbalance_df['Total Imbalance [GWh] - IBA|DK1'] = imbalance_df['Total Imbalance [MWh] - IBA|DK1'] * 0.001

    # Filter only the data from 2025 onwards
    dmi_df['Datetime'] = pd.to_datetime(dmi_df['Datetime'])
    dmi_df = dmi_df[dmi_df['Datetime'] >= "2025-01-01"]

    total_generation = df['Total Generation [GW]'].values  # Now in GW
    mean_temp = dmi_df['mean_temp'].values
    mean_wind_speed = dmi_df['mean_wind_speed'].values
    imbalance = imbalance_df['Total Imbalance [GWh] - IBA|DK1'].values  # Now in GWh
    
    min_length = min(len(total_generation), len(mean_temp), len(mean_wind_speed), len(imbalance))
    
    return (
        total_generation[:min_length],
        mean_temp[:min_length],
        mean_wind_speed[:min_length],
        imbalance[:min_length]
    )

def create_sequences(total_generation, mean_temp, mean_wind_speed, imbalance, seq_length=24):
    """
    Creates sequences where X contains seq_length hours of:
      - Total Generation (in GW)
      - Mean Temperature
      - Mean Wind Speed
      - Imbalance (in GWh)
      
    Returns X with shape (n_days, 4, seq_length).
    """
    X = []
    # Step through the data in increments of seq_length (one day)
    for i in range(0, len(total_generation) - seq_length + 1, seq_length):
        X_seq = np.stack([
            total_generation[i:i+seq_length],   # Production in GW
            mean_temp[i:i+seq_length],          # Temperature
            mean_wind_speed[i:i+seq_length],      # Wind Speed
            imbalance[i:i+seq_length]             # Imbalance in GWh
        ], axis=0)  # Combined shape: (4, 24)
        X.append(X_seq)
    return np.array(X)

if __name__ == "__main__":
    df = pd.read_csv("data/Actual Generation per Production Type_DK1.csv")
    dmi_df = pd.read_csv("data/DMI_data.csv")
    imbalance_df = pd.read_csv("data/Imbalance_year_DK1.csv")
    
    total_generation, mean_temp, mean_wind_speed, imbalance = load_and_trim_data(df, dmi_df, imbalance_df)
    X = create_sequences(total_generation, mean_temp, mean_wind_speed, imbalance, seq_length=24)
    
    np.save("data/X.npy", X)
    
    print("Processed data saved: X.npy with shape", X.shape)
