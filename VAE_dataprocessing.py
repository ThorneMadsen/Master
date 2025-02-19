import numpy as np
import pandas as pd

def load_and_trim_data(df, dmi_df, imbalance_df, seq_length=24):
    """
    Loads and trims the datasets to the same length.
    """
    numerical_columns = df.columns[2:]
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col].replace('n/e', 0), errors='coerce').fillna(0)

    # Sum all numerical columns
    df['Total Generation [MW]'] = df[numerical_columns].sum(axis=1)

    # Convert 'n/e' to 0 in the imbalance data and make it numeric
    imbalance_df['Total Imbalance [MWh] - IBA|DK1'] = pd.to_numeric(
        imbalance_df['Total Imbalance [MWh] - IBA|DK1'].replace('n/e', 0), errors='coerce'
    ).fillna(0)

    # Adjust sign based on "Deficit" or "Surplus"
    imbalance_df['Total Imbalance [MWh] - IBA|DK1'] = imbalance_df.apply(
        lambda row: -row['Total Imbalance [MWh] - IBA|DK1'] if row['Situation'] == 'Deficit' else row['Total Imbalance [MWh] - IBA|DK1'],
        axis=1
    )

    for i in range(len(df)):
        df.loc[i, 'Total Generation [MW]'] += imbalance_df.loc[i, 'Total Imbalance [MWh] - IBA|DK1']
    
    # Filter only the data from 2025 onwards
    dmi_df['Datetime'] = pd.to_datetime(dmi_df['Datetime'])
    dmi_df = dmi_df[dmi_df['Datetime'] >= "2025-01-01"]

    total_generation = df['Total Generation [MW]'].values
    mean_temp = dmi_df['mean_temp'].values
    mean_wind_speed = dmi_df['mean_wind_speed'].values
    imbalance = imbalance_df['Total Imbalance [MWh] - IBA|DK1'].values
    
    min_length = min(len(total_generation), len(mean_temp), len(mean_wind_speed), len(imbalance))
    
    return (
        total_generation[:min_length],
        mean_temp[:min_length],
        mean_wind_speed[:min_length],
        imbalance[:min_length]
    )

def create_sequences(total_generation, mean_temp, mean_wind_speed, imbalance, seq_length=24):
    """
    Creates 24-hour sequences for input (X) and target (Y).
    """
    X, Y = [], []
    
    for i in range(len(total_generation) - seq_length - 1):
        X_seq = np.stack([
            total_generation[i:i+seq_length],
            mean_temp[i:i+seq_length],
            mean_wind_speed[i:i+seq_length]
        ], axis=0)  # Shape (3, 24)
        
        Y_seq = imbalance[i+1:i+1+seq_length]  # Shape (24,)
        
        X.append(X_seq)
        Y.append(Y_seq)
    
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("data/Actual Generation per Production Type_DK1.csv")
    dmi_df = pd.read_csv("data/DMI_data.csv")
    imbalance_df = pd.read_csv("data/Imbalance_year_DK1.csv")
    
    total_generation, mean_temp, mean_wind_speed, imbalance = load_and_trim_data(df, dmi_df, imbalance_df)
    X, Y = create_sequences(total_generation, mean_temp, mean_wind_speed, imbalance)

    for i in range(3):  # Normalize each feature separately
        X[:, i, :] = (X[:, i, :] - np.mean(X[:, i, :])) / np.std(X[:, i, :])

    # Normalize Y separately
    Y = (Y - np.mean(Y)) / np.std(Y)

    print("After Normalization:")
    print("X[0] mean (Total Generation):", np.mean(X[0]))
    print("X[1] mean (Mean Temp):", np.mean(X[1]))
    print("X[2] mean (Mean Wind Speed):", np.mean(X[2]))
    print("Y mean:", np.mean(Y))
    print("X min:", np.min(X), " | X max:", np.max(X))
    print("Y min:", np.min(Y), " | Y max:", np.max(Y))
    
    # Save processed data for training
    np.save("data/X.npy", X)
    np.save("data/Y.npy", Y)
    
    print("Processed data saved: X.npy (", X.shape, ") Y.npy (", Y.shape, ")")
