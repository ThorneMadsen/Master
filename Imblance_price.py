import pandas as pd

# --- 1. Load the CSV files manually ---
imbalance_df_2023 = pd.read_csv("data/Imbalance_DK2_2023.csv")
imbalance_df_2024 = pd.read_csv("data/Imbalance_DK2_2024.csv")
imbalance_df_2025 = pd.read_csv("data/Imbalance_DK2_2025.csv")

imbalance_price_df_2023 = pd.read_csv("data/Imbalance Prices_DK2_2023.csv")
imbalance_price_df_2024 = pd.read_csv("data/Imbalance Prices_DK2_2024.csv")
imbalance_price_df_2025 = pd.read_csv("data/Imbalance Prices_DK2_2025.csv")

# Concatenate the yearly dataframes for volumes and prices
df_vol = pd.concat([imbalance_df_2023, imbalance_df_2024, imbalance_df_2025], ignore_index=True)
df_price = pd.concat([imbalance_price_df_2023, imbalance_price_df_2024, imbalance_price_df_2025], ignore_index=True)

# --- 2. Standardize the settlement period column and extract start datetime ---
df_vol = df_vol.rename(columns={"Imbalance settlement period CET/CEST": "settlement_period"})
df_price = df_price.rename(columns={"Imbalance settlement period (CET/CEST)": "settlement_period"})

def extract_start_datetime(period_str):
    # Assumes format "dd.mm.yyyy HH:MM - dd.mm.yyyy HH:MM"
    start_str = period_str.split(" - ")[0]
    return pd.to_datetime(start_str, format="%d.%m.%Y %H:%M")

df_vol['start_datetime'] = df_vol['settlement_period'].apply(extract_start_datetime)
df_price['start_datetime'] = df_price['settlement_period'].apply(extract_start_datetime)

# --- 3. Filter out any 2025 rows after January 5th ---
cutoff = pd.to_datetime("05.01.2025 23:59", format="%d.%m.%Y %H:%M")
df_vol = df_vol[~((df_vol['start_datetime'].dt.year == 2025) & (df_vol['start_datetime'] > cutoff))]
df_price = df_price[~((df_price['start_datetime'].dt.year == 2025) & (df_price['start_datetime'] > cutoff))]

# --- Debug: Print entries for 2024-11-17 before merging ---
target_date = pd.to_datetime("2024-11-17").date()

print("Volume entries for 2024-11-17:")
print(df_vol[df_vol['start_datetime'].dt.date == target_date][['settlement_period', 'start_datetime', "Total Imbalance [MWh] - IBA|DK2"]])

print("\nPrice entries for 2024-11-17:")
print(df_price[df_price['start_datetime'].dt.date == target_date][['settlement_period', 'start_datetime']])

# --- 4. Merge the volume and price data and calculate the cost ---
df_vol["Total Imbalance [MWh] - IBA|DK2"] = pd.to_numeric(df_vol["Total Imbalance [MWh] - IBA|DK2"], errors="coerce")

# Merge data on settlement_period.
merged = pd.merge(df_vol, df_price, on="settlement_period", suffixes=('_vol', '_price'))

# --- Debug: Print merged entries for 2024-11-17 ---
print("\nMerged entries for 2024-11-17:")
merged_17 = merged[merged['start_datetime_vol'].dt.date == target_date]
print(merged_17[['settlement_period', 'start_datetime_vol', "Total Imbalance [MWh] - IBA|DK2", 
                   "+ Imbalance Price [EUR/MWh] - IPA|DK2", "- Imbalance Price [EUR/MWh] - IPA|DK2"]])

merged["+ Imbalance Price [EUR/MWh] - IPA|DK2"] = pd.to_numeric(merged["+ Imbalance Price [EUR/MWh] - IPA|DK2"], errors="coerce")
merged["- Imbalance Price [EUR/MWh] - IPA|DK2"] = pd.to_numeric(merged["- Imbalance Price [EUR/MWh] - IPA|DK2"], errors="coerce")

merged['imbalance_price'] = merged["+ Imbalance Price [EUR/MWh] - IPA|DK2"].combine_first(merged["- Imbalance Price [EUR/MWh] - IPA|DK2"])
# Replace any missing price values with 0
merged['imbalance_price'] = merged['imbalance_price'].fillna(0)

merged['imbalance_cost'] = merged["Total Imbalance [MWh] - IBA|DK2"] * merged['imbalance_price']

# --- 5. Sum up total imbalance volume and cost ---
total_volume = merged["Total Imbalance [MWh] - IBA|DK2"].sum()
total_cost = merged['imbalance_cost'].sum()

print("\nOverall Totals:")
print("Total Imbalance Volume (MWh):", total_volume)
print("Total Imbalance Cost (EUR):", total_cost)

# --- 6. Daily Aggregation ---
# Use the volume start_datetime column for the date
merged['date'] = merged['start_datetime_vol'].dt.date

daily_agg = merged.groupby('date').agg({
    "Total Imbalance [MWh] - IBA|DK2": "sum",
    "imbalance_cost": "sum"
}).reset_index()

# --- Debug: Print daily aggregation for 2024-11-17 ---
daily_agg_17 = daily_agg[daily_agg['date'] == target_date]
print("\nDaily aggregated values for 2024-11-17:")
print(daily_agg_17)

# --- 7. Find the day with the highest total imbalance and its cost ---
idx_max = daily_agg["Total Imbalance [MWh] - IBA|DK2"].abs().idxmax()
max_day = daily_agg.loc[idx_max]

print("\nDay with highest total imbalance:")
print("Date:", max_day['date'])
print("Total Imbalance Volume (MWh):", max_day["Total Imbalance [MWh] - IBA|DK2"])
print("Total Imbalance Cost (EUR):", max_day["imbalance_cost"])
print(f"\nOn {max_day['date']}, the total imbalance was {max_day['Total Imbalance [MWh] - IBA|DK2']} MWh, costing {max_day['imbalance_cost']} EUR.")
