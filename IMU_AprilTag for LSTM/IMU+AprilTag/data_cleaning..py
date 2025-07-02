import pandas as pd
import numpy as np

# === Load CSV ===
df = pd.read_csv("april_imu_log.csv", na_values="null")

# === Drop rows where rel_x/y/z is missing ===
df_clean = df.dropna(subset=["rel_x", "rel_y", "rel_z"])

# === Optional: Reset index ===
df_clean.reset_index(drop=True, inplace=True)

# === Convert data types (if not already float) ===
float_cols = [
    "pose_x", "pose_y", "pose_z",
    "rel_x", "rel_y", "rel_z",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z"
]
df_clean[float_cols] = df_clean[float_cols].astype(float)

# === Save to cleaned CSV for LSTM ===
df_clean.to_csv("cleaned_data.csv", index=False)

print("Cleaned data saved to 'cleaned_data.csv'")
print(df_clean.head())
