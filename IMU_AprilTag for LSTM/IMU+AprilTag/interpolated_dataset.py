import pandas as pd

# Load original CSV
df = pd.read_csv("cleaned_data.csv")

# Convert numeric columns
for col in ["pose_x", "pose_y", "pose_z", "rel_x", "rel_y", "rel_z"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert timestamp_ms to datetime and set as index
df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit='ms')
df.set_index("datetime", inplace=True)

# Interpolate AprilTag data
df_interp = df.copy()
df_interp[["pose_x", "pose_y", "pose_z", "rel_x", "rel_y", "rel_z"]] = df_interp[[
    "pose_x", "pose_y", "pose_z", "rel_x", "rel_y", "rel_z"
]].interpolate(method="time", limit_direction="both")

# Drop rows where rel_x/rel_y/rel_z are still NaN
df_interp.dropna(subset=["rel_x", "rel_y", "rel_z"], inplace=True)

# Optional: Reset index and keep timestamp_ms
df_interp.reset_index(drop=True, inplace=True)

# Save
df_interp.to_csv("interpolated_dataset.csv", index=False)
print("Interpolated dataset saved as interpolated_dataset.csv")
