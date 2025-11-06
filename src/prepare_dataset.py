import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from fetch_himawari import fetch_recent_hours

print("📂 Loading Haryana NASA dataset...")

# === 1️⃣ Load NASA dataset ===
nasa_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "haryana_nasa_daily.csv"))
nasa_df = pd.read_csv(nasa_path)

# Validate columns
for col in ["ghi", "temp", "humidity"]:
    if col not in nasa_df.columns:
        raise ValueError(f"❌ Missing column '{col}' in NASA dataset")

# Rename and standardize columns
nasa_df = nasa_df[["ghi", "temp", "humidity"]].rename(columns={
    "ghi": "solar_irradiance",
    "temp": "temperature",
    "humidity": "humidity"
})
nasa_df["date"] = pd.date_range(start="2015-01-01", periods=len(nasa_df), freq="D")

# === 2️⃣ Fetch recent Himawari satellite data ===
lat, lon = 28.45, 77.02  # Gurugram center
print("🛰️ Fetching Himawari satellite data (past 12 hours)...")
try:
    hima_records = fetch_recent_hours(lat, lon, hours=12)
    him_df = pd.DataFrame(hima_records)
    him_df["time"] = pd.to_datetime(him_df["time"])
    # Average satellite reflectance and cloud-top temp
    reflectance_mean = him_df["B03"].mean(skipna=True)
    bt_cloudtop_mean = him_df["B13"].mean(skipna=True)
except Exception as e:
    print(f"[WARN] Himawari fetch failed: {e}")
    reflectance_mean = np.nan
    bt_cloudtop_mean = np.nan

# === 3️⃣ Add derived features ===
merged_df = nasa_df.copy()
merged_df["direct"] = merged_df["solar_irradiance"] * 0.7
merged_df["diffuse"] = merged_df["solar_irradiance"] * 0.3
merged_df["cloud"] = np.clip(100 - (merged_df["solar_irradiance"] / merged_df["solar_irradiance"].max() * 100), 0, 100)

# Add averaged Himawari features
merged_df["reflectance"] = reflectance_mean
merged_df["bt_cloudtop"] = bt_cloudtop_mean

print("✅ Fusion complete. Columns:", merged_df.columns.tolist())

# === 4️⃣ Normalize and prepare sequences ===
scaler = MinMaxScaler()
scaled = scaler.fit_transform(merged_df[["solar_irradiance", "direct", "diffuse", "temperature", "humidity", "cloud"]])

seq_length = 7
X, y = [], []
for i in range(len(scaled) - seq_length):
    X.append(scaled[i:i + seq_length])
    y.append(scaled[i + seq_length][0])

X, y = np.array(X), np.array(y)

# === 5️⃣ Save processed outputs ===
processed_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
os.makedirs(processed_dir, exist_ok=True)

merged_df.to_csv(os.path.join(processed_dir, "haryana_processed.csv"), index=False)
np.save(os.path.join(processed_dir, "X.npy"), X)
np.save(os.path.join(processed_dir, "y.npy"), y)

print("\n✅ Haryana fused dataset ready!")
print(f"📁 Saved to: {processed_dir}/haryana_processed.csv")
print(f"X shape: {X.shape}, Y shape: {y.shape}")
