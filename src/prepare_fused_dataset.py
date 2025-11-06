import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, UTC
from fetch_himawari import fetch_himawari_point

# =========================================================
# 1️⃣ Load latest NASA POWER hourly Haryana dataset
# =========================================================
print("📂 Loading Haryana NASA dataset...")

# Use the hourly file if it exists, otherwise fall back to daily
if os.path.exists("data/haryana_nasa_hourly.csv"):
    df = pd.read_csv("data/haryana_nasa_hourly.csv")
    print("✅ Using hourly NASA POWER dataset.")
else:
    df = pd.read_csv("data/haryana_nasa_daily.csv")
    print("⚠️ Using daily NASA POWER dataset (hourly file not found).")

# Select and rename key columns
expected_cols = ["ghi", "temp", "humidity", "wind", "cloud", "rain"]
present_cols = [c for c in expected_cols if c in df.columns]
df = df[present_cols].rename(
    columns={
        "ghi": "solar_irradiance",
        "temp": "temperature",
        "humidity": "humidity",
        "wind": "wind_speed",
        "cloud": "cloud_amount",
        "rain": "precipitation",
    }
)

print("✅ Columns selected & renamed:", df.columns.tolist())

# =========================================================
# 2️⃣ Fetch latest Himawari features
# =========================================================
lat, lon = 28.45, 77.02  # Gurugram for reference

print("🛰️ Fetching latest Himawari data for Gurugram...")
try:
    hima = fetch_himawari_point(lat, lon, datetime.now(UTC))
    hima_df = pd.DataFrame(
        [{
            "time": hima.get("time"),
            "reflectance": hima.get("B03", np.nan),
            "bt_cloudtop": hima.get("B13", np.nan),
            "source": hima.get("source", "")
        }]
    )
    print(f"✅ Himawari data fetched from source: {hima.get('source', 'unknown')}")
except Exception as e:
    print(f"[WARN] Himawari fetch failed: {e}")
    hima_df = pd.DataFrame(
        [{"time": None, "reflectance": np.nan, "bt_cloudtop": np.nan, "source": "none"}]
    )

# =========================================================
# 3️⃣ Merge / fuse datasets
# =========================================================
# Use mean of Himawari values as a placeholder for fusion
ref_val = hima_df["reflectance"].mean(skipna=True)
bt_val = hima_df["bt_cloudtop"].mean(skipna=True)

# If those values are still NaN, use a neutral placeholder (0.5 normalized space)
if pd.isna(ref_val):
    ref_val = 0.5
if pd.isna(bt_val):
    bt_val = 0.5

df["reflectance"] = ref_val
df["bt_cloudtop"] = bt_val


print("✅ Columns after fusion:", df.columns.tolist())

# =========================================================
# 4️⃣ Normalize and build sequences
# =========================================================
# Replace all-NaN or constant columns with mean-safe values
numeric_df = df.select_dtypes(include=[np.number])
numeric_df = numeric_df.fillna(numeric_df.mean(numeric_only=True))

# Drop any columns with zero variance (constant)
variance = numeric_df.std(numeric_only=True)
numeric_df = numeric_df.loc[:, variance > 1e-6]

# Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(numeric_df)

print("✅ Columns retained for scaling:", numeric_df.columns.tolist())


seq_length = 24  # last 24 hours (or 24 days if daily)
X, y = [], []
for i in range(len(scaled) - seq_length):
    X.append(scaled[i : i + seq_length])
    y.append(scaled[i + seq_length][0])  # predict next irradiance

X, y = np.array(X), np.array(y)

# =========================================================
# 5️⃣ Save processed data
# =========================================================
os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/X.npy", X)
np.save("data/processed/y.npy", y)

# Save normalization info for prediction scaling
norm_info = {
    "min_vals": scaler.data_min_,
    "max_vals": scaler.data_max_,
    "columns": df.columns.tolist(),
}
np.save("models/norm.npy", norm_info, allow_pickle=True)

print("✅ Fused dataset ready! Saved to data/processed/")
print("X shape:", X.shape, "Y shape:", y.shape)
