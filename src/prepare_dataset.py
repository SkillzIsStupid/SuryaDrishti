import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

print("📂 Loading Haryana NASA dataset...")

df = pd.read_csv("data/haryana_nasa_daily.csv")

# Select relevant columns
df = df[["ghi", "temp", "humidity"]]

df = df.rename(columns={
    "ghi": "solar_irradiance",
    "temp": "temperature",
})

print("✅ Columns selected & renamed:", df.columns)

# Normalization
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

seq_length = 7  # last 7 days input

X, y = [], []
for i in range(len(scaled) - seq_length):
    X.append(scaled[i:i+seq_length])
    y.append(scaled[i+seq_length][0])  # predict irradiance only

X, y = np.array(X), np.array(y)

os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/X.npy", X)
np.save("data/processed/y.npy", y)

print("✅ Dataset ready! Saved to data/processed/")
print("X shape:", X.shape, "Y shape:", y.shape)
