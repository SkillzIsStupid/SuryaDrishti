import os
import requests
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from urllib.parse import quote

# ===============================
# 1️⃣ Geocode user input (OSM)
# ===============================

def geocode_location(name: str):
    """Convert user text input into (lat, lon) coordinates via OpenStreetMap."""
    url = f"https://nominatim.openstreetmap.org/search?q={quote(name + ', Haryana, India')}&format=json&limit=1"
    headers = {"User-Agent": "SuryaDrishti-Geocoder"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError(f"Location '{name}' not found.")
    return float(data[0]["lat"]), float(data[0]["lon"])


# ===============================
# 2️⃣ Fetch live solar & weather data (Open-Meteo)
# ===============================

def fetch_openmeteo(lat, lon):
    """Fetch live 1-km irradiance & meteorological data from Open-Meteo."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&hourly=shortwave_radiation,direct_radiation,diffuse_radiation,"
        "temperature_2m,relative_humidity_2m,cloud_cover"
        "&timezone=auto"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()["hourly"]

    # Select the latest available hour
    i = -1
    return {
        "ghi": data["shortwave_radiation"][i],
        "direct": data["direct_radiation"][i],
        "diffuse": data["diffuse_radiation"][i],
        "temp": data["temperature_2m"][i],
        "humidity": data["relative_humidity_2m"][i],
        "cloud": data["cloud_cover"][i],
    }


# ===============================
# 3️⃣ Load trained model + normalization
# ===============================

model_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
norm_path = os.path.join(model_dir, "norm.npy")
model_path = os.path.join(model_dir, "lstm_model_fused.pth")

norm_data = np.load(norm_path, allow_pickle=True).item()
min_vals = np.array(norm_data["min_vals"])
max_vals = np.array(norm_data["max_vals"])
columns = norm_data["columns"]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, out_size=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, out_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

input_size = len(columns)
model = LSTMModel(input_size=input_size)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()


# ===============================
# 4️⃣ Load past Haryana dataset for sequence context
# ===============================

import pandas as pd

data_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "processed", "haryana_processed.csv")
)
df = pd.read_csv(data_path)
numeric_df = df.select_dtypes(include=[np.number])
features = numeric_df.values.astype(float)
features = (features - min_vals) / (max_vals - min_vals + 1e-9)
SEQ_LEN = min(24, len(features) - 1)


# ===============================
# 5️⃣ Prediction function for any user-input location
# ===============================

def predict_irradiance(location: str):
    """Main callable function for frontend integration."""
    try:
        lat, lon = geocode_location(location)
        print(f"📍 {location}: lat={lat:.4f}, lon={lon:.4f}")

        live = fetch_openmeteo(lat, lon)
        live_vector = np.array([
            live["ghi"], live["direct"], live["diffuse"],
            live["temp"], live["humidity"], live["cloud"]
        ])
        live_vector_norm = (live_vector - min_vals[:6]) / (max_vals[:6] - min_vals[:6] + 1e-9)

        # Prepare LSTM input
        X = torch.tensor(features[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
        X[0, -1, :len(live_vector_norm)] = torch.tensor(live_vector_norm, dtype=torch.float32)

        # Run prediction
        with torch.no_grad():
            pred = model(X).item()

        pred_ghi = pred * (max_vals[0] - min_vals[0]) + min_vals[0]

        return {
            "location": location,
            "lat": lat,
            "lon": lon,
            "predicted_GHI": round(pred_ghi, 2),
            "live_data": live,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        return {"error": str(e)}


# ===============================
# 6️⃣ Run interactively or via API call
# ===============================

if __name__ == "__main__":
    user_input = input("Enter a Haryana location (e.g., 'Sector 31 Gurugram', 'Panipat City', 'Hisar Sector 16'): ")
    result = predict_irradiance(user_input)

    if "error" in result:
        print("❌ Error:", result["error"])
    else:
        live = result["live_data"]
        print("\n=== ☀️  Solar Irradiance Forecast ===")
        print(f"📍 Location: {result['location']}")
        print(f"🌐 Coordinates: {result['lat']:.4f}, {result['lon']:.4f}")
        print(f"⏰ Timestamp: {result['timestamp']}")
        print(f"🔆 Predicted Next-Hour GHI: {result['predicted_GHI']} W/m²")
        print("\n--- Live Conditions ---")
        print(f"Current GHI: {live['ghi']} W/m²")
        print(f"Direct Radiation: {live['direct']} W/m²")
        print(f"Diffuse Radiation: {live['diffuse']} W/m²")
        print(f"Temperature: {live['temp']} °C")
        print(f"Humidity: {live['humidity']} %")
        print(f"Cloud Cover: {live['cloud']} %")
