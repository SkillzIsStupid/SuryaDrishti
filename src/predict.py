import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# ==== 1️⃣ Load processed Haryana dataset ====
data_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "haryana_processed.csv"))
print("📂 Loading dataset from:", data_path)

df = pd.read_csv(data_path)
features = df[["ghi", "temp", "humidity"]].values.astype(float)

# ==== 2️⃣ Load normalization info from training ====
norm_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models", "norm.npy"))
norm_data = np.load(norm_path, allow_pickle=True).item()
min_vals, max_vals = norm_data["min_vals"], norm_data["max_vals"]

features = (features - min_vals) / (max_vals - min_vals + 1e-9)

SEQ_LEN = min(24, len(features) - 1)
X = torch.tensor(features[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)

# ==== 3️⃣ Define same model architecture as training ====
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, out_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()

# ==== 4️⃣ Load trained weights ====
model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models", "lstm_model.pth"))
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# ==== 5️⃣ Predict next solar value ====
with torch.no_grad():
    pred = model(X).item()

# Convert back to original scale
pred = pred * (max_vals[0] - min_vals[0]) + min_vals[0]

print(f"☀️  Predicted next solar irradiance (GHI): {pred:.2f}")
