import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ================================
# 1️⃣ Load the processed dataset
# ================================
data_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "processed", "haryana_processed.csv")
)
print(f"📂 Loading dataset from: {data_path}")

df = pd.read_csv(data_path)

# Expected columns for fused 6-feature model
expected_cols = ["ghi", "direct", "diffuse", "temp", "humidity", "cloud"]

# Ensure all required columns exist
for col in expected_cols:
    if col not in df.columns:
        print(f"[WARN] Column '{col}' missing — creating placeholder zeros.")
        df[col] = 0.0

df = df[expected_cols]
print(f"✅ Using columns: {list(df.columns)}")

# ================================
# 2️⃣ Normalize all features
# ================================
min_vals = df.min(axis=0).values
max_vals = df.max(axis=0).values
scaled = (df - min_vals) / (max_vals - min_vals + 1e-9)
print("📊 Normalization complete.")

# ================================
# 3️⃣ Create sequences for LSTM
# ================================
SEQ_LEN = 24  # previous 24 samples (hours/days)
X, Y = [], []
values = scaled.values

for i in range(len(values) - SEQ_LEN):
    X.append(values[i:i + SEQ_LEN])
    Y.append(values[i + SEQ_LEN, 0])  # predict next-step GHI

X = np.array(X)
Y = np.array(Y)
print(f"✅ Created dataset with {len(X)} sequences (X shape: {X.shape}, Y shape: {Y.shape})")

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# ================================
# 4️⃣ Define the LSTM model
# ================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, out_size=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

input_size = len(expected_cols)
model = LSTMModel(input_size=input_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ================================
# 5️⃣ Train the model
# ================================
EPOCHS = 25
print("\n🚀 Starting LSTM training...\n")

for epoch in range(EPOCHS):
    total_loss = 0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"📉 Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.6f}")

print("\n🎯 Training complete.\n")

# ================================
# 6️⃣ Save model & normalization info
# ================================
model_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "lstm_model_fused.pth")
torch.save(model.state_dict(), model_path)

norm_data = {
    "min_vals": min_vals,
    "max_vals": max_vals,
    "columns": expected_cols
}
norm_path = os.path.join(model_dir, "norm.npy")
np.save(norm_path, norm_data)

print(f"✅ Model saved to: {model_path}")
print(f"✅ Normalization info saved to: {norm_path}")
print(f"🧠 Trained on features: {expected_cols}")
