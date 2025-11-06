import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ==== 1️⃣ Load the processed Haryana dataset ====
data_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "haryana_processed.csv"))
print("📂 Loading dataset:", data_path)

df = pd.read_csv(data_path)
print("✅ Columns:", df.columns.tolist())

# Use only the numeric features
features = df[["ghi", "temp", "humidity"]].values.astype(float)

# Normalize each column
min_vals = features.min(axis=0)
max_vals = features.max(axis=0)
features = (features - min_vals) / (max_vals - min_vals + 1e-9)

# ==== 2️⃣ Create sequences for LSTM ====
SEQ_LEN = 24  # using 1 day of hourly data (if daily, it’s 24 days)
X, Y = [], []
for i in range(len(features) - SEQ_LEN):
    X.append(features[i:i+SEQ_LEN])
    Y.append(features[i+SEQ_LEN, 0])  # predict next ghi

X = np.array(X)
Y = np.array(Y)

print(f"✅ Created dataset with {len(X)} sequences, shape: X={X.shape}, Y={Y.shape}")

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# ==== 3️⃣ Define the LSTM model ====
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, out_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==== 4️⃣ Train the model ====
EPOCHS = 25
for epoch in range(EPOCHS):
    total_loss = 0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"📉 Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(dataloader):.6f}")

# ==== 5️⃣ Save model and normalization info ====
model_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "lstm_model.pth")
torch.save(model.state_dict(), model_path)

# Save normalization data so prediction can reverse scale
norm_path = os.path.join(model_dir, "norm.npy")
np.save(norm_path, {"min_vals": min_vals, "max_vals": max_vals})

print(f"✅ Model saved to: {model_path}")
print(f"✅ Normalization saved to: {norm_path}")
print("🎉 Training complete!")
