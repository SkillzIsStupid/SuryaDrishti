# 🌞 SuryaDrishti — Real-Time Solar Irradiance Prediction for Haryana

**SuryaDrishti** is an AI-powered solar irradiance forecasting system that fuses live satellite imagery (Himawari 8/9 via JAXA / AWS) with NASA POWER meteorological data to predict near-real-time solar irradiance across Haryana, India — down to the sector level.

---

## 🚀 Features
- 🌍 **Multivariate Dataset Fusion**  
  Combines NASA POWER hourly weather data with satellite-derived reflectance and cloud-top temperature from Himawari 8/9 and JAXA P-Tree.

- 🤖 **Deep Learning (LSTM / BiLSTM)**  
  Predicts solar irradiance (GHI) using temporal sequences of meteorological and satellite data.

- 🛰️ **Real-Time Integration**  
  Fetches current conditions via APIs and FTP, supports fallback to NASA POWER if satellite data is unavailable.

- 📊 **Fully Automated Pipeline**
  - `fetch_himawari.py` → Fetch satellite data  
  - `fetch_nasa_haryana.py` → Download NASA POWER dataset  
  - `prepare_fused_dataset.py` → Fuse and normalize data  
  - `train_lstm.py` / `train_bilstm.py` → Train models  
  - `predict.py` → Generate live irradiance predictions

---

## ⚙️ Setup

1. **Clone this repo:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/SuryaDrishti.git
   cd SuryaDrishti
