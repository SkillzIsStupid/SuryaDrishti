import pandas as pd
import requests
import time
from datetime import datetime

# Haryana district lat/lon
districts = {
    "Gurugram": (28.45, 77.02),
    "Faridabad": (28.41, 77.31),
    "Rewari": (28.20, 76.61),
    "Mahendragarh": (28.28, 76.15),
    "Bhiwani": (28.79, 76.13),
    "Hisar": (29.15, 75.72),
    "Sirsa": (29.53, 75.03),
    "Fatehabad": (29.52, 75.45),
    "Rohtak": (28.89, 76.59),
    "Jhajjar": (28.61, 76.65),
    "Karnal": (29.69, 76.99),
    "Panipat": (29.39, 76.97),
    "Ambala": (30.38, 76.78),
    "Yamunanagar": (30.12, 77.28),
    "Kurukshetra": (29.97, 76.85),
    "Palwal": (28.15, 77.33),
}

def fetch_nasa(lat, lon):
    """Fetch hourly NASA POWER data for given lat/lon."""
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point"
        f"?parameters=ALLSKY_SFC_SW_DWN,T2M,RH2M,WS2M,CLOUD_AMT,PRECTOTCORR"
        f"&community=RE&longitude={lon}&latitude={lat}"
        "&start=20241001&end=20241107&format=JSON"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        props = r.json().get("properties", {})
        params = props.get("parameter", {})
        if not params:
            return None

        # Flatten hourly keys like "2024110105" into datetime
        records = []
        for i, key in enumerate(params["ALLSKY_SFC_SW_DWN"].keys()):
            try:
                t = datetime.strptime(key, "%Y%m%d%H")
                row = {
                    "time": t,
                    "ghi": params["ALLSKY_SFC_SW_DWN"][key],
                    "temp": params["T2M"][key],
                    "humidity": params["RH2M"][key],
                    "wind": params["WS2M"][key],
                    "cloud": params["CLOUD_AMT"][key],
                    "rain": params["PRECTOTCORR"][key],
                }
                records.append(row)
            except Exception:
                continue

        df = pd.DataFrame(records)
        df["lat"] = lat
        df["lon"] = lon
        return df if not df.empty else None
    except Exception as e:
        print(f"[WARN] Failed for {lat},{lon}: {e}")
        return None

full_df = []
for district, (lat, lon) in districts.items():
    print(f"🔹 Fetching NASA hourly data for {district} ({lat},{lon}) ...")
    df = fetch_nasa(lat, lon)
    if df is None:
        print(f"⚠️  Skipped {district}, no data returned.")
        continue
    df["district"] = district
    full_df.append(df)
    time.sleep(1)

if not full_df:
    print("❌ No data fetched — check API response or network.")
else:
    final = pd.concat(full_df, ignore_index=True)
    out_path = "data/haryana_nasa_hourly.csv"
    final.to_csv(out_path, index=False)
    print(f"✅ Saved {len(final)} hourly records -> {out_path}")
