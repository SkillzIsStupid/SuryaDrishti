import pandas as pd
import requests
import time

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
    "Palwal": (28.15, 77.33)
}

def fetch_nasa(lat, lon):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=ALLSKY_SFC_SW_DWN,T2M,RH2M,CLRSKY_SFC_SW_DWN"
        f"&community=RE&longitude={lon}&latitude={lat}"
        "&start=2015&end=2023&format=JSON"
    )
    r = requests.get(url)
    if r.status_code != 200:
        return None
    data = r.json()["properties"]["parameter"]
    df = pd.DataFrame({
        "ghi": data["ALLSKY_SFC_SW_DWN"],
        "clear_sky_ghi": data["CLRSKY_SFC_SW_DWN"],
        "temp": data["T2M"],
        "humidity": data["RH2M"],
    })
    return df

full_df = []

for district, (lat, lon) in districts.items():
    print(f"Fetching NASA data for {district}...")
    df = fetch_nasa(lat, lon)
    if df is None:
        print(f"Failed to fetch {district}")
        continue
    df["district"] = district
    full_df.append(df)
    time.sleep(1)  # avoid rate limit

final = pd.concat(full_df)
final.to_csv("data/haryana_nasa_daily.csv", index=False)

print("✅ Haryana NASA dataset saved to data/haryana_nasa_daily.csv")
