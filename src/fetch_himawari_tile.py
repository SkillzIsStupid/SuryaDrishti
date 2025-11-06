import requests
import datetime as dt
import numpy as np

def fetch_realtime_power(lat, lon):
    """
    Fetch current irradiance, temperature, humidity from NASA POWER API.
    """
    end = dt.datetime.utcnow().strftime("%Y%m%d")
    url = (
        f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"parameters=ALLSKY_SFC_SW_DWN,T2M,RH2M,CLRSKY_SFC_SW_DWN,CLOUD_AMT"
        f"&community=RE&longitude={lon}&latitude={lat}"
        f"&start={end}&end={end}&format=JSON"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]

    # get last available hour
    key = sorted(data["ALLSKY_SFC_SW_DWN"].keys())[-1]
    return {
        "ghi": data["ALLSKY_SFC_SW_DWN"][key],
        "temp": data["T2M"][key],
        "humidity": data["RH2M"][key],
        "clear_sky": data["CLRSKY_SFC_SW_DWN"][key],
        "cloud": data.get("CLOUD_AMT", {}).get(key, np.nan)
    }

if __name__ == "__main__":
    print(fetch_realtime_power(28.46, 77.03))
