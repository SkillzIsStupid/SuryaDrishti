"""
fetch_himawari.py
-----------------
Lightweight Himawari-8/9 data fetcher for real-time solar irradiance features.
Grabs Band 3 (visible) and Band 13 (IR) radiances for a given lat/lon and UTC time.
"""

import datetime as dt
import requests
import xarray as xr
import numpy as np
from io import BytesIO
from pathlib import Path

# Simple coordinate map for testing
TEST_COORDS = {"Gurugram": (28.45, 77.02), "Hisar": (29.15, 75.72)}

AWS_ROOT = "https://noaa-himawari8.s3.amazonaws.com"


def _nearest_slot(timestamp: dt.datetime) -> dt.datetime:
    """Round timestamp to nearest 10-minute Himawari slot."""
    ts = timestamp.replace(second=0, microsecond=0)
    minute = (ts.minute // 10) * 10
    return ts.replace(minute=minute)


def fetch_himawari_point(lat: float, lon: float, timestamp=None):
    """
    Return a dict with reflectance (B03) and brightness temperature (B13)
    nearest to given lat/lon at given UTC timestamp.
    """
    if timestamp is None:
        timestamp = dt.datetime.utcnow()
    slot = _nearest_slot(timestamp)
    tdir = slot.strftime("%Y/%m/%d/%H%M00")

    bands = {
        "B03": "R03",  # visible reflectance
        "B13": "IR13"  # IR brightness temp
    }

    results = {"time": slot.isoformat()}
    for band, key in bands.items():
        url = f"{AWS_ROOT}/{tdir}/{band}/{band}-{slot.strftime('%Y%m%d%H%M')}_R20_FLDK.nc"
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            ds = xr.open_dataset(BytesIO(r.content))
            if "latitude" in ds and "longitude" in ds:
                lat_name, lon_name = "latitude", "longitude"
            elif "lat" in ds and "lon" in ds:
                lat_name, lon_name = "lat", "lon"
            else:
                raise KeyError("No coordinate names found")

            val = (
                ds[key]
                .sel({lat_name: lat, lon_name: lon}, method="nearest")
                .item()
            )
            results[band] = float(val)
        except Exception as e:
            print(f"[WARN] {band} fetch failed: {e}")
            results[band] = np.nan
    return results


def fetch_recent_hours(lat, lon, hours=1):
    """Fetch a rolling window of Himawari slots for the past N hours."""
    now = dt.datetime.utcnow()
    slots = [now - dt.timedelta(minutes=10 * i) for i in range(int(hours * 6))]
    data = [fetch_himawari_point(lat, lon, s) for s in reversed(slots)]
    return data


if __name__ == "__main__":
    Path("data/satellite").mkdir(parents=True, exist_ok=True)
    for name, (lat, lon) in TEST_COORDS.items():
        print(f"⛅ Fetching recent Himawari data for {name} ...")
        frames = fetch_recent_hours(lat, lon, hours=2)
        import pandas as pd

        df = pd.DataFrame(frames)
        outfile = Path(f"data/satellite/{name.lower()}_himawari.csv")
        df.to_csv(outfile, index=False)
        print(f"✅ Saved {len(df)} records -> {outfile}")
