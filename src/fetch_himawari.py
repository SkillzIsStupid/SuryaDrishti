"""
fetch_himawari.py — night-safe unified fetcher
----------------------------------------------
Tries JAXA → AWS → NASA POWER.
If it's night: fetch B13 only and reuse last daylight B03.
"""

import datetime as dt
import os
import numpy as np
import xarray as xr
import requests
import ftplib
from io import BytesIO
from dotenv import load_dotenv
from astral import LocationInfo
from astral.sun import sun

# --- Load env credentials
load_dotenv()
JAXA_UID = os.getenv("JAXA_UID")
JAXA_PW = os.getenv("JAXA_PW")

FTP_HOST = "ftp.ptree.jaxa.jp"
AWS_ROOT = "https://noaa-himawari8.s3.amazonaws.com"

CACHE_FILE = "data/satellite/last_daylight_b03.npy"

def _nearest_slot(timestamp: dt.datetime) -> dt.datetime:
    ts = timestamp.replace(second=0, microsecond=0)
    minute = (ts.minute // 10) * 10
    return ts.replace(minute=minute)

def _is_valid(val): return np.isfinite(val) and not np.isnan(val)

# -------- helpers --------
def _sun_is_up(lat, lon, when=None):
    """Return True if it's daylight at location."""
    from datetime import UTC
    city = LocationInfo(latitude=lat, longitude=lon)
    when = when or dt.datetime.now(UTC)
    s = sun(city.observer, date=when.date())
    return s["sunrise"] <= when <= s["sunset"]

# ---------- JAXA ----------
def _fetch_from_jaxa(lat, lon, timestamp):
    slot = _nearest_slot(timestamp)
    results = {"time": slot.isoformat()}
    with ftplib.FTP(FTP_HOST) as ftp:
        ftp.login(JAXA_UID, JAXA_PW)
        for band in ["B03", "B13"]:
            remote_file = f"/himawari8/FD/{band}/{slot.strftime('%Y/%m/%d/%H%M')}/{band}-{slot.strftime('%Y%m%d%H%M')}_FLDK.nc"
            try:
                bio = BytesIO()
                ftp.retrbinary(f"RETR {remote_file}", bio.write)
                bio.seek(0)
                ds = xr.open_dataset(bio)
                lat_name = "latitude" if "latitude" in ds else "lat"
                lon_name = "longitude" if "longitude" in ds else "lon"
                val = ds[list(ds.data_vars)[0]].sel({lat_name: lat, lon_name: lon}, method="nearest").item()
                results[band] = float(val)
            except Exception:
                results[band] = np.nan
    return results

# ---------- AWS ----------
def _fetch_from_aws(lat, lon, timestamp):
    slot = _nearest_slot(timestamp)
    results = {"time": slot.isoformat()}
    for band in ["B03", "B13"]:
        url = f"{AWS_ROOT}/{slot.strftime('%Y/%m/%d/%H%M00')}/{band}/{band}-{slot.strftime('%Y%m%d%H%M')}_R20_FLDK.nc"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            ds = xr.open_dataset(BytesIO(r.content))
            lat_name = "latitude" if "latitude" in ds else "lat"
            lon_name = "longitude" if "longitude" in ds else "lon"
            val = ds[list(ds.data_vars)[0]].sel({lat_name: lat, lon_name: lon}, method="nearest").item()
            results[band] = float(val)
        except Exception:
            results[band] = np.nan
    return results

# ---------- NASA POWER ----------
def _fetch_from_nasa_power(lat, lon, timestamp):
    ymd = timestamp.strftime("%Y%m%d")
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point"
        f"?parameters=ALLSKY_SFC_SW_DWN,CLOUD_AMT"
        f"&community=RE&longitude={lon}&latitude={lat}"
        f"&start={ymd}&end={ymd}&format=JSON"
    )
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()["properties"]["parameter"]
        ghi = [v for v in data.get("ALLSKY_SFC_SW_DWN", {}).values() if v != -999.0]
        cloud = [v for v in data.get("CLOUD_AMT", {}).values() if v != -999.0]
        return {
            "B03": float(np.nanmean(ghi)) if ghi else np.nan,
            "B13": float(np.nanmean(cloud)) if cloud else np.nan,
            "time": timestamp.isoformat(),
        }
    except Exception:
        return {"B03": np.nan, "B13": np.nan, "time": timestamp.isoformat()}

# ---------- Unified ----------
def fetch_himawari_point(lat, lon, timestamp=None):
    if timestamp is None:
        timestamp = dt.datetime.now(dt.UTC)
    daylight = _sun_is_up(lat, lon, timestamp)

    # 1️⃣ Try JAXA
    try:
        data = _fetch_from_jaxa(lat, lon, timestamp)
        source = "JAXA"
    except Exception:
        data = {"B03": np.nan, "B13": np.nan}
        source = "JAXA_fail"

    # 2️⃣ If invalid → try AWS
    if not (_is_valid(data.get("B03")) or _is_valid(data.get("B13"))):
        try:
            data = _fetch_from_aws(lat, lon, timestamp)
            source = "AWS"
        except Exception:
            source = "AWS_fail"

    # 3️⃣ Fallback → NASA POWER
    if not (_is_valid(data.get("B03")) or _is_valid(data.get("B13"))):
        data = _fetch_from_nasa_power(lat, lon, timestamp)
        source = "NASA_POWER"

    # ---- Night handling ----
    if not daylight:
        # Keep live IR
        b13 = data.get("B13", np.nan)
        # Replace visible with last cached daylight
        if os.path.exists(CACHE_FILE):
            last_b03 = np.load(CACHE_FILE).item()
        else:
            last_b03 = np.nan
        data["B03"] = last_b03
        data["B13"] = b13
        data["source"] = source + "_night"
    else:
        # Cache today's daylight visible band
        if _is_valid(data.get("B03")):
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            np.save(CACHE_FILE, data["B03"])
        data["source"] = source

    return data

if __name__ == "__main__":
    lat, lon = 28.45, 77.02
    print(f"🛰️ Testing fetch for Gurugram ({lat},{lon}) ...")
    out = fetch_himawari_point(lat, lon)
    print(out)
