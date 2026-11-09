# ==============================================================================
# METHODOLOGY STEP: Step 1 — Data Collection (Weather)
# DESCRIPTION: Fetches historical daily temperature, precipitation, and wind
#              speed from the NASA POWER API for the 6 target Indian states.
#              Date range updated to 2021-01-01 → 2024-12-31 for recency.
# OUTPUT: data/weather_2021_2024.csv
# ==============================================================================

import requests
import pandas as pd
import time
import os
from config import STATES

os.makedirs("data", exist_ok=True)

# ── NASA POWER API parameters ──────────────────────────────────────────────────
NASA_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = {
    "T2M_MAX":      "Max Temperature 2m (°C)",
    "T2M_MIN":      "Min Temperature 2m (°C)",
    "PRECTOTCORR":  "Corrected Precipitation (mm/day)",
    "WS10M":        "Wind Speed 10m (m/s)",
    "RH2M":         "Relative Humidity 2m (%)",   # Extra: humidity adds to heat stress
}

START_DATE = "20130331"
END_DATE   = "20260524"


def fetch_nasa_weather(state: str, lat: float, lon: float) -> pd.DataFrame | None:
    """Fetch daily weather parameters from NASA POWER for a single location."""
    query_params = {
        "parameters": ",".join(PARAMS.keys()),
        "community":  "RE",
        "longitude":  lon,
        "latitude":   lat,
        "start":      START_DATE,
        "end":        END_DATE,
        "format":     "JSON",
    }
    try:
        resp = requests.get(NASA_BASE, params=query_params, timeout=60)
        resp.raise_for_status()
        raw = resp.json()["properties"]["parameter"]

        df = pd.DataFrame(raw)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.index.name = "Date"
        df["State"] = state

        # Rename columns to human-readable
        df.rename(columns={k: v.split(" (")[0].replace(" ", "_")
                            for k, v in PARAMS.items()}, inplace=True)

        # Replace NASA fill-value (-999) with NaN
        df.replace(-999.0, float("nan"), inplace=True)

        print(f"  ✓ {state}: {len(df)} days fetched")
        return df

    except requests.exceptions.RequestException as exc:
        print(f"  ✗ {state}: network error — {exc}")
        return None
    except (KeyError, ValueError) as exc:
        print(f"  ✗ {state}: parse error — {exc}")
        return None


def main():
    all_frames = []
    print(f"\nFetching NASA POWER data ({START_DATE} → {END_DATE})\n{'─'*50}")

    for state, meta in STATES.items():
        print(f"→ {state}")
        df = fetch_nasa_weather(state, meta["lat"], meta["lon"])
        if df is not None:
            all_frames.append(df)
        time.sleep(2.0)   # respect NASA POWER rate limit

    if not all_frames:
        raise RuntimeError("No weather data fetched — check internet connection.")

    combined = pd.concat(all_frames)
    combined.reset_index(inplace=True)

    # Forward-fill small gaps (≤ 3 days) introduced by API anomalies
    combined.set_index(["State", "Date"], inplace=True)
    numeric_cols = combined.select_dtypes("number").columns
    combined[numeric_cols] = (
        combined.groupby("State")[numeric_cols]
                .transform(lambda s: s.fillna(method="ffill", limit=3)
                                      .fillna(method="bfill", limit=3))
    )
    combined.reset_index(inplace=True)

    out_path = "data/weather_2021_2024.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n{'─'*50}\n✅  Saved → {out_path}")
    print(f"   Shape : {combined.shape}")
    print(f"   States: {combined['State'].nunique()}")
    print(f"   Date range: {combined['Date'].min()} → {combined['Date'].max()}")


if __name__ == "__main__":
    main()
