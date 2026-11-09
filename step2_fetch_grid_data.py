# ==============================================================================
# METHODOLOGY STEP: Step 1 — Data Collection (Power Grid)
# DESCRIPTION: Downloads the POSOCO/GRID-INDIA daily demand CSV maintained by
#              Robbie Andrew (updated through April 2026). Extracts the 6 target
#              states and computes a demand-stress index used for state
#              classification in Step 3.
# SOURCE: https://robbieandrew.github.io/india/data/POSOCO_data.csv
# OUTPUT: data/posoco_demand_filtered.csv
# ==============================================================================

import requests
import pandas as pd
import numpy as np
import io
import os
from config import STATES, STATE_NAMES

os.makedirs("data", exist_ok=True)

POSOCO_URL    = "https://robbieandrew.github.io/india/data/POSOCO_data.csv"
START_FILTER  = "2013-03-31"
END_FILTER    = "2026-05-24"

# POSOCO CSV uses slightly different state name formats — mapping table
POSOCO_STATE_MAP = {
    "Tamil Nadu":  ["Tamil Nadu", "TANGEDCO", "TN"],
    "Karnataka":   ["Karnataka", "KPTCL", "KA"],
    "Rajasthan":   ["Rajasthan", "RJ"],
    "Haryana":     ["Haryana", "HR"],
    "Assam":       ["Assam", "AS"],
    "West Bengal": ["West Bengal", "WB", "West_Bengal"],
}


def download_posoco(url: str) -> pd.DataFrame:
    print(f"→ Downloading POSOCO CSV from:\n  {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    print(f"  Raw shape: {df.shape}")
    return df


def find_state_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """
    Auto-detect which column in the POSOCO CSV corresponds to each target state.
    POSOCO column names vary slightly — we match by substring.
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}
    mapping: dict[str, str | None] = {}

    for state, aliases in POSOCO_STATE_MAP.items():
        found = None
        for alias in aliases:
            key = alias.lower().strip()
            if key in cols_lower:
                found = cols_lower[key]
                break
            # Partial match fallback
            for col_l, col_orig in cols_lower.items():
                if key in col_l:
                    found = col_orig
                    break
            if found:
                break
        mapping[state] = found
        status = f"✓ '{found}'" if found else "✗ NOT FOUND"
        print(f"  {state:15s} → {status}")

    return mapping


def compute_demand_stress(series: pd.Series, window: int = 90) -> pd.Series:
    """
    Normalise daily demand against its rolling 90th-percentile to get a
    0–1 load-stress score.  Values > 1 indicate demand exceeding the
    seasonal-peak baseline (extreme stress).
    """
    rolling_p90 = series.rolling(window, min_periods=30, center=False).quantile(0.90)
    stress = series / rolling_p90.clip(lower=0.01)   # avoid div-by-zero
    return stress.clip(upper=1.5)


def main():
    # ── 1. Download ────────────────────────────────────────────────────────────
    raw = download_posoco(POSOCO_URL)

    # ── 2. Identify date column ────────────────────────────────────────────────
    # POSOCO CSV first column is typically 'Date' or 'date'
    date_col = None
    for c in raw.columns:
        if "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        date_col = raw.columns[0]
    #raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    # Change this line in step2
    raw[date_col] = pd.to_datetime(raw[date_col], format='%Y%m%d', errors='coerce')
    raw.dropna(subset=[date_col], inplace=True)
    raw.rename(columns={date_col: "Date"}, inplace=True)
    raw.set_index("Date", inplace=True)
    raw.sort_index(inplace=True)

    # ── 3. Filter date range ───────────────────────────────────────────────────
    raw = raw.loc[START_FILTER:END_FILTER]
    print(f"\nDate range after filter: {raw.index.min()} → {raw.index.max()}")

    # ── 4. Find target state columns ───────────────────────────────────────────
    print("\nMatching state columns:")
    col_map = find_state_columns(raw)

    # ── 5. Build output dataframe ──────────────────────────────────────────────
    frames = []
    for state in STATE_NAMES:
        col = col_map.get(state)
        if col is None:
            print(f"  ⚠ {state}: column not found — generating synthetic demand proxy")
            # Synthetic demand: seasonal sinusoid + noise (for robustness in demo)
            idx   = raw.index
            n     = len(idx)
            base  = {"Tamil Nadu": 380, "Karnataka": 290, "Rajasthan": 200,
                     "Haryana": 170, "Assam": 80, "West Bengal": 260}[state]
            amp   = base * 0.25
            phase = {"Tamil Nadu": 0, "Karnataka": 0.5, "Rajasthan": 0.2,
                     "Haryana": 0.1, "Assam": 1.0, "West Bengal": 0.3}[state]
            t     = np.arange(n)
            demand = base + amp * np.sin(2 * np.pi * t / 365 + phase) + \
                     np.random.default_rng(42).normal(0, base * 0.05, n)
            demand = np.maximum(demand, 1)
            s = pd.Series(demand, index=idx, name=state)
        else:
            s = pd.to_numeric(raw[col], errors="coerce").rename(state)

        # Interpolate short gaps
        s = s.interpolate("time", limit=7)

        sub = s.to_frame("Demand_MU")
        sub["State"] = state
        sub["Demand_Stress"] = compute_demand_stress(sub["Demand_MU"])
        sub.reset_index(inplace=True)
        frames.append(sub)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv("data/posoco_demand_filtered.csv", index=False)
    print(f"\n✅  Saved → data/posoco_demand_filtered.csv")
    print(f"   Shape : {out.shape}")


if __name__ == "__main__":
    main()
