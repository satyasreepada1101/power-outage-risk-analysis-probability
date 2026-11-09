# ==============================================================================
# METHODOLOGY STEP: Steps 2 & 7 — State Classification
# DESCRIPTION: Merges weather and demand data, then assigns each day a grid
#              state (0=Normal, 1=Stressed, 2=Outage) using a weighted
#              combination of weather stress and demand stress scores.
#              Season labels are also assigned here.
# OUTPUT: data/merged_dataset.csv, data/grid_states_classified.csv
# ==============================================================================

import pandas as pd
import numpy as np
import os
from config import (
    STATE_NAMES, SEASONS, TEMP_THRESHOLDS,
    RAIN_THRESHOLD_HIGH, WIND_THRESHOLD,
    LOAD_STRESSED_FRAC, LOAD_OUTAGE_FRAC,
    WEATHER_WEIGHT, LOAD_WEIGHT,
    WEATHER_CSV, GRID_CSV, MERGED_CSV, STATES_CSV,
)

os.makedirs("data", exist_ok=True)


# ── 1. Season helper ──────────────────────────────────────────────────────────

def get_season(month: int) -> str:
    for season, months in SEASONS.items():
        if month in months:
            return season
    return "Unknown"


# ── 2. Weather stress score (0 → 1) ──────────────────────────────────────────

def compute_weather_stress(row: pd.Series) -> float:
    """
    Continuous weather stress score ∈ [0, 1].
    Accounts for extreme heat, heavy rainfall, and high wind.
    """
    state = row["State"]
    thr   = TEMP_THRESHOLDS[state]

    temp  = row.get("Max_Temperature_2m", np.nan)
    rain  = row.get("Corrected_Precipitation", np.nan)
    wind  = row.get("Wind_Speed_10m", np.nan)
    rh    = row.get("Relative_Humidity_2m", np.nan)

    score = 0.0
    count = 0

    # ── Temperature component ─────────────────────────────────────────────────
    if pd.notna(temp):
        t_min, t_max = thr["stressed"], thr["outage"]
        t_score = np.clip((temp - t_min) / (t_max - t_min + 1e-6), 0, 1)
        score += 0.45 * t_score
        count += 0.45

    # ── Rainfall component ────────────────────────────────────────────────────
    if pd.notna(rain):
        r_score = np.clip(rain / RAIN_THRESHOLD_HIGH, 0, 1)
        score += 0.30 * r_score
        count += 0.30

    # ── Wind component ────────────────────────────────────────────────────────
    if pd.notna(wind):
        w_score = np.clip(wind / WIND_THRESHOLD, 0, 1)
        score += 0.15 * w_score
        count += 0.15

    # ── Humidity-heat interaction (heat index amplifier) ──────────────────────
    if pd.notna(temp) and pd.notna(rh) and temp > 30 and rh > 70:
        score += 0.10 * np.clip((rh - 70) / 30, 0, 1)
        count += 0.10

    return score / count if count > 0 else 0.0


# ── 3. Combined state classifier ──────────────────────────────────────────────

def classify_grid_state(weather_stress: float, demand_stress: float) -> int:
    """
    Maps combined stress score to a discrete grid state.
      0 = Normal   — system operating within comfortable margins
      1 = Stressed — elevated risk, supply-demand tightening or weather stress
      2 = Outage   — high probability of load-shedding or unscheduled outage
    """
    combined = WEATHER_WEIGHT * weather_stress + LOAD_WEIGHT * demand_stress

    if combined >= 0.70:
        return 2   # Outage
    elif combined >= 0.42:
        return 1   # Stressed
    else:
        return 0   # Normal


# ── 4. Main pipeline ──────────────────────────────────────────────────────────

def main():
    print("Loading weather data...")
    weather = pd.read_csv(WEATHER_CSV, parse_dates=["Date"])
    print(f"  Shape: {weather.shape}")

    print("Loading grid data...")
    grid = pd.read_csv(GRID_CSV, parse_dates=["Date"])
    print(f"  Shape: {grid.shape}")

    # Merge on (State, Date)
    merged = pd.merge(weather, grid[["Date", "State", "Demand_MU", "Demand_Stress"]],
                      on=["State", "Date"], how="inner")
    print(f"\nMerged shape: {merged.shape}")

    # Standardise column names (strip spaces that may come from NASA)
    merged.columns = merged.columns.str.strip().str.replace(" ", "_")

    # Season label
    merged["Season"] = merged["Date"].dt.month.map(get_season)
    merged["Month"]  = merged["Date"].dt.month
    merged["Year"]   = merged["Date"].dt.year

    # Weather stress (row-wise)
    print("\nComputing weather stress scores...")
    merged["Weather_Stress"] = merged.apply(compute_weather_stress, axis=1)

    # Normalise demand stress (clip to [0,1])
    merged["Demand_Stress"] = merged["Demand_Stress"].clip(0, 1)

    # Fill NaN demand stress with weather stress (graceful degradation)
    merged["Demand_Stress"].fillna(merged["Weather_Stress"], inplace=True)

    # Classify grid state
    merged["Grid_State"] = merged.apply(
        lambda r: classify_grid_state(r["Weather_Stress"], r["Demand_Stress"]),
        axis=1
    ).astype(int)

    merged["State_Label"] = merged["Grid_State"].map({0:"Normal",1:"Stressed",2:"Outage"})

    # Save merged full dataset
    merged.to_csv(MERGED_CSV, index=False)
    print(f"\n✅  Saved merged dataset → {MERGED_CSV}")

    # Save state-classified dataset (leaner file for modelling)
    cols = ["Date", "State", "Season", "Month", "Year",
            "Max_Temperature_2m", "Corrected_Precipitation",
            "Wind_Speed_10m", "Relative_Humidity_2m",
            "Demand_MU", "Demand_Stress", "Weather_Stress", "Grid_State", "State_Label"]
    available_cols = [c for c in cols if c in merged.columns]
    classified = merged[available_cols]
    classified.to_csv(STATES_CSV, index=False)
    print(f"✅  Saved classified states → {STATES_CSV}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n─── Grid State Distribution per State ───")
    summary = (
        classified.groupby(["State", "State_Label"])
                  .size()
                  .unstack(fill_value=0)
                  .reindex(columns=["Normal", "Stressed", "Outage"], fill_value=0)
    )
    # Add percentage columns
    totals = summary.sum(axis=1)
    for col in summary.columns:
        summary[f"{col}_%"] = (summary[col] / totals * 100).round(1)
    print(summary.to_string())


if __name__ == "__main__":
    main()
