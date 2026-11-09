# ==============================================================================
# config.py — Shared configuration for all pipeline steps
# ==============================================================================

# ── Target States ──────────────────────────────────────────────────────────────
# Selected from geographic extremes of India for climatic diversity
STATES = {
    "Tamil Nadu":  {"lat": 13.08, "lon": 80.27, "region": "Southern"},
    "Karnataka":   {"lat": 12.97, "lon": 77.59, "region": "Southern"},
    "Rajasthan":   {"lat": 26.91, "lon": 75.78, "region": "Western"},
    "Haryana":     {"lat": 28.67, "lon": 77.32, "region": "Northern"},
    "Assam":       {"lat": 26.14, "lon": 91.74, "region": "North-Eastern"},
    "West Bengal": {"lat": 22.57, "lon": 88.36, "region": "Eastern"},
}

STATE_NAMES  = list(STATES.keys())
STATE_COLORS = {
    "Tamil Nadu":  "#E63946",
    "Karnataka":   "#2A9D8F",
    "Rajasthan":   "#E9C46A",
    "Haryana":     "#264653",
    "Assam":       "#F4A261",
    "West Bengal": "#A8DADC",
}

# ── Grid States ────────────────────────────────────────────────────────────────
GRID_STATES  = {0: "Normal", 1: "Stressed", 2: "Outage"}
STATE_LABELS = ["Normal", "Stressed", "Outage"]

# ── Seasons ────────────────────────────────────────────────────────────────────
SEASONS = {
    "Summer":  [3, 4, 5],          # Mar–May
    "Monsoon": [6, 7, 8, 9],       # Jun–Sep
    "Winter":  [10, 11, 12, 1, 2], # Oct–Feb
}

# ── State Classification Thresholds ────────────────────────────────────────────
# Weather stress thresholds (state-specific hot/wet climates)
TEMP_THRESHOLDS = {
    "Tamil Nadu":  {"stressed": 35, "outage": 40},
    "Karnataka":   {"stressed": 33, "outage": 38},
    "Rajasthan":   {"stressed": 38, "outage": 44},
    "Haryana":     {"stressed": 37, "outage": 42},
    "Assam":       {"stressed": 32, "outage": 36},
    "West Bengal": {"stressed": 34, "outage": 39},
}
RAIN_THRESHOLD_HIGH = 50   # mm/day — storm stress
WIND_THRESHOLD      = 12   # m/s   — high-wind stress

# Load stress: fraction of rolling 90th-percentile demand
LOAD_STRESSED_FRAC = 0.88
LOAD_OUTAGE_FRAC   = 0.97

# Weights for combining weather and load stress
WEATHER_WEIGHT = 0.45
LOAD_WEIGHT    = 0.55

# ── Data Paths ─────────────────────────────────────────────────────────────────
WEATHER_CSV = "data/weather_2021_2024.csv"
GRID_CSV    = "data/posoco_demand_filtered.csv"
MERGED_CSV  = "data/merged_dataset.csv"
STATES_CSV  = "data/grid_states_classified.csv"

# ── Prediction Parameters ──────────────────────────────────────────────────────
DEFAULT_PREDICTION_STEPS = 30  # days ahead

# ── Bayesian Parameters ────────────────────────────────────────────────────────
# Prior outage rates per season (from historical CEA reports, approximate)
PRIOR_OUTAGE = {
    "Summer":  0.18,
    "Monsoon": 0.22,
    "Winter":  0.08,
}

# Likelihood P(extreme weather | outage) per season
LIKELIHOOD_WEATHER_GIVEN_OUTAGE = {
    "Summer":  0.70,
    "Monsoon": 0.65,
    "Winter":  0.45,
}
LIKELIHOOD_WEATHER_GIVEN_NO_OUTAGE = {
    "Summer":  0.25,
    "Monsoon": 0.30,
    "Winter":  0.10,
}
