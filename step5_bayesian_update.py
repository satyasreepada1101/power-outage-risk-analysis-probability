# ==============================================================================
# METHODOLOGY STEP: Step 4 — Bayesian Weather Integration (PROJECT NOVELTY)
# DESCRIPTION: Computes a Bayesian-updated outage probability for any given
#              state, date, and observed weather conditions.
#
#   Bayes' rule:
#     P(Outage | Weather) = P(Weather | Outage) × P(Outage)
#                          ─────────────────────────────────
#                                    P(Weather)
#
#   P(Weather) = P(W|Out)·P(Out) + P(W|¬Out)·P(¬Out)
#
#   The prior P(Outage) comes from the Markov steady-state vector.
#   The likelihoods are calibrated per season from historical weather patterns.
# ==============================================================================

import numpy as np
import pandas as pd
import pickle
import os
from config import (
    STATE_NAMES, SEASONS, PRIOR_OUTAGE,
    LIKELIHOOD_WEATHER_GIVEN_OUTAGE,
    LIKELIHOOD_WEATHER_GIVEN_NO_OUTAGE,
    TEMP_THRESHOLDS, RAIN_THRESHOLD_HIGH, WIND_THRESHOLD,
)

os.makedirs("models", exist_ok=True)


# ── Weather extremity detector ────────────────────────────────────────────────

def is_extreme_weather(temp: float, rain: float, wind: float,
                       state: str) -> bool:
    """
    Returns True if the weather reading qualifies as 'extreme' for the given
    state, triggering Bayesian updating.
    """
    thr = TEMP_THRESHOLDS[state]
    extreme_temp = pd.notna(temp) and temp >= thr["stressed"]
    extreme_rain = pd.notna(rain) and rain >= RAIN_THRESHOLD_HIGH * 0.6
    extreme_wind = pd.notna(wind) and wind >= WIND_THRESHOLD * 0.8
    return extreme_temp or extreme_rain or extreme_wind


def weather_extremity_score(temp: float, rain: float, wind: float,
                             state: str) -> float:
    """
    Continuous version: returns a score in [0, 1] capturing how extreme
    the weather is.  Used to blend likelihood smoothly.
    """
    thr = TEMP_THRESHOLDS[state]
    t_score = np.clip((temp - thr["stressed"]) /
                      (thr["outage"] - thr["stressed"] + 1e-6), 0, 1) \
              if pd.notna(temp) else 0.0
    r_score = np.clip(rain / RAIN_THRESHOLD_HIGH, 0, 1) \
              if pd.notna(rain) else 0.0
    w_score = np.clip(wind / WIND_THRESHOLD, 0, 1) \
              if pd.notna(wind) else 0.0
    return max(t_score, r_score, w_score)


# ── Bayesian updater ──────────────────────────────────────────────────────────

def bayesian_update(prior_outage: float, season: str,
                    weather_score: float) -> dict:
    """
    Apply Bayes' theorem to update outage probability given weather evidence.

    Parameters
    ----------
    prior_outage   : P(Outage) from Markov steady-state or seasonal prior
    season         : 'Summer' | 'Monsoon' | 'Winter'
    weather_score  : 0–1 weather extremity (0 = benign, 1 = extremely severe)

    Returns
    -------
    dict with keys: prior, likelihood_given_outage, likelihood_given_no_outage,
                    posterior, bayes_factor, update_direction
    """
    p_w_out   = LIKELIHOOD_WEATHER_GIVEN_OUTAGE[season]
    p_w_noout = LIKELIHOOD_WEATHER_GIVEN_NO_OUTAGE[season]

    # Scale likelihoods by weather extremity score (smooth interpolation)
    # When weather_score=0 → P(W|Out) ≈ base-rate, no update
    # When weather_score=1 → full seasonal likelihood applies
    alpha = weather_score
    p_w_out_eff   = (1 - alpha) * prior_outage + alpha * p_w_out
    p_w_noout_eff = (1 - alpha) * (1 - prior_outage) / 2 + alpha * p_w_noout

    # Normalising constant P(Weather)
    p_weather = p_w_out_eff * prior_outage + p_w_noout_eff * (1 - prior_outage)
    p_weather  = max(p_weather, 1e-9)

    posterior = (p_w_out_eff * prior_outage) / p_weather
    posterior = np.clip(posterior, 0.0, 1.0)

    bayes_factor = (p_w_out_eff / max(p_w_noout_eff, 1e-9))

    return {
        "prior":                     round(prior_outage, 4),
        "likelihood_given_outage":   round(p_w_out_eff, 4),
        "likelihood_given_no_outage":round(p_w_noout_eff, 4),
        "p_evidence":                round(p_weather, 4),
        "posterior":                 round(posterior, 4),
        "bayes_factor":              round(bayes_factor, 3),
        "update_direction":          "↑ increased" if posterior > prior_outage else "↓ decreased",
    }


# ── Full state-level Bayesian time series ─────────────────────────────────────

def compute_bayesian_series(df: pd.DataFrame, matrices: dict) -> pd.DataFrame:
    """
    For each row in the classified dataset, compute the Bayesian-updated
    outage probability.  The prior is taken from the steady-state of the
    corresponding state's Markov model.
    """
    rows = []
    for _, row in df.iterrows():
        state   = row["State"]
        season  = row["Season"]
        temp    = row.get("Max_Temperature_2m", np.nan)
        rain    = row.get("Corrected_Precipitation", np.nan)
        wind    = row.get("Wind_Speed_10m", np.nan)

        # Prior: Markov steady-state outage probability
        if state in matrices:
            prior = matrices[state]["steady"][2]
        else:
            prior = PRIOR_OUTAGE.get(season, 0.10)

        w_score = weather_extremity_score(temp, rain, wind, state)
        result  = bayesian_update(prior, season, w_score)

        rows.append({
            "Date":              row["Date"],
            "State":             state,
            "Season":            season,
            "Grid_State":        row["Grid_State"],
            "Weather_Score":     round(w_score, 4),
            "Prior_Outage":      result["prior"],
            "Posterior_Outage":  result["posterior"],
            "Bayes_Factor":      result["bayes_factor"],
        })

    return pd.DataFrame(rows)


def main():
    from config import STATES_CSV

    print("Loading classified states...")
    df = pd.read_csv(STATES_CSV, parse_dates=["Date"])

    print("Loading Markov transition matrices...")
    with open("models/transition_matrices.pkl", "rb") as f:
        matrices = pickle.load(f)

    print("Computing Bayesian-updated outage probabilities...")
    bayes_df = compute_bayesian_series(df, matrices)
    bayes_df.to_csv("models/bayesian_outage_series.csv", index=False)
    print("✅  Saved → models/bayesian_outage_series.csv")

    # ── Summary statistics ─────────────────────────────────────────────────────
    print("\n─── Bayesian Update Summary by State ───")
    summary = bayes_df.groupby("State").agg(
        Prior_Mean       =("Prior_Outage",    "mean"),
        Posterior_Mean   =("Posterior_Outage","mean"),
        Posterior_Max    =("Posterior_Outage","max"),
        High_Risk_Days   =("Posterior_Outage", lambda x: (x > 0.5).sum()),
    ).round(4)
    print(summary.to_string())

    # ── Interactive prediction helper ─────────────────────────────────────────
    print("\n─── Sample Bayesian Inference ───")
    for state in STATE_NAMES[:3]:
        if state not in matrices:
            continue
        prior  = matrices[state]["steady"][2]
        result = bayesian_update(prior, "Summer", weather_score=0.85)
        print(f"\n  {state}:")
        print(f"    Prior P(Outage)     = {result['prior']}")
        print(f"    Posterior P(Outage) = {result['posterior']}")
        print(f"    Bayes Factor        = {result['bayes_factor']}")
        print(f"    Direction           = {result['update_direction']}")


if __name__ == "__main__":
    main()
