# ==============================================================================
# METHODOLOGY STEP: Step 6 — Prediction Engine
# DESCRIPTION: Given a target state and a date, predicts:
#   1. Grid state probabilities for the next N days using Markov chains
#      (S(n) = S(0) × P^n, seasonal P selected from target month)
#   2. Bayesian-updated outage probability for that specific day
#   3. Poisson-distributed outage frequency over the forecast window
#   4. Exponential-distributed outage duration estimate
#
# USAGE:
#   python step6_prediction_engine.py --state "Tamil Nadu" --date 2024-06-15 --days 30
# ==============================================================================
import math
import argparse
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from config import SEASONS, TEMP_THRESHOLDS, STATE_NAMES, STATES_CSV
from step5_bayesian_update import bayesian_update, weather_extremity_score

os.makedirs("results", exist_ok=True)


# ── Helper: month → season ────────────────────────────────────────────────────

def month_to_season(month: int) -> str:
    for season, months in SEASONS.items():
        if month in months:
            return season
    return "Winter"


# ── Poisson: expected outage frequency ───────────────────────────────────────

def poisson_outage_frequency(outage_prob_per_day: float, days: int) -> dict:
    """
    Model outage count over `days` as Poisson(λ).
    λ = outage_prob_per_day × days
    """
    lam = outage_prob_per_day * days
    k_values = np.arange(0, min(int(lam * 4) + 6, 15))
    pmf = (np.exp(-lam) * lam**k_values) / np.array(
        [math.factorial(k) for k in k_values]
    )
    return {
        "lambda":           round(lam, 4),
        "expected_outages": round(lam, 2),
        "k_values":         k_values.tolist(),
        "pmf":              pmf.round(4).tolist(),
        "prob_at_least_one":round(1 - np.exp(-lam), 4),
        "most_likely_k":    int(k_values[np.argmax(pmf)]),
    }


# ── Exponential: outage duration ─────────────────────────────────────────────

def exponential_outage_duration(mean_duration_hrs: float) -> dict:
    """
    Model outage duration as Exponential(μ = 1/mean_duration).
    State-specific mean durations calibrated from CEA reports.
    """
    mu = 1.0 / mean_duration_hrs
    quantiles = {
        "P25": round(-np.log(0.75) / mu, 2),
        "P50": round(-np.log(0.50) / mu, 2),   # median
        "P75": round(-np.log(0.25) / mu, 2),
        "P90": round(-np.log(0.10) / mu, 2),
    }
    return {
        "mean_hrs":   mean_duration_hrs,
        "rate_mu":    round(mu, 6),
        "quantiles":  quantiles,
        "prob_gt_2h": round(np.exp(-mu * 2), 4),
        "prob_gt_4h": round(np.exp(-mu * 4), 4),
        "prob_gt_8h": round(np.exp(-mu * 8), 4),
    }


# Mean outage durations per state (hours) — calibrated from CEA annual reports
MEAN_OUTAGE_HOURS = {
    "Tamil Nadu":  4.2,
    "Karnataka":   3.8,
    "Rajasthan":   6.5,
    "Haryana":     5.1,
    "Assam":       7.8,
    "West Bengal": 5.3,
}


# ── Core prediction function ──────────────────────────────────────────────────

def predict(state: str, target_date: str, n_days: int = 30,
            temp: float = None, rain: float = None, wind: float = None,
            verbose: bool = True) -> dict:
    """
    Full prediction pipeline for a given state and date.

    Parameters
    ----------
    state       : Indian state name (must be in STATE_NAMES)
    target_date : ISO date string e.g. '2024-06-15'
    n_days      : forecast horizon in days
    temp        : observed/forecast temperature (°C); None → use seasonal average
    rain        : precipitation (mm/day); None → use 0
    wind        : wind speed (m/s); None → use seasonal average

    Returns
    -------
    dict with Markov trajectory, Bayesian posterior, Poisson, Exponential results
    """
    if state not in STATE_NAMES:
        raise ValueError(f"Unknown state '{state}'. Choose from: {STATE_NAMES}")

    date_obj = datetime.strptime(target_date, "%Y-%m-%d")
    season   = month_to_season(date_obj.month)

    # Load transition matrices
    with open("models/transition_matrices.pkl", "rb") as f:
        matrices = pickle.load(f)

    if state not in matrices:
        raise FileNotFoundError(
            f"No Markov model for '{state}'. Run step4_markov_model.py first."
        )

    mats = matrices[state]

    # ── Select seasonal or global matrix ─────────────────────────────────────
    season_data = mats["seasonal"].get(season, {})
    P = season_data.get("matrix", mats["global"])

    # ── Determine initial state from last known data ──────────────────────────
    # Load the classified dataset and get the most recent state
    df_hist = pd.read_csv(STATES_CSV, parse_dates=["Date"])
    df_state_hist = df_hist[df_hist["State"] == state].sort_values("Date")
    if not df_state_hist.empty:
        last_state = int(df_state_hist.iloc[-1]["Grid_State"])
    else:
        last_state = 0   # default to Normal if no history

    s0 = np.zeros(3)
    s0[last_state] = 1.0

    # ── Markov prediction trajectory ─────────────────────────────────────────
    trajectory = np.zeros((n_days + 1, 3))
    trajectory[0] = s0
    s = s0.copy()
    for i in range(1, n_days + 1):
        s = s @ P
        trajectory[i] = s

    # ── Fill in weather defaults if not provided ──────────────────────────────
    seasonal_temp_defaults = {
        "Summer":  {"Tamil Nadu": 38, "Karnataka": 36, "Rajasthan": 42,
                    "Haryana": 41, "Assam": 34, "West Bengal": 37},
        "Monsoon": {"Tamil Nadu": 32, "Karnataka": 29, "Rajasthan": 34,
                    "Haryana": 33, "Assam": 31, "West Bengal": 32},
        "Winter":  {"Tamil Nadu": 27, "Karnataka": 24, "Rajasthan": 22,
                    "Haryana": 20, "Assam": 18, "West Bengal": 22},
    }
    if temp is None:
        temp = seasonal_temp_defaults[season][state]
    if rain is None:
        rain = 30.0 if season == "Monsoon" else 2.0
    if wind is None:
        wind = 8.0 if season == "Summer" else 5.0

    # ── Bayesian update ───────────────────────────────────────────────────────
    prior_outage = mats["steady"][2]
    w_score      = weather_extremity_score(temp, rain, wind, state)
    bayes_result = bayesian_update(prior_outage, season, w_score)

    # Use target-day trajectory outage probability as additional prior signal
    day_offset = min(n_days, 1)
    markov_outage_prob = trajectory[day_offset, 2]
    blended_prior = 0.5 * prior_outage + 0.5 * markov_outage_prob
    bayes_refined = bayesian_update(blended_prior, season, w_score)

    # ── Poisson and Exponential models ────────────────────────────────────────
    daily_outage_prob = bayes_refined["posterior"]
    poisson_result    = poisson_outage_frequency(daily_outage_prob, n_days)
    exp_result        = exponential_outage_duration(
        MEAN_OUTAGE_HOURS.get(state, 5.0)
    )

    result = {
        "state":            state,
        "target_date":      target_date,
        "season":           season,
        "n_days":           n_days,
        "initial_state":    {0: "Normal", 1: "Stressed", 2: "Outage"}[last_state],
        "transition_matrix": P.tolist(),
        "trajectory": {
            "Normal":   trajectory[:, 0].round(4).tolist(),
            "Stressed": trajectory[:, 1].round(4).tolist(),
            "Outage":   trajectory[:, 2].round(4).tolist(),
        },
        "weather_input": {
            "temperature": temp,
            "rainfall":    rain,
            "wind_speed":  wind,
            "extremity_score": round(w_score, 4),
        },
        "bayesian": bayes_refined,
        "poisson":  poisson_result,
        "exponential": exp_result,
        "summary": {
            "outage_prob_day1":       round(trajectory[1, 2], 4),
            "outage_prob_day7":       round(trajectory[min(7, n_days), 2], 4),
            "outage_prob_day30":      round(trajectory[min(30, n_days), 2], 4),
            "bayesian_posterior":     bayes_refined["posterior"],
            "expected_outages":       poisson_result["expected_outages"],
            "median_outage_duration": exp_result["quantiles"]["P50"],
            "steady_state_outage":    round(mats["steady"][2], 4),
        },
    }

    if verbose:
        _print_report(result)

    return result


def _print_report(r: dict):
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  POWER OUTAGE PREDICTION REPORT")
    print(f"  State : {r['state']}")
    print(f"  Date  : {r['target_date']}  ({r['season']})")
    print(f"  Horizon: {r['n_days']} days")
    print(sep)
    print(f"\n  ── Current Grid Status ──")
    print(f"  Initial state     : {r['initial_state']}")
    print(f"\n  ── Markov Chain Forecast ──")
    s = r["summary"]
    print(f"  P(Outage) Day+1   : {s['outage_prob_day1']:.2%}")
    print(f"  P(Outage) Day+7   : {s['outage_prob_day7']:.2%}")
    print(f"  P(Outage) Day+30  : {s['outage_prob_day30']:.2%}")
    print(f"  Long-run P(Outage): {s['steady_state_outage']:.2%}")
    print(f"\n  ── Bayesian Update (Weather Integration) ──")
    b = r["bayesian"]
    print(f"  Weather extremity : {r['weather_input']['extremity_score']:.2f}  "
          f"({r['weather_input']['temperature']}°C, "
          f"{r['weather_input']['rainfall']}mm, "
          f"{r['weather_input']['wind_speed']}m/s)")
    print(f"  Prior P(Outage)   : {b['prior']:.2%}")
    print(f"  Posterior P(Out)  : {b['posterior']:.2%}  {b['update_direction']}")
    print(f"  Bayes Factor      : {b['bayes_factor']}")
    print(f"\n  ── Poisson Distribution ──")
    p = r["poisson"]
    print(f"  λ (expected outages/{r['n_days']}d): {p['lambda']}")
    print(f"  Most likely count : {p['most_likely_k']}")
    print(f"  P(≥1 outage)      : {p['prob_at_least_one']:.2%}")
    print(f"\n  ── Exponential Duration ──")
    e = r["exponential"]
    print(f"  Mean duration     : {e['mean_hrs']} hrs")
    print(f"  Median duration   : {e['quantiles']['P50']} hrs")
    print(f"  P(> 4 hrs)        : {e['prob_gt_4h']:.2%}")
    print(f"{sep}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Power Outage Prediction Engine")
    parser.add_argument("--state",  default="Tamil Nadu",
                        help=f"State name. Options: {STATE_NAMES}")
    parser.add_argument("--date",   default="2024-06-15",
                        help="Target date (YYYY-MM-DD)")
    parser.add_argument("--days",   type=int, default=30,
                        help="Forecast horizon in days")
    parser.add_argument("--temp",   type=float, default=None,
                        help="Observed temperature (°C)")
    parser.add_argument("--rain",   type=float, default=None,
                        help="Precipitation (mm/day)")
    parser.add_argument("--wind",   type=float, default=None,
                        help="Wind speed (m/s)")
    args = parser.parse_args()

    result = predict(
        state=args.state,
        target_date=args.date,
        n_days=args.days,
        temp=args.temp,
        rain=args.rain,
        wind=args.wind,
        verbose=True,
    )

    # Save result
    import json
    out_path = f"results/prediction_{args.state.replace(' ','_')}_{args.date}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved → {out_path}")


if __name__ == "__main__":
    main()
