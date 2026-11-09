#!/usr/bin/env python3
# ==============================================================================
# run_pipeline.py — Master runner for the full Power Outage Modelling pipeline
#
# USAGE:
#   python run_pipeline.py                          # full pipeline
#   python run_pipeline.py --skip-fetch             # skip API calls (use cached)
#   python run_pipeline.py --predict-only           # prediction only
#   python run_pipeline.py --state "Rajasthan" \
#                          --date 2024-05-20 \
#                          --temp 45 --rain 0 --wind 10
# ==============================================================================

import argparse
import subprocess
import sys
import os

STEPS = [
    ("step1_fetch_weather.py",    "Fetching NASA POWER weather data"),
    ("step2_fetch_grid_data.py",  "Fetching POSOCO grid demand data"),
    ("step3_classify_states.py",  "Classifying grid states (Normal/Stressed/Outage)"),
    ("step4_markov_model.py",     "Building Markov transition matrices"),
    ("step5_bayesian_update.py",  "Computing Bayesian-updated outage probabilities"),
    ("step7_visualizations.py",   "Generating all plots"),
    ("step8_map_visualization.py","Building interactive HTML map"),
]


def run_step(script: str, desc: str):
    print(f"\n{'═'*65}")
    print(f"  ▶  {desc}")
    print(f"     ({script})")
    print(f"{'═'*65}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\n❌  FAILED: {script} exited with code {result.returncode}")
        sys.exit(result.returncode)


def run_prediction(state, date, n_days, temp, rain, wind):
    from step6_prediction_engine import predict
    predict(
        state=state,
        target_date=date,
        n_days=n_days,
        temp=temp,
        rain=rain,
        wind=wind,
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch",    action="store_true",
                        help="Skip Steps 1-2 (use already-downloaded data)")
    parser.add_argument("--skip-plots",    action="store_true",
                        help="Skip visualisation steps")
    parser.add_argument("--predict-only",  action="store_true",
                        help="Run prediction engine only (requires built models)")
    parser.add_argument("--state",         default="Tamil Nadu")
    parser.add_argument("--date",          default="2024-06-15")
    parser.add_argument("--days",          type=int, default=30)
    parser.add_argument("--temp",          type=float, default=None)
    parser.add_argument("--rain",          type=float, default=None)
    parser.add_argument("--wind",          type=float, default=None)
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.predict_only:
        run_prediction(args.state, args.date, args.days,
                       args.temp, args.rain, args.wind)
        return

    for script, desc in STEPS:
        if args.skip_fetch and script in (
            "step1_fetch_weather.py", "step2_fetch_grid_data.py"
        ):
            print(f"  ⏭  Skipping {script} (--skip-fetch)")
            continue
        if args.skip_plots and script in (
            "step7_visualizations.py", "step8_map_visualization.py"
        ):
            print(f"  ⏭  Skipping {script} (--skip-plots)")
            continue
        run_step(script, desc)

    # Always run prediction at end
    print(f"\n{'═'*65}")
    print(f"  ▶  Running prediction demo")
    print(f"{'═'*65}")
    run_prediction(args.state, args.date, args.days,
                   args.temp, args.rain, args.wind)

    print(f"\n{'═'*65}")
    print("  ✅  PIPELINE COMPLETE")
    print(f"  📊  Plots  : plots/")
    print(f"  🗺   Map    : plots/india_power_outage_risk_map.html")
    print(f"  📦  Models : models/")
    print(f"  📄  Data   : data/")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
