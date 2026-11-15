# Power Grid Outage-Risk Modeling

An academic Probability and Random Processes mini-project that models weather- and demand-related grid stress across six Indian states:
**Assam, Haryana, Karnataka, Rajasthan, Tamil Nadu, and West Bengal**.

The project combines rule-based state classification, Markov chains, Bayesian updating, Poisson counts, and exponential duration estimates. It includes both Python and MATLAB analysis paths.

> **Important:** This is a research prototype, not an operational forecasting system. The `Outage` label is inferred from weather and demand stress thresholds; it is not a verified record of real outage events.

## Workflow

1. Fetch daily weather data from NASA POWER.
2. Fetch and filter state-level demand data from POSOCO/GRID-INDIA.
3. Classify each state-day as `Normal`, `Stressed`, or `Outage`.
4. Build global and seasonal Markov transition matrices.
5. Estimate steady-state and Bayesian-adjusted outage probabilities.
6. Generate predictions, plots, dashboards, and an interactive risk map.

## Repository layout

```text
.
├── config.py                    # States, thresholds, seasons, and paths
├── run_pipeline.py              # Main command-line entry point
├── step1_fetch_weather.py       # NASA POWER data
├── step2_fetch_grid_data.py     # POSOCO/GRID-INDIA data
├── step3_classify_states.py     # Stress classification
├── step4_markov_model.py        # Markov matrices and steady state
├── step5_bayesian_update.py     # Bayesian calculations
├── step6_prediction_engine.py   # Prediction logic
├── step7_visualizations.py      # Python figures
├── step8_map_visualization.py   # Interactive HTML map
├── markov_analysis.m            # MATLAB Markov/Bayesian analysis
├── predict_outage.m             # MATLAB prediction function
├── requirements.txt
├── data/                        # Input and intermediate CSV files
├── models/                      # Generated model files
├── plots/                       # Python figures and interactive map
├── plots_matlab/                # MATLAB figures
└── results/                     # Example prediction JSON files
```

Generated Python caches, virtual environments, MATLAB autosave files, and local logs are excluded by `.gitignore`. The supplied datasets and example outputs remain available for reproducibility.

## Python setup

Run these commands from this directory. Python 3.10 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the pipeline

Run the complete workflow, including fresh downloads:

```powershell
python run_pipeline.py
```

Use the included CSV files without making network requests:

```powershell
python run_pipeline.py --skip-fetch
```

Skip plot generation:

```powershell
python run_pipeline.py --skip-fetch --skip-plots
```

Run only a prediction after the models have been generated:

```powershell
python run_pipeline.py --predict-only `
  --state "Rajasthan" --date 2024-05-20 `
  --days 30 --temp 45 --rain 0 --wind 10
```

Run `python run_pipeline.py --help` for all available options. Outputs are written to `models/`, `plots/`, and `results/`.

## MATLAB

Run MATLAB from the project directory after generating `data/grid_states_classified.csv` with Python:

```matlab
markov_analysis
predict_outage('Rajasthan', '2024-05-20', 45, 0, 10, 30)
```

MATLAB figures are saved in `plots_matlab/`. The MATLAB scripts expect the `data/` directory to be relative to the current working directory.

## Data sources

- [NASA POWER](https://power.larc.nasa.gov/) — daily weather observations.
- [POSOCO/GRID-INDIA demand data](https://robbieandrew.github.io/india/data/POSOCO_data.csv) — electricity demand source used by the fetch script.

The included CSV files allow the project to run with `--skip-fetch`. Check upstream licensing, attribution, and availability before redistributing or refreshing the datasets.

## Assumptions and limitations

- Weather and load stress are combined using weights in `config.py` (currently 0.45 and 0.55).
- Markov matrices are estimated from consecutive classified daily states.
- Bayesian priors, likelihoods, and outage-duration values are illustrative assumptions.
- Missing data, threshold choices, and changes in upstream datasets can materially change the results.

The results should be used to understand the implemented probability models, not as validated outage forecasts.
