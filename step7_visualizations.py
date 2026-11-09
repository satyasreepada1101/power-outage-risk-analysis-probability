# ==============================================================================
# METHODOLOGY STEP: Step 8 — Visualizations
# DESCRIPTION: Generates all publication-quality graphs for the project:
#   1.  Transition probability heatmaps   (one per state)
#   2.  Seasonal transition matrix comparison (3 seasons × 6 states)
#   3.  State probability trajectory curves (n-step forecast per state)
#   4.  Outage probability time-series per state
#   5.  Weather vs outage probability scatter (per state, seasonal colour)
#   6.  Steady-state distribution bar chart (all states)
#   7.  Poisson PMF curves per state
#   8.  Exponential duration CDFs per state
#   9.  Bayesian update waterfall plot (prior → posterior by season)
#   10. Monthly outage probability heatmap calendar (per state)
#   11. Season-wise comparison bar chart (outage rate)
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
import os
from matplotlib.colors import LinearSegmentedColormap
from config import STATE_NAMES, STATE_COLORS, SEASONS

os.makedirs("plots", exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
})
CMAP_OUTAGE = LinearSegmentedColormap.from_list(
    "outage", ["#2ecc71", "#f39c12", "#e74c3c"]
)
GRID_STATE_COLORS = ["#2ecc71", "#f39c12", "#e74c3c"]
LABEL_MAP = {0: "Normal", 1: "Stressed", 2: "Outage"}


# ── 1. Per-state transition heatmaps ─────────────────────────────────────────

def plot_transition_heatmaps(matrices: dict):
    """6 subplots in 2×3 grid, one per state."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("3×3 Transition Probability Matrices — All States\n"
                 "(Row = current state, Column = next state)", fontsize=14, y=1.01)

    for ax, state in zip(axes.flatten(), STATE_NAMES):
        P = matrices[state]["global"]
        labels = ["Normal", "Stressed", "Outage"]
        df_p = pd.DataFrame(P, index=labels, columns=labels)

        sns.heatmap(df_p, ax=ax, annot=True, fmt=".3f",
                    cmap="RdYlGn_r", vmin=0, vmax=1,
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "Probability"})
        ax.set_title(state, fontsize=12, fontweight="bold",
                     color=STATE_COLORS[state])
        ax.set_xlabel("Next State")
        ax.set_ylabel("Current State")

    plt.tight_layout()
    plt.savefig("plots/01_transition_heatmaps_all_states.png",
                bbox_inches="tight")
    plt.close()
    print("  ✓ plots/01_transition_heatmaps_all_states.png")


# ── 2. Seasonal matrices per state ────────────────────────────────────────────

def plot_seasonal_matrices(matrices: dict):
    """One figure per state, 3 subplots (Summer / Monsoon / Winter)."""
    seasons = list(SEASONS.keys())
    for state in STATE_NAMES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"{state} — Seasonal Transition Matrices",
                     fontsize=13, fontweight="bold",
                     color=STATE_COLORS[state])
        labels = ["Normal", "Stressed", "Outage"]

        for ax, season in zip(axes, seasons):
            P = matrices[state]["seasonal"][season]["matrix"]
            df_p = pd.DataFrame(P, index=labels, columns=labels)
            sns.heatmap(df_p, ax=ax, annot=True, fmt=".3f",
                        cmap="RdYlGn_r", vmin=0, vmax=1,
                        linewidths=0.5, linecolor="white",
                        cbar=(ax == axes[-1]))
            ax.set_title(season, fontsize=11)
            ax.set_xlabel("Next State")
            ax.set_ylabel("Current State" if ax == axes[0] else "")

        plt.tight_layout()
        fname = f"plots/02_seasonal_matrices_{state.replace(' ','_')}.png"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
    print("  ✓ plots/02_seasonal_matrices_[state].png (6 files)")


# ── 3. State probability trajectory forecast ─────────────────────────────────

def plot_markov_trajectories(matrices: dict, n_steps: int = 60):
    """One figure per state: P(Normal/Stressed/Outage) over n_steps days."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
    fig.suptitle(f"Markov Chain Forecast — State Probability over {n_steps} Days\n"
                 "(Starting from Normal state)", fontsize=14)

    for ax, state in zip(axes.flatten(), STATE_NAMES):
        P  = matrices[state]["global"]
        s0 = np.array([1.0, 0.0, 0.0])
        traj = np.zeros((n_steps + 1, 3))
        traj[0] = s0
        s = s0.copy()
        for i in range(1, n_steps + 1):
            s = s @ P
            traj[i] = s

        days = np.arange(n_steps + 1)
        ax.fill_between(days, traj[:, 0], alpha=0.25, color=GRID_STATE_COLORS[0])
        ax.fill_between(days, traj[:, 1], alpha=0.25, color=GRID_STATE_COLORS[1])
        ax.fill_between(days, traj[:, 2], alpha=0.35, color=GRID_STATE_COLORS[2])

        for i, (label, color) in enumerate(zip(LABEL_MAP.values(), GRID_STATE_COLORS)):
            ax.plot(days, traj[:, i], color=color, label=label, linewidth=2.2)

        ax.set_title(state, fontweight="bold", color=STATE_COLORS[state])
        ax.set_xlabel("Days Ahead")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper right")

        # Steady-state lines (dashed)
        pi = matrices[state]["steady"]
        for i, color in enumerate(GRID_STATE_COLORS):
            ax.axhline(pi[i], linestyle="--", color=color, linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    plt.savefig("plots/03_markov_trajectory_all_states.png", bbox_inches="tight")
    plt.close()
    print("  ✓ plots/03_markov_trajectory_all_states.png")


# ── 4. Outage probability time-series ─────────────────────────────────────────

def plot_outage_timeseries(bayes_df: pd.DataFrame):
    """Daily posterior outage probability per state over the full data window."""
    fig, axes = plt.subplots(3, 2, figsize=(18, 14), sharex=False)
    fig.suptitle("Daily Bayesian-Updated Outage Probability per State\n"
                 "(Dashed = Markov steady-state baseline)", fontsize=14)

    for ax, state in zip(axes.flatten(), STATE_NAMES):
        df_s = bayes_df[bayes_df["State"] == state].sort_values("Date")
        color = STATE_COLORS[state]

        ax.fill_between(df_s["Date"], df_s["Posterior_Outage"],
                        alpha=0.30, color=color)
        ax.plot(df_s["Date"], df_s["Posterior_Outage"].rolling(14).mean(),
                color=color, linewidth=1.8, label="14-day rolling mean")
        ax.plot(df_s["Date"], df_s["Posterior_Outage"],
                color=color, alpha=0.25, linewidth=0.5)

        prior_mean = df_s["Prior_Outage"].mean()
        ax.axhline(prior_mean, linestyle="--", color="gray",
                   linewidth=1.2, label=f"Steady-state ({prior_mean:.2f})")

        # Shade seasons
        for _, row in df_s.iterrows():
            if row["Season"] == "Summer":
                ax.axvspan(row["Date"], row["Date"] + pd.Timedelta("1D"),
                           alpha=0.03, color="red")
            elif row["Season"] == "Monsoon":
                ax.axvspan(row["Date"], row["Date"] + pd.Timedelta("1D"),
                           alpha=0.03, color="blue")

        ax.set_title(state, fontweight="bold", color=color)
        ax.set_ylabel("P(Outage)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    plt.savefig("plots/04_outage_timeseries_all_states.png", bbox_inches="tight")
    plt.close()
    print("  ✓ plots/04_outage_timeseries_all_states.png")


# ── 5. Weather vs outage scatter ─────────────────────────────────────────────

def plot_weather_vs_outage(merged_df: pd.DataFrame, bayes_df: pd.DataFrame):
    """Temperature vs Posterior Outage probability, coloured by season, per state."""
    combined = merged_df.merge(
        bayes_df[["Date", "State", "Posterior_Outage"]],
        on=["Date", "State"], how="inner"
    )

    season_colors = {"Summer": "#e74c3c", "Monsoon": "#3498db", "Winter": "#2ecc71"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Max Temperature vs Bayesian Outage Probability\n"
                 "(Coloured by Season)", fontsize=14)

    for ax, state in zip(axes.flatten(), STATE_NAMES):
        df_s = combined[combined["State"] == state].dropna(
            subset=["Max_Temperature_2m", "Posterior_Outage"]
        )
        for season, sc in season_colors.items():
            df_sea = df_s[df_s["Season"] == season]
            ax.scatter(df_sea["Max_Temperature_2m"], df_sea["Posterior_Outage"],
                       c=sc, alpha=0.35, s=8, label=season)

        ax.set_title(state, fontweight="bold", color=STATE_COLORS[state])
        ax.set_xlabel("Max Temperature (°C)")
        ax.set_ylabel("P(Outage) — Bayesian Posterior")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, markerscale=2)

    plt.tight_layout()
    plt.savefig("plots/05_weather_vs_outage_scatter.png", bbox_inches="tight")
    plt.close()
    print("  ✓ plots/05_weather_vs_outage_scatter.png")


# ── 6. Steady-state distribution bar chart ───────────────────────────────────

def plot_steady_state(matrices: dict):
    """Grouped bar chart: steady-state π for all 6 states."""
    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(len(STATE_NAMES))
    w = 0.25
    bar_colors = GRID_STATE_COLORS

    for i, (label, bc) in enumerate(zip(LABEL_MAP.values(), bar_colors)):
        vals = [matrices[s]["steady"][i] for s in STATE_NAMES]
        bars = ax.bar(x + i * w, vals, w, label=label, color=bc, alpha=0.85,
                      edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x + w)
    ax.set_xticklabels([s.replace(" ", "\n") for s in STATE_NAMES], fontsize=10)
    ax.set_ylabel("Steady-State Probability π")
    ax.set_title("Long-Run (Steady-State) Grid State Distribution\nAll States",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Grid State", fontsize=10)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("plots/06_steady_state_distribution.png", bbox_inches="tight")
    plt.close()
    print("  ✓ plots/06_steady_state_distribution.png")


# ── 7. Poisson PMF curves ─────────────────────────────────────────────────────

def plot_poisson_pmf(matrices: dict, n_days: int = 30):
    """Poisson PMF for expected outage count over n_days, per state."""
    import math

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(f"Poisson PMF — Expected Number of Outages over {n_days} Days\n"
                 "λ = steady-state outage probability × days", fontsize=13)

    for ax, state in zip(axes.flatten(), STATE_NAMES):
        lam = matrices[state]["steady"][2] * n_days
        k_max = max(int(lam * 3.5) + 4, 8)
        k = np.arange(0, k_max + 1)
        pmf = np.array([math.exp(-lam) * lam**ki / math.factorial(ki) for ki in k])

        bars = ax.bar(k, pmf, color=STATE_COLORS[state],
                      alpha=0.8, edgecolor="white")
        ax.axvline(lam, color="black", linestyle="--", linewidth=1.5,
                   label=f"λ = {lam:.2f}")
        ax.set_title(state, fontweight="bold", color=STATE_COLORS[state])
        ax.set_xlabel("Number of Outage Days")
        ax.set_ylabel("Probability")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("plots/07_poisson_pmf_all_states.png", bbox_inches="tight")
    plt.close()
    print("  ✓ plots/07_poisson_pmf_all_states.png")


# ── 8. Exponential CDF per state ─────────────────────────────────────────────

def plot_exponential_cdf():
    MEAN_HRS = {
        "Tamil Nadu": 4.2, "Karnataka": 3.8, "Rajasthan": 6.5,
        "Haryana": 5.1, "Assam": 7.8, "West Bengal": 5.3,
    }
    fig, ax = plt.subplots(figsize=(12, 7))
    t = np.linspace(0, 20, 500)

    for state in STATE_NAMES:
        mu  = 1 / MEAN_HRS[state]
        cdf = 1 - np.exp(-mu * t)
        ax.plot(t, cdf, label=f"{state} (mean {MEAN_HRS[state]}h)",
                color=STATE_COLORS[state], linewidth=2.2)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="50% quantile")
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=1, label="90% quantile")
    ax.set_xlabel("Outage Duration (hours)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Exponential Distribution — Outage Duration CDF per State",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("plots/08_exponential_duration_cdf.png", bbox_inches="tight")
    plt.close()
    print("  ✓ plots/08_exponential_duration_cdf.png")


# ── 9. Bayesian waterfall: prior → posterior by season & state ────────────────

def plot_bayesian_waterfall(matrices: dict):
    from step5_bayesian_update import bayesian_update

    WEATHER_SCORES = [0.2, 0.5, 0.8]
    season_list = list(SEASONS.keys())
    fig, axes = plt.subplots(len(STATE_NAMES), len(season_list),
                              figsize=(15, 22), sharey=False)
    fig.suptitle("Bayesian Update: Prior → Posterior P(Outage)\n"
                 "(Each bar group = weather extremity 0.2 / 0.5 / 0.8)",
                 fontsize=13, y=1.005)

    x = np.arange(len(WEATHER_SCORES))
    for row_i, state in enumerate(STATE_NAMES):
        prior = matrices[state]["steady"][2]
        for col_j, season in enumerate(season_list):
            ax = axes[row_i, col_j]
            posts = []
            for ws in WEATHER_SCORES:
                br = bayesian_update(prior, season, ws)
                posts.append(br["posterior"])

            bar_c = [GRID_STATE_COLORS[2] if p > 0.5 else
                     GRID_STATE_COLORS[1] if p > 0.25 else
                     GRID_STATE_COLORS[0] for p in posts]
            ax.bar(x, posts, color=bar_c, alpha=0.85, edgecolor="white")
            ax.axhline(prior, color="black", linestyle="--", linewidth=1.2,
                       label=f"Prior={prior:.3f}")
            ax.set_ylim(0, 1)
            ax.set_xticks(x)
            ax.set_xticklabels(["Low", "Med", "High"], fontsize=8)
            if col_j == 0:
                ax.set_ylabel(state.replace(" ", "\n"), fontsize=8,
                              color=STATE_COLORS[state], fontweight="bold")
            if row_i == 0:
                ax.set_title(season, fontsize=10, fontweight="bold")
            ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    plt.savefig("plots/09_bayesian_waterfall.png", bbox_inches="tight")
    plt.close()
    print("  ✓ plots/09_bayesian_waterfall.png")


# ── 10. Monthly outage heatmap calendar ───────────────────────────────────────

def plot_monthly_outage_heatmap(bayes_df: pd.DataFrame):
    """12×4 heatmap (Month × Year) of mean posterior outage probability per state."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Monthly Mean Outage Probability — Calendar Heatmap\n"
                 "(Rows = Months, Columns = Years)", fontsize=14)

    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    for ax, state in zip(axes.flatten(), STATE_NAMES):
        df_s = bayes_df[bayes_df["State"] == state].copy()
        df_s["Month"] = df_s["Date"].dt.month
        df_s["Year"]  = df_s["Date"].dt.year

        pivot = df_s.groupby(["Month", "Year"])["Posterior_Outage"] \
                    .mean().unstack(fill_value=np.nan)
        pivot.index = [months[m-1] for m in pivot.index]

        sns.heatmap(pivot, ax=ax, cmap=CMAP_OUTAGE, vmin=0, vmax=1,
                    annot=True, fmt=".2f", linewidths=0.4,
                    linecolor="white", cbar_kws={"label": "P(Outage)"})
        ax.set_title(state, fontweight="bold", color=STATE_COLORS[state])
        ax.set_xlabel("Year")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig("plots/10_monthly_outage_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  ✓ plots/10_monthly_outage_heatmap.png")


# ── 11. Season-wise outage rate comparison ────────────────────────────────────

def plot_seasonal_comparison(bayes_df: pd.DataFrame):
    """Bar chart per season, grouped by state."""
    fig, ax = plt.subplots(figsize=(14, 7))
    season_list = list(SEASONS.keys())
    n_states = len(STATE_NAMES)
    w = 0.22
    x = np.arange(n_states)
    season_hatch = [None, "///", "xxx"]
    season_edge  = ["#e74c3c", "#3498db", "#bdc3c7"]

    for si, season in enumerate(season_list):
        vals = []
        for state in STATE_NAMES:
            df_sub = bayes_df[(bayes_df["State"] == state) &
                              (bayes_df["Season"] == season)]
            vals.append(df_sub["Posterior_Outage"].mean() if len(df_sub) else 0)

        ax.bar(x + si * w, vals, w,
               label=season,
               color=season_edge[si], alpha=0.82,
               edgecolor="white",
               hatch=season_hatch[si])
        for xi, v in enumerate(vals):
            ax.text(x[xi] + si * w, v + 0.005, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=45)

    ax.set_xticks(x + w)
    ax.set_xticklabels([s.replace(" ", "\n") for s in STATE_NAMES], fontsize=10)
    ax.set_ylabel("Mean Posterior P(Outage)")
    ax.set_title("Seasonal Outage Probability Comparison — All States",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Season", fontsize=10)
    ax.set_ylim(0, 0.85)
    plt.tight_layout()
    plt.savefig("plots/11_seasonal_comparison.png", bbox_inches="tight")
    plt.close()
    print("  ✓ plots/11_seasonal_comparison.png")


# ── Master runner ─────────────────────────────────────────────────────────────

def main():
    print("Loading data and models...")

    with open("models/transition_matrices.pkl", "rb") as f:
        matrices = pickle.load(f)

    bayes_df  = pd.read_csv("models/bayesian_outage_series.csv",
                             parse_dates=["Date"])
    merged_df = pd.read_csv("data/merged_dataset.csv", parse_dates=["Date"])

    print("\nGenerating plots:")
    plot_transition_heatmaps(matrices)
    plot_seasonal_matrices(matrices)
    plot_markov_trajectories(matrices)
    plot_outage_timeseries(bayes_df)
    plot_weather_vs_outage(merged_df, bayes_df)
    plot_steady_state(matrices)
    plot_poisson_pmf(matrices)
    plot_exponential_cdf()
    plot_bayesian_waterfall(matrices)
    plot_monthly_outage_heatmap(bayes_df)
    plot_seasonal_comparison(bayes_df)

    print(f"\n✅  All plots saved to plots/")


if __name__ == "__main__":
    main()
