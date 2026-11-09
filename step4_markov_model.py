# ==============================================================================
# METHODOLOGY STEP: Steps 3 & 6 — Markov Chain Model
# DESCRIPTION: Builds 3×3 transition probability matrices from consecutive-day
#              state pairs.  Produces:
#                • One global matrix per Indian state
#                • Season-specific matrices (Summer / Monsoon / Winter)
#                • Steady-state probability vectors
#                • Short-term state probability forecasts  S(n) = S(0) × P^n
# OUTPUT: models/transition_matrices.pkl
#         models/steady_state.csv
# ==============================================================================

import numpy as np
import pandas as pd
import pickle
import os
from config import STATE_NAMES, SEASONS, STATES_CSV

os.makedirs("models", exist_ok=True)

N_STATES = 3   # 0=Normal, 1=Stressed, 2=Outage


# ── Core Markov utilities ─────────────────────────────────────────────────────

def count_transitions(states: np.ndarray) -> np.ndarray:
    """Count transition frequencies from consecutive day pairs."""
    counts = np.zeros((N_STATES, N_STATES), dtype=int)
    for i in range(len(states) - 1):
        s_now  = int(states[i])
        s_next = int(states[i + 1])
        if 0 <= s_now < N_STATES and 0 <= s_next < N_STATES:
            counts[s_now, s_next] += 1
    return counts


def counts_to_matrix(counts: np.ndarray) -> np.ndarray:
    """
    Convert raw counts to row-stochastic probability matrix.
    Rows with zero observations are assigned uniform distribution (1/3 each)
    to avoid absorbing states and ensure ergodicity.
    """
    P = counts.astype(float)
    row_sums = P.sum(axis=1, keepdims=True)
    # Rows with no observations → uniform
    zero_rows = (row_sums.flatten() == 0)
    P[zero_rows] = 1.0 / N_STATES
    row_sums[zero_rows] = 1.0
    P /= row_sums
    return P


def steady_state(P: np.ndarray, tol: float = 1e-10, max_iter: int = 10_000) -> np.ndarray:
    """
    Compute steady-state distribution by power iteration.
    π P = π,  sum(π) = 1
    """
    pi = np.ones(N_STATES) / N_STATES
    for _ in range(max_iter):
        pi_new = pi @ P
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        pi = pi_new
    return pi


def predict_state_vector(s0: np.ndarray, P: np.ndarray, n_steps: int) -> np.ndarray:
    """
    S(n) = S(0) × P^n
    Returns shape (n_steps+1, N_STATES) — rows are time steps 0…n.
    """
    trajectory = np.zeros((n_steps + 1, N_STATES))
    trajectory[0] = s0
    s = s0.copy()
    for i in range(1, n_steps + 1):
        s = s @ P
        trajectory[i] = s
    return trajectory


# ── Build matrices ────────────────────────────────────────────────────────────

def build_matrices_for_state(df_state: pd.DataFrame) -> dict:
    """
    Returns a dict with:
      'global'  : overall 3×3 transition matrix
      'counts'  : raw transition counts (global)
      'seasonal': {season_name: {'matrix', 'counts', 'n_days'}}
      'steady'  : steady-state vector for global matrix
    """
    result = {}

    # Global matrix
    seq_global           = df_state.sort_values("Date")["Grid_State"].values
    counts_global        = count_transitions(seq_global)
    P_global             = counts_to_matrix(counts_global)
    result["global"]     = P_global
    result["counts"]     = counts_global
    result["steady"]     = steady_state(P_global)
    result["n_global"]   = len(seq_global)

    # Seasonal matrices
    result["seasonal"] = {}
    for season in SEASONS:
        df_s  = df_state[df_state["Season"] == season].sort_values("Date")
        seq_s = df_s["Grid_State"].values

        if len(seq_s) < 10:
            # Fall back to global for seasons with very little data
            result["seasonal"][season] = {
                "matrix": P_global.copy(),
                "counts": counts_global.copy(),
                "n_days": len(seq_s),
                "fallback": True,
            }
        else:
            counts_s = count_transitions(seq_s)
            P_s      = counts_to_matrix(counts_s)
            result["seasonal"][season] = {
                "matrix": P_s,
                "counts": counts_s,
                "n_days": len(seq_s),
                "fallback": False,
            }

    return result


def main():
    print("Loading classified states...")
    df = pd.read_csv(STATES_CSV, parse_dates=["Date"])
    print(f"  Shape: {df.shape}")

    all_matrices = {}
    steady_rows  = []

    print("\n─── Building Transition Matrices ───")
    for state in STATE_NAMES:
        df_s = df[df["State"] == state]
        if df_s.empty:
            print(f"  ⚠ {state}: no data, skipping")
            continue

        mats = build_matrices_for_state(df_s)
        all_matrices[state] = mats

        print(f"\n  {state} ({mats['n_global']} days)")
        print(f"  Global transition matrix P:")
        labels = ["Norm", "Strs", "Out"]
        header = "         " + "  ".join(f"{l:>7}" for l in labels)
        print(header)
        for i, row_label in enumerate(labels):
            row = "  ".join(f"{mats['global'][i,j]:7.4f}" for j in range(N_STATES))
            print(f"  {row_label:>7}: {row}")

        print(f"  Steady-state π: {mats['steady']}")

        steady_rows.append({
            "State":       state,
            "π_Normal":    round(mats["steady"][0], 4),
            "π_Stressed":  round(mats["steady"][1], 4),
            "π_Outage":    round(mats["steady"][2], 4),
        })

    # Save pickle for downstream use
    with open("models/transition_matrices.pkl", "wb") as f:
        pickle.dump(all_matrices, f)
    print("\n✅  Saved → models/transition_matrices.pkl")

    # Save steady-state table
    pd.DataFrame(steady_rows).to_csv("models/steady_state.csv", index=False)
    print("✅  Saved → models/steady_state.csv")

    # ── Sample prediction demo ─────────────────────────────────────────────────
    demo_state = "Tamil Nadu"
    if demo_state in all_matrices:
        P  = all_matrices[demo_state]["global"]
        s0 = np.array([1.0, 0.0, 0.0])   # start in Normal state
        traj = predict_state_vector(s0, P, n_steps=30)
        print(f"\n── 30-day prediction demo ({demo_state}, starting Normal) ──")
        print(f"  Day 0  : Normal={traj[0,0]:.3f}  Stressed={traj[0,1]:.3f}  Outage={traj[0,2]:.3f}")
        print(f"  Day 7  : Normal={traj[7,0]:.3f}  Stressed={traj[7,1]:.3f}  Outage={traj[7,2]:.3f}")
        print(f"  Day 30 : Normal={traj[30,0]:.3f}  Stressed={traj[30,1]:.3f}  Outage={traj[30,2]:.3f}")

    return all_matrices


if __name__ == "__main__":
    main()
