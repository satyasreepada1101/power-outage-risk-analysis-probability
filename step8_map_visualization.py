# ==============================================================================
# METHODOLOGY STEP: Step 8 — Map Visualization
# DESCRIPTION: Generates an interactive Folium choropleth-style map of India
#              showing per-state outage risk metrics. Includes:
#              • Circle markers sized by outage probability
#              • Colour-coded by risk level (green/orange/red)
#              • Popup cards with full prediction summary per state
#              • Season toggle buttons (Summer/Monsoon/Winter/Overall)
# OUTPUT: plots/india_power_outage_risk_map.html
# ==============================================================================

import folium
import pandas as pd
import numpy as np
import pickle
import os
from folium.plugins import FloatImage, GroupedLayerControl
from config import STATE_NAMES, STATES

os.makedirs("plots", exist_ok=True)


# ── State geo-centres (for circle placement) ──────────────────────────────────
STATE_CENTERS = {
    "Tamil Nadu":  [10.9,  78.7],
    "Karnataka":   [15.3,  75.7],
    "Rajasthan":   [27.0,  74.2],
    "Haryana":     [29.2,  76.1],
    "Assam":       [26.2,  92.9],
    "West Bengal": [23.4,  87.9],
}


def risk_color(prob: float) -> str:
    if prob >= 0.50:
        return "#e74c3c"   # red
    elif prob >= 0.30:
        return "#e67e22"   # orange
    elif prob >= 0.15:
        return "#f1c40f"   # yellow
    else:
        return "#2ecc71"   # green


def risk_label(prob: float) -> str:
    if prob >= 0.50: return "HIGH"
    elif prob >= 0.30: return "MODERATE-HIGH"
    elif prob >= 0.15: return "MODERATE"
    else: return "LOW"


def build_popup_html(state: str, seasonal: dict, overall: dict,
                     steady: np.ndarray, season_data: dict) -> str:
    """Build a styled HTML popup card for each state marker."""
    p_out  = overall["posterior"]
    p_str  = overall["prior"]
    color  = risk_color(p_out)

    season_rows = ""
    for s_name, s_val in seasonal.items():
        sc = risk_color(s_val)
        season_rows += (
            f"<tr><td style='padding:3px 8px'>{s_name}</td>"
            f"<td style='padding:3px 8px;color:{sc};font-weight:bold'>"
            f"{s_val:.1%}</td></tr>"
        )

    n_days = season_data.get("n_global", "—")
    pi     = f"{steady[0]:.3f} / {steady[1]:.3f} / {steady[2]:.3f}"

    html = f"""
    <div style='font-family:Arial,sans-serif;width:290px;font-size:12px'>
      <div style='background:{color};padding:10px;border-radius:6px 6px 0 0;'>
        <span style='color:white;font-size:15px;font-weight:bold'>{state}</span>
        <span style='color:white;float:right;font-size:18px;font-weight:bold'>
          {risk_label(p_out)}
        </span>
      </div>
      <div style='padding:10px;border:1px solid #ddd;border-top:none;
                  border-radius:0 0 6px 6px'>
        <p style='margin:4px 0'><b>Overall Outage Risk:</b>
          <span style='color:{color};font-size:14px;font-weight:bold'>
            {p_out:.1%}
          </span>
        </p>
        <p style='margin:4px 0'><b>Steady-State (Normal / Stressed / Outage):</b><br/>
          {pi}
        </p>
        <p style='margin:4px 0'><b>Seasonal Breakdown:</b></p>
        <table style='border-collapse:collapse;width:100%'>
          <tr style='background:#f5f5f5'>
            <th style='padding:3px 8px;text-align:left'>Season</th>
            <th style='padding:3px 8px;text-align:left'>P(Outage)</th>
          </tr>
          {season_rows}
        </table>
        <p style='margin:6px 0 2px;color:#666;font-size:10px'>
          Based on {n_days} days of data (2021–2024) · Markov + Bayesian model
        </p>
      </div>
    </div>
    """
    return html


def make_map(matrices: dict, bayes_df: pd.DataFrame) -> folium.Map:
    m = folium.Map(
        location=[22.0, 82.0],
        zoom_start=5,
        tiles="CartoDB positron",
    )

    # ── Title box ─────────────────────────────────────────────────────────────
    title_html = """
    <div style='position:fixed;top:12px;left:55px;z-index:9999;
                background:white;padding:12px 18px;border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.25);font-family:Arial;'>
      <b style='font-size:15px'>⚡ India Power Outage Risk Map</b><br>
      <span style='font-size:11px;color:#555'>
        Bayesian-Markov Model · States: 6 · Data: 2021–2024
      </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = """
    <div style='position:fixed;bottom:30px;left:30px;z-index:9999;
                background:white;padding:10px 14px;border-radius:8px;
                box-shadow:0 2px 6px rgba(0,0,0,0.2);font-family:Arial;
                font-size:11px;'>
      <b>Risk Level</b><br>
      <span style='color:#e74c3c'>&#9679;</span> HIGH (≥50%)<br>
      <span style='color:#e67e22'>&#9679;</span> MOD-HIGH (30–50%)<br>
      <span style='color:#f1c40f'>&#9679;</span> MODERATE (15–30%)<br>
      <span style='color:#2ecc71'>&#9679;</span> LOW (&lt;15%)<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── One feature group per season + overall ─────────────────────────────────
    seasons = ["Overall", "Summer", "Monsoon", "Winter"]
    groups  = {}
    for s in seasons:
        groups[s] = folium.FeatureGroup(name=s, show=(s == "Overall"))
        m.add_child(groups[s])

    # Pre-compute seasonal probabilities from bayes_df
    seasonal_probs: dict[str, dict[str, float]] = {}
    for state in STATE_NAMES:
        df_s = bayes_df[bayes_df["State"] == state]
        seasonal_probs[state] = {}
        for season in ["Summer", "Monsoon", "Winter"]:
            df_sea = df_s[df_s["Season"] == season]
            seasonal_probs[state][season] = (
                df_sea["Posterior_Outage"].mean()
                if len(df_sea) > 0 else 0.0
            )
        seasonal_probs[state]["Overall"] = df_s["Posterior_Outage"].mean()

    for state in STATE_NAMES:
        lat, lon = STATE_CENTERS[state]
        mats     = matrices[state]
        steady   = mats["steady"]

        for season in seasons:
            p_out = seasonal_probs[state][season]
            color = risk_color(p_out)
            radius = 20 + p_out * 45   # size proportional to risk

            # Season-level prior/posterior for popup
            overall = {"prior": steady[2], "posterior": p_out}

            popup_html = build_popup_html(
                state, seasonal_probs[state], overall, steady, mats
            )

            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.55,
                opacity=0.9,
                popup=folium.Popup(popup_html, max_width=310),
                tooltip=f"{state}: P(Outage)={p_out:.1%} [{season}]",
            ).add_to(groups[season])

            # State label
            folium.Marker(
                location=[lat + 0.5, lon],
                icon=folium.DivIcon(
                    html=f"<div style='font-size:9px;font-family:Arial;"
                         f"font-weight:bold;color:#333;white-space:nowrap;"
                         f"text-shadow:1px 1px 2px white'>{state}</div>",
                    icon_size=(120, 20),
                    icon_anchor=(60, 0),
                ),
            ).add_to(groups[season])

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main():
    print("Loading models and data...")

    with open("models/transition_matrices.pkl", "rb") as f:
        matrices = pickle.load(f)

    bayes_df = pd.read_csv("models/bayesian_outage_series.csv",
                            parse_dates=["Date"])

    print("Building map...")
    m = make_map(matrices, bayes_df)

    out_path = "plots/india_power_outage_risk_map.html"
    m.save(out_path)
    print(f"\n✅  Saved → {out_path}")
    print("   Open in browser · Click each state bubble for details")
    print("   Use layer toggle (top-right) to switch seasons")


if __name__ == "__main__":
    main()
