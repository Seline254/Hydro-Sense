"""
src/visualization.py
---------------------
Scientific visualization functions for HydroSense-Kenya.
Produces all 5 required Level 4 plots plus simulation/MC plots for Level 5.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ZONE_COLORS = {"Zone_A": "#2196F3", "Zone_B": "#4CAF50", "Zone_C": "#FF9800"}
FIG_DPI = 130


#  Plot 1: Rainfall Time Series 

def plot_rainfall(weather_df: pd.DataFrame, save_path: str = None):
    """
    Line + bar chart of daily rainfall with flagged extreme events.
    """
    fig, ax = plt.subplots(figsize=(12, 4), dpi=FIG_DPI)
    ax.bar(weather_df["date"], weather_df["rainfall_mm"],
           color="#4FC3F7", alpha=0.7, label="Daily rainfall")

    # Highlight flagged extreme values if column exists
    if "rainfall_flag" in weather_df.columns:
        extremes = weather_df[weather_df["rainfall_flag"]]
        ax.bar(extremes["date"], extremes["rainfall_mm"],
               color="#E53935", alpha=0.9, label="Extreme event (flagged)")

    ax.set_title("Daily Rainfall — March 2026", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rainfall (mm)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


#  Plot 2: Soil Moisture by Zone 

def plot_soil_moisture_by_zone(soil_df: pd.DataFrame,
                                params_df: pd.DataFrame = None,
                                save_path: str = None):
    """
    Line chart of soil moisture per zone with optional stress threshold lines.
    """
    fig, ax = plt.subplots(figsize=(12, 5), dpi=FIG_DPI)

    for zone, grp in soil_df.groupby("zone_id"):
        grp = grp.sort_values("timestamp")
        ax.plot(grp["timestamp"], grp["soil_moisture_pct"],
                color=ZONE_COLORS.get(zone, "gray"),
                linewidth=2, label=zone, marker="o", markersize=3)

    # Minimum thresholds
    if params_df is not None:
        for _, row in params_df.iterrows():
            ax.axhline(row["min_moisture_pct"],
                       color=ZONE_COLORS.get(row["zone_id"], "gray"),
                       linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_title("Soil Moisture by Zone — March 2026", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Soil Moisture (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(title="Zone")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


#  Plot 3: ET Estimates 

def plot_et(weather_df: pd.DataFrame, et_values: np.ndarray,
            save_path: str = None):
    """
    Overlay ET and rainfall to show water demand vs supply.
    """
    fig, ax1 = plt.subplots(figsize=(12, 4), dpi=FIG_DPI)

    ax1.plot(weather_df["date"], et_values,
             color="#E53935", linewidth=2, label="ET (mm/day)")
    ax1.set_ylabel("Evapotranspiration (mm/day)", color="#E53935")

    ax2 = ax1.twinx()
    ax2.bar(weather_df["date"], weather_df["rainfall_mm"].fillna(0),
            alpha=0.3, color="#4FC3F7", label="Rainfall (mm)")
    ax2.set_ylabel("Rainfall (mm)", color="#4FC3F7")

    ax1.set_title("Daily ET vs Rainfall — March 2026", fontsize=14, fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


#  Plot 4: Cumulative Water Deficit 

def plot_water_deficit(dates, deficit_trap: np.ndarray,
                       deficit_simp: np.ndarray = None,
                       save_path: str = None):
    """
    Cumulative water deficit estimated by trapezoidal (and optionally Simpson's) rule.
    """
    fig, ax = plt.subplots(figsize=(12, 4), dpi=FIG_DPI)
    ax.fill_between(dates, deficit_trap, alpha=0.3, color="#FF7043")
    ax.plot(dates, deficit_trap, color="#FF7043", linewidth=2,
            label="Trapezoidal estimate")

    if deficit_simp is not None:
        ax.plot(dates, deficit_simp, color="#7B1FA2", linewidth=1.5,
                linestyle="--", label="Simpson's estimate")

    ax.set_title("Cumulative Water Deficit — March 2026",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Deficit (mm)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


#  Plot 5: Correlation Heatmap 

def plot_correlation_heatmap(weather_df: pd.DataFrame, save_path: str = None):
    """
    Heatmap of correlations between weather variables.
    """
    import matplotlib.colors as mcolors

    cols = ["rainfall_mm", "temperature_c", "humidity_pct",
            "wind_speed_mps", "solar_index"]
    corr = weather_df[cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6), dpi=FIG_DPI)
    cax = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")

    ax.set_title("Weather Variable Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


#  Plot 6: Euler vs RK4 Simulation 

def plot_simulation_comparison(S_euler: np.ndarray, S_rk4: np.ndarray,
                                zone_id: str, min_moisture: float,
                                target_moisture: float,
                                save_path: str = None):
    """
    Compare Euler and RK4 soil moisture simulations for one zone.
    """
    days = np.arange(len(S_euler))
    fig, ax = plt.subplots(figsize=(12, 5), dpi=FIG_DPI)

    ax.plot(days, S_euler, label="Euler", color="#1565C0", linewidth=2)
    ax.plot(days, S_rk4,   label="RK4",   color="#2E7D32", linewidth=2,
            linestyle="--")
    ax.axhline(min_moisture, color="#C62828", linestyle=":", linewidth=1.5,
               label=f"Min threshold ({min_moisture}%)")
    ax.axhline(target_moisture, color="#F57F17", linestyle=":", linewidth=1.5,
               label=f"Target ({target_moisture}%)")

    ax.set_title(f"Soil Moisture Simulation — {zone_id}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Day")
    ax.set_ylabel("Soil Moisture (%)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


#  Plot 7: Monte Carlo Distribution 

def plot_monte_carlo(mc_results: np.ndarray, min_moisture: float,
                     zone_id: str, save_path: str = None):
    """
    Distribution of final day soil moisture across Monte Carlo scenarios.
    """
    finals = mc_results[:, -1]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=FIG_DPI)

    ax.hist(finals, bins=50, color="#7986CB", edgecolor="white", linewidth=0.4)
    ax.axvline(min_moisture, color="#E53935", linestyle="--",
               linewidth=2, label=f"Min threshold ({min_moisture}%)")
    ax.axvline(np.percentile(finals, 5), color="#FF8F00", linestyle="--",
               linewidth=1.5, label="5th percentile (worst-case)")

    shortage_pct = 100 * (finals < min_moisture).mean()
    ax.set_title(
        f"Monte Carlo Soil Moisture Distribution — {zone_id}\n"
        f"P(shortage) = {shortage_pct:.1f}%",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Final Day Soil Moisture (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
