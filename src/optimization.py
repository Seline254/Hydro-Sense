"""
src/optimization.py
--------------------
Irrigation schedule optimization for HydroSense-Kenya.
Used in Level 5.

Goal: minimize total irrigation water used over n_days
      subject to: soil_moisture(t) >= min_moisture for all t and zones
"""

import numpy as np
from src.simulation import euler_soil_moisture


def compute_et(T: float, W: float, Solar: float, H: float) -> float:
    """
    Simplified empirical evapotranspiration estimate (from project brief).

    ET = max(0, 0.12*T + 0.35*W + 2.4*Solar - 0.025*H)

    Parameters
    ----------
    T     : temperature (°C)
    W     : wind speed (m/s)
    Solar : solar index (0–1)
    H     : humidity (%)

    Returns
    -------
    ET    : float, mm/day
    """
    return max(0.0, 0.12 * T + 0.35 * W + 2.4 * Solar - 0.025 * H)


def water_balance(S: float, R: float, I: float,
                  ET: float, D: float) -> float:
    """
    Single-step water balance equation.

    S(t+1) = S(t) + R(t) + I(t) - ET(t) - D(t)

    Parameters
    ----------
    S  : current soil moisture
    R  : rainfall
    I  : irrigation applied
    ET : evapotranspiration
    D  : drainage

    Returns
    -------
    S_next : float
    """
    return S + R + I - ET - D


def greedy_irrigation_schedule(rainfall: np.ndarray,
                                ET: np.ndarray,
                                S0: float,
                                min_moisture: float,
                                target_moisture: float,
                                field_capacity: float,
                                drainage_coef: float,
                                n_days: int = 30) -> np.ndarray:
    """
    Greedy algorithm: irrigate only when soil moisture would drop below
    min_moisture, applying just enough to reach target_moisture.

    This minimizes water use while guaranteeing no crop stress.

    Parameters
    ----------
    rainfall      : daily rainfall array (length n_days)
    ET            : daily ET array (length n_days)
    S0            : initial soil moisture
    min_moisture  : lower threshold (crop stress if breached)
    target_moisture: irrigation target
    field_capacity : zone field capacity
    drainage_coef  : drainage coefficient
    n_days         : planning horizon

    Returns
    -------
    irrigation : np.ndarray of shape (n_days,) — daily irrigation amounts
    """
    irrigation = np.zeros(n_days)
    S = S0

    for t in range(n_days):
        D = drainage_coef * max(0.0, S - field_capacity)
        S_next_no_irr = S + rainfall[t] - ET[t] - D

        if S_next_no_irr < min_moisture:
            # Irrigate just enough to reach target
            irrigation[t] = max(0.0, target_moisture - S_next_no_irr)

        D_updated = drainage_coef * max(0.0, S - field_capacity)
        S = max(0.0, S + rainfall[t] + irrigation[t] - ET[t] - D_updated)

    return irrigation


def optimise_all_zones(weather_df, params_df, n_days: int = 30) -> dict:
    """
    Run the greedy optimizer for all three crop zones.

    Parameters
    ----------
    weather_df : cleaned weather DataFrame
    params_df  : crop_zone_parameters DataFrame
    n_days     : simulation horizon

    Returns
    -------
    dict: zone_id → {'irrigation': np.ndarray, 'total_water': float}
    """
    import numpy as np

    # Build ET array from weather
    weather = weather_df.head(n_days).copy()
    ET_arr = np.array([
        compute_et(row.temperature_c, row.wind_speed_mps,
                   row.solar_index, row.humidity_pct)
        for _, row in weather.iterrows()
    ])
    R_arr = weather["rainfall_mm"].fillna(0).values[:n_days]

    results = {}
    for _, zone in params_df.iterrows():
        zid = zone["zone_id"]
        irr = greedy_irrigation_schedule(
            rainfall=R_arr,
            ET=ET_arr,
            S0=zone["target_moisture_pct"],
            min_moisture=zone["min_moisture_pct"],
            target_moisture=zone["target_moisture_pct"],
            field_capacity=zone["field_capacity_pct"],
            drainage_coef=zone["drainage_coefficient"],
            n_days=n_days,
        )
        results[zid] = {
            "irrigation": irr,
            "total_water_mm": round(irr.sum(), 2),
        }
        print(f"  {zid}: total irrigation = {irr.sum():.2f} mm over {n_days} days")

    return results
