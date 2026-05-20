"""
src/simulation.py
------------------
Simulation engine for HydroSense-Kenya.
Used in Level 5.

Functions
---------
euler_soil_moisture(S0, rainfall, ET, irrigation, drainage_coef, field_capacity, n_days)
runge_kutta_soil_moisture(S0, rainfall, ET, irrigation, drainage_coef, field_capacity, n_days)
monte_carlo_rainfall(mean_rain, std_rain, n_days, n_scenarios, seed)
analyse_monte_carlo(mc_results, min_moisture, target_moisture)
"""

import numpy as np
import pandas as pd


#  Water Balance Dynamics 

def _drainage(S: float, field_capacity: float, coef: float) -> float:
    """Drainage term: only occurs when soil exceeds field capacity."""
    return coef * max(0.0, S - field_capacity)


def _dSdt(S, R, I, ET, field_capacity, drainage_coef):
    """Rate of change of soil moisture per day."""
    D = _drainage(S, field_capacity, drainage_coef)
    return R + I - ET - D


#  Euler Method 

def euler_soil_moisture(S0: float,
                        rainfall: np.ndarray,
                        ET: np.ndarray,
                        irrigation: np.ndarray,
                        drainage_coef: float,
                        field_capacity: float,
                        n_days: int = 30) -> np.ndarray:
    """
    Simulate soil moisture over n_days using the forward Euler method.

    S(t+1) = S(t) + dS/dt * dt   where dt = 1 day

    Parameters
    ----------
    S0            : initial soil moisture (%)
    rainfall      : array of daily rainfall (mm → rescaled to %)
    ET            : array of daily evapotranspiration
    irrigation    : array of daily irrigation applied
    drainage_coef : zone-specific drainage coefficient
    field_capacity: zone field capacity (%)
    n_days        : number of simulation days

    Returns
    -------
    S : np.ndarray of soil moisture values, shape (n_days + 1,)
    """
    S = np.zeros(n_days + 1)
    S[0] = S0
    for t in range(n_days):
        dS = _dSdt(S[t], rainfall[t], irrigation[t], ET[t],
                   field_capacity, drainage_coef)
        S[t + 1] = max(0.0, S[t] + dS)
    return S


#  4th-Order Runge-Kutta 

def runge_kutta_soil_moisture(S0: float,
                               rainfall: np.ndarray,
                               ET: np.ndarray,
                               irrigation: np.ndarray,
                               drainage_coef: float,
                               field_capacity: float,
                               n_days: int = 30) -> np.ndarray:
    """
    Simulate soil moisture using the 4th-order Runge-Kutta method (RK4).

    Parameters — same as euler_soil_moisture.

    Returns
    -------
    S : np.ndarray of shape (n_days + 1,)
    """
    S = np.zeros(n_days + 1)
    S[0] = S0
    dt = 1.0   # 1 day

    for t in range(n_days):
        R, I, E = rainfall[t], irrigation[t], ET[t]
        f = lambda s: _dSdt(s, R, I, E, field_capacity, drainage_coef)

        k1 = f(S[t])
        k2 = f(S[t] + 0.5 * dt * k1)
        k3 = f(S[t] + 0.5 * dt * k2)
        k4 = f(S[t] + dt * k3)

        S[t + 1] = max(0.0, S[t] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4))

    return S


#  Monte Carlo Rainfall 

def monte_carlo_rainfall(mean_rain: float,
                         std_rain: float,
                         n_days: int = 30,
                         n_scenarios: int = 1000,
                         seed: int = 42) -> np.ndarray:
    """
    Generate stochastic rainfall scenarios.

    Parameters
    ----------
    mean_rain   : mean daily rainfall (mm)
    std_rain    : standard deviation of daily rainfall
    n_days      : days per scenario
    n_scenarios : number of Monte Carlo scenarios
    seed        : random seed for reproducibility

    Returns
    -------
    rainfall_mc : np.ndarray of shape (n_scenarios, n_days)
                  All values clipped to >= 0
    """
    rng = np.random.default_rng(seed)
    rainfall_mc = rng.normal(loc=mean_rain, scale=std_rain,
                             size=(n_scenarios, n_days))
    return np.clip(rainfall_mc, 0, None)


def analyse_monte_carlo(mc_soil_results: np.ndarray,
                        min_moisture: float,
                        target_moisture: float) -> dict:
    """
    Analyse Monte Carlo simulation outcomes.

    Parameters
    ----------
    mc_soil_results : np.ndarray, shape (n_scenarios, n_days+1)
                      Soil moisture trace per scenario
    min_moisture    : minimum acceptable moisture threshold
    target_moisture : target moisture level

    Returns
    -------
    dict with keys:
        p_shortage     : probability any day falls below min_moisture
        p_over_irrig   : probability any day exceeds target by >10%
        expected_deficit : mean total water deficit (mm)
        worst_case_min : 5th percentile of minimum moisture across scenarios
    """
    # Shortage: any day below minimum
    any_shortage = (mc_soil_results < min_moisture).any(axis=1)
    p_shortage = any_shortage.mean()

    # Over-irrigation: any day more than 10% above target
    any_over = (mc_soil_results > target_moisture * 1.10).any(axis=1)
    p_over = any_over.mean()

    # Total deficit per scenario
    deficit = np.maximum(0, min_moisture - mc_soil_results).sum(axis=1)
    expected_deficit = deficit.mean()

    # Worst-case: 5th percentile of scenario minimums
    scenario_mins = mc_soil_results.min(axis=1)
    worst_case_min = np.percentile(scenario_mins, 5)

    return {
        "p_shortage":       round(p_shortage, 4),
        "p_over_irrigation": round(p_over, 4),
        "expected_deficit": round(expected_deficit, 4),
        "worst_case_min_moisture": round(worst_case_min, 4),
    }
