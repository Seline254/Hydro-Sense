"""
src/data_cleaning.py
---------------------
Reusable data cleaning functions for HydroSense-Kenya.
Used primarily in Level 4.

Functions
---------
load_datasets()
flag_outliers(df, col, low, high)
impute_missing(df, col, method)
clean_weather(df)
clean_soil(df)
cleaning_report(df, name)
"""

import pandas as pd
import numpy as np


# Loaders 

def load_datasets(data_dir="data/raw"):
    """
    Load all three HydroSense datasets.

    Returns
    -------
    weather : pd.DataFrame
    soil    : pd.DataFrame
    params  : pd.DataFrame
    """
    weather = pd.read_csv(f"{data_dir}/weather_daily.csv",
                          parse_dates=["date"], na_values=["NA", ""])
    soil    = pd.read_csv(f"{data_dir}/soil_sensor_data.csv",
                          parse_dates=["timestamp"], na_values=["NA", ""])
    params  = pd.read_csv(f"{data_dir}/crop_zone_parameters.csv",
                          na_values=["NA", ""])
    return weather, soil, params


#  Diagnostics 

def cleaning_report(df: pd.DataFrame, name: str) -> None:
    """Print a concise data-quality report."""
    print(f"\n{'='*55}")
    print(f"  Data Quality Report: {name}")
    print(f"{'='*55}")
    print(f"  Shape      : {df.shape}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  Missing    : none")
    else:
        print("  Missing values:")
        for col, n in missing.items():
            pct = 100 * n / len(df)
            print(f"    {col:30s} {n:4d} ({pct:.1f}%)")
    dups = df.duplicated().sum()
    print(f"  Duplicates : {dups}")


# Outlier Detection 

def flag_outliers(df: pd.DataFrame, col: str,
                  low: float, high: float) -> pd.Series:
    """
    Return a boolean mask where values in `col` fall outside [low, high].

    Parameters
    ----------
    df   : DataFrame to check
    col  : column name
    low  : minimum plausible value
    high : maximum plausible value

    Returns
    -------
    pd.Series of bool (True = outlier)
    """
    return (df[col] < low) | (df[col] > high)


#  Imputation 

def impute_missing(df: pd.DataFrame, col: str,
                   method: str = "median") -> pd.DataFrame:
    """
    Fill missing values in `col` using the chosen method.

    Parameters
    ----------
    df     : DataFrame (copied internally)
    col    : column name
    method : 'median', 'mean', or 'ffill'

    Returns
    -------
    DataFrame with column imputed
    """
    df = df.copy()
    before = df[col].isnull().sum()
    if method == "median":
        df[col] = df[col].fillna(df[col].median())
    elif method == "mean":
        df[col] = df[col].fillna(df[col].mean())
    elif method == "ffill":
        df[col] = df[col].ffill()
    else:
        raise ValueError(f"Unknown method: {method}")
    after = df[col].isnull().sum()
    print(f"  Imputed '{col}' ({method}): {before} → {after} missing")
    return df


#  Dataset-specific Cleaners 

def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the weather_daily dataset.

    Issues handled
    --------------
    - Missing rainfall_mm  → impute with median
    - Missing humidity_pct → impute with median
    - Outlier temperature  → flag rows where temp > 40°C (sensor fault)
    - Outlier rainfall     → flag rows where rainfall > 80 mm (extreme event, kept but flagged)
    """
    df = df.copy()
    print("\n[clean_weather]")

    # Missing values
    df = impute_missing(df, "rainfall_mm", method="median")
    df = impute_missing(df, "humidity_pct", method="median")

    # Outlier: temperature spike on 2026-03-14 (45.8°C — implausible for Kenya highlands)
    temp_mask = flag_outliers(df, "temperature_c", low=10, high=40)
    n = temp_mask.sum()
    if n:
        med_temp = df.loc[~temp_mask, "temperature_c"].median()
        df.loc[temp_mask, "temperature_c"] = med_temp
        print(f"  Replaced {n} temperature outlier(s) with median ({med_temp:.1f}°C)")

    # Flag extreme rainfall (85 mm on 2026-03-26 — kept as genuine event)
    rain_mask = flag_outliers(df, "rainfall_mm", low=0, high=80)
    df["rainfall_flag"] = rain_mask
    if rain_mask.sum():
        print(f"  Flagged {rain_mask.sum()} extreme rainfall event(s) (kept, flagged)")

    print(f"  ✓ Clean weather shape: {df.shape}")
    return df


def clean_soil(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the soil_sensor_data dataset.

    Issues handled
    --------------
    - Missing soil_moisture_pct (Zone_B, 2026-03-06) → ffill per zone
    - Outlier soil_moisture_pct = 8.5 (Zone_B, 2026-03-25) → sensor fault, replace with zone median
    - Outlier tank_level_liters = 9900 (Zone_C, 2026-03-14) → replace with zone median
    - pump_flow_lpm = 0 with sensor_status=CHECK → flag but keep
    """
    df = df.copy()
    print("\n[clean_soil]")

    # Per-zone forward-fill for missing soil moisture
    df["soil_moisture_pct"] = (
        df.groupby("zone_id")["soil_moisture_pct"]
        .transform(lambda s: s.ffill())
    )
    print("  Forward-filled missing soil_moisture_pct per zone")

    # Outlier: soil moisture 8.5% in Zone_B (below any agronomic minimum — sensor fault)
    sm_mask = flag_outliers(df, "soil_moisture_pct", low=10, high=60)
    if sm_mask.sum():
        zone_medians = df.groupby("zone_id")["soil_moisture_pct"].transform("median")
        df.loc[sm_mask, "soil_moisture_pct"] = zone_medians[sm_mask]
        print(f"  Replaced {sm_mask.sum()} soil moisture outlier(s) with zone median")

    # Outlier: tank level 9900L (impossible — sensor overflow flag)
    tank_mask = flag_outliers(df, "tank_level_liters", low=0, high=6000)
    if tank_mask.sum():
        zone_medians = df.groupby("zone_id")["tank_level_liters"].transform("median")
        df.loc[tank_mask, "tank_level_liters"] = zone_medians[tank_mask]
        print(f"  Replaced {tank_mask.sum()} tank level outlier(s) with zone median")

    # Flag CHECK rows (pump fault on Zone_B 2026-03-21)
    check_mask = df["sensor_status"] == "CHECK"
    df["sensor_fault_flag"] = check_mask
    print(f"  Flagged {check_mask.sum()} sensor fault row(s)")

    print(f"  ✓ Clean soil shape: {df.shape}")
    return df
