"""
test_data_cleaning.py
=====================
Tests that verify the data cleaning decisions made in Level 4
produce a valid, consistent cleaned dataset.

Run with: pytest tests/ -v
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest


def load_raw():
    weather = pd.read_csv('../data/raw/weather_daily.csv',   na_values=['NA', ''])
    soil    = pd.read_csv('../data/raw/soil_sensor_data.csv', na_values=['NA', ''])
    params  = pd.read_csv('../data/raw/crop_zone_parameters.csv')
    return weather, soil, params


def apply_cleaning(weather, soil, params):
    """Reproduce the exact cleaning steps from Level 4."""
    weather_clean = weather.copy()
    soil_clean    = soil.copy()

    # Weather cleaning
    weather_clean['rainfall_mm']   = weather_clean['rainfall_mm'].fillna(0.0)
    rolling_mean = weather_clean['temperature_c'].rolling(7, center=True, min_periods=3).mean()
    outlier_idx  = weather_clean[weather_clean['temperature_c'] > 40].index
    weather_clean.loc[outlier_idx, 'temperature_c'] = rolling_mean[outlier_idx].values
    mean_humidity = weather_clean['humidity_pct'].mean()
    weather_clean['humidity_pct'].fillna(mean_humidity, inplace=True)
    weather_clean['rainfall_flag'] = weather_clean['rainfall_mm'].apply(
        lambda x: 'EXTREME' if x > 60 else 'OK'
    )

    # Soil cleaning
    fault_mask = (soil_clean['zone_id'] == 'Zone_B') & (soil_clean['soil_moisture_pct'] < 10)
    soil_clean.loc[fault_mask, 'soil_moisture_pct'] = np.nan
    for zone in ['Zone_A', 'Zone_B', 'Zone_C']:
        mask = soil_clean['zone_id'] == zone
        soil_clean.loc[mask, 'soil_moisture_pct'] = (
            soil_clean.loc[mask, 'soil_moisture_pct'].interpolate(method='linear')
        )
    check_mask = soil_clean['sensor_status'] == 'CHECK'
    soil_clean.loc[check_mask, 'pump_flow_lpm'] = np.nan
    tank_fault = (soil_clean['zone_id'] == 'Zone_C') & (soil_clean['tank_level_liters'] > 9000)
    zone_c_mean = soil_clean[
        (soil_clean['zone_id'] == 'Zone_C') & (~tank_fault)
    ]['tank_level_liters'].mean()
    soil_clean.loc[tank_fault, 'tank_level_liters'] = round(zone_c_mean)
    for zone in ['Zone_A', 'Zone_B', 'Zone_C']:
        mask = soil_clean['zone_id'] == zone
        soil_clean.loc[mask, 'soil_moisture_pct'] = (
            soil_clean.loc[mask, 'soil_moisture_pct'].interpolate(method='linear')
        )

    return weather_clean, soil_clean


# ──────────────────────────────────────────────────────────────────────────────
# RAW DATA AUDIT TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestRawDataAudit:
    """Verify we correctly identify all the known issues in the raw data."""

    def setup_method(self):
        self.weather, self.soil, self.params = load_raw()

    def test_weather_has_missing_rainfall(self):
        assert self.weather['rainfall_mm'].isnull().sum() > 0, \
            "Raw weather should have at least one missing rainfall value"

    def test_weather_has_missing_humidity(self):
        assert self.weather['humidity_pct'].isnull().sum() > 0

    def test_weather_has_temperature_outlier(self):
        assert (self.weather['temperature_c'] > 40).sum() > 0, \
            "Raw weather should have temperature outlier > 40°C"

    def test_weather_has_extreme_rainfall(self):
        assert (self.weather['rainfall_mm'] > 60).sum() > 0, \
            "Raw weather should have at least one extreme rainfall event"

    def test_soil_has_missing_moisture(self):
        assert self.soil['soil_moisture_pct'].isnull().sum() > 0

    def test_soil_has_implausible_moisture(self):
        # Zone B moisture of 8.5% is far below Zone B minimum of 24%
        assert (self.soil['soil_moisture_pct'] < 10).sum() > 0

    def test_soil_has_check_status(self):
        assert (self.soil['sensor_status'] == 'CHECK').sum() > 0

    def test_soil_has_tank_spike(self):
        assert (self.soil['tank_level_liters'] > 9000).sum() > 0

    def test_weather_has_30_rows(self):
        assert len(self.weather) == 30

    def test_soil_has_90_rows(self):
        assert len(self.soil) == 90  # 30 days × 3 zones

    def test_params_has_3_zones(self):
        assert len(self.params) == 3


# ──────────────────────────────────────────────────────────────────────────────
# CLEANED DATA TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestCleanedWeather:

    def setup_method(self):
        weather_raw, soil_raw, params = load_raw()
        self.weather, self.soil = apply_cleaning(weather_raw, soil_raw, params)

    def test_no_missing_rainfall(self):
        assert self.weather['rainfall_mm'].isnull().sum() == 0

    def test_no_missing_humidity(self):
        assert self.weather['humidity_pct'].isnull().sum() == 0

    def test_temperature_outlier_removed(self):
        assert (self.weather['temperature_c'] > 40).sum() == 0, \
            "Cleaned weather should have no temperature > 40°C"

    def test_temperature_in_plausible_range(self):
        assert self.weather['temperature_c'].min() >= 15.0
        assert self.weather['temperature_c'].max() <= 40.0

    def test_rainfall_non_negative(self):
        assert (self.weather['rainfall_mm'] < 0).sum() == 0

    def test_rainfall_flag_column_exists(self):
        assert 'rainfall_flag' in self.weather.columns

    def test_extreme_rainfall_flagged(self):
        extreme_rows = self.weather[self.weather['rainfall_flag'] == 'EXTREME']
        assert len(extreme_rows) >= 1, "At least one row should be flagged EXTREME"

    def test_extreme_rainfall_retained(self):
        # We keep the 85mm event - verify it's still in the data
        assert self.weather['rainfall_mm'].max() > 60

    def test_humidity_in_valid_range(self):
        assert self.weather['humidity_pct'].min() >= 0
        assert self.weather['humidity_pct'].max() <= 100

    def test_still_30_rows(self):
        assert len(self.weather) == 30


class TestCleanedSoil:

    def setup_method(self):
        weather_raw, soil_raw, params = load_raw()
        self.weather, self.soil = apply_cleaning(weather_raw, soil_raw, params)
        self.params = params

    def test_no_missing_moisture_zone_a(self):
        zone_a = self.soil[self.soil['zone_id'] == 'Zone_A']
        assert zone_a['soil_moisture_pct'].isnull().sum() == 0

    def test_no_missing_moisture_zone_b(self):
        zone_b = self.soil[self.soil['zone_id'] == 'Zone_B']
        assert zone_b['soil_moisture_pct'].isnull().sum() == 0

    def test_no_implausible_moisture_zone_b(self):
        zone_b = self.soil[self.soil['zone_id'] == 'Zone_B']
        assert (zone_b['soil_moisture_pct'] < 10).sum() == 0

    def test_no_tank_spike(self):
        assert (self.soil['tank_level_liters'] > 9000).sum() == 0

    def test_moisture_within_field_capacity_zone_a(self):
        zp      = self.params[self.params['zone_id'] == 'Zone_A'].iloc[0]
        zone_a  = self.soil[self.soil['zone_id'] == 'Zone_A']
        assert (zone_a['soil_moisture_pct'] > zp['field_capacity_pct']).sum() == 0

    def test_moisture_non_negative_all_zones(self):
        assert (self.soil['soil_moisture_pct'] < 0).sum() == 0

    def test_still_90_rows(self):
        assert len(self.soil) == 90

    def test_check_status_pump_flow_is_nan(self):
        # Rows with sensor_status=CHECK should have pump_flow set to NaN
        check_rows = self.soil[self.soil['sensor_status'] == 'CHECK']
        assert check_rows['pump_flow_lpm'].isnull().all(), \
            "All CHECK-status rows should have pump_flow set to NaN"

    def test_zone_b_moisture_interpolated(self):
        # After cleaning, Zone B day 25 should be between day 24 and day 26 values
        zone_b = self.soil[self.soil['zone_id'] == 'Zone_B'].reset_index(drop=True)
        val_24 = zone_b.iloc[23]['soil_moisture_pct']  # day 24
        val_25 = zone_b.iloc[24]['soil_moisture_pct']  # day 25 (was 8.5)
        val_26 = zone_b.iloc[25]['soil_moisture_pct']  # day 26
        min_bound = min(val_24, val_26) - 1.0  # small tolerance
        max_bound = max(val_24, val_26) + 1.0
        assert min_bound <= val_25 <= max_bound, \
            f"Day 25 Zone B moisture ({val_25:.2f}) should be between day 24 ({val_24:.2f}) and day 26 ({val_26:.2f})"
