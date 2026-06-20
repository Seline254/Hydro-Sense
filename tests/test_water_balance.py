"""
test_water_balance.py
=====================
Pytest tests for the water balance and ET functions from Level 1.

Tests verify:
  - ET formula correctness
  - Water balance physics (conservation, bounds, drainage)
  - Edge cases (zero rainfall, oversaturated soil, zero ET)

Run with: pytest tests/ -v
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


# ── Inline the functions (they live in the notebook; we reproduce here for testing) ──

def compute_et(temperature, wind_speed, solar_index, humidity):
    et = 0.12 * temperature + 0.35 * wind_speed + 2.4 * solar_index - 0.025 * humidity
    return max(0.0, et)

def water_balance(S_t, rainfall, irrigation, ET, drainage_coeff, field_capacity):
    drainage = drainage_coeff * max(0.0, S_t - field_capacity)
    S_next   = S_t + rainfall + irrigation - ET - drainage
    return max(0.0, min(S_next, field_capacity))

def is_stressed(soil_moisture, min_moisture):
    return soil_moisture < min_moisture

def irrigation_needed(S_t, target_moisture):
    return max(0.0, target_moisture - S_t)


# ──────────────────────────────────────────────────────────────────────────────
# ET TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeET:

    def test_day1_value(self):
        # Manual calculation: 0.12*23.8 + 0.35*2.28 + 2.4*0.78 - 0.025*69.7
        expected = 0.12*23.8 + 0.35*2.28 + 2.4*0.78 - 0.025*69.7
        result   = compute_et(23.8, 2.28, 0.78, 69.7)
        assert abs(result - expected) < 1e-10

    def test_never_negative(self):
        # With extreme humidity, formula could go negative - must return 0
        result = compute_et(temperature=10.0, wind_speed=0.1,
                            solar_index=0.1, humidity=99.0)
        assert result >= 0.0

    def test_zero_inputs_gives_zero(self):
        result = compute_et(0.0, 0.0, 0.0, 0.0)
        assert result == 0.0

    def test_increases_with_temperature(self):
        et_low  = compute_et(20.0, 2.0, 0.6, 60.0)
        et_high = compute_et(35.0, 2.0, 0.6, 60.0)
        assert et_high > et_low

    def test_increases_with_wind(self):
        et_calm  = compute_et(25.0, 0.5, 0.6, 60.0)
        et_windy = compute_et(25.0, 5.0, 0.6, 60.0)
        assert et_windy > et_calm

    def test_increases_with_solar(self):
        et_cloudy = compute_et(25.0, 2.0, 0.2, 60.0)
        et_sunny  = compute_et(25.0, 2.0, 0.9, 60.0)
        assert et_sunny > et_cloudy

    def test_decreases_with_humidity(self):
        et_dry  = compute_et(25.0, 2.0, 0.6, 40.0)
        et_moist = compute_et(25.0, 2.0, 0.6, 80.0)
        assert et_dry > et_moist

    def test_returns_float(self):
        result = compute_et(25.0, 2.0, 0.7, 65.0)
        assert isinstance(result, float)

    def test_high_solar_index(self):
        # Solar index max is 1.0
        result = compute_et(30.0, 3.0, 1.0, 50.0)
        expected = max(0, 0.12*30 + 0.35*3 + 2.4*1.0 - 0.025*50)
        assert abs(result - expected) < 1e-10

    def test_typical_range(self):
        # For Kenyan conditions, ET should typically be between 1 and 8 mm/day
        result = compute_et(26.0, 2.0, 0.7, 65.0)
        assert 1.0 <= result <= 8.0, f"ET={result} outside typical Kenyan range [1, 8]"


# ──────────────────────────────────────────────────────────────────────────────
# WATER BALANCE TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestWaterBalance:

    def test_basic_calculation(self):
        # S=30, R=5, I=0, ET=4, drain=0.18, fc=41
        # drainage = 0.18 * max(0, 30-41) = 0
        # S_next = 30 + 5 + 0 - 4 - 0 = 31
        result = water_balance(30.0, 5.0, 0.0, 4.0, 0.18, 41.0)
        assert abs(result - 31.0) < 1e-10

    def test_never_exceeds_field_capacity(self):
        # Heavy rain should not exceed field capacity
        result = water_balance(38.0, 50.0, 10.0, 0.5, 0.18, 41.0)
        assert result <= 41.0

    def test_never_negative(self):
        # Extreme ET and no rain should not give negative moisture
        result = water_balance(5.0, 0.0, 0.0, 20.0, 0.18, 41.0)
        assert result >= 0.0

    def test_drainage_activates_above_field_capacity(self):
        # S_t = 45 > fc = 41: drainage should kick in
        # drainage = 0.18 * (45 - 41) = 0.72
        # S_next = 45 + 0 + 0 - 0 - 0.72 = 44.28, clipped to 41
        result = water_balance(45.0, 0.0, 0.0, 0.0, 0.18, 41.0)
        assert result == 41.0  # Clipped to field capacity

    def test_no_drainage_below_field_capacity(self):
        # S_t = 30 < fc = 41: no drainage
        S_t = 30.0; R = 2.0; I = 0.0; ET = 3.0; dc = 0.18; fc = 41.0
        expected = S_t + R + I - ET
        result   = water_balance(S_t, R, I, ET, dc, fc)
        assert abs(result - expected) < 1e-10

    def test_zero_et_zero_rain_stable(self):
        # With nothing happening, moisture should stay the same
        result = water_balance(30.0, 0.0, 0.0, 0.0, 0.0, 41.0)
        assert abs(result - 30.0) < 1e-10

    def test_irrigation_raises_moisture(self):
        without = water_balance(25.0, 0.0, 0.0, 3.0, 0.18, 41.0)
        with_irr = water_balance(25.0, 0.0, 10.0, 3.0, 0.18, 41.0)
        assert with_irr > without

    def test_result_in_valid_range(self):
        np.random.seed(5)
        for _ in range(100):
            S   = np.random.uniform(0, 41)
            R   = np.random.uniform(0, 20)
            I   = np.random.uniform(0, 15)
            ET  = np.random.uniform(0, 8)
            dc  = np.random.uniform(0, 0.3)
            fc  = 41.0
            result = water_balance(S, R, I, ET, dc, fc)
            assert 0.0 <= result <= fc, f"Result {result} out of bounds [0, {fc}]"

    def test_zone_a_day1(self):
        # Verify zone A, day 1 water balance
        ET = compute_et(23.8, 2.28, 0.78, 69.7)
        result = water_balance(33.2, 3.2, 0.0, ET, 0.18, 41.0)
        assert 0 <= result <= 41.0


# ──────────────────────────────────────────────────────────────────────────────
# STRESS AND IRRIGATION TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestStressAndIrrigationNeeded:

    def test_stressed_below_min(self):
        assert is_stressed(19.0, 20.0) == True

    def test_not_stressed_above_min(self):
        assert is_stressed(25.0, 20.0) == False

    def test_exactly_at_min_not_stressed(self):
        assert is_stressed(20.0, 20.0) == False

    def test_irrigation_needed_positive(self):
        result = irrigation_needed(25.0, 33.0)
        assert abs(result - 8.0) < 1e-10

    def test_irrigation_needed_zero_when_above_target(self):
        result = irrigation_needed(36.0, 33.0)
        assert result == 0.0

    def test_irrigation_needed_zero_at_target(self):
        result = irrigation_needed(33.0, 33.0)
        assert result == 0.0

    def test_irrigation_needed_nonnegative(self):
        for S in [10, 20, 30, 35, 40]:
            assert irrigation_needed(float(S), 33.0) >= 0.0
