"""
test_simulation.py
==================
Pytest tests for the simulation and optimisation module.

Tests verify:
  - Euler and RK4 produce physically valid results
  - Monte Carlo generates correct output structure
  - Optimiser reduces stress days and saves water
  - Edge cases: zero rain, fully saturated, all-dry month

Run with: pytest tests/ -v
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.simulation import euler_simulate, rk4_simulate, monte_carlo_simulate, optimise_irrigation


# ── Shared test fixtures ──────────────────────────────────────────────────────

N_DAYS        = 30
FIELD_CAP     = 41.0
DRAIN_COEFF   = 0.18
MIN_MOISTURE  = 22.0
TARGET        = 33.0
S0            = 30.0

np.random.seed(0)
RAIN_FLAT   = np.full(N_DAYS, 3.0)       # constant 3mm/day
ET_FLAT     = np.full(N_DAYS, 4.0)       # constant 4mm/day (net loss)
RAIN_ZERO   = np.zeros(N_DAYS)           # drought scenario
ET_ZERO     = np.zeros(N_DAYS)
IRR_ZERO    = np.zeros(N_DAYS)
IRR_CONST   = np.full(N_DAYS, 2.0)       # 2mm/day irrigation


# ──────────────────────────────────────────────────────────────────────────────
# EULER TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestEulerSimulate:

    def test_output_length(self):
        result = euler_simulate(S0, RAIN_FLAT, ET_FLAT, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        assert len(result) == N_DAYS + 1, "Output should have n_days + 1 values (including S0)"

    def test_first_value_is_S0(self):
        result = euler_simulate(S0, RAIN_FLAT, ET_FLAT, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        assert result[0] == S0

    def test_never_negative(self):
        result = euler_simulate(S0, RAIN_ZERO, ET_FLAT, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        assert np.all(result >= 0.0), "Soil moisture cannot be negative"

    def test_never_exceeds_field_capacity(self):
        # Heavy rain every day should not exceed field capacity
        heavy_rain = np.full(N_DAYS, 50.0)
        result     = euler_simulate(S0, heavy_rain, ET_ZERO, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        assert np.all(result <= FIELD_CAP), "Soil moisture cannot exceed field capacity"

    def test_net_loss_decreases_moisture(self):
        # ET > rainfall → moisture should generally decline
        result = euler_simulate(S0, RAIN_ZERO, ET_FLAT, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        assert result[-1] < result[0], "Net loss scenario should reduce moisture"

    def test_net_gain_increases_moisture(self):
        # Irrigation > ET and no drainage → moisture should increase (until field cap)
        big_irr = np.full(N_DAYS, 10.0)
        result  = euler_simulate(10.0, RAIN_ZERO, ET_ZERO, big_irr, DRAIN_COEFF, FIELD_CAP)
        assert result[-1] >= result[0]

    def test_zero_everything_stable(self):
        # With no rain, no ET, no irrigation, no drainage: moisture stays constant
        result = euler_simulate(S0, RAIN_ZERO, ET_ZERO, IRR_ZERO, 0.0, FIELD_CAP)
        assert np.allclose(result, S0), "Moisture should be stable with no inputs or losses"

    def test_returns_numpy_array(self):
        result = euler_simulate(S0, RAIN_FLAT, ET_FLAT, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        assert isinstance(result, np.ndarray)

    def test_drought_reaches_zero(self):
        # With no rain, high ET, no irrigation: should eventually hit 0
        high_et = np.full(N_DAYS, 8.0)
        result  = euler_simulate(20.0, RAIN_ZERO, high_et, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        assert result[-1] == 0.0

    def test_drainage_prevents_oversaturation(self):
        # Start above field capacity - drainage should bring it down
        result = euler_simulate(50.0, RAIN_ZERO, ET_ZERO, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        assert result[1] <= FIELD_CAP


# ──────────────────────────────────────────────────────────────────────────────
# RK4 TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestRK4Simulate:

    def test_output_length(self):
        result = rk4_simulate(S0, RAIN_FLAT, ET_FLAT, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        assert len(result) == N_DAYS + 1

    def test_first_value_is_S0(self):
        result = rk4_simulate(S0, RAIN_FLAT, ET_FLAT, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        assert result[0] == S0

    def test_never_negative(self):
        result = rk4_simulate(S0, RAIN_ZERO, ET_FLAT, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        assert np.all(result >= 0.0)

    def test_never_exceeds_field_capacity(self):
        heavy_rain = np.full(N_DAYS, 50.0)
        result     = rk4_simulate(S0, heavy_rain, ET_ZERO, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        assert np.all(result <= FIELD_CAP)

    def test_net_loss_decreases_moisture(self):
        result = rk4_simulate(S0, RAIN_ZERO, ET_FLAT, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        assert result[-1] < result[0]

    def test_zero_everything_stable(self):
        result = rk4_simulate(S0, RAIN_ZERO, ET_ZERO, IRR_ZERO, 0.0, FIELD_CAP)
        assert np.allclose(result, S0)


class TestEulerVsRK4:
    """Both methods should agree closely for smooth daily timestep."""

    def test_close_agreement_smooth_input(self):
        euler = euler_simulate(S0, RAIN_FLAT, ET_FLAT, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        rk4   = rk4_simulate(S0, RAIN_FLAT, ET_FLAT, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        max_diff = np.abs(euler - rk4).max()
        assert max_diff < 0.5, f"Euler and RK4 differ by {max_diff:.4f}% - too large"

    def test_same_initial_value(self):
        euler = euler_simulate(S0, RAIN_FLAT, ET_FLAT, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        rk4   = rk4_simulate(S0, RAIN_FLAT, ET_FLAT, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        assert euler[0] == rk4[0] == S0

    def test_same_trend_direction(self):
        # Both should show same direction: moisture goes up or down together
        euler = euler_simulate(S0, RAIN_FLAT, ET_FLAT, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        rk4   = rk4_simulate(S0, RAIN_FLAT, ET_FLAT, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        assert (euler[-1] < euler[0]) == (rk4[-1] < rk4[0])


# ──────────────────────────────────────────────────────────────────────────────
# MONTE CARLO TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestMonteCarlo:

    def setup_method(self):
        self.result = monte_carlo_simulate(
            S0=S0, base_rainfall=RAIN_FLAT, ET_series=ET_FLAT,
            irrigation_series=IRR_ZERO, drainage_coeff=DRAIN_COEFF,
            field_capacity=FIELD_CAP, min_moisture=MIN_MOISTURE,
            n_scenarios=200, noise_std=0.3, seed=42
        )

    def test_returns_dict(self):
        assert isinstance(self.result, dict)

    def test_required_keys(self):
        required = {'scenarios', 'mean', 'p5', 'p95', 'shortage_prob', 'expected_demand', 'worst_case_demand'}
        assert required.issubset(self.result.keys())

    def test_scenarios_shape(self):
        shape = self.result['scenarios'].shape
        assert shape == (200, N_DAYS + 1), f"Expected (200, 31), got {shape}"

    def test_mean_length(self):
        assert len(self.result['mean']) == N_DAYS + 1

    def test_p5_less_than_mean(self):
        assert np.all(self.result['p5'] <= self.result['mean'] + 0.01)

    def test_p95_greater_than_mean(self):
        assert np.all(self.result['p95'] >= self.result['mean'] - 0.01)

    def test_shortage_prob_between_0_and_1(self):
        p = self.result['shortage_prob']
        assert 0.0 <= p <= 1.0, f"Shortage probability {p} is out of [0, 1]"

    def test_scenarios_non_negative(self):
        assert np.all(self.result['scenarios'] >= 0.0)

    def test_scenarios_within_field_capacity(self):
        assert np.all(self.result['scenarios'] <= FIELD_CAP)

    def test_reproducible_with_same_seed(self):
        result2 = monte_carlo_simulate(
            S0=S0, base_rainfall=RAIN_FLAT, ET_series=ET_FLAT,
            irrigation_series=IRR_ZERO, drainage_coeff=DRAIN_COEFF,
            field_capacity=FIELD_CAP, min_moisture=MIN_MOISTURE,
            n_scenarios=200, noise_std=0.3, seed=42
        )
        assert np.allclose(self.result['mean'], result2['mean'])

    def test_more_scenarios_gives_similar_mean(self):
        r500 = monte_carlo_simulate(
            S0=S0, base_rainfall=RAIN_FLAT, ET_series=ET_FLAT,
            irrigation_series=IRR_ZERO, drainage_coeff=DRAIN_COEFF,
            field_capacity=FIELD_CAP, min_moisture=MIN_MOISTURE,
            n_scenarios=500, noise_std=0.3, seed=99
        )
        # Means should be in the same ballpark (within 2%)
        assert abs(r500['mean'][-1] - self.result['mean'][-1]) < 3.0

    def test_drought_raises_shortage_prob(self):
        # With zero rainfall and high ET, shortage should be very likely
        r_drought = monte_carlo_simulate(
            S0=S0, base_rainfall=RAIN_ZERO, ET_series=ET_FLAT,
            irrigation_series=IRR_ZERO, drainage_coeff=DRAIN_COEFF,
            field_capacity=FIELD_CAP, min_moisture=MIN_MOISTURE,
            n_scenarios=100, noise_std=0.1, seed=0
        )
        assert r_drought['shortage_prob'] > 0.5, "Drought should give high shortage probability"


# ──────────────────────────────────────────────────────────────────────────────
# OPTIMISATION TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestOptimiseIrrigation:

    def setup_method(self):
        self.result = optimise_irrigation(
            S0=S0, rainfall_series=RAIN_FLAT, ET_series=ET_FLAT,
            drainage_coeff=DRAIN_COEFF, field_capacity=FIELD_CAP,
            min_moisture=MIN_MOISTURE, target_moisture=TARGET,
            max_daily_irrigation=20.0
        )

    def test_returns_dict(self):
        assert isinstance(self.result, dict)

    def test_required_keys(self):
        required = {'schedule', 'moisture', 'total_water', 'stress_days', 'savings_vs_max'}
        assert required.issubset(self.result.keys())

    def test_schedule_length(self):
        assert len(self.result['schedule']) == N_DAYS

    def test_moisture_length(self):
        assert len(self.result['moisture']) == N_DAYS + 1

    def test_schedule_non_negative(self):
        assert np.all(self.result['schedule'] >= 0.0)

    def test_schedule_within_max(self):
        assert np.all(self.result['schedule'] <= 20.0)

    def test_total_water_matches_schedule(self):
        assert abs(self.result['total_water'] - self.result['schedule'].sum()) < 1e-8

    def test_moisture_non_negative(self):
        assert np.all(self.result['moisture'] >= 0.0)

    def test_moisture_within_field_capacity(self):
        assert np.all(self.result['moisture'] <= FIELD_CAP)

    def test_optimised_uses_less_water_than_always_on(self):
        # Always irrigate to target uses more water than greedy minimum
        assert self.result['savings_vs_max'] >= 0.0, "Optimised should save water vs always-on"

    def test_fewer_stress_days_than_no_irrigation(self):
        no_irr = euler_simulate(S0, RAIN_FLAT, ET_FLAT, IRR_ZERO, DRAIN_COEFF, FIELD_CAP)
        stress_no_irr = (no_irr[1:] < MIN_MOISTURE).sum()
        assert self.result['stress_days'] <= stress_no_irr

    def test_drought_scenario_uses_irrigation(self):
        result_drought = optimise_irrigation(
            S0=S0, rainfall_series=RAIN_ZERO, ET_series=ET_FLAT,
            drainage_coeff=DRAIN_COEFF, field_capacity=FIELD_CAP,
            min_moisture=MIN_MOISTURE, target_moisture=TARGET,
            max_daily_irrigation=20.0
        )
        assert result_drought['total_water'] > 0.0, "Drought scenario should require irrigation"
