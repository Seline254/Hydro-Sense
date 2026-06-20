"""
test_numerical_methods.py
=========================
Pytest test suite for HydroSense-Kenya numerical methods.

Run from the project root:
    pytest tests/ -v
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.numerical_methods import (
    bisection, newton_raphson, secant,
    forward_difference, backward_difference, central_difference, all_differences,
    trapezoidal, simpson,
    gaussian_elimination
)


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def f_quadratic(x):
    """f(x) = x^2 - 4, roots at x = +2 and x = -2."""
    return x**2 - 4

def df_quadratic(x):
    """f'(x) = 2x."""
    return 2 * x

def f_linear(x):
    """f(x) = 3x - 9, root at x = 3."""
    return 3 * x - 9

def df_linear(x):
    return 3.0

def f_irrigation(I, S_t=25.0, R_t=3.2, ET_t=4.03, drain=0.18, fc=41.0, target=33.0):
    """Water balance root-finding function."""
    drainage = drain * max(0.0, S_t - fc)
    return S_t + R_t + I - ET_t - drainage - target

def df_irrigation(I):
    return 1.0


# ──────────────────────────────────────────────────────────────────────────────
# ROOT FINDING TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestBisection:
    def test_quadratic_positive_root(self):
        root, iters, history = bisection(f_quadratic, a=0, b=10)
        assert abs(root - 2.0) < 1e-5, f"Expected root ~2.0, got {root}"

    def test_quadratic_negative_root(self):
        root, iters, history = bisection(f_quadratic, a=-10, b=0)
        assert abs(root + 2.0) < 1e-5, f"Expected root ~-2.0, got {root}"

    def test_linear_root(self):
        root, iters, history = bisection(f_linear, a=0, b=10)
        assert abs(root - 3.0) < 1e-5

    def test_irrigation_root(self):
        root, iters, history = bisection(f_irrigation, a=0, b=50)
        assert root >= 0, "Irrigation cannot be negative"
        assert abs(f_irrigation(root)) < 1e-5

    def test_convergence_history_decreasing(self):
        _, _, history = bisection(f_quadratic, a=0, b=10)
        # Error should generally decrease
        assert history[-1] < history[0], "Error should decrease over iterations"

    def test_returns_three_values(self):
        result = bisection(f_quadratic, a=0, b=10)
        assert len(result) == 3

    def test_tolerance_respected(self):
        root, iters, history = bisection(f_quadratic, a=0, b=10, tol=1e-8)
        assert abs(f_quadratic(root)) < 1e-6

    def test_raises_on_same_sign(self):
        with pytest.raises(ValueError):
            bisection(f_quadratic, a=3, b=5)  # Both f(3)=5 and f(5)=21 positive

    def test_iterations_positive(self):
        _, iters, _ = bisection(f_quadratic, a=0, b=10)
        assert iters > 0


class TestNewtonRaphson:
    def test_quadratic_root(self):
        root, iters, history = newton_raphson(f_quadratic, df_quadratic, x0=3.0)
        assert abs(root - 2.0) < 1e-5

    def test_linear_root(self):
        root, iters, history = newton_raphson(f_linear, df_linear, x0=0.0)
        assert abs(root - 3.0) < 1e-5

    def test_irrigation_root(self):
        root, iters, history = newton_raphson(f_irrigation, df_irrigation, x0=10.0)
        assert root >= 0
        assert abs(f_irrigation(root)) < 1e-5

    def test_fast_convergence(self):
        # Newton-Raphson should converge quickly for smooth functions
        _, iters, _ = newton_raphson(f_quadratic, df_quadratic, x0=3.0)
        assert iters < 20, f"Newton-Raphson took too many iterations: {iters}"

    def test_negative_initial_guess(self):
        root, iters, history = newton_raphson(f_quadratic, df_quadratic, x0=-3.0)
        assert abs(root + 2.0) < 1e-5

    def test_returns_three_values(self):
        result = newton_raphson(f_quadratic, df_quadratic, x0=3.0)
        assert len(result) == 3

    def test_history_length_equals_iters(self):
        _, iters, history = newton_raphson(f_quadratic, df_quadratic, x0=3.0)
        assert len(history) == iters


class TestSecant:
    def test_quadratic_root(self):
        root, iters, history = secant(f_quadratic, x0=1.0, x1=3.0)
        assert abs(root - 2.0) < 1e-5

    def test_linear_root(self):
        root, iters, history = secant(f_linear, x0=0.0, x1=5.0)
        assert abs(root - 3.0) < 1e-5

    def test_irrigation_root(self):
        root, iters, history = secant(f_irrigation, x0=0.0, x1=20.0)
        assert root >= 0
        assert abs(f_irrigation(root)) < 1e-5

    def test_returns_three_values(self):
        result = secant(f_quadratic, x0=1.0, x1=3.0)
        assert len(result) == 3

    def test_convergence(self):
        _, iters, _ = secant(f_quadratic, x0=1.0, x1=3.0)
        assert iters < 30


class TestRootConsistency:
    """All three methods should agree to within tolerance."""
    def test_all_methods_agree_quadratic(self):
        root_b, _, _ = bisection(f_quadratic, 0, 10)
        root_n, _, _ = newton_raphson(f_quadratic, df_quadratic, 3.0)
        root_s, _, _ = secant(f_quadratic, 1.0, 3.0)
        assert np.isclose(root_b, root_n, atol=1e-4)
        assert np.isclose(root_b, root_s, atol=1e-4)

    def test_all_methods_agree_irrigation(self):
        root_b, _, _ = bisection(f_irrigation, 0, 50)
        root_n, _, _ = newton_raphson(f_irrigation, df_irrigation, 10.0)
        root_s, _, _ = secant(f_irrigation, 0.0, 20.0)
        assert np.isclose(root_b, root_n, atol=1e-3)
        assert np.isclose(root_b, root_s, atol=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# FINITE DIFFERENCE TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestFiniteDifferences:
    """Test derivatives against known analytical values."""

    def setup_method(self):
        # f(x) = x^2, f'(x) = 2x, on x = 0, 1, 2, 3, 4
        self.x  = np.arange(0, 5, 1.0)
        self.y  = self.x ** 2           # [0, 1, 4, 9, 16]
        self.dy = 2 * self.x            # [0, 2, 4, 6, 8]

    def test_forward_interior_accuracy(self):
        fd = forward_difference(self.y, h=1.0)
        # Forward diff at x=1: (4-1)/1 = 3  (true = 2, first-order error ~1)
        # For x^2, forward diff gives: 2x + h = 2*1 + 1 = 3
        assert abs(fd[1] - 3.0) < 1e-10

    def test_backward_interior_accuracy(self):
        bd = backward_difference(self.y, h=1.0)
        # Backward diff at x=2: (4-1)/1 = 3 (true = 4, first-order error)
        assert abs(bd[2] - 3.0) < 1e-10

    def test_central_more_accurate(self):
        cd = central_difference(self.y, h=1.0)
        # Central diff at x=2: (9-1)/2 = 4.0 (true = 4, second-order: exact for poly!)
        assert abs(cd[2] - 4.0) < 1e-10

    def test_forward_last_is_nan(self):
        fd = forward_difference(self.y, h=1.0)
        assert np.isnan(fd[-1])

    def test_backward_first_is_nan(self):
        bd = backward_difference(self.y, h=1.0)
        assert np.isnan(bd[0])

    def test_central_endpoints_are_nan(self):
        cd = central_difference(self.y, h=1.0)
        assert np.isnan(cd[0])
        assert np.isnan(cd[-1])

    def test_output_length_matches_input(self):
        n = 15
        data = np.random.randn(n)
        assert len(forward_difference(data))  == n
        assert len(backward_difference(data)) == n
        assert len(central_difference(data))  == n

    def test_all_differences_returns_dict(self):
        result = all_differences(self.y)
        assert set(result.keys()) == {'forward', 'backward', 'central'}

    def test_constant_array_zero_derivative(self):
        const = np.ones(10) * 5.0
        cd = central_difference(const, h=1.0)
        # All interior values should be zero
        assert np.allclose(cd[1:-1], 0.0)

    def test_linear_array_constant_derivative(self):
        linear = np.arange(0, 10, 1.0)  # y = x, dy/dx = 1
        cd = central_difference(linear, h=1.0)
        assert np.allclose(cd[1:-1], 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# NUMERICAL INTEGRATION TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestNumericalIntegration:
    """Test integration against known analytical values."""

    def test_trapezoidal_constant(self):
        # Integral of f=5 from 0 to 10 = 50
        y = np.ones(11) * 5.0  # 11 points, 10 intervals
        result = trapezoidal(y, h=1.0)
        assert abs(result - 50.0) < 1e-10

    def test_trapezoidal_linear(self):
        # Integral of f=x from 0 to 4 = 8
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = trapezoidal(y, h=1.0)
        assert abs(result - 8.0) < 1e-10

    def test_trapezoidal_quadratic_error(self):
        # Integral of x^2 from 0 to 4 = 64/3 ≈ 21.333
        x = np.linspace(0, 4, 100)
        y = x ** 2
        result = trapezoidal(y, h=x[1] - x[0])
        assert abs(result - 64/3) < 0.01   # Trapezoidal has O(h^2) error

    def test_simpson_constant(self):
        y = np.ones(11) * 5.0
        result = simpson(y, h=1.0)
        assert abs(result - 50.0) < 1e-10

    def test_simpson_linear(self):
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = simpson(y, h=1.0)
        assert abs(result - 8.0) < 1e-10

    def test_simpson_more_accurate_than_trapezoidal(self):
        # For x^2, Simpson is exact; trapezoidal has error
        x = np.linspace(0, 4, 11)
        y = x ** 2
        h = x[1] - x[0]
        trap = trapezoidal(y, h)
        simp = simpson(y, h)
        exact = 64 / 3
        assert abs(simp - exact) < abs(trap - exact)

    def test_trapezoidal_matches_numpy(self):
        np.random.seed(1)
        y = np.random.rand(20)
        our_result = trapezoidal(y, h=1.0)
        np_result  = np.trapz(y, dx=1.0)
        assert abs(our_result - np_result) < 1e-10

    def test_zero_array_gives_zero(self):
        y = np.zeros(10)
        assert trapezoidal(y) == 0.0
        assert simpson(y) == 0.0

    def test_nonnegative_deficit_integral(self):
        # Water deficit can't be negative
        deficit = np.maximum(0, np.random.randn(30))
        assert trapezoidal(deficit) >= 0
        assert simpson(deficit) >= 0


# ──────────────────────────────────────────────────────────────────────────────
# GAUSSIAN ELIMINATION TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestGaussianElimination:

    def test_2x2_system(self):
        # 2x + y = 5, x + 3y = 10 → x=1, y=3
        A = [[2, 1], [1, 3]]
        b = [5, 10]
        x = gaussian_elimination(A, b)
        assert abs(x[0] - 1.0) < 1e-10
        assert abs(x[1] - 3.0) < 1e-10

    def test_3x3_system(self):
        A = [[1, 0, -1.3],
             [0, 1,  1.0],
             [1, 1,  1.0]]
        b = [0.0, 33.0, 55.0]
        x = gaussian_elimination(A, b)
        x_np = np.linalg.solve(np.array(A, dtype=float), np.array(b, dtype=float))
        assert np.allclose(x, x_np, atol=1e-8)

    def test_identity_matrix(self):
        A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = [3.0, 7.0, 2.0]
        x = gaussian_elimination(A, b)
        assert np.allclose(x, b, atol=1e-10)

    def test_residual_near_zero(self):
        A = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
        b = [7, 10, 10]
        x = gaussian_elimination(A, b)
        A_np = np.array(A, dtype=float)
        b_np = np.array(b, dtype=float)
        residual = np.linalg.norm(A_np @ x - b_np)
        assert residual < 1e-8

    def test_matches_numpy_random(self):
        np.random.seed(0)
        A = np.random.randn(5, 5)
        b = np.random.randn(5)
        x_our = gaussian_elimination(A.tolist(), b.tolist())
        x_np  = np.linalg.solve(A, b)
        assert np.allclose(x_our, x_np, atol=1e-8)

    def test_singular_raises(self):
        A = [[1, 2], [2, 4]]  # Rows are linearly dependent - singular
        b = [3, 6]
        with pytest.raises(ValueError):
            gaussian_elimination(A, b)

    def test_negative_rhs(self):
        A = [[2, -1], [-1, 3]]
        b = [-1, 4]
        x = gaussian_elimination(A, b)
        x_np = np.linalg.solve(np.array(A, dtype=float), np.array(b, dtype=float))
        assert np.allclose(x, x_np, atol=1e-8)
