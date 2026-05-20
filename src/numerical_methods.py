"""
src/numerical_methods.py
-------------------------
Core numerical methods implemented from scratch for HydroSense-Kenya.
No SciPy for the core implementations — NumPy used only for verification.

Functions
---------
Root finding
    bisection(f, a, b, tol, max_iter)
    newton_raphson(f, df, x0, tol, max_iter)
    secant(f, x0, x1, tol, max_iter)
    compare_root_methods(f, df, a, b, x0, x1)

Differentiation
    forward_diff(f, x, h)
    backward_diff(f, x, h)
    central_diff(f, x, h)

Integration
    trapezoidal(y, x)
    simpsons(y, x)

Linear systems
    gaussian_elimination(A, b)
"""

import numpy as np
import time


#  Helpers 

def _check_convergence(label, converged, root, iterations, tol):
    status = "✓ converged" if converged else "✗ did not converge"
    print(f"  [{label}] {status} | root={root:.8f} | iters={iterations} | tol={tol}")


#  Root Finding 

def bisection(f, a: float, b: float,
              tol: float = 1e-6, max_iter: int = 200):
    """
    Bisection method for root finding.

    Parameters
    ----------
    f        : callable — target function f(x) = 0
    a, b     : initial bracket (f(a) and f(b) must have opposite signs)
    tol      : convergence tolerance
    max_iter : maximum iterations

    Returns
    -------
    root       : float
    iterations : int
    converged  : bool
    errors     : list of |f(mid)| at each step
    """
    if f(a) * f(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")

    errors = []
    for i in range(1, max_iter + 1):
        mid = (a + b) / 2.0
        err = abs(f(mid))
        errors.append(err)
        if err < tol or (b - a) / 2.0 < tol:
            return mid, i, True, errors
        if f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid
    return (a + b) / 2.0, max_iter, False, errors


def newton_raphson(f, df, x0: float,
                   tol: float = 1e-6, max_iter: int = 200):
    """
    Newton-Raphson method for root finding.

    Parameters
    ----------
    f        : callable — target function
    df       : callable — derivative of f
    x0       : initial guess
    tol      : convergence tolerance
    max_iter : maximum iterations

    Returns
    -------
    root, iterations, converged, errors
    """
    x = x0
    errors = []
    for i in range(1, max_iter + 1):
        fx = f(x)
        errors.append(abs(fx))
        if abs(fx) < tol:
            return x, i, True, errors
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError("Derivative is zero — Newton-Raphson failed.")
        x = x - fx / dfx
    return x, max_iter, False, errors


def secant(f, x0: float, x1: float,
           tol: float = 1e-6, max_iter: int = 200):
    """
    Secant method for root finding (no derivative needed).

    Parameters
    ----------
    f        : callable
    x0, x1  : two initial guesses
    tol      : convergence tolerance
    max_iter : maximum iterations

    Returns
    -------
    root, iterations, converged, errors
    """
    errors = []
    for i in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)
        errors.append(abs(f1))
        if abs(f1) < tol:
            return x1, i, True, errors
        if f1 - f0 == 0:
            raise ZeroDivisionError("Secant denominator is zero.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, x1 = x1, x2
    return x1, max_iter, False, errors


def compare_root_methods(f, df_func, a, b, x0, x1, tol=1e-6):
    """
    Run all three root-finding methods and return a comparison DataFrame.

    Parameters
    ----------
    f        : target function
    df_func  : derivative of f (for Newton-Raphson)
    a, b     : bisection bracket
    x0, x1  : secant / Newton initial guesses
    tol      : shared tolerance

    Returns
    -------
    pd.DataFrame with columns: method, root, iterations, converged, time_ms
    """
    import pandas as pd

    results = []
    for name, fn, args in [
        ("Bisection",       bisection,       (f, a, b, tol)),
        ("Newton-Raphson",  newton_raphson,  (f, df_func, x0, tol)),
        ("Secant",          secant,          (f, x0, x1, tol)),
    ]:
        t0 = time.perf_counter()
        root, iters, conv, errs = fn(*args)
        elapsed = (time.perf_counter() - t0) * 1000
        results.append({
            "Method":     name,
            "Root":       round(root, 8),
            "Iterations": iters,
            "Converged":  conv,
            "Final |f(x)|": round(errs[-1], 2e-10 if errs else 0),
            "Time (ms)":  round(elapsed, 4),
        })
    return pd.DataFrame(results)


# Numerical Differentiation 

def forward_diff(f, x: float, h: float = 1e-5) -> float:
    """Forward difference: f'(x) ≈ [f(x+h) - f(x)] / h"""
    return (f(x + h) - f(x)) / h


def backward_diff(f, x: float, h: float = 1e-5) -> float:
    """Backward difference: f'(x) ≈ [f(x) - f(x-h)] / h"""
    return (f(x) - f(x - h)) / h


def central_diff(f, x: float, h: float = 1e-5) -> float:
    """Central difference: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)  — O(h²) accurate"""
    return (f(x + h) - f(x - h)) / (2 * h)


# Numerical Integration 

def trapezoidal(y: np.ndarray, x: np.ndarray) -> float:
    """
    Trapezoidal rule integration.

    Parameters
    ----------
    y : array of function values
    x : array of corresponding x values (need not be uniform)

    Returns
    -------
    Approximate integral as float
    """
    return float(np.trapezoid(y, x))   


def simpsons(y: np.ndarray, x: np.ndarray) -> float:
    """
    Simpson's 1/3 rule integration (manual implementation).
    Requires an even number of intervals (odd number of points).

    Parameters
    ----------
    y : array of function values
    x : array of x values (assumed uniform spacing)

    Returns
    -------
    Approximate integral as float
    """
    n = len(y) - 1
    if n % 2 != 0:
        raise ValueError("Simpson's rule requires an even number of intervals "
                         "(odd number of points).")
    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]
    for i in range(1, n):
        result += (4 if i % 2 != 0 else 2) * y[i]
    return result * h / 3


#  Linear Systems 

def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    Implemented entirely from scratch.

    Parameters
    ----------
    A : (n x n) coefficient matrix
    b : (n,) right-hand side vector

    Returns
    -------
    x : (n,) solution vector
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot row
        max_row = col + int(np.argmax(np.abs(A[col:, col])))
        if A[max_row, col] == 0:
            raise ValueError("Matrix is singular — no unique solution.")
        # Swap rows
        A[[col, max_row]] = A[[max_row, col]]
        b[[col, max_row]] = b[[max_row, col]]
        # Eliminate below pivot
        for row in range(col + 1, n):
            factor = A[row, col] / A[col, col]
            A[row, col:] -= factor * A[col, col:]
            b[row]       -= factor * b[col]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x
