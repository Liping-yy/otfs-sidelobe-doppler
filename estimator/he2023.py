# -*- coding: utf-8 -*-
"""
Baseline: He 2023 two-stage fractional Doppler estimation
=========================================================

Reference: Xiaolin He, Pingzhi Fan, Qianli Wang,
"A Two-Stage Channel Estimation Algorithm for OTFS in Fractional
Doppler Channels," IEEE Communications Letters, vol. 27, no. 7,
pp. 1839-1843, July 2023.
DOI: 10.1109/LCOMM.2023.3270296

Algorithm flow:
  Stage 1 (coarse estimation):
    - For candidate fractional Doppler grid nu_grid, generate complete
      OTFS link pilot row template
    - Compute correlation (matched filtering) between received pilot
      row and template
    - Take the nu with maximum correlation peak as coarse estimate

  Stage 2 (refinement):
    - Using nu_coarse as initial value, use quasi-Newton iteration
      to minimize residual
    - Residual function: f(nu) = ||y - h_est(nu)*template(nu)||^2
    - BFGS approximate Hessian

Difference from MP/Prony:
  - MP/Prony: time-domain super-resolution (SVD/polynomial roots),
    requires a clean full pilot row
  - He 2023: template correlation + iterative optimization,
    requires complete OTFS link template
  - Under single-path, the three give different estimates (not equivalent)

Note: The original literature assumes fractional Doppler causes
DD-domain peak shift, but in our OTFS link the peak does not move
with nuf (nuf applied along n-axis). Therefore Stage 1 is modified
to full-link template matching (instead of basis function correlation).
This is an adaptation of the original algorithm; the core idea of
"coarse correlation + iterative refinement" is unchanged.
"""

import numpy as np
from scipy.optimize import minimize


# === Link template generation (lazy import to avoid circular dependency) ===

def _generate_pilot_template(nu, M, N, lp, kp, A):
    """
    Generate DD-domain pilot row template for given nuf.

    Flow: Pure pilot frame -> ISFFT -> Apply Doppler nu -> SFFT -> Take pilot row

    Args:
        nu: Candidate fractional Doppler
        M, N: OTFS frame dimension
        lp, kp: Pilot position
        A: Pilot amplitude

    Returns:
        template: Pilot row, shape (N,)
    """
    # Lazy import (to avoid circular dependency)
    import sys, os
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    from modem.otfs_modem import isfft_fft, sfft_fft, apply_fractional_doppler_tf
    from frame.frame_gen import generate_pure_frame

    X_dd = generate_pure_frame(M, N, lp, kp, A)
    X_tf = isfft_fft(X_dd)
    R_tf = apply_fractional_doppler_tf(X_tf, nu)
    Y_dd = sfft_fft(R_tf)
    return Y_dd[lp, :]


def _template_correlation(y_row, M, N, lp, kp, A, nu_grid):
    """
    Stage 1: Template matching correlation coarse estimation.

    For each candidate nu, generate complete OTFS link pilot row template,
    compute matched filtering output with received signal.

    Args:
        y_row: Pilot delay row, shape (N,)
        M, N: OTFS frame dimension
        lp, kp: Pilot position
        A: Pilot amplitude
        nu_grid: Candidate fractional Doppler grid

    Returns:
        nu_coarse: Coarse estimate
    """
    best_corr = -np.inf
    nu_coarse = 0.0

    for nu in nu_grid:
        tmpl = _generate_pilot_template(nu, M, N, lp, kp, A)
        corr = np.abs(np.vdot(y_row, tmpl))
        if corr > best_corr:
            best_corr = corr
            nu_coarse = nu

    return nu_coarse


def _residual_objective(nu, y_row, M, N, lp, kp, A):
    """
    Stage 2: Residual objective function f(nu) = ||y - h_est*template(nu)||^2

    Single-path model: model(nu) = h_est * template(nu)
    h_est obtained via least-squares fitting.

    Args:
        nu: Candidate fractional Doppler
        y_row: Pilot row received signal
        M, N, lp, kp, A: Link parameters

    Returns:
        residual: Scalar residual
    """
    tmpl = _generate_pilot_template(nu, M, N, lp, kp, A)

    # Least-squares fitting h_est = <y, tmpl> / <tmpl, tmpl>
    tt = np.vdot(tmpl, tmpl)
    if np.abs(tt) < 1e-30:
        return 1e30

    h_est = np.vdot(tmpl, y_row) / tt

    # Residual
    residual_vec = y_row - h_est * tmpl
    return float(np.real(np.vdot(residual_vec, residual_vec)))


def _quasi_newton_refine(nu_coarse, y_row, M, N, lp, kp, A):
    """
    Stage 2: quasi-Newton (BFGS) refinement of coarse estimate.

    Uses scipy.optimize.minimize (method='BFGS') to minimize residual.
    Constrains nu within (-0.5, 0.5).

    Args:
        nu_coarse: Coarse estimate
        y_row: Pilot row
        M, N, lp, kp, A: Link parameters

    Returns:
        nu_refined: Refined estimate
    """
    def obj(x):
        nu = float(x[0])
        if nu < -0.5:
            nu = -0.5
        elif nu > 0.5:
            nu = 0.5
        return _residual_objective(nu, y_row, M, N, lp, kp, A)

    result = minimize(
        obj,
        x0=[nu_coarse],
        method='BFGS',
        options={
            'maxiter': 30,
            'gtol': 1e-10,
        },
    )

    nu_refined = float(result.x[0])

    if nu_refined > 0.5:
        nu_refined = 0.5
    elif nu_refined < -0.5:
        nu_refined = -0.5

    return nu_refined


def estimate_nu_f(
    y_pilot_row: np.ndarray,
    kp: int,
    N: int,
    M: int = 64,
    A: float = 3.0,
    lp: int = 5,
    grid_size: int = 101,
    **kwargs,
) -> float:
    """
    He 2023 two-stage fractional Doppler estimation
    (adapted to our OTFS link).

    Interface unified with proposed/MP/Prony:
      estimate_nu_f(y_pilot_row, kp, N, **kwargs) -> float

    Stage 1: Template matching correlation coarse estimation (grid search)
    Stage 2: quasi-Newton iterative refinement

    Args:
        y_pilot_row: Pilot delay row y[lp, :], shape (N,)
        kp: Pilot Doppler column index
        N: Doppler dimension size
        M: Delay dimension size (default 64)
        A: Pilot amplitude (default 3.0)
        lp: Pilot delay row index (default 5)
        grid_size: Coarse estimation grid point count
            (default 101, covering [-0.5,0.5] with step 0.01)

    Returns:
        nu_f_hat: Fractional Doppler estimate in (-0.5, 0.5)
    """
    # Stage 1: Coarse estimation
    nu_grid = np.linspace(-0.5, 0.5, grid_size)
    nu_coarse = _template_correlation(
        y_pilot_row, M, N, lp, kp, A, nu_grid
    )

    # Stage 2: quasi-Newton refinement
    nu_f_hat = _quasi_newton_refine(
        nu_coarse, y_pilot_row, M, N, lp, kp, A
    )

    return nu_f_hat
