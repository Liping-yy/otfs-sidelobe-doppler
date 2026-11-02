# -*- coding: utf-8 -*-
"""
Baseline: Prony two-stage fractional Doppler estimation
=======================================================

Reference: arXiv:2506.17599 (2025)

Algorithm flow:
  1. Pilot row y[lp,:] -> IDFT -> DT domain sequence
  2. Prony method: construct linear prediction equations
  3. Solve characteristic polynomial roots
  4. Extract fractional Doppler from root phase

Similar to MP, Prony also assumes pure complex exponential
superposition. IDI contamination in engineering frame also
breaks model assumption. Prony noise resistance is slightly
weaker than MP (no SVD denoising step).
"""

import numpy as np


def _prony_core(y_dt: np.ndarray, n_paths: int = 1) -> float:
    """
    Prony core solver.

    Prony method: y[n] = Sum_p A_p * z_p^n
    where z_p = exp(j*2*pi*nu_p/N), obtained by finding roots
    of the linear prediction characteristic polynomial.

    Args:
        y_dt: DT domain received sequence
        n_paths: Number of signal paths

    Returns:
        nu_f_hat: Fractional Doppler estimate
    """
    N = len(y_dt)

    # Construct linear prediction matrix (Yule-Walker type)
    L = max(3, N // 3)

    # Solve linear prediction coefficients using least squares
    # [y[L], y[L+1], ..., y[N-1]] = [y[L-1]...y[0], y[L]...y[1], ...] @ a
    n_rows = N - L
    n_cols = L

    if n_rows < n_cols:
        # Insufficient data, degenerate to simple FFT peak
        spec = np.fft.fft(y_dt)
        peak = np.argmax(np.abs(spec))
        nu_est = peak / N if peak <= N // 2 else (peak - N) / N
        return float(nu_est)

    R = np.zeros((n_rows, n_cols), dtype=complex)
    for i in range(n_rows):
        for j in range(n_cols):
            R[i, j] = y_dt[i + L - 1 - j]

    r_vec = y_dt[L:L + n_rows]

    # Least-squares for linear prediction coefficients
    a_coef, *_ = np.linalg.lstsq(R, r_vec, rcond=None)

    # Find characteristic polynomial roots: z^L - a_1*z^{L-1} - ... - a_L = 0
    poly = np.zeros(L + 1, dtype=complex)
    poly[0] = 1.0
    for i in range(L):
        poly[i + 1] = -a_coef[i]
    roots = np.roots(poly)

    # Filter valid roots (near unit circle)
    valid = np.isfinite(roots) & (np.abs(roots) > 0.1) & (np.abs(roots) < 10.0)
    roots = roots[valid]
    if len(roots) == 0:
        return 0.0

    # Extract Doppler from root phase
    angles = np.angle(roots)
    nu_est_all = N * angles / (2.0 * np.pi)
    nu_est_all = ((nu_est_all + 0.5) % 1.0) - 0.5

    # Select root with maximum amplitude (least-squares fit)
    Z = np.zeros((N, len(roots)), dtype=complex)
    for p_idx, r in enumerate(roots):
        Z[:, p_idx] = r ** np.arange(N)
    amp_est, *_ = np.linalg.lstsq(Z, y_dt, rcond=None)
    best_idx = np.argmax(np.abs(amp_est))

    return nu_est_all[best_idx]


def estimate_nu_f(
    y_pilot_row: np.ndarray,
    kp: int,
    N: int,
    n_paths: int = 1,
    **kwargs,
) -> float:
    """
    Prony two-stage fractional Doppler estimation.

    Interface unified with proposed/sidetone_only.

    Args:
        y_pilot_row: Pilot delay row y[lp, :], shape (N,)
        kp: Pilot Doppler column (interface uniformity, Prony does not use)
        N: Doppler dimension size
        n_paths: Number of paths (single path = 1)

    Returns:
        nu_f_hat: Fractional Doppler estimate in (-0.5, 0.5)
    """
    y_dt = np.fft.ifft(y_pilot_row)
    return _prony_core(y_dt, n_paths=n_paths)
