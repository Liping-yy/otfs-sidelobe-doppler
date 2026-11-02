# -*- coding: utf-8 -*-
"""
Baseline: Matrix Pencil (MP) fractional Doppler estimation
==========================================================

Reference: Liang et al., "Matrix Pencil-Based Channel Estimation
Algorithm for OTFS," IEEE Comm. Lett., 2024.

Algorithm flow:
  1. Pilot row y[lp,:] -> IDFT -> DT domain sequence
  2. Construct Hankel matrix H
  3. Generalized eigenvalue decomposition: A*v = lambda*B*v
  4. Extract fractional Doppler from eigenvalue phase

MP requires a clean full pilot row (model assumes input is pure complex
exponential superposition). IDI contamination in engineering frame
breaks model assumption, leading to performance degradation.
"""

import numpy as np
from scipy.linalg import eig as scipy_eig


def _build_hankel(y_dt: np.ndarray, L: int) -> np.ndarray:
    """Construct Hankel matrix."""
    N = len(y_dt)
    n_rows = N - L
    n_cols = L + 1
    H = np.zeros((n_rows, n_cols), dtype=complex)
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i + j
            if idx < N:
                H[i, j] = y_dt[idx]
    return H


def _mp_core(y_dt: np.ndarray, n_paths: int = 1) -> float:
    """
    Matrix Pencil core solver.

    Args:
        y_dt: DT domain received sequence
        n_paths: Number of paths to estimate (single path = 1)

    Returns:
        nu_f_hat: Fractional Doppler estimate
    """
    N = len(y_dt)
    L = max(3, N // 3)
    H = _build_hankel(y_dt, L)

    H1 = H[:, :-1]
    H2 = H[:, 1:]

    # Generalized eigenvalue A*v = lambda*B*v
    A = H1.conj().T @ H2
    B = H1.conj().T @ H1

    eigvals, _ = scipy_eig(A, b=B)

    # Filter outliers
    valid_mask = np.isfinite(eigvals)
    eigvals = eigvals[valid_mask]
    if len(eigvals) == 0:
        return 0.0

    # Extract Doppler from eigenvalue phase
    angles = np.angle(eigvals)
    nu_est_all = N * angles / (2.0 * np.pi)
    nu_est_all = ((nu_est_all + 0.5) % 1.0) - 0.5

    # Select root with maximum amplitude
    Z = np.zeros((N, len(eigvals)), dtype=complex)
    for p_idx, lam in enumerate(eigvals):
        Z[:, p_idx] = lam ** np.arange(N)
    amp_est = np.linalg.pinv(Z) @ y_dt
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
    Matrix Pencil fractional Doppler estimation.

    Interface unified with proposed/sidetone_only:
      Input y_pilot_row + kp + N -> output nu_f_hat

    Note: MP does not use kp and delta_k_list; it uses the full row
    y[lp,:] for estimation. kp is kept only for interface uniformity
    and does not participate in computation.

    Args:
        y_pilot_row: Pilot delay row y[lp, :], shape (N,)
        kp: Pilot Doppler column (interface uniformity, MP does not use)
        N: Doppler dimension size
        n_paths: Number of paths (single path = 1)

    Returns:
        nu_f_hat: Fractional Doppler estimate in (-0.5, 0.5)
    """
    # Pilot row -> IDFT -> DT domain
    y_dt = np.fft.ifft(y_pilot_row)
    return _mp_core(y_dt, n_paths=n_paths)
