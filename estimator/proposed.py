# -*- coding: utf-8 -*-
"""
Core estimator: sidelobe + peak self-multiplication closed-form
fractional Doppler estimation
===============================================================

Corresponding derivation docx Section 4
(sidelobe + peak self-multiplication weighted fusion closed-form estimator).

Core formula (derivation docx Section 4.3):
  psi_bar = arg{ y[lp,kp]^2 + Sum_{Dk=1..K}(-y[lp,kp+Dk]*y[lp,kp-Dk]) }
  nu_f_hat = psi_bar * N / (2*pi*(N-1))

Features: no grid search, no iteration, no singularity, no phase wrapping,
no cyclic ambiguity.
"""

import numpy as np


def estimate_nu_f(
    y_pilot_row: np.ndarray,
    kp: int,
    N: int,
    delta_k_list: tuple = (1, 2, 3, 4, 5, 6, 7),
) -> float:
    """
    Sidelobe + peak self-multiplication weighted fusion closed-form estimator
    (derivation docx Section 4.3 final expression).

    Steps:
      1. Peak self-multiplication (Dk=0): Z0 = y[kp]^2, compensate pi
      2. Sidelobe pairs (Dk>=1): Z_Dk = -y[kp+Dk]*y[kp-Dk]
      3. Weighted fusion: psi_bar = arg{ Sum w_Dk * exp(j*psi_Dk) }
      4. Closed-form solution: nu_f_hat = psi_bar * N / (2*pi*(N-1))

    Simplified equivalent form (derivation docx Section 4.3):
      psi_bar = arg{ y[kp]^2 + Sum(-y_plus * y_minus) }
      (Weights automatically reflected in magnitude of complex multiplication)

    Args:
        y_pilot_row: Pilot delay row y[lp, :], shape (N,)
        kp: Pilot Doppler column index
        N: Doppler dimension size
        delta_k_list: Sidelobe offset list
            (Dk=0 handled by peak self-multiplication, not in this list)

    Returns:
        nu_f_hat: Fractional Doppler estimate in (-0.5, 0.5)
    """
    # === Peak self-multiplication term (Dk=0) ===
    y_peak = y_pilot_row[kp]
    # psi(0) = arg{-y0^2} - pi, after compensation = 2*pi*nuf*(N-1)/N
    prod0 = -y_peak * y_peak
    psi0 = np.angle(prod0) - np.pi
    # Wrap to (-pi, pi]
    psi0 = ((psi0 + np.pi) % (2.0 * np.pi)) - np.pi
    w0 = np.abs(prod0)

    # === Sidelobe pair terms (Dk>=1) ===
    sum_vec = w0 * np.exp(1j * psi0)  # Peak term

    for dk in delta_k_list:
        k_plus = (kp + dk) % N
        k_minus = (kp - dk) % N
        yp = y_pilot_row[k_plus]
        ym = y_pilot_row[k_minus]
        prod = -yp * ym
        psi_dk = np.angle(prod)
        w_dk = np.abs(prod)
        sum_vec += w_dk * np.exp(1j * psi_dk)

    # === Closed-form solution ===
    psi_bar = np.angle(sum_vec)
    nu_f_hat = psi_bar * N / (2.0 * np.pi * (N - 1))

    # Limit to (-0.5, 0.5)
    if nu_f_hat > 0.5:
        nu_f_hat = 0.5
    elif nu_f_hat < -0.5:
        nu_f_hat = -0.5

    return nu_f_hat


def estimate_nu_f_vectorized(
    Y_dd: np.ndarray,
    lp: int,
    kp: int,
    N: int,
    delta_k_list: tuple = (1, 2, 3, 4, 5, 6, 7),
) -> float:
    """
    Vectorized version: directly extract pilot row from DD-domain matrix
    and estimate.

    Args:
        Y_dd: [M, N] DD-domain received matrix
        lp: Pilot delay row
        kp: Pilot Doppler column
        N: Doppler dimension size
        delta_k_list: Sidelobe offset list

    Returns:
        nu_f_hat: Fractional Doppler estimate
    """
    y_pilot_row = Y_dd[lp, :]
    return estimate_nu_f(y_pilot_row, kp, N, delta_k_list)
