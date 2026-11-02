# -*- coding: utf-8 -*-
"""
Ablation control: Sidelobe-only phase-sum estimator
====================================================

Does not include peak self-multiplication (Dk=0); uses only sidelobe
pairs (Dk>=1). Used to quantify information gain of peak
self-multiplication (experiment outline Experiment 2 ablation control).

Core formula (derivation docx Section 3.3):
  psi(Dk) = arg{-y_plus * y_minus} = 2*pi*nuf*(N-1)/N, Dk = 1,...,K
  psi_bar = arg{ Sum w_Dk * exp(j*psi_Dk) }
  nu_f_hat = psi_bar * N / (2*pi*(N-1))

Known limitation: when nuf -> 0, sidelobes are all zero, arg fails
-> singularity (derivation docx Section 3.5).
"""

import numpy as np


def estimate_nu_f(
    y_pilot_row: np.ndarray,
    kp: int,
    N: int,
    delta_k_list: tuple = (1, 2, 3, 4),
) -> float:
    """
    Sidelobe-only phase-sum estimator (ablation baseline,
    derivation docx Section 3.3).

    Args:
        y_pilot_row: Pilot delay row y[lp, :], shape (N,)
        kp: Pilot Doppler column index
        N: Doppler dimension size
        delta_k_list: Sidelobe offset list

    Returns:
        nu_f_hat: Fractional Doppler estimate in (-0.5, 0.5)
    """
    psis = []
    weights = []

    for dk in delta_k_list:
        k_plus = (kp + dk) % N
        k_minus = (kp - dk) % N
        yp = y_pilot_row[k_plus]
        ym = y_pilot_row[k_minus]
        prod = -yp * ym
        psis.append(np.angle(prod))
        weights.append(np.abs(prod))

    if len(psis) == 0:
        return 0.0

    sum_vec = sum(w * np.exp(1j * p) for w, p in zip(weights, psis))
    nu_f_hat = np.angle(sum_vec) * N / (2.0 * np.pi * (N - 1))

    if nu_f_hat > 0.5:
        nu_f_hat = 0.5
    elif nu_f_hat < -0.5:
        nu_f_hat = -0.5

    return nu_f_hat
