# -*- coding: utf-8 -*-
"""
Single-path LOS channel model
=============================

Corresponding derivation docx §1.2 Channel assumptions:
  - Ka-band LEO-SISO single-path LOS
  - Constant complex channel gain h0
  - Constant Doppler within frame (nudot=0)
  - Only residual fractional Doppler nuf in (-0.5, 0.5)

Channel effect in OTFS link:
  TF domain: R[n,m] = h0 * X[n,m] * exp(j*2*pi*nuf*n/N) + W[n,m]
  (h0 is absorbed into pilot amplitude A; current implementation uses h0=1)
"""

import numpy as np


def apply_los_channel(
    X_tf: np.ndarray,
    nu_f: float,
    h0: complex = 1.0 + 0j,
) -> np.ndarray:
    """
    Single-path LOS channel: apply Doppler + channel gain in TF domain.

    R[n,m] = h0 * X[n,m] * exp(j*2*pi*nuf*n/N)

    Args:
        X_tf: [N, M] TF-domain transmit signal
        nu_f: fractional Doppler
        h0: Channel complex gain (default 1.0)

    Returns:
        R_tf: [N, M] TF-domain received signal (noiseless)
    """
    N = X_tf.shape[0]
    n_arr = np.arange(N)
    phase = np.exp(1j * 2 * np.pi * nu_f * n_arr / N)
    return h0 * X_tf * phase[:, np.newaxis]


def compute_doppler_freq(nu_f: float, N: int, delta_f: float) -> float:
    """
    Compute physical Doppler frequency shift from normalized fractional Doppler.

    f_D = nu_f * N * delta_f

    Args:
        nu_f: Normalized fractional Doppler
        N: Doppler dimension size
        delta_f: Subcarrier spacing (Hz)

    Returns:
        f_D: Doppler frequency shift (Hz)
    """
    return nu_f * N * delta_f


def compute_leo_max_doppler(fc: float, altitude: float, elevation: float = 90.0) -> float:
    """
    Compute LEO satellite max Doppler frequency shift (for parameter verification).

    f_D,max approx (v_sat / c) * f_c

    Args:
        fc: Carrier frequency (Hz)
        altitude: Satellite altitude (m)
        elevation: Elevation angle (degrees, 90=zenith)

    Returns:
        f_D_max: Max Doppler frequency shift (Hz)
    """
    # Earth radius and gravitational parameter
    R_E = 6371e3  # m
    GM = 3.986e14  # m^3/s^2

    # Orbital radius and velocity
    r_orbit = R_E + altitude
    v_sat = np.sqrt(GM / r_orbit)  # Circular orbit velocity

    c = 3e8  # Speed of light
    return (v_sat / c) * fc


if __name__ == "__main__":
    # LEO parameter verification
    f_D_max = compute_leo_max_doppler(28e9, 600e3)
    print(f"LEO 28GHz/600km max Doppler: {f_D_max/1e3:.0f} kHz")
    print(f"Normalized nuf_max (N=16, df=120kHz): {f_D_max / (16 * 120e3):.2f}")
