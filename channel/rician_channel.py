# -*- coding: utf-8 -*-
"""
Rician / multipath channel model for OTFS-LEO (revision extension)
=================================================================

Adds the multipath capability requested by Reviewers R1-7 and R3-2:
  - Finite-K Rician channel: dominant LOS path + P_s scattered paths
  - Each path p has:
      * integer delay tap l_p   (delay row in DD domain)
      * fractional Doppler nu_p in (-0.5, 0.5)
      * complex gain h_p
  - TF-domain exact implementation:
      H[n,m] = sum_p h_p * exp(j*2*pi*nu_p*n/N) * exp(-j*2*pi*l_p*m/M)

  The per-path TF response follows the standard OTFS channel model
  (Raviteja 2019 / Hadani 2017), extended with fractional Doppler.

Normalization conventions:
  - LOS path (p=0) gain magnitude: sqrt(K/(K+1))
  - Scattered paths (p>=1) gain magnitudes: sqrt(1/(K+1)) / sqrt(P_s)
  - K is the Rician K-factor in linear scale (K_dB = 10*log10(K)).
  - With K -> inf, reduces to the single-path LOS channel (h0=1).

Residual timing offset (fractional delay) lambda:
  - Optional fractional delay in units of the delay resolution.
  - TF-domain effect: exp(-j*2*pi*lambda*m/M), which is NOT separable
    per-delay-row and therefore creates genuine cross-row leakage
    (the exact phenomenon discussed in Section V of the revision).
"""

import numpy as np


def rician_paths(
    M: int,
    N: int,
    K_db: float = 10.0,
    P_s: int = 2,
    nu_f: float = 0.3,
    nu_scatter_std: float = 0.1,
    l_pilot: int = 5,
    los_phase: float = 0.0,
    seed: int = 2026,
    rng: np.random.Generator = None,
):
    """
    Generate a finite-K Rician channel path set.

    Args:
        M: Delay dimension (subcarrier count)
        N: Doppler dimension (time slot count)
        K_db: Rician K-factor in dB
        P_s: Number of scattered paths (>= 1)
        nu_f: Fractional Doppler of the LOS path
        nu_scatter_std: Std of scattered-path fractional Doppler around nu_f
        l_pilot: Pilot delay row
        los_phase: Fixed phase of the LOS path (rad). The receiver typically
            uses the pilot peak as phase reference, so the LOS phase can be
            absorbed into the reference: default 0.
        seed / rng: Random generator

    Returns:
        paths: list of dicts, one per path, each with keys
            l:   delay tap (float; fractional delays create cross-row leakage)
            nu:  fractional Doppler in (-0.5, 0.5)
            gain: complex gain
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    K_lin = 10 ** (K_db / 10.0)

    # --- LOS path (p=0) ---
    los_gain_mag = np.sqrt(K_lin / (K_lin + 1.0))
    los_gain = los_gain_mag * np.exp(1j * los_phase)

    # --- Scattered paths (p=1..P_s) ---
    scat_gain_mag = np.sqrt(1.0 / (K_lin + 1.0) / P_s)
    scat_gains = []
    scat_nus = []
    scat_ls = []
    for _ in range(P_s):
        scat_gains.append(
            scat_gain_mag * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
        )
        # Scattered Doppler: around LOS Doppler, within (-0.5, 0.5)
        nu_s = nu_f + rng.normal(0.0, nu_scatter_std)
        # wrap into (-0.5, 0.5)
        nu_s = ((nu_s + 0.5) % 1.0) - 0.5
        scat_nus.append(nu_s)
        # Scattered delay: fractional delay in a few delay-tap units.
        # A fractional delay breaks the delay-row orthogonality, so the
        # scattered path leaks into the pilot row (genuine cross-row
        # interference, cf. Section V of the revision).
        l_s = float(rng.uniform(0.2, 3.5))
        scat_ls.append(l_s)

    paths = [{"l": 0.0, "nu": nu_f, "gain": los_gain}]
    for i in range(P_s):
        paths.append({"l": scat_ls[i], "nu": scat_nus[i], "gain": scat_gains[i]})

    return paths


def apply_rician_channel(
    X_tf: np.ndarray,
    paths: list,
    lambda_delay: float = 0.0,
) -> np.ndarray:
    """
    Apply a multipath Rician channel in the TF domain.

    R[n,m] = sum_p h_p * X[n,m] * exp(j*2*pi*nu_p*n/N) * exp(-j*2*pi*l_p*m/M)
             * exp(-j*2*pi*lambda*m/M)   (optional residual fractional delay)

    Args:
        X_tf: [N, M] TF-domain transmit signal
        paths: list of path dicts from rician_paths()
        lambda_delay: residual fractional delay (in delay-tap units)

    Returns:
        R_tf: [N, M] TF-domain received signal (noiseless)
    """
    N, M = X_tf.shape
    n_arr = np.arange(N)[:, np.newaxis]     # [N, 1]
    m_arr = np.arange(M)[np.newaxis, :]     # [1, M]

    R_tf = np.zeros_like(X_tf)
    for path in paths:
        l_p = path["l"]
        nu_p = path["nu"]
        h_p = path["gain"]
        phase = (
            np.exp(1j * 2.0 * np.pi * nu_p * n_arr / N)
            * np.exp(-1j * 2.0 * np.pi * l_p * m_arr / M)
        )
        R_tf += h_p * X_tf * phase

    # Residual fractional delay (timing offset) — common to all paths
    if lambda_delay != 0.0:
        R_tf *= np.exp(-1j * 2.0 * np.pi * lambda_delay * m_arr / M)

    return R_tf


def apply_rician_channel_dd(
    X_dd: np.ndarray,
    paths: list,
    lambda_delay: float = 0.0,
) -> np.ndarray:
    """
    Apply Rician channel with the full DD->TF->channel->SFFT round trip.

    Args:
        X_dd: [M, N] DD-domain transmit frame
        paths: list of path dicts
        lambda_delay: residual fractional delay

    Returns:
        Y_dd: [M, N] DD-domain received frame (noiseless)
    """
    from modem.otfs_modem import isfft_fft, sfft_fft

    X_tf = isfft_fft(X_dd)
    R_tf = apply_rician_channel(X_tf, paths, lambda_delay=lambda_delay)
    return sfft_fft(R_tf)


# ============================================================
# self-test
# ============================================================

def self_test():
    """Verify Rician channel reduces to LOS at K->inf and sanity checks."""
    M, N = 64, 16
    lp, kp = 5, 8
    A = 3.0
    nu_f = 0.3

    from modem.otfs_modem import isfft_fft, sfft_fft
    from frame.frame_gen import generate_pure_frame

    X_dd = generate_pure_frame(M, N, lp, kp, A)
    X_tf = isfft_fft(X_dd)

    # Test 1: K -> inf should approach the single-path LOS channel
    # (fix LOS phase to 0 so it aligns with the h0=1 single-path model)
    paths_inf = rician_paths(M, N, K_db=40.0, P_s=2, nu_f=nu_f, seed=1)
    paths_inf[0]["gain"] = np.abs(paths_inf[0]["gain"])  # force LOS phase 0
    R_ric = apply_rician_channel(X_tf, paths_inf)
    R_los = X_tf * np.exp(1j * 2 * np.pi * nu_f * np.arange(N)[:, np.newaxis] / N)
    err = np.max(np.abs(R_ric - R_los))
    print(f"[1] K=40dB vs single-path LOS max diff: {err:.2e}",
          "PASS" if err < 0.05 else "FAIL")

    # Test 2: noiseless DD-domain pilot row under K->inf matches single path
    Y_ric = apply_rician_channel_dd(X_dd, paths_inf)
    Y_los = sfft_fft(R_los)
    err2 = np.max(np.abs(Y_ric[lp, :] - Y_los[lp, :]))
    print(f"[2] DD pilot row (K=40dB) vs LOS max diff: {err2:.2e}",
          "PASS" if err2 < 0.05 else "FAIL")

    # Test 3: cross-row leakage appears under finite K (scattered delay rows)
    paths_fin = rician_paths(M, N, K_db=10.0, P_s=2, nu_f=nu_f, seed=1)
    Y_fin = apply_rician_channel_dd(X_dd, paths_fin)
    # energy on non-pilot delay rows should be non-zero now
    off_rows = [l for l in range(M) if l != lp]
    energy_off = np.sum(np.abs(Y_fin[off_rows, :]) ** 2)
    print(f"[3] off-pilot-row energy at K=10dB: {energy_off:.3e} "
          f"(should be > 0; single-path gives ~1e-30)")
    print("    PASS" if energy_off > 1e-8 else "FAIL")

    # Test 4: residual fractional delay creates cross-row leakage even for LOS
    paths_los = rician_paths(M, N, K_db=40.0, P_s=2, nu_f=nu_f, seed=1)
    paths_los[0]["gain"] = np.abs(paths_los[0]["gain"])
    Y_lamb = apply_rician_channel_dd(X_dd, paths_los, lambda_delay=0.2)
    energy_lamb = np.sum(np.abs(Y_lamb[0, :]) ** 2)
    print(f"[4] off-pilot-row energy with lambda=0.2: {energy_lamb:.3e} "
          f"(should be > 0: fractional delay leaks across rows)")
    print("    PASS" if energy_lamb > 1e-8 else "FAIL")

    # Test 5: path count / normalization
    K_db = 10.0
    P_s = 3
    paths = rician_paths(M, N, K_db=K_db, P_s=P_s, nu_f=nu_f, seed=7)
    total_power = sum(np.abs(p["gain"]) ** 2 for p in paths)
    # LOS mag^2 = K/(K+1); scat per-path = 1/(K+1)/P_s; sum = K/(K+1) + P_s*1/(K+1)/P_s = 1
    print(f"[5] total path power = {total_power:.6f} (expected ~1)", "PASS" if abs(total_power - 1) < 1e-6 else "FAIL")
    print(f"    paths = {len(paths)} (expected 1 + {P_s})")

    print("\nRician channel self-test done.")


if __name__ == "__main__":
    self_test()
