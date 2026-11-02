# -*- coding: utf-8 -*-
"""
OTFS Common Utilities for Baseline Methods
==========================================
Shared components: OTFS modem (textcontributionstandarddefinition), AWGN channel, 
                   Dirichlet kernel, pilot frame generation.

All baseline methods use the same standard OTFS I/O relationship:
  Y_dd[l,k] = Sum_{l'} Sum_{k'} x[l',k'] * delta[l-l'] * zeta_N(k - k' - nu_f*N)
where zeta_N(x) = sin(πx) / sin(πx/N) is the Dirichlet kernel (periodic sinc).

References:
  [1] Raviteja et al., IEEE TVT 68(5):4906-4917, 2019
  [2] Wei et al., arXiv:2101.11770, 2021
  [3] Liang et al., IEEE Comm. Lett. 28(9):2191-2195, 2024
  [4] Jiang et al., Physical Communication 72:102788, 2025

"""

import numpy as np


# ============================================================
# OTFS Modem (Standard Literature Definition)
# ============================================================
# Convention (Raviteja 2019 / Wei 2021):
#   X_dd[l, k]: l = delay bin (0..M-1), k = Doppler bin (0..N-1)
#   X_tf[n, m]: n = time slot (0..N-1), m = subcarrier (0..M-1)
#
# ISFFT: X_tf[n, m] = (1/sqrt(MN)) Sum_l Sum_k X_dd[l,k] * exp(j2π(nk/N - ml/M))
# SFFT:  X_dd[l, k] = (1/sqrt(MN)) Sum_n Sum_m X_tf[n,m] * exp(-j2π(nk/N - ml/M))
#
# With standard numpy FFT convention:
#   ISFFT: X_tf = FFT(N * IFFT(X_dd, axis=0), axis=1) / (M*N)  [simplified]
#   We use the direct formula via fft/ifft for clarity.


def isfft(X_dd):
    """
    Inverse Symplectic FFT: DD -> TF (matrix implementation, mathematically exact).
    
    X_dd: [M, N]  l=delay(row), k=Doppler(col)
    X_tf: [N, M]  n=timeslot(row), m=subcarrier(col)
    
    X_tf[n,m] = Sum_l Sum_k X_dd[l,k] * exp(j*2π*(n*k/N - m*l/M))
    
    Matrix form: X_tf = F_N @ X_dd @ F_M^H
    where F_P[a,b] = exp(-j*2π*a*b/P) is the standard DFT matrix.
    
    Derivation:
      Sum_l Sum_k X_dd[l,k] exp(j2π(nk/N - ml/M))
      = Sum_k exp(j2πnk/N) Sum_l X_dd[l,k] exp(-j2πml/M)
      = Sum_k exp(j2πnk/N) [F_M @ X_dd][m, k]      (FFT along l-axis of X_dd)
      = [F_M @ X_dd @ F_N^H][m, n]  (but need [n,m])
    
    Actually: 
      temp = F_M @ X_dd        -> [M,N], temp[m,k] = Sum_l X_dd[l,k] exp(-j2πml/M)
      X_tf = conj(F_N) @ temp.T -> [N,M], X_tf[n,m] = Sum_k temp.T[m,k] exp(+j2πnk/N)
    
    F_N^H = conj(F_N) (inverse DFT without normalization).
    """
    M, N = X_dd.shape
    F_M = np.fft.fft(np.eye(M))    # [M,M] DFT matrix: exp(-j2πml/M)
    F_N = np.fft.fft(np.eye(N))    # [N,N] DFT matrix: exp(-j2πnk/N)
    temp = F_M @ X_dd              # [M,N]: FFT along l -> m
    X_tf = np.conj(F_N) @ temp.T   # [N,M]: conj(DFT) = exp(+j2πnk/N), temp.T swaps m,k
    return X_tf


def sfft(X_tf):
    """
    Symplectic FFT: TF -> DD (inverse of ISFFT, matrix implementation).
    
    X_dd[l,k] = Sum_n Sum_m X_tf[n,m] * exp(-j*2π*(n*k/N - m*l/M))
             = Sum_m exp(+j2πml/M) Sum_n X_tf[n,m] exp(-j2πnk/N)
    
    Matrix form: X_dd = F_M^H @ X_tf^T @ F_N
    """
    N, M = X_tf.shape
    F_M = np.fft.fft(np.eye(M))
    F_N = np.fft.fft(np.eye(N))
    temp = np.conj(F_M) @ X_tf.T   # [M,N]: conj(DFT) along m -> l
    X_dd = temp @ F_N              # [M,N]: DFT along n -> k
    return X_dd / (M * N)         # normalize round-trip to identity





def otfs_modulate(X_dd):
    """OTFS modulation: DD -> time domain signal (flatten)."""
    X_tf = isfft(X_dd)          # DD -> TF (no normalization)
    # Heisenberg: IFFT along m (subcarrier) for each time slot
    # numpy ifft includes 1/M factor
    x_time = np.fft.ifft(X_tf, axis=1)  # [N, M]
    return x_time.flatten()


def otfs_demodulate(y_rx, M, N):
    """OTFS demodulation: time domain signal -> DD domain."""
    y_mat = y_rx.reshape(N, M)
    # Wigner: FFT along m for each time slot
    # numpy fft has no normalization factor
    Y_tf = np.fft.fft(y_mat, axis=1)  # [N, M]
    Y_dd = sfft(Y_tf)                  # TF -> DD (no normalization)
    return Y_dd


# ============================================================
# Fractional Doppler Application (Time Domain)
# ============================================================

def apply_fractional_doppler(s, nu_f, M, N):
    """
    Apply fractional Doppler shift in time domain.
    
    Standard model (Raviteja 2019, Wei 2021):
      y[n*M + m] = s[n*M + m] * exp(j * 2π * nu_f * n / N)
    
    The Doppler shift is applied per time-slot index n (not per sample).
    This is the standard OTFS convention where Doppler operates on the
    time-slot axis, causing DD-domain sinc (Dirichlet) spreading along
    the Doppler (k) dimension.
    
    Args:
        s: [N*M] time-domain signal
        nu_f: fractional Doppler shift, nu_f ∈ (-0.5, 0.5)
        M: subcarriers
        N: time slots
    
    Returns:
        y: [N*M] Doppler-shifted signal
    """
    n = np.arange(N)  # time slot indices
    phase = np.exp(1j * 2 * np.pi * nu_f * n / N)  # [N]
    # Repeat phase for each subcarrier within each slot
    phase_full = np.repeat(phase, M)  # [N*M]
    return s * phase_full


# ============================================================
# AWGN Channel
# ============================================================

def add_awgn(signal, snr_db, rng=None):
    """Add complex AWGN to achieve target SNR (dB)."""
    if rng is None:
        rng = np.random.default_rng()
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )
    return signal + noise


# ============================================================
# Dirichlet Kernel (Periodic Sinc)
# ============================================================

def dirichlet_kernel(x, N):
    """
    Dirichlet kernel (periodic sinc function):
      D_N(x) = sin(πx) / sin(πx/N)
    
    This is the DD-domain basis function for fractional Doppler in OTFS
    (Wei 2021, eq. for single-path response in Doppler domain).
    
    At integer x: D_N(k) = N if k ≡ 0 (mod N), else 0.
    At fractional x: produces sinc-like spreading (IDI).
    
    Args:
        x: array of positions (can be fractional)
        N: period (number of Doppler bins)
    
    Returns:
        D_N(x): Dirichlet kernel values
    """
    x = np.asarray(x, dtype=float)
    # Handle near-integer cases to avoid 0/0
    result = np.zeros_like(x, dtype=complex)
    
    # Where x is near integer multiple of N
    mask_near_int = np.abs(np.sin(np.pi * x / N)) < 1e-12
    mask_near_int &= np.abs(np.sin(np.pi * x)) < 1e-12
    
    # General case
    mask_general = ~mask_near_int
    result[mask_general] = np.sin(np.pi * x[mask_general]) / np.sin(np.pi * x[mask_general] / N)
    
    # Near integer: limit is N (when x -> 0 mod N) or 0 (other integers)
    # More precisely, D_N(x) -> N when x -> 0 (mod N), -> 0 when x -> k (mod N), k!=0
    x_near = x[mask_near_int]
    mod_result = np.where(np.abs(x_near % N) < 1e-10, float(N), 0.0)
    result[mask_near_int] = mod_result
    
    return result


# ============================================================
# Pilot Frame Generation
# ============================================================

def generate_impulse_pilot_frame(M, N, lp=0, kp=0, pilot_value=1.0):
    """
    Generate DD-domain frame with single impulse pilot.
    
    Single impulse at (lp, kp): X_dd[lp, kp] = pilot_value, rest = 0.
    This is the simplest pilot for channel estimation (Raviteja 2019).
    
    Args:
        M: delay bins
        N: Doppler bins
        lp: pilot delay index
        kp: pilot Doppler index
        pilot_value: complex pilot symbol
    
    Returns:
        X_dd: [M, N] DD-domain frame
    """
    X_dd = np.zeros((M, N), dtype=complex)
    X_dd[lp, kp] = pilot_value
    return X_dd


def generate_embedded_pilot_frame(M, N, lp=0, kp=0, pilot_value=1.0, 
                                   guard_delay=0, guard_doppler=0, rng=None):
    """
    Generate DD-domain frame with embedded pilot + guard + QPSK data.
    
    Layout (Raviteja 2019):
      - Pilot at (lp, kp)
      - Guard zeros around pilot: |l-lp| <= guard_delay, |k-kp| <= guard_doppler
      - QPSK data in remaining positions
    
    Args:
        M, N: frame dimensions
        lp, kp: pilot position
        pilot_value: pilot symbol
        guard_delay: guard zeros in delay dimension
        guard_doppler: guard zeros in Doppler dimension
        rng: random generator
    
    Returns:
        X_dd: [M, N] DD-domain frame
    """
    if rng is None:
        rng = np.random.default_rng()
    
    X_dd = np.zeros((M, N), dtype=complex)
    
    # QPSK data everywhere
    bits = rng.integers(0, 2, size=(M, N, 2))
    qpsk = ((1 - 2 * bits[:, :, 0]) + 1j * (1 - 2 * bits[:, :, 1])) / np.sqrt(2)
    X_dd = qpsk.copy()
    
    # Place guard zeros
    for l in range(M):
        for k in range(N):
            if abs(l - lp) <= guard_delay and abs(k - kp) <= guard_doppler:
                X_dd[l, k] = 0.0
    
    # Place pilot
    X_dd[lp, kp] = pilot_value
    
    return X_dd


# ============================================================
# Full Transmission Chain
# ============================================================

def transmit_otfs(X_dd, nu_f, snr_db, M, N, rng=None):
    """
    Full OTFS transmission: DD -> modulate -> Doppler -> AWGN -> demodulate -> DD.
    
    Args:
        X_dd: [M, N] DD-domain transmit frame
        nu_f: fractional Doppler shift
        snr_db: SNR in dB
        M, N: frame dimensions
        rng: random generator
    
    Returns:
        Y_dd: [M, N] DD-domain received frame
    """
    if rng is None:
        rng = np.random.default_rng()
    
    s = otfs_modulate(X_dd)
    s_doppler = apply_fractional_doppler(s, nu_f, M, N)
    y_rx = add_awgn(s_doppler, snr_db, rng)
    Y_dd = otfs_demodulate(y_rx, M, N)
    return Y_dd


# ============================================================
# Verification / Self-Test
# ============================================================

def self_test():
    """Verify ISFFT/SFFT round-trip and Dirichlet kernel properties."""
    M, N = 64, 16
    
    # Test 1: ISFFT -> SFFT round-trip
    X_dd = np.random.randn(M, N) + 1j * np.random.randn(M, N)
    X_tf = isfft(X_dd)
    X_dd_back = sfft(X_tf)
    rt_error = np.max(np.abs(X_dd - X_dd_back))
    print(f"[Test 1] ISFFT->SFFT round-trip error: {rt_error:.2e}", 
          "PASS" if rt_error < 1e-12 else "FAIL")
    
    # Test 2: Impulse pilot -> Dirichlet kernel in Doppler
    X_imp = generate_impulse_pilot_frame(M, N, lp=5, kp=8)
    Y_dd = transmit_otfs(X_imp, nu_f=0.3, snr_db=100, M=M, N=N,
                         rng=np.random.default_rng(42))
    
    # Check that energy spreads along Doppler axis at l=lp
    pilot_row = Y_dd[5, :]  # l=5, all k
    peak_idx = np.argmax(np.abs(pilot_row))
    peak_val = np.abs(pilot_row[peak_idx])
    total_energy = np.sum(np.abs(pilot_row) ** 2)
    peak_ratio = peak_val ** 2 / total_energy
    
    print(f"[Test 2] Impulse pilot with nu_f=0.3:")
    print(f"  Peak at k={peak_idx} (expected k=8, should NOT move)")
    print(f"  Peak energy ratio: {peak_ratio:.4f}")
    print(f"  Energy spread across Doppler bins: {'YES' if peak_ratio < 0.9 else 'NO'}")
    
    # Test 3: nu_f=0 -> no spreading
    Y_dd0 = transmit_otfs(X_imp, nu_f=0.0, snr_db=100, M=M, N=N,
                          rng=np.random.default_rng(42))
    pilot_row0 = Y_dd0[5, :]
    peak_idx0 = np.argmax(np.abs(pilot_row0))
    peak_ratio0 = np.abs(pilot_row0[peak_idx0]) ** 2 / np.sum(np.abs(pilot_row0) ** 2)
    print(f"[Test 3] nu_f=0.0: peak at k={peak_idx0}, ratio={peak_ratio0:.4f}",
          "PASS" if peak_ratio0 > 0.99 else "FAIL")
    
    print("\nAll tests completed.")


if __name__ == '__main__':
    self_test()
