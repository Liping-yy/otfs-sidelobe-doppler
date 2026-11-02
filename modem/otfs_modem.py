# -*- coding: utf-8 -*-
"""
OTFS modulation and demodulation module
==============

Implement standard OTFS transform chain (Hadani-Raviteja 2017 definition)：
  ISFFT: DD domain → TF domain
  SFFT:  TF domain → DD domain
  Time-domain Doppler application
  AWGN noise addition

providetwokindachieves：
  1. Matrix DFT implementation（isfft/sfft）——Exact but slower, for verification
  2. FFT-accelerated implementation（isfft_fft/sfft_fft）——Mathematically equivalent, for large-scale simulation

Convention (strictly consistent with derivation docx §1.1)：
  X_dd[l, k]: l=delay(row, 0..M-1), k=Doppler(column, 0..N-1)
  X_tf[n, m]: n=time slot(row, 0..N-1), m=subcarrier(column, 0..M-1)

  ISFFT: X[n,m] = Σ_l Σ_k x[l,k] · exp(j2π(nk/N - ml/M))
  SFFT:  y[l,k] = (1/MN) Σ_n Σ_m Y[n,m] · exp(-j2π(nk/N - ml/M))

"""

import numpy as np


# ============================================================
# Matrix DFT implementation（exact，used forverify/Unit test）
# ============================================================

def isfft(X_dd: np.ndarray) -> np.ndarray:
    """
    Inverse Symplectic FFT: DD -> TF (matrix DFT implementation)。

    X_dd: [M, N]  l=delay(row), k=Doppler(col)
    X_tf: [N, M]  n=timeslot(row), m=subcarrier(col)

    numberlearnformula: X[n,m] = Σ_l Σ_k X_dd[l,k] · exp(j2π(nk/N - ml/M))
    matrix form: X_tf = conj(F_N) @ (F_M @ X_dd).T
    """
    M, N = X_dd.shape
    F_M = np.fft.fft(np.eye(M))      # [M,M] DFTmatrix
    F_N = np.fft.fft(np.eye(N))      # [N,N] DFTmatrix
    temp = F_M @ X_dd                # FFT along l -> m
    X_tf = np.conj(F_N) @ temp.T     # conj(DFT) = exp(+j2πnk/N)
    return X_tf


def sfft(X_tf: np.ndarray) -> np.ndarray:
    """
    Symplectic FFT: TF -> DD (matrix DFT implementation, inverse of ISFFT)。

    X_tf: [N, M]  n=timeslot(row), m=subcarrier(col)
    X_dd: [M, N]  l=delay(row), k=Doppler(col)

    numberlearnformula: y[l,k] = (1/MN) Σ_n Σ_m Y[n,m] · exp(-j2π(nk/N - ml/M))
    matrix form: X_dd = conj(F_M) @ X_tf.T @ F_N / (MN)
    """
    N, M = X_tf.shape
    F_M = np.fft.fft(np.eye(M))
    F_N = np.fft.fft(np.eye(N))
    temp = np.conj(F_M) @ X_tf.T     # conj(DFT) along m -> l
    X_dd = temp @ F_N                # DFT along n -> k
    return X_dd / (M * N)


# ============================================================
# FFT-accelerated implementation（Mathematically equivalent, for large-scale simulation）
# ============================================================

def isfft_fft(x: np.ndarray) -> np.ndarray:
    """
    FFT-accelerated ISFFT (mathematically strictly equivalent to isfft())。

    Derivation mapping:
      Along k-axis(axis=1): Σ_k x[l,k]·exp(j2πnk/N) = N·ifft(x, axis=1)[l,n]
      Along l-axis(axis=0): Σ_l f[l,n]·exp(-j2πml/M) = fft(f, axis=0)[m,n]
      Combined: X = fft(N·ifft(x, axis=1), axis=0).T

    x: [M, N] DD domain
    return: [N, M] TF domain
    """
    M, N = x.shape
    t1 = N * np.fft.ifft(x, axis=1)   # [M,N]
    t2 = np.fft.fft(t1, axis=0)        # [M,N]
    return t2.T                        # [N,M]


def sfft_fft(Y: np.ndarray) -> np.ndarray:
    """
    FFT-accelerated SFFT (mathematically strictly equivalent to sfft())。

    Derivation mapping:
      Along m-axis(axis=1): Σ_m Y[n,m]·exp(j2πml/M) = M·ifft(Y, axis=1)[n,l]
      Along n-axis(axis=0): Σ_n g[n,l]·exp(-j2πnk/N) = fft(g, axis=0)[k,l]
      Combined: y = fft(M·ifft(Y, axis=1), axis=0).T / (MN)

    Y: [N, M] TF domain
    return: [M, N] DD domain
    """
    N, M = Y.shape
    t1 = M * np.fft.ifft(Y, axis=1)   # [N,M]
    t2 = np.fft.fft(t1, axis=0)        # [N,M]
    return t2.T / (M * N)              # [M,N]


# ============================================================
# Time-domain Doppler application
# ============================================================

def apply_fractional_doppler_tf(X_tf: np.ndarray, nu_f: float) -> np.ndarray:
    """
    Apply fractional Doppler phase rotation in TF domain。

    Standard model (derivation docx §2.3):
      R[n,m] = X[n,m] · exp(j2π·νf·n/N)

    Doppler applied along time slot n-axis (corresponding to DD-domain Doppler k-dimension spreading)。

    Args:
        X_tf: [N, M] TF-domain transmit signal
        nu_f: fractional Doppler，νf ∈ (-0.5, 0.5)

    Returns:
        R_tf: [N, M] TF domainreceived signal（Doppler effect only, noiseless）
    """
    N = X_tf.shape[0]
    n_arr = np.arange(N)
    phase = np.exp(1j * 2 * np.pi * nu_f * n_arr / N)
    return X_tf * phase[:, np.newaxis]


def apply_fractional_doppler_time(s: np.ndarray, nu_f: float, M: int, N: int) -> np.ndarray:
    """
    Apply fractional Doppler in time domain (physical implementation, equivalent to TF-domain application)。

    standard: y[n*M+m] = s[n*M+m] · exp(j2π·νf·n/N)

    Args:
        s: [N*M] Time-domain signal
        nu_f: fractional Doppler
        M: subcarriernumber
        N: time slotnumber

    Returns:
        [N*M] Time-domain signal after Doppler shift
    """
    n = np.arange(N)
    phase = np.exp(1j * 2 * np.pi * nu_f * n / N)
    return s * np.repeat(phase, M)


# ============================================================
# AWGNNoise
# ============================================================

def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator = None) -> np.ndarray:
    """
    Add complex Gaussian white noise to achieve target SNR。

    SNR definition: signal_power / noise_power = 10^(snr_db/10)
    Noise: CN(0, σ²), Variance per complex sample σ² = signal_power / snr_linear

    Args:
        signal: Input signal (any shape)
        snr_db: objectiveSNR (dB)
        rng: Random generator

    Returns:
        Noisy signal
    """
    if rng is None:
        rng = np.random.default_rng()
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )
    return signal + noise


def add_awgn_local_snr(
    signal: np.ndarray,
    snr_db: float,
    local_indices: list,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Define SNR using local sample point power (engineering frame specific)。

    Engineering frame contains QPSK data; full-frame average power dominated by data (~27dB higher than sidelobe power)，
    Using full-frame power for noise would make sidelobe actual SNR far lower than set value。
    Use sidelobe sample point local power to compute noise variance instead。

    Args:
        signal: DD-domain received signal [M, N]
        snr_db: objectiveSNR (dB)
        local_indices: Local sample point list [(lp, kp+dk), (lp, kp-dk), ...]
        rng: Random generator

    Returns:
        Noisy signal
    """
    if rng is None:
        rng = np.random.default_rng()
    local_powers = [np.abs(signal[r, c]) ** 2 for r, c in local_indices]
    power_signal = np.mean(local_powers)
    sigma_noise = np.sqrt(power_signal / (10.0 ** (snr_db / 10.0)))
    noise = (rng.standard_normal(signal.shape) +
             1j * rng.standard_normal(signal.shape)) / np.sqrt(2)
    return signal + sigma_noise * noise


# ============================================================
# completeTransmission link
# ============================================================

def transmit_pure_frame(
    X_dd: np.ndarray,
    nu_f: float,
    snr_db: float,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Pure frame complete transmission link (time-domain Doppler + AWGN)。

    link: DD → ISFFT → time domainDoppler → AWGN → SFFT → DD
    SNR defined using full-frame average power (pure frame has no data, no bias)。

    Args:
        X_dd: [M, N] DD-domain transmit frame
        nu_f: fractional Doppler
        snr_db: SNR (dB)
        rng: Random generator

    Returns:
        Y_dd: [M, N] DD-domain receive frame
    """
    if rng is None:
        rng = np.random.default_rng()
    M, N = X_dd.shape
    # DD -> TF -> time domain
    X_tf = isfft_fft(X_dd)
    # Time-domain Doppler (TF-domain phase rotation, mathematically equivalent)
    R_tf = apply_fractional_doppler_tf(X_tf, nu_f)
    # TF -> DD
    y_clean = sfft_fft(R_tf)
    # AWGN (full-frame power)
    y_noisy = add_awgn(y_clean, snr_db, rng)
    return y_noisy


def transmit_engineering_frame(
    X_dd: np.ndarray,
    nu_f: float,
    snr_db: float,
    lp: int,
    kp: int,
    delta_k_list: tuple,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Engineering frame complete transmission link (local sidelobe power SNR definition)。

    link: DD → ISFFT(FFTversion) → TF-domain Doppler → SFFT(FFTversion) → DD → localSNRaddnoise
    SNR defined using sidelobe sample point local power, avoiding overestimation of noise by full-frame average power。

    Args:
        X_dd: [M, N] DD-domain transmit frame (pilot + guard zero + QPSK data)
        nu_f: fractional Doppler
        snr_db: SNR (dB)
        lp: pilotdelayrow
        kp: pilot Doppler column
        delta_k_list: Sidelobe offset list（Used for local power computation）
        rng: Random generator

    Returns:
        Y_dd: [M, N] DD-domain receive frame
    """
    if rng is None:
        rng = np.random.default_rng()
    # DD -> TF
    X_tf = isfft_fft(X_dd)
    # TF-domain Doppler
    R_tf = apply_fractional_doppler_tf(X_tf, nu_f)
    # TF -> DD
    y_clean = sfft_fft(R_tf)
    # Local sidelobe power SNR noise addition
    local_idx = []
    for dk in delta_k_list:
        local_idx.append((lp, kp + dk))
        local_idx.append((lp, kp - dk))
    y_noisy = add_awgn_local_snr(y_clean, snr_db, local_idx, rng)
    return y_noisy


# ============================================================
# self-test
# ============================================================

def self_test():
    """Verify ISFFT/SFFT round-trip and FFT/matrix implementation equivalence。"""
    M, N = 64, 16
    rng = np.random.default_rng(42)

    # Test 1: matrixachievesround-trip
    X = rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))
    X_tf = isfft(X)
    X_back = sfft(X_tf)
    err1 = np.max(np.abs(X - X_back))
    print(f"[1] Matrix ISFFT/SFFT round-trip error: {err1:.2e}",
          "PASS" if err1 < 1e-12 else "FAIL")

    # Test 2: FFTachievesround-trip
    X_tf2 = isfft_fft(X)
    X_back2 = sfft_fft(X_tf2)
    err2 = np.max(np.abs(X - X_back2))
    print(f"[2] FFT ISFFT/SFFT round-trip error: {err2:.2e}",
          "PASS" if err2 < 1e-12 else "FAIL")

    # Test 3: twokindachievesequivalentproperty
    err3 = np.max(np.abs(isfft(X) - isfft_fft(X)))
    print(f"[3] Matrix vs FFT implementation equivalence error: {err3:.2e}",
          "PASS" if err3 < 1e-12 else "FAIL")

    # Test 4: TF domainvstime domainDopplerequivalentproperty
    from modem.otfs_modem import apply_fractional_doppler_tf, apply_fractional_doppler_time, otfs_modulate, otfs_demodulate
    s = otfs_modulate(X)
    s_dop_time = apply_fractional_doppler_time(s, 0.3, M, N)
    y_time = otfs_demodulate(s_dop_time, M, N)

    X_tf = isfft_fft(X)
    R_tf = apply_fractional_doppler_tf(X_tf, 0.3)
    y_tf = sfft_fft(R_tf)

    err4 = np.max(np.abs(y_time - y_tf))
    print(f"[4] TF-domain vs time-domain Doppler equivalence error: {err4:.2e}",
          "PASS" if err4 < 1e-12 else "FAIL")

    print("\nSelf-test done。")


def otfs_modulate(X_dd: np.ndarray) -> np.ndarray:
    """OTFS modulation: DD → Time-domain signal（extendtranslation）。"""
    X_tf = isfft_fft(X_dd)
    x_time = np.fft.ifft(X_tf, axis=1)  # Heisenberg: IFFTalongsubcarrier
    return x_time.flatten()


def otfs_demodulate(y_rx: np.ndarray, M: int, N: int) -> np.ndarray:
    """OTFS demodulation: Time-domain signal → DD domain。"""
    y_mat = y_rx.reshape(N, M)
    Y_tf = np.fft.fft(y_mat, axis=1)  # Wigner: FFTalongsubcarrier
    return sfft_fft(Y_tf)


if __name__ == "__main__":
    self_test()
