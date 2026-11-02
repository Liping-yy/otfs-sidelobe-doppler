# -*- coding: utf-8 -*-
"""
Experiment 5（v2）：linkBER-SNRperformance —— zero-paddedframeversion
============================================

Based on exp5_ber_snr.py modified，adaptednewpaperFrame structure：
  - engineering framechangeaspilot row fully zero-padded（No Kmax concept）
  - datalocset：allnon-pilot rowlocset（l ≠ lp）

5Curves:
  1. ProposedCompensation
  2. MPCompensation
  3. He2023Compensation
  4. Ideal sync upper bound
  5. No-compensation lower bound

SNR: -5~25dB step size5dB, MC=2000frame/SNR

"""

import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.params import DEFAULT
from modem.otfs_modem import (
    isfft_fft, sfft_fft,
    apply_fractional_doppler_tf, add_awgn,
)
from frame.frame_gen import generate_engineering_frame_zp
from estimator import get_estimator
from utils.metrics import save_results


# ============================================================
# QPSKmodulation/demodulation
# ============================================================

def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """QPSKmodulation (Gray mapping, powernormalized=1)。"""
    bits = bits.reshape(-1, 2)
    return ((1 - 2 * bits[:, 0]) + 1j * (1 - 2 * bits[:, 1])) / np.sqrt(2)


def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """QPSK hard decisiondemodulation (Gray mappingcounter)。"""
    bits_re = (symbols.real < 0).astype(int)
    bits_im = (symbols.imag < 0).astype(int)
    return np.stack([bits_re, bits_im], axis=-1).flatten()


# ============================================================
# extractdata symbollocset（zero-paddedframeversion）
# ============================================================

def get_data_positions_zp(M, N, lp):
    """
    getzero-paddedframemidQPSKdatadataposition indices。

    zero-paddedframe：pilot row(lp)integerrowzero-padded，only(lp,kp)rationalpilot。
    datalocset = allnon-pilot rowlocset（l ≠ lpall(l,k)）。

    Args:
        M, N: Frame dimension
        lp: pilotdelayrow

    Returns:
        positions: [(l, k), ...] dataPosition list
    """
    positions = []
    for l in range(M):
        if l == lp:
            continue  # skippilot row（integerrowzero-padded）
        for k in range(N):
            positions.append((l, k))
    return positions


# ============================================================
# unitframeBER trial
# ============================================================

def _run_single_frame_ber(
    M, N, lp, kp, A, nu_f, snr_db,
    est_names, delta_k_list, delta_k_sidetone,
    rng,
):
    """
    unitframeBER trial（zero-paddedframeversion）。

    Returns:
        results: dict {
            "total_bits": int,
            "errors": {est_name: int},
        }
    """
    # 1. generatezero-paddedframe
    X_dd = generate_engineering_frame_zp(M, N, lp, kp, A, rng)

    # 2. extractoriginaldatalocsetsumSymbols
    data_pos = get_data_positions_zp(M, N, lp)
    original_symbols = np.array([X_dd[l, k] for l, k in data_pos])
    original_bits = qpsk_demodulate(original_symbols)
    total_bits = len(original_bits)

    # 3. Transmission link
    X_tf = isfft_fft(X_dd)
    R_tf = apply_fractional_doppler_tf(X_tf, nu_f)
    R_tf_noisy = add_awgn(R_tf, snr_db, rng)
    Y_dd = sfft_fft(R_tf_noisy)

    # 4. pilot rowused forEstimate
    y_row = Y_dd[lp, :]

    # 5. eachestimatorEstimate + Compensation + demodulation
    errors = {}

    for name in est_names:
        fn = get_estimator(name)

        if name == "sidetone_only":
            nu_hat = fn(y_row, kp, N, delta_k_list=delta_k_sidetone)
        elif name == "he2023":
            nu_hat = fn(y_row, kp, N, M=M, A=A, lp=lp)
        elif name == "matrix_pencil":
            nu_hat = fn(y_row, kp, N)
        else:
            nu_hat = fn(y_row, kp, N, delta_k_list=delta_k_list)

        # TF domain compensation
        R_comp = R_tf_noisy * np.exp(
            -1j * 2 * np.pi * nu_hat * np.arange(N) / N
        )[:, np.newaxis]
        Y_comp = sfft_fft(R_comp)

        rx_symbols = np.array([Y_comp[l, k] for l, k in data_pos])
        rx_bits = qpsk_demodulate(rx_symbols)
        errors[name] = int(np.sum(rx_bits != original_bits))

    # 6. Ideal sync upper bound
    R_ideal = R_tf_noisy * np.exp(
        -1j * 2 * np.pi * nu_f * np.arange(N) / N
    )[:, np.newaxis]
    Y_ideal = sfft_fft(R_ideal)
    rx_symbols_ideal = np.array([Y_ideal[l, k] for l, k in data_pos])
    rx_bits_ideal = qpsk_demodulate(rx_symbols_ideal)
    errors["ideal"] = int(np.sum(rx_bits_ideal != original_bits))

    # 7. No-compensation lower bound
    rx_symbols_nocomp = np.array([Y_dd[l, k] for l, k in data_pos])
    rx_bits_nocomp = qpsk_demodulate(rx_symbols_nocomp)
    errors["no_comp"] = int(np.sum(rx_bits_nocomp != original_bits))

    return {
        "total_bits": total_bits,
        "errors": errors,
    }


# ============================================================
# main function
# ============================================================

def run_experiment5(
    mc_times=None,
    snr_list=None,
    est_names=None,
):
    """
    Experiment 5（v2）main function：BER-SNRlinkperformance（zero-paddedframeversion）。

    Difference from original：engineering frameusezero-paddedversion，irrationalKmaxparameter。
    datalocset = allnon-pilot row。
    """
    cfg = DEFAULT
    M, N = cfg.M, cfg.N
    lp, kp = cfg.pilot_delay, cfg.pilot_doppler
    A = cfg.pilot_amp
    nu_f = cfg.nu_f_default
    delta_k_list = cfg.delta_k_list
    delta_k_sidetone = cfg.delta_k_sidetone

    if mc_times is None:
        mc_times = cfg.mc_times
    if snr_list is None:
        snr_list = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
    if est_names is None:
        est_names = ["proposed", "matrix_pencil", "he2023"]

    all_curves = est_names + ["ideal", "no_comp"]

    print("=" * 80)
    print(f"  Experiment 5(v2): link BER-SNR performance [zero-paddedframe]")
    print(f"  nu_f={nu_f}, SNR={snr_list} dB, MC={mc_times} frames")
    print(f"  frame: engineering_zp (A={A}, pilot row zero-padded)")
    print(f"  datalocset: allnon-pilot row (l != lp)")
    print(f"  eachframedataBits: {2 * (M - 1) * N}")
    print(f"  Curves: {all_curves}")
    print("=" * 80)

    results = {
        "meta": {
            "mc_times": mc_times,
            "snr_list": list(snr_list),
            "nu_f": nu_f,
            "frame_type": "engineering_zp",
            "M": M, "N": N, "lp": lp, "kp": kp, "A": A,
            "curves": all_curves,
            "frame_note": "pilot row zero-padded, no Kmax",
        },
        "ber": {curve: {} for curve in all_curves},
    }

    for snr_db in snr_list:
        t_start = time.time()

        total_bits_accum = 0
        errors_accum = {curve: 0 for curve in all_curves}

        for trial in range(mc_times):
            trial_rng = np.random.default_rng(
                cfg.seed + trial + int(abs(snr_db) * 1000) + 1
            )

            result = _run_single_frame_ber(
                M, N, lp, kp, A, nu_f, snr_db,
                est_names, delta_k_list, delta_k_sidetone,
                trial_rng,
            )

            total_bits_accum += result["total_bits"]
            for curve in all_curves:
                errors_accum[curve] += result["errors"][curve]

        for curve in all_curves:
            ber = errors_accum[curve] / total_bits_accum if total_bits_accum > 0 else 0.0
            results["ber"][curve][f"{snr_db:.1f}"] = {
                "ber": ber,
                "errors": errors_accum[curve],
                "total_bits": total_bits_accum,
            }

        ber_strs = []
        for curve in all_curves:
            ber_val = results["ber"][curve][f"{snr_db:.1f}"]["ber"]
            if ber_val < 1e-6:
                ber_strs.append(f"{curve}={ber_val:.2e}")
            else:
                ber_strs.append(f"{curve}={ber_val:.4e}")

        elapsed = time.time() - t_start
        print(f"  SNR={snr_db:+5.1f}dB | " + " | ".join(ber_strs) + f" | {elapsed:.1f}s")

    output_path = os.path.join(PROJECT_ROOT, "results", "exp5_ber_snr.json")
    save_results(results, output_path)
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_experiment5()
