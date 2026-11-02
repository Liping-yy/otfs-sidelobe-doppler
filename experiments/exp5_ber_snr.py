# -*- coding: utf-8 -*-
"""
Experiment 5: link BER-SNR performance
========================

Corresponding to experiment outline Experiment 5。

Design:
  engineering frame(QPSK+guard zero), νf=0.3
  5Curves:
    1. ProposedCompensation: Sidelobe + peak self-multiplication closed-formEstimate → Compensation → demodulation
    2. MPCompensation: Matrix PencilEstimate → Compensation → demodulation
    3. He2023Compensation: twoordersegmentEstimate → Compensation → demodulation
    4. Ideal sync upper bound: completeUSalreadyknowνfCompensation → demodulation
    5. No-compensation lower bound: noCompensation → demodulation

  SNR: -5~25dB step size5dB (7 points)
  MC: eachSNRrepeatmc_timesframe, counttotalBitserrorrate

Compensation method:
  atTF domainuseEstimatenu_f_hatdofeedbackvectorphaserotation:
    R_compensated[n,m] = R[n,m] * exp(-j*2*pi*nu_f_hat*n/N)
  Then SFFT back to DD domain for demodulation

QPSK demodulation:
  DD domaindata symbol → extractnon-pilotnon-guard zerolocset → QPSK hard decision → JacobifororiginalBits

"""

import os
import sys
import time
import numpy as np

# project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.params import DEFAULT
from modem.otfs_modem import (
    isfft_fft, sfft_fft,
    apply_fractional_doppler_tf, add_awgn,
)
from frame.frame_gen import generate_engineering_frame
from estimator import get_estimator
from utils.metrics import save_results


# ============================================================
# QPSKmodulation/demodulation
# ============================================================

def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    QPSKmodulation (Gray mapping, powernormalized=1)。

    (0,0)->(1+j)/sqrt2, (0,1)->(1-j)/sqrt2,
    (1,0)->(-1+j)/sqrt2, (1,1)->(-1-j)/sqrt2

    Args:
        bits: 2KBits, shape (2K,)

    Returns:
        symbols: KQPSKSymbols, shape (K,)
    """
    bits = bits.reshape(-1, 2)
    return ((1 - 2 * bits[:, 0]) + 1j * (1 - 2 * bits[:, 1])) / np.sqrt(2)


def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """
    QPSK hard decisiondemodulation (Gray mappingcounter)。

    Decision rule:
      Re>0, Im>0 -> (0,0)
      Re>0, Im<0 -> (0,1)
      Re<0, Im>0 -> (1,0)
      Re<0, Im<0 -> (1,1)

    Args:
        symbols: Received symbols, shape (K,)

    Returns:
        bits: Demodulated bits, shape (2K,)
    """
    bits_re = (symbols.real < 0).astype(int)
    bits_im = (symbols.imag < 0).astype(int)
    return np.stack([bits_re, bits_im], axis=-1).flatten()


# ============================================================
# extractdata symbollocset（non-pilotnon-guard zero）
# ============================================================

def get_data_positions(M, N, lp, kp, Kmax):
    """
    getengineering framemidQPSKdatadataposition indices。

    excludepilotdelayrowlpupperguard zeroregion(|k-kp|<=Kmax)。
    otherall(l,k)locsetallasdatadata。

    Args:
        M, N: Frame dimension
        lp: pilotdelayrow
        kp: pilot Doppler column
        Kmax: Guard zero one-sided length

    Returns:
        positions: [(l, k), ...] dataPosition list
    """
    positions = []
    for l in range(M):
        for k in range(N):
            if l == lp:
                # pilot row：excludeguard zeroregion
                dk = (k - kp) % N
                if dk > N // 2:
                    dk = dk - N
                if abs(dk) <= Kmax:
                    continue
            positions.append((l, k))
    return positions


# ============================================================
# unitframeBER trial
# ============================================================

def _run_single_frame_ber(
    M, N, lp, kp, A, Kmax, nu_f, snr_db,
    est_names, delta_k_list, delta_k_sidetone,
    rng,
):
    """
    unitframeBER trial：Generate engineering frame → Doppler+Noise → eachestimatorCompensation → demodulation → Count error bits。

    Returns:
        results: dict {
            "total_bits": int,
            "errors": {est_name: int},
        }
    """
    # 1. Generate engineering frame
    X_dd = generate_engineering_frame(M, N, lp, kp, A, Kmax, rng)

    # 2. extractoriginaldatalocsetsumSymbols
    data_pos = get_data_positions(M, N, lp, kp, Kmax)
    n_data = len(data_pos)

    # 3. extractoriginalBits（fromQPSKSymbolsfeedbackrecurrence）
    original_symbols = np.array([X_dd[l, k] for l, k in data_pos])
    # fromQPSKSymbolsrecoverBits（used forBERcompute）
    original_bits = qpsk_demodulate(original_symbols)

    total_bits = len(original_bits)

    # 4. Transmission link: ISFFT → Doppler → AWGN
    X_tf = isfft_fft(X_dd)
    R_tf = apply_fractional_doppler_tf(X_tf, nu_f)
    R_tf_noisy = add_awgn(R_tf, snr_db, rng)
    Y_dd = sfft_fft(R_tf_noisy)

    # 5. pilot rowused forEstimate
    y_row = Y_dd[lp, :]

    # 6. eachestimatorEstimate + Compensation + demodulation
    errors = {}

    for name in est_names:
        fn = get_estimator(name)

        # Estimate nu_f
        if name == "sidetone_only":
            nu_hat = fn(y_row, kp, N, delta_k_list=delta_k_sidetone)
        elif name in ("matrix_pencil", "he2023"):
            if name == "he2023":
                nu_hat = fn(y_row, kp, N, M=M, A=A, lp=lp)
            else:
                nu_hat = fn(y_row, kp, N)
        else:
            nu_hat = fn(y_row, kp, N, delta_k_list=delta_k_list)

        # TF domain compensation: R_comp = R * exp(-j*2*pi*nu_hat*n/N)
        R_comp = R_tf_noisy * np.exp(-1j * 2 * np.pi * nu_hat * np.arange(N) / N)[:, np.newaxis]
        Y_comp = sfft_fft(R_comp)

        # extractdata symboluniondemodulation
        rx_symbols = np.array([Y_comp[l, k] for l, k in data_pos])
        rx_bits = qpsk_demodulate(rx_symbols)

        # Count error bits
        errors[name] = int(np.sum(rx_bits != original_bits))

    # 7. Ideal sync upper bound
    R_ideal = R_tf_noisy * np.exp(-1j * 2 * np.pi * nu_f * np.arange(N) / N)[:, np.newaxis]
    Y_ideal = sfft_fft(R_ideal)
    rx_symbols_ideal = np.array([Y_ideal[l, k] for l, k in data_pos])
    rx_bits_ideal = qpsk_demodulate(rx_symbols_ideal)
    errors["ideal"] = int(np.sum(rx_bits_ideal != original_bits))

    # 8. No-compensation lower bound
    Y_nocomp = Y_dd  # Already have uncompensated DD domain
    rx_symbols_nocomp = np.array([Y_nocomp[l, k] for l, k in data_pos])
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
    runExperiment 5: BER-SNRlinkperformance。

    Args:
        mc_times: eachSNRMonte Carloframe(default defaultcfg.mc_times=2000)
        snr_list: SNRlistdB（defaultcfg.snr_range + 25dB）
        est_names: estimatorlist（default["proposed","matrix_pencil","he2023"]）
    """
    cfg = DEFAULT
    M, N = cfg.M, cfg.N
    lp, kp = cfg.pilot_delay, cfg.pilot_doppler
    A = cfg.pilot_amp
    Kmax = cfg.kmax_default
    nu_f = cfg.nu_f_default
    delta_k_list = cfg.delta_k_list
    delta_k_sidetone = cfg.delta_k_sidetone

    if mc_times is None:
        mc_times = cfg.mc_times
    if snr_list is None:
        snr_list = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
    if est_names is None:
        est_names = ["proposed", "matrix_pencil", "he2023"]

    # All curve names
    all_curves = est_names + ["ideal", "no_comp"]

    print("=" * 80)
    print(f"  Experiment 5: link BER-SNR performance")
    print(f"  nu_f={nu_f}, SNR={snr_list} dB, MC={mc_times} frames")
    print(f"  frame: engineering (Kmax={Kmax}, A={A})")
    print(f"  Curves: {all_curves}")
    print("=" * 80)

    # resultstructure
    results = {
        "meta": {
            "mc_times": mc_times,
            "snr_list": list(snr_list),
            "nu_f": nu_f,
            "frame_type": "engineering",
            "M": M, "N": N, "lp": lp, "kp": kp, "A": A, "Kmax": Kmax,
            "curves": all_curves,
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
                M, N, lp, kp, A, Kmax, nu_f, snr_db,
                est_names, delta_k_list, delta_k_sidetone,
                trial_rng,
            )

            total_bits_accum += result["total_bits"]
            for curve in all_curves:
                errors_accum[curve] += result["errors"][curve]

        # BER = totalerror bits / totalBits
        for curve in all_curves:
            ber = errors_accum[curve] / total_bits_accum if total_bits_accum > 0 else 0.0
            results["ber"][curve][f"{snr_db:.1f}"] = {
                "ber": ber,
                "errors": errors_accum[curve],
                "total_bits": total_bits_accum,
            }

        # print
        ber_strs = []
        for curve in all_curves:
            ber_val = results["ber"][curve][f"{snr_db:.1f}"]["ber"]
            if ber_val < 1e-6:
                ber_strs.append(f"{curve}={ber_val:.2e}")
            else:
                ber_strs.append(f"{curve}={ber_val:.4e}")

        elapsed = time.time() - t_start
        print(f"  SNR={snr_db:+5.1f}dB | " + " | ".join(ber_strs) + f" | {elapsed:.1f}s")

    # saveresult
    output_path = os.path.join(PROJECT_ROOT, "results", "exp5_ber_snr.json")
    save_results(results, output_path)
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_experiment5()
