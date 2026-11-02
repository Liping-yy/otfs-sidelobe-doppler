# -*- coding: utf-8 -*-
"""
Experiment 4（v2）：pilot powerAsensitivity —— zero-paddedframeversion
=============================================

Based on exp4_kmax_a_heatmap.py modified，adaptednewpaper：
  - deleteKmaxdimension，onlySweepA
  - engineering framechangeaspilot row fully zero-padded

Design:
  Fixed SNR=10dB, nu_f=0.3, zero-paddedframe
  Sweep: A ∈ {1.0, 1.5, 3.0}
  3estimator: Proposed / MP / He2023
  MC=1000 trials/point

  output: MSE(A) Curves（3curvesestimatorCurves）

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
from utils.metrics import save_results, compute_mse_bias


# ============================================================
# Single A value trial
# ============================================================

def _run_single_a(
    M, N, lp, kp, A, nu_f, snr_db,
    est_names, delta_k_list, delta_k_sidetone,
    mc_times, seed_base,
):
    """
    unitA valuerunmc_times times Monte Carlo trial。

    Returns:
        results: {est_name: {"mse":..., "bias":...}}
    """
    errors = {name: [] for name in est_names}

    for trial in range(mc_times):
        trial_rng = np.random.default_rng(
            seed_base + trial + int(abs(snr_db) * 1000) + int(A * 1000)
        )

        # generatezero-paddedframe
        X_dd = generate_engineering_frame_zp(M, N, lp, kp, A, trial_rng)

        # Transmission link
        X_tf = isfft_fft(X_dd)
        R_tf = apply_fractional_doppler_tf(X_tf, nu_f)
        R_tf_noisy = add_awgn(R_tf, snr_db, trial_rng)
        Y_dd = sfft_fft(R_tf_noisy)

        # pilot row
        y_row = Y_dd[lp, :]

        # eachestimator
        for name in est_names:
            fn = get_estimator(name)

            if name == "sidetone_only":
                nu_hat = fn(y_row, kp, N, delta_k_list=delta_k_sidetone)
            elif name == "matrix_pencil":
                nu_hat = fn(y_row, kp, N)
            elif name == "he2023":
                nu_hat = fn(y_row, kp, N, M=M, A=A, lp=lp)
            else:
                nu_hat = fn(y_row, kp, N, delta_k_list=delta_k_list)

            errors[name].append(nu_hat - nu_f)

    # computeMSE/bias
    results = {}
    for name in est_names:
        err_arr = np.array(errors[name])
        results[name] = compute_mse_bias(err_arr)

    return results


# ============================================================
# main function
# ============================================================

def run_experiment4_v2(
    mc_times=None,
    snr_db=10.0,
    a_list=None,
    est_names=None,
):
    """
    Experiment 4（v2）main function：pilot powerAsensitivity（zero-paddedframeversion）。

    Difference from original：
      - deleteKmaxSweepdimension，onlySweepA
      - engineering frameusezero-paddedversion(irrationalKmaxparameter)

    Args:
        mc_times: eachgrid pointMC times(default default1000)
        snr_db: FixedSNR（default10dB）
        a_list: pilotmagnitudeSweeplist（default[1.0, 1.5, 3.0]）
        est_names: estimatorlist
    """
    cfg = DEFAULT
    M, N = cfg.M, cfg.N
    lp, kp = cfg.pilot_delay, cfg.pilot_doppler
    nu_f = cfg.nu_f_default
    delta_k_list = cfg.delta_k_list
    delta_k_sidetone = cfg.delta_k_sidetone

    if mc_times is None:
        mc_times = 1000
    if a_list is None:
        a_list = [1.0, 1.5, 3.0]
    if est_names is None:
        est_names = ["proposed", "matrix_pencil", "he2023"]

    print("=" * 80)
    print(f"  Experiment 4(v2): pilot powerAsensitivity [zero-paddedframe]")
    print(f"  nu_f={nu_f}, SNR={snr_db}dB, MC={mc_times}/point")
    print(f"  A={a_list}")
    print(f"  estimator: {est_names}")
    print("=" * 80)

    # resultstructure: results_by_a[a][est_name] = {mse, bias}
    results_by_a = {}
    meta = {
        "mc_times": mc_times,
        "snr_db": snr_db,
        "nu_f": nu_f,
        "frame_type": "engineering_zp",
        "M": M, "N": N, "lp": lp, "kp": kp,
        "a_list": list(a_list),
        "est_names": list(est_names),
    }

    for idx, a in enumerate(a_list):
        t_start = time.time()

        cell_results = _run_single_a(
            M, N, lp, kp, a, nu_f, snr_db,
            est_names, delta_k_list, delta_k_sidetone,
            mc_times, cfg.seed,
        )

        results_by_a[f"{a:.1f}"] = cell_results

        mse_strs = [f"{name}={cell_results[name]['mse']:.2e}" for name in est_names]
        elapsed = time.time() - t_start
        print(f"  [{idx+1}/{len(a_list)}] A={a:.1f} | " +
              " | ".join(mse_strs) + f" | {elapsed:.1f}s")

    # ==================== Print summary ====================
    print("\n" + "=" * 80)
    print(f"  pilotpowerAsensitivitySummary (SNR={snr_db}dB, nu_f={nu_f})")
    print("=" * 80)

    header = f"{'A':>6s}"
    for name in est_names:
        header += f" | {name:>14s} MSE"
    header += f" | {'Prop/MP':>8s}"
    print(header)
    print("-" * (6 + 16 * len(est_names) + 10))

    for a in a_list:
        key = f"{a:.1f}"
        row = f"{a:>6.1f}"
        for name in est_names:
            mse_val = results_by_a[key][name]["mse"]
            row += f" | {mse_val:>14.4e}"
        prop_mse = results_by_a[key]["proposed"]["mse"]
        mp_mse = results_by_a[key]["matrix_pencil"]["mse"]
        ratio = prop_mse / mp_mse if mp_mse > 0 else float("inf")
        row += f" | {ratio:>7.2f}x"
        print(row)

    # biastable
    print(f"\n  Bias Summary")
    print("-" * (6 + 16 * len(est_names)))
    header = f"{'A':>6s}"
    for name in est_names:
        header += f" | {name:>14s} bias"
    print(header)
    print("-" * (6 + 16 * len(est_names)))
    for a in a_list:
        key = f"{a:.1f}"
        row = f"{a:>6.1f}"
        for name in est_names:
            bias_val = results_by_a[key][name]["bias"]
            row += f" | {bias_val:>+14.4e}"
        print(row)

    print("=" * 80)

    # ==================== saveresult ====================
    output = {
        "experiment": "exp4_a_sensitivity_v2",
        "config": meta,
        "results_by_a": results_by_a,
    }

    output_path = os.path.join(PROJECT_ROOT, "results", "exp4_a_sensitivity_v2.json")
    save_results(output, output_path)
    print(f"\nresultalreadysaveto: {output_path}")

    return output


if __name__ == "__main__":
    run_experiment4_v2()
