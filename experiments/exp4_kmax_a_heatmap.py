# -*- coding: utf-8 -*-
"""
Experiment 4: engineeringsincefactorsensitivity (Kmax x A heatmap)
=========================================

Corresponding to experiment outline Experiment 4。

Design:
  Fixed SNR=10dB, nu_f=0.3, engineering frame
  Sweeptwoengineeringparameter:
    Kmax ∈ {2, 5, 10, 15}  (Guard zero one-sided length)
    A   ∈ {1.0, 1.5, 3.0}  (pilot amplitude boost)

  3estimator: Proposed / MP / He2023
  MC=1000 trials/cell

  output: 3heatmap (MSE over Kmax x A)
        eachfigureas1estimator MSE(Kmax, A) matrix

  Physical meaning:
    - Kmaxlarge → guard zero wider → IDIinterferencefew → MSElow
    - Alarge → pilot power boost → sidelobeSNRhigh → MSElow
    - Constraint: Kmax >= 1/(rho*A), rho=0.1 → Kmax_min = 10(A=1), 6.67(A=1.5), 3.33(A=3)
    - Kmax < Kmax_min timeguard zero insufficient, systematicbiasoutputappear

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
from utils.metrics import save_results, compute_mse_bias


# ============================================================
# unitcell trial: given(Kmax, A, SNR, nu_f)lowerMC timesEstimate
# ============================================================

def _run_single_cell(
    M, N, lp, kp, A, Kmax, nu_f, snr_db,
    est_names, delta_k_list, delta_k_sidetone,
    mc_times, seed_base,
):
    """
    unit(Kmax, A)grid pointrunmc_times times Monte Carlo trial。

    Returns:
        results: {est_name: {"mse":..., "bias":...}}
    """
    errors = {name: [] for name in est_names}

    for trial in range(mc_times):
        trial_rng = np.random.default_rng(
            seed_base + trial + int(abs(snr_db) * 1000) + int(A * 1000) + Kmax * 100
        )

        # Generate engineering frame
        X_dd = generate_engineering_frame(M, N, lp, kp, A, Kmax, trial_rng)

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

def run_experiment4(
    mc_times=None,
    snr_db=10.0,
    kmax_list=None,
    a_list=None,
    est_names=None,
):
    """
    runExperiment 4: engineeringsincefactorsensitivity (Kmax x A heatmap)。

    Args:
        mc_times: eachgrid pointMC times(default default1000)
        snr_db: FixedSNR（default10dB）
        kmax_list: KmaxSweeplist（default[2,5,10,15]）
        a_list: pilotmagnitudeSweeplist（default[1.0,1.5,3.0]）
        est_names: estimatorlist（default["proposed","matrix_pencil","he2023"]）
    """
    cfg = DEFAULT
    M, N = cfg.M, cfg.N
    lp, kp = cfg.pilot_delay, cfg.pilot_doppler
    nu_f = cfg.nu_f_default
    delta_k_list = cfg.delta_k_list
    delta_k_sidetone = cfg.delta_k_sidetone

    if mc_times is None:
        mc_times = 1000
    if kmax_list is None:
        kmax_list = [2, 5, 10, 15]
    if a_list is None:
        a_list = [1.0, 1.5, 3.0]
    if est_names is None:
        est_names = ["proposed", "matrix_pencil", "he2023"]

    print("=" * 80)
    print(f"  Experiment 4: engineeringsincefactorsensitivity (Kmax x A heatmap)")
    print(f"  nu_f={nu_f}, SNR={snr_db}dB, MC={mc_times}/cell")
    print(f"  Kmax={kmax_list}, A={a_list}")
    print(f"  estimator: {est_names}")
    print(f"  Total grid points: {len(kmax_list)} x {len(a_list)} = {len(kmax_list)*len(a_list)}")
    print("=" * 80)

    # resultstructure: heatmap[est_name][Kmax][A] = {mse, bias}
    results = {
        "meta": {
            "mc_times": mc_times,
            "snr_db": snr_db,
            "nu_f": nu_f,
            "frame_type": "engineering",
            "M": M, "N": N, "lp": lp, "kp": kp,
            "kmax_list": list(kmax_list),
            "a_list": list(a_list),
            "est_names": list(est_names),
        },
        "heatmap": {name: {} for name in est_names},
    }

    # theoreticalminimumKmaxreference
    rho = 0.1
    print(f"  theoreticalminimumKmax (rho={rho}):")
    for a in a_list:
        kmin = 1.0 / (rho * a)
        print(f"    A={a:.1f} -> Kmax_min={kmin:.2f}")

    total_cells = len(kmax_list) * len(a_list)
    cell_idx = 0

    for kmax in kmax_list:
        for a in a_list:
            cell_idx += 1
            t_start = time.time()

            cell_results = _run_single_cell(
                M, N, lp, kp, a, kmax, nu_f, snr_db,
                est_names, delta_k_list, delta_k_sidetone,
                mc_times, cfg.seed,
            )

            # Store
            for name in est_names:
                if str(kmax) not in results["heatmap"][name]:
                    results["heatmap"][name][str(kmax)] = {}
                results["heatmap"][name][str(kmax)][f"{a:.1f}"] = cell_results[name]

            # print
            mse_strs = [f"{name}={cell_results[name]['mse']:.2e}" for name in est_names]
            elapsed = time.time() - t_start
            print(f"  [{cell_idx}/{total_cells}] Kmax={kmax:2d}, A={a:.1f} | " +
                  " | ".join(mse_strs) + f" | {elapsed:.1f}s")

    # printheatmapSummary
    print("\n" + "=" * 80)
    print("  heatmapSummary (MSE)")
    print("=" * 80)

    for name in est_names:
        print(f"\n  === {name} ===")
        header_label = "Kmax\\A"
        print(f"  {header_label:<8s}", end="")
        for a in a_list:
            print(f"  A={a:<5.1f}", end="")
        print()
        print("  " + "-" * (8 + 10 * len(a_list)))

        for kmax in kmax_list:
            print(f"  {kmax:<8d}", end="")
            for a in a_list:
                mse_val = results["heatmap"][name][str(kmax)][f"{a:.1f}"]["mse"]
                print(f"  {mse_val:<10.2e}", end="")
            # mark guard zero zero insufficient
            kmin = 1.0 / (rho * a) if a > 0 else 999
            print()

    # saveresult
    output_path = os.path.join(PROJECT_ROOT, "results", "exp4_kmax_a_heatmap.json")
    save_results(results, output_path)
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_experiment4()
