# -*- coding: utf-8 -*-
"""
Experiment 6（v2）：complexitycomparison —— zero-paddedframeversion
=======================================

Based on exp6_complexity.py modified，adaptednewpaperFrame structure：
  - engineering framechangeaspilot row fully zero-padded（No Kmax concept）
  - timinglogicunchanged，onlyframe generationequationformulachangetransform

"""

import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.params import DEFAULT
from modem.otfs_modem import isfft_fft, sfft_fft, apply_fractional_doppler_tf, add_awgn
from frame.frame_gen import generate_engineering_frame_zp
from estimator import get_estimator
from utils.metrics import save_results


# ============================================================
# theoreticalcomplexityparametersnapshot
# ============================================================

THEORETICAL_COMPLEXITY = {
    "proposed": {
        "big_o": "O(K)",
        "flops_per_call": "7*(2 mul + 1 angle) + 1 angle + 1 div",
        "description": "Ksidelobeforcomplex multiplication+amplitudeangle+weightedvectorsummation+1 timesclosed-formremovemethod",
        "note": "K=7 (delta_k_list), irrationalsearchirrationaliteration",
    },
    "sidetone_only": {
        "big_o": "O(K)",
        "flops_per_call": "4*(2 mul + 1 angle) + 1 angle + 1 div",
        "description": "Ksidelobeforcomplex multiplication+amplitudeangle+weightedvectorsummation+1 timesclosed-formremovemethod",
        "note": "K=4 (delta_k_sidetone), irrationalsearchirrationaliteration, irrationalPeak self-multiplication",
    },
    "matrix_pencil": {
        "big_o": "O(N^3)",
        "flops_per_call": "N^3 (Hankel SVD) + N^2 (Generalized eigenvalue)",
        "description": "IDFT + Hankelmatrixconstruct + SVD/Generalized eigenvalue decomposition",
        "note": "N=16, needintegerrowpilot, scipy.linalg.eig",
    },
    "he2023": {
        "big_o": "O(G * N * log N)",
        "flops_per_call": "G*(ISFFT + Doppler + SFFT + inner_product) + BFGS_iter*(residual_eval)",
        "description": "G=101grid ptsmodeltemplatematch(each ptscompleteOTFSlink) + BFGSiteration(approximately10-30step)",
        "note": "G=101, eachgrid ptsneedISFFT+SFFT, BFGSeachstepneed1 timesmodeltemplategenerate+Residualcompute",
    },
}


# ============================================================
# timingexperiment
# ============================================================

def run_experiment6_v2(
    n_trials=500,
    snr_db=10.0,
    nu_f=0.3,
):
    """
    Experiment 6（v2）main function：complexitycomparison（zero-paddedframeversion）。

    Difference from original：frame generationusezero-paddedversion，irrationalKmaxparameter。
    timinglogiccompleteallsame。
    """
    cfg = DEFAULT
    M, N = cfg.M, cfg.N
    lp, kp = cfg.pilot_delay, cfg.pilot_doppler
    A = cfg.pilot_amp
    delta_k_list = cfg.delta_k_list
    delta_k_sidetone = cfg.delta_k_sidetone

    est_names = ["proposed", "sidetone_only", "matrix_pencil", "he2023"]

    print("=" * 80)
    print(f"  Experiment 6(v2): complexitycomparison [zero-paddedframe]")
    print(f"  n_trials={n_trials}, SNR={snr_db}dB, nu_f={nu_f}")
    print(f"  frame: engineering_zp (A={A}, pilot row zero-padded)")
    print(f"  estimator: {est_names}")
    print("=" * 80)

    # preventgeneratealltestframe
    print("  preventgeneratetestframe...")
    rng_master = np.random.default_rng(cfg.seed)
    y_rows = []
    for _ in range(n_trials + 10):
        trial_rng = np.random.default_rng(rng_master.integers(0, 2**31))
        X_dd = generate_engineering_frame_zp(M, N, lp, kp, A, trial_rng)
        X_tf = isfft_fft(X_dd)
        R_tf = apply_fractional_doppler_tf(X_tf, nu_f)
        R_tf_noisy = add_awgn(R_tf, snr_db, trial_rng)
        Y_dd = sfft_fft(R_tf_noisy)
        y_rows.append(Y_dd[lp, :])
    print(f"  alreadygenerate {len(y_rows)} framepilot row")

    # timing
    results = {
        "meta": {
            "n_trials": n_trials,
            "snr_db": snr_db,
            "nu_f": nu_f,
            "frame_type": "engineering_zp",
            "M": M, "N": N, "lp": lp, "kp": kp, "A": A,
            "delta_k_list": list(delta_k_list),
            "delta_k_sidetone": list(delta_k_sidetone),
            "frame_note": "pilot row zero-padded, no Kmax",
        },
        "timing": {},
        "theoretical": THEORETICAL_COMPLEXITY,
    }

    for name in est_names:
        fn = get_estimator(name)

        # warmup
        for i in range(10):
            y_row = y_rows[i]
            if name == "sidetone_only":
                fn(y_row, kp, N, delta_k_list=delta_k_sidetone)
            elif name == "matrix_pencil":
                fn(y_row, kp, N)
            elif name == "he2023":
                fn(y_row, kp, N, M=M, A=A, lp=lp)
            else:
                fn(y_row, kp, N, delta_k_list=delta_k_list)

        # positiveformulatiming
        times = []
        for i in range(n_trials):
            y_row = y_rows[10 + i]

            t_start = time.perf_counter()

            if name == "sidetone_only":
                fn(y_row, kp, N, delta_k_list=delta_k_sidetone)
            elif name == "matrix_pencil":
                fn(y_row, kp, N)
            elif name == "he2023":
                fn(y_row, kp, N, M=M, A=A, lp=lp)
            else:
                fn(y_row, kp, N, delta_k_list=delta_k_list)

            t_end = time.perf_counter()
            times.append((t_end - t_start) * 1e6)

        times = np.array(times)

        results["timing"][name] = {
            "mean_us": float(np.mean(times)),
            "std_us": float(np.std(times)),
            "median_us": float(np.median(times)),
            "min_us": float(np.min(times)),
            "max_us": float(np.max(times)),
            "p95_us": float(np.percentile(times, 95)),
            "n_trials": n_trials,
        }

        mean_val = results["timing"][name]["mean_us"]
        std_val = results["timing"][name]["std_us"]
        median_val = results["timing"][name]["median_us"]
        big_o = THEORETICAL_COMPLEXITY[name]["big_o"]
        print(f"  {name:20s} | mean={mean_val:10.2f} us | std={std_val:8.2f} | median={median_val:10.2f} us | {big_o}")

    # Summarytable
    print("\n" + "=" * 80)
    print("  Summarytable")
    print("=" * 80)
    print(f"  {'Estimator':<20s} | {'Mean (us)':<12s} | {'Median (us)':<12s} | {'Complexity':<12s} | {'vs Proposed':<12s}")
    print("  " + "-" * 76)

    prop_mean = results["timing"]["proposed"]["mean_us"]
    for name in est_names:
        mean_val = results["timing"][name]["mean_us"]
        median_val = results["timing"][name]["median_us"]
        big_o = THEORETICAL_COMPLEXITY[name]["big_o"]
        ratio = mean_val / prop_mean if prop_mean > 0 else float("inf")
        print(f"  {name:<20s} | {mean_val:<12.2f} | {median_val:<12.2f} | {big_o:<12s} | {ratio:<12.1f}x")

    output_path = os.path.join(PROJECT_ROOT, "results", "exp6_complexity_v2.json")
    save_results(results, output_path)
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_experiment6_v2()
