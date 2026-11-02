# -*- coding: utf-8 -*-
"""
Experiment 8 (revision): Matched-sample MP comparison (fairness)
================================================================

Addresses Reviewer 1 Comment 6 and Reviewer 3 Comment 3:
  - Clarify what observations each method uses
  - Add a matrix-pencil comparison using the SAME observations as proposed

Key observation-set facts (single-path LOS, zero-padded pilot row, N=16):
  - Proposed:      sidelobe pairs (kp ± dk, dk=1..7) + peak kp
                   -> indices {1,...,15} (15 samples; index 0 unused)
  - MP (original): full pilot row indices {0,...,15} (16 samples)
  - MP (matched):  same 15-sample window {1,...,15} as proposed

Two frame types (pure / engineering_zp) x SNR sweep, all four estimators,
plus the matched-sample MP variant. The comparison isolates the effect of
the *method* (phase-sum vs subspace decomposition) from the *sample set*.

Run:
    python experiments/exp8_matched_mp.py
"""

import sys
import os
import time
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.params import DEFAULT
from modem.otfs_modem import isfft_fft, sfft_fft, add_awgn
from frame.frame_gen import generate_pure_frame, generate_engineering_frame_zp
from estimator import get_estimator
from estimator import matrix_pencil
from utils.metrics import compute_mse_bias


def _mp_matched(y_pilot_row, kp, N, n_paths=1, **kwargs):
    """
    MP on the SAME sample window as the proposed estimator.

    Proposed uses indices {kp-dk, ..., kp+dk} for dk=1..K (i.e., for
    N=16, kp=8, K=7: indices 1..15). We pass exactly this contiguous
    window (y_pilot_row[kp-K : kp+K+1]) to the MP routine. Since MP's
    internal length is derived from len(y_dt), passing the 15-sample
    window yields a matched-observation comparison.
    """
    K = (N - 1) // 2
    start = kp - K
    end = kp + K + 1
    window = y_pilot_row[start:end]          # contiguous slice, length 2K+1 = N-1
    # verify the window is exactly the proposed sample set
    assert len(window) == N - 1, f"expected {N-1} samples, got {len(window)}"
    y_dt = np.fft.ifft(window)
    from estimator.matrix_pencil import _mp_core
    return _mp_core(y_dt, n_paths=n_paths)


def _run_single_trial(X_dd, nu_f, snr_db, lp, kp, N, M, rng,
                      est_names, delta_k_list, delta_k_sidetone, A=3.0):
    """One trial: frame -> Doppler+noise -> all estimators (incl. MP-matched)."""
    X_tf = isfft_fft(X_dd)
    R_tf = X_tf * np.exp(1j * 2 * np.pi * nu_f * np.arange(N)[:, np.newaxis] / N)
    R_tf_noisy = add_awgn(R_tf, snr_db, rng)
    Y_dd = sfft_fft(R_tf_noisy)
    y_row = Y_dd[lp, :]

    estimates = {}
    for name in est_names:
        if name == "sidetone_only":
            estimates[name] = get_estimator(name)(y_row, kp, N, delta_k_list=delta_k_sidetone)
        elif name == "matrix_pencil":
            estimates[name] = get_estimator(name)(y_row, kp, N)
        elif name == "matrix_pencil_matched":
            estimates[name] = _mp_matched(y_row, kp, N)
        elif name == "he2023":
            estimates[name] = get_estimator(name)(y_row, kp, N, M=M, A=A, lp=lp)
        else:
            estimates[name] = get_estimator(name)(y_row, kp, N, delta_k_list=delta_k_list)
    return estimates


def run_exp8(cfg=None, nu_f=None, snr_list=None, mc_times=None,
             frame_types=None, est_names=None):
    """Experiment 8 main: matched-sample MP comparison."""
    if cfg is None:
        cfg = DEFAULT
    if nu_f is None:
        nu_f = cfg.nu_f_default
    if snr_list is None:
        snr_list = cfg.snr_range
    if mc_times is None:
        mc_times = cfg.mc_times
    if frame_types is None:
        frame_types = ["pure", "engineering_zp"]
    if est_names is None:
        est_names = ["proposed", "sidetone_only", "matrix_pencil",
                     "matrix_pencil_matched", "he2023"]

    M, N = cfg.M, cfg.N
    lp, kp = cfg.pilot_delay, cfg.pilot_doppler
    A = cfg.pilot_amp
    delta_k_list = cfg.delta_k_list
    delta_k_sidetone = cfg.delta_k_sidetone

    print("#" * 78)
    print("#  Experiment 8: matched-sample MP comparison (fairness)")
    print(f"#  nu_f={nu_f}, SNR={list(snr_list)} dB, MC={mc_times}")
    print(f"#  frame: {frame_types}")
    print(f"#  estimators: {est_names}")
    print("#" * 78)

    all_results = {}
    for frame_type in frame_types:
        print(f"\n--- {frame_type} frame ---")
        frame_data = {}
        for snr_db in snr_list:
            errors = {name: np.zeros(mc_times) for name in est_names}
            t0 = time.time()
            for trial in range(mc_times):
                rng = np.random.default_rng(cfg.seed + trial + int(abs(snr_db) * 1000) + 1)
                if frame_type == "pure":
                    X_dd = generate_pure_frame(M, N, lp, kp, A)
                else:
                    X_dd = generate_engineering_frame_zp(M, N, lp, kp, A, rng)
                ests = _run_single_trial(
                    X_dd, nu_f, snr_db, lp, kp, N, M, rng,
                    est_names, delta_k_list, delta_k_sidetone, A=A,
                )
                for name in est_names:
                    errors[name][trial] = ests[name] - nu_f

            metrics = {name: compute_mse_bias(errors[name]) for name in est_names}
            frame_data[f"snr_{snr_db:.0f}dB"] = {
                "snr_db": float(snr_db),
                "metrics": metrics,
                "elapsed_s": time.time() - t0,
            }
            mp = metrics["matrix_pencil"]["mse"]
            mpm = metrics["matrix_pencil_matched"]["mse"]
            prop = metrics["proposed"]["mse"]
            print(f"  SNR={snr_db:>+5.0f}dB | Prop={prop:.3e} | "
                  f"MP-full={mp:.3e} | MP-matched={mpm:.3e} | "
                  f"MP-full/MP-matched={mp/mpm:.1f}x | {time.time()-t0:.1f}s")

        all_results[frame_type] = frame_data

    # ===== summary table =====
    for frame_type in frame_types:
        print(f"\n{'='*88}")
        print(f"  {frame_type} frame: MSE summary (nu_f={nu_f}, MC={mc_times})")
        print(f"{'='*88}")
        header = f"{'SNR':>6s}"
        for name in est_names:
            header += f" | {name:>20s}"
        print(header)
        print("-" * 88)
        for snr_db in snr_list:
            key = f"snr_{snr_db:.0f}dB"
            metrics = all_results[frame_type][key]["metrics"]
            row = f"{snr_db:>+6.0f}"
            for name in est_names:
                row += f" | {metrics[name]['mse']:>20.3e}"
            print(row)
        # bias
        print(f"\n  Bias summary")
        print("-" * 88)
        for snr_db in snr_list:
            key = f"snr_{snr_db:.0f}dB"
            metrics = all_results[frame_type][key]["metrics"]
            row = f"{snr_db:>+6.0f}"
            for name in est_names:
                row += f" | {metrics[name]['bias']:>+20.3e}"
            print(row)

    output = {
        "experiment": "exp8_matched_mp",
        "config": {
            "M": M, "N": N, "lp": lp, "kp": kp, "A": A,
            "nu_f": nu_f, "snr_list": list(snr_list),
            "mc_times": mc_times,
            "delta_k_list": list(delta_k_list),
            "delta_k_sidetone": list(delta_k_sidetone),
            "frame_types": frame_types,
            "est_names": est_names,
            "seed": cfg.seed,
            "observation_facts": (
                "proposed uses indices {kp-K..kp+K} = {1..15} (15 samples); "
                "MP-full uses all {0..15} (16 samples); "
                "MP-matched uses the same {1..15} window as proposed"
            ),
        },
        "results": all_results,
    }
    results_path = os.path.join(PROJECT_ROOT, "results", "exp8_matched_mp.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    return output


if __name__ == "__main__":
    run_exp8()
