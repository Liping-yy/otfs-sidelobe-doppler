# -*- coding: utf-8 -*-
"""
Experiment 3：Full-range nufrobustnessandSingularity pointelimination
=================================

Corresponding to experiment outline Experiment 3。

pure frame，SNR=10dBFixed，νf∈(-0.5, 0.5)full rangeSweepstep size0.01，
zoom-in(-0.1, 0.1)step size0.005。
2Curves：Proposed(sidelobe+Peak self-multiplication) vs onlysidelobe(Ablation)。
MC=1000 times/point。

Expected：ProposedSmooth throughout；onlysidelobeat nu_f=0nearMSEjumpsnumberorders of magnitude。

Output：1figure（Full-range nuf-MSE，Embedded singularity neighborhood zoom subplot）。

"""

import sys
import os
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.params import DEFAULT
from modem.otfs_modem import (
    isfft_fft, sfft_fft,
    apply_fractional_doppler_tf,
    add_awgn,
)
from frame.frame_gen import generate_pure_frame
from estimator import get_estimator
from utils.metrics import save_results, compute_mse_bias


# ============================================================
# unit times trial
# ============================================================

def _run_single_trial(nu_f, snr_db, M, N, lp, kp, A, rng,
                      delta_k_list, delta_k_sidetone, est_names):
    """pure frameunit times trial，returneachestimatornu_f_hat。"""
    X_dd = generate_pure_frame(M, N, lp, kp, A)
    X_tf = isfft_fft(X_dd)
    R_tf = apply_fractional_doppler_tf(X_tf, nu_f)
    R_tf_noisy = add_awgn(R_tf, snr_db, rng)
    Y_dd = sfft_fft(R_tf_noisy)
    y_row = Y_dd[lp, :]

    estimates = {}
    for name in est_names:
        fn = get_estimator(name)
        if name == "sidetone_only":
            estimates[name] = fn(y_row, kp, N, delta_k_list=delta_k_sidetone)
        else:
            estimates[name] = fn(y_row, kp, N, delta_k_list=delta_k_list)
    return estimates


# ============================================================
# Sweep1groupsνf
# ============================================================

def _sweep_nu_f(nu_f_list, snr_db, M, N, lp, kp, A, seed,
                mc_times, delta_k_list, delta_k_sidetone, est_names):
    """for1groupsνfvalueSweep，return {nu_f: {est_name: metrics}}。"""
    results = {}

    for i, nu_f in enumerate(nu_f_list):
        errors = {name: np.zeros(mc_times) for name in est_names}

        for trial in range(mc_times):
            trial_rng = np.random.default_rng(seed + trial + int(abs(nu_f) * 1e6) + 1)
            ests = _run_single_trial(
                nu_f, snr_db, M, N, lp, kp, A, trial_rng,
                delta_k_list, delta_k_sidetone, est_names
            )
            for name in est_names:
                errors[name][trial] = ests[name] - nu_f

        metrics = {}
        for name in est_names:
            metrics[name] = compute_mse_bias(errors[name])
        results[nu_f] = metrics

    return results


# ============================================================
# main function
# ============================================================

def run_experiment3(cfg=None, snr_db=10.0, mc_times=1000,
                    nu_f_step=0.01, nu_f_local_step=0.005,
                    est_names=None):
    """
    Experiment 3main function：Full-range nufrobustnessandSingularity pointelimination。

    Args:
        cfg: SimConfigInstance
        snr_db: FixedSNR（default10dB）
        mc_times: eachnu_f ptsMC times(default default1000)
        nu_f_step: full rangestep size（default0.01）
        nu_f_local_step: zoom-instep size（default0.005）
        est_names: estimator list（defaultProposed + onlysidelobe）

    Returns:
        results: dict
    """
    if cfg is None:
        cfg = DEFAULT
    if est_names is None:
        est_names = ["proposed", "sidetone_only"]

    M, N = cfg.M, cfg.N
    lp, kp = cfg.pilot_delay, cfg.pilot_doppler
    A = cfg.pilot_amp
    seed = cfg.seed
    delta_k_list = cfg.delta_k_list
    delta_k_sidetone = cfg.delta_k_sidetone

    # generatesweep points
    nu_f_full = np.arange(-0.5 + nu_f_step, 0.5, nu_f_step)
    nu_f_local = np.arange(-0.1, 0.1 + nu_f_local_step, nu_f_local_step)
    # Merge and deduplicate（np.uniqueautohandling+0.0/-0.0）
    nu_f_all = np.unique(np.concatenate([nu_f_full, nu_f_local]))

    print("#" * 75)
    print("#  Experiment 3: Full-range nu_f robustnessandSingularity pointelimination")
    print(f"#  SNR={snr_db}dB, MC={mc_times}/point")
    print(f"#  Full-range: {len(nu_f_full)} points, step size{nu_f_step}")
    print(f"#  zoom-in: {len(nu_f_local)} points, step size{nu_f_local_step}")
    print(f"#  total: {len(nu_f_all)} pts")
    print(f"#  estimator: {est_names}")
    print("#" * 75)

    t_start = time.time()

    # Full-rangeSweep（includingzoom-in points)
    all_data = _sweep_nu_f(
        nu_f_all, snr_db, M, N, lp, kp, A, seed,
        mc_times, delta_k_list, delta_k_sidetone, est_names
    )

    t_elapsed = time.time() - t_start
    print(f"\nTotal elapsed time: {t_elapsed:.1f}s ({t_elapsed/60:.1f}min)")

    # ==================== Print key points ====================
    key_points = [-0.45, -0.30, -0.10, -0.05, -0.02, 0.00, 0.02, 0.05, 0.10, 0.30, 0.45]
    print(f"\n{'='*70}")
    print(f"  Key points MSE (SNR={snr_db}dB, MC={mc_times})")
    print(f"{'='*70}")
    print(f"{'nu_f':>8s} | {'Proposed MSE':>14s} | {'SideOnly MSE':>14s} | {'Ratio':>8s}")
    print("-" * 70)

    for nu_f in key_points:
        # Find nearest sweep point
        idx = np.argmin(np.abs(nu_f_all - nu_f))
        actual_nu = nu_f_all[idx]
        m = all_data[actual_nu]
        prop_mse = m["proposed"]["mse"]
        side_mse = m["sidetone_only"]["mse"]
        ratio = prop_mse / side_mse if side_mse > 0 else 0
        print(f"{actual_nu:>+8.3f} | {prop_mse:>14.4e} | {side_mse:>14.4e} | {ratio:>7.4f}x")

    print(f"{'='*70}")

    # Singularity neighborhood detailed print
    local_zone = nu_f_all[(nu_f_all >= -0.1) & (nu_f_all <= 0.1)]
    print(f"\n  singularityneighborhood nu_f in [-0.1, 0.1] detailed ({len(local_zone)} points)")
    print(f"{'nu_f':>8s} | {'Proposed MSE':>14s} | {'SideOnly MSE':>14s} | {'Prop bias':>12s} | {'Side bias':>12s}")
    print("-" * 70)
    for nu_f in local_zone:
        m = all_data[nu_f]
        print(f"{nu_f:>+8.3f} | {m['proposed']['mse']:>14.4e} | "
              f"{m['sidetone_only']['mse']:>14.4e} | "
              f"{m['proposed']['bias']:>+12.4e} | "
              f"{m['sidetone_only']['bias']:>+12.4e}")

    print(f"{'='*70}")

    # ==================== saveresult ====================
    # serialize：nu_fas stringkey
    results_serializable = {}
    for nu_f in nu_f_all:
        results_serializable[f"{nu_f:.4f}"] = {
            "nu_f": float(nu_f),
            "metrics": all_data[nu_f],
        }

    output = {
        "experiment": "exp3_full_nu_sweep",
        "config": {
            "M": M, "N": N, "lp": lp, "kp": kp, "A": A,
            "snr_db": snr_db,
            "mc_times": mc_times,
            "nu_f_step": nu_f_step,
            "nu_f_local_step": nu_f_local_step,
            "nu_f_full_range": [-0.5, 0.5],
            "nu_f_local_range": [-0.1, 0.1],
            "delta_k_list": delta_k_list,
            "delta_k_sidetone": delta_k_sidetone,
            "est_names": est_names,
            "seed": seed,
        },
        "n_points": len(nu_f_all),
        "elapsed_s": t_elapsed,
        "results": results_serializable,
    }

    results_path = os.path.join(PROJECT_ROOT, "results", "exp3_full_nu_sweep.json")
    save_results(output, results_path)

    return output


if __name__ == "__main__":
    # completerun：MC=1000, Full-range+zoom-in
    # quicktest: run_experiment3(mc_times=50, nu_f_step=0.1, nu_f_local_step=0.02)
    results = run_experiment3()
