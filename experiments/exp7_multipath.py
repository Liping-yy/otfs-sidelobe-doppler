# -*- coding: utf-8 -*-
"""
Experiment: Multipath / Rician channel and residual synchronization error
=========================================================================

Four sub-experiments:
  - Finite-K Rician evaluation (K = 0, 5, 10, 15, 20 dB)
  - Weak second-path deterministic scenario (K = 10 dB single echo)
  - Residual timing offset (fractional delay lambda sweep)
  - Residual frequency offset (delta_nu sweep)

Channel model: channel/rician_channel.py (TF-domain exact multipath)

Key physics:
  - Under the single-path TF-multiplicative channel, cross-row interference
    is exactly zero (Lemma 1 of the paper).
  - Under Rician/multipath, scattered paths carry their own delay rows and
    leak into the pilot row -> genuine cross-row interference appears.
  - Residual fractional delay (timing offset) breaks the delay-row
    orthogonality even for a single path -> cross-row leakage.
  - The proposed estimator's bias scales as ~ 1/K (NLOS/LOS power ratio).

Run:
    python experiments/exp7_multipath.py
"""

import sys
import os
import time
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.params import DEFAULT, get_config
from modem.otfs_modem import isfft_fft, sfft_fft, add_awgn
from frame.frame_gen import generate_pure_frame, generate_engineering_frame_zp
from channel.rician_channel import rician_paths, apply_rician_channel
from estimator import get_estimator
from utils.metrics import compute_mse_bias


# ============================================================
# Single-trial runner
# ============================================================

def _run_single_trial(X_dd, paths, lambda_delay, snr_db, lp, kp, N, M, rng,
                      est_names, delta_k_list, delta_k_sidetone, A=3.0):
    """One trial: frame -> Rician channel (+ residual delay) -> noise -> all estimators."""
    X_tf = isfft_fft(X_dd)
    R_tf = apply_rician_channel(X_tf, paths, lambda_delay=lambda_delay)
    R_tf_noisy = add_awgn(R_tf, snr_db, rng)
    Y_dd = sfft_fft(R_tf_noisy)
    y_row = Y_dd[lp, :]

    estimates = {}
    for name in est_names:
        fn = get_estimator(name)
        if name == "sidetone_only":
            estimates[name] = fn(y_row, kp, N, delta_k_list=delta_k_sidetone)
        elif name in ("matrix_pencil", "prony"):
            estimates[name] = fn(y_row, kp, N)
        elif name == "he2023":
            estimates[name] = fn(y_row, kp, N, M=M, A=A, lp=lp)
        else:
            estimates[name] = fn(y_row, kp, N, delta_k_list=delta_k_list)
    return estimates


# ============================================================
# Experiment 7a: Rician K-factor sweep
# ============================================================

def run_k_sweep(cfg=None, k_db_list=None, snr_db=10.0, mc_times=None,
                nu_f=0.3, p_s=2, frame_type="engineering_zp"):
    """
    MSE of all estimators vs Rician K-factor at fixed SNR.

    Args:
        cfg: SimConfig
        k_db_list: K values in dB (default [0, 5, 10, 15, 20])
        snr_db: fixed global SNR
        mc_times: Monte Carlo count
        nu_f: LOS fractional Doppler
        p_s: number of scattered paths
        frame_type: "pure" or "engineering_zp"
    """
    if cfg is None:
        cfg = DEFAULT
    if k_db_list is None:
        k_db_list = [0.0, 5.0, 10.0, 15.0, 20.0]
    if mc_times is None:
        mc_times = 1000

    M, N = cfg.M, cfg.N
    lp, kp = cfg.pilot_delay, cfg.pilot_doppler
    A = cfg.pilot_amp
    delta_k_list = cfg.delta_k_list
    delta_k_sidetone = cfg.delta_k_sidetone
    est_names = ["proposed", "sidetone_only", "matrix_pencil", "he2023"]

    print("#" * 78)
    print("#  Exp 7a: Rician K-factor sweep  (SNR=%g dB, MC=%d, P_s=%d, nu_f=%g)"
          % (snr_db, mc_times, p_s, nu_f))
    print(f"#  frame = {frame_type}")
    print("#" * 78)

    results = {}
    for k_db in k_db_list:
        errors = {name: np.zeros(mc_times) for name in est_names}
        t0 = time.time()
        for trial in range(mc_times):
            rng = np.random.default_rng(cfg.seed + trial + int(k_db * 100))

            if frame_type == "pure":
                X_dd = generate_pure_frame(M, N, lp, kp, A)
            else:
                X_dd = generate_engineering_frame_zp(M, N, lp, kp, A, rng)

            # Rician path set (fixed seed per K so the comparison is fair
            # across estimators; re-drawn per trial otherwise)
            paths = rician_paths(
                M, N, K_db=k_db, P_s=p_s, nu_f=nu_f,
                nu_scatter_std=0.05, l_pilot=lp,
                seed=cfg.seed + trial,
            )
            ests = _run_single_trial(
                X_dd, paths, 0.0, snr_db, lp, kp, N, M, rng,
                est_names, delta_k_list, delta_k_sidetone, A=A,
            )
            for name in est_names:
                errors[name][trial] = ests[name] - nu_f

        metrics = {name: compute_mse_bias(errors[name]) for name in est_names}
        results[f"K_{k_db:.0f}dB"] = {
            "k_db": float(k_db),
            "metrics": metrics,
            "elapsed_s": time.time() - t0,
        }
        prop = metrics["proposed"]["mse"]
        mp = metrics["matrix_pencil"]["mse"]
        print(f"  K={k_db:>5.1f}dB | Prop={prop:.3e} | SideOnly="
              f"{metrics['sidetone_only']['mse']:.3e} | MP={mp:.3e} | "
              f"He={metrics['he2023']['mse']:.3e} | Prop/MP={prop/mp:.1f}x | "
              f"{time.time()-t0:.1f}s")

    return {"experiment": "exp7a_k_sweep", "config": {
        "snr_db": snr_db, "mc_times": mc_times, "nu_f": nu_f,
        "p_s": p_s, "frame_type": frame_type,
        "k_db_list": list(k_db_list),
    }, "results": results}


# ============================================================
# Experiment 7b: Weak second-path deterministic scenario
# ============================================================

def run_weak_echo(cfg=None, k_db=10.0, snr_db=10.0, mc_times=None,
                  nu_f=0.3, delay_offsets=None):
    """
    Weak second-path (single echo) at fixed K = 10 dB.

    Scattered Doppler offset from LOS and delay-row offset are swept to
    show the bias dependence on the echo geometry.

    Args:
        cfg: SimConfig
        k_db: fixed K-factor (default 10 dB)
        snr_db: fixed global SNR
        delay_offsets: list of echo delay-row offsets from lp
    """
    if cfg is None:
        cfg = DEFAULT
    if mc_times is None:
        mc_times = 1000
    if delay_offsets is None:
        delay_offsets = [1, 3, 5, 8]

    M, N = cfg.M, cfg.N
    lp, kp = cfg.pilot_delay, cfg.pilot_doppler
    A = cfg.pilot_amp
    delta_k_list = cfg.delta_k_list
    delta_k_sidetone = cfg.delta_k_sidetone
    est_names = ["proposed", "sidetone_only", "matrix_pencil", "he2023"]

    print("#" * 78)
    print(f"#  Exp 7b: Weak second-path (K={k_db}dB single echo, SNR={snr_db}dB, MC={mc_times})")
    print("#" * 78)

    results = {}
    for d_off in delay_offsets:
        errors = {name: np.zeros(mc_times) for name in est_names}
        for trial in range(mc_times):
            rng = np.random.default_rng(cfg.seed + trial + d_off * 100)

            X_dd = generate_engineering_frame_zp(M, N, lp, kp, A, rng)
            paths = rician_paths(
                M, N, K_db=k_db, P_s=1, nu_f=nu_f,
                nu_scatter_std=0.05, l_pilot=lp,
                seed=cfg.seed + trial,
            )
            # force the single scattered path to a specific delay offset
            echo_delay = (lp + d_off) % M
            paths[1]["l"] = echo_delay

            ests = _run_single_trial(
                X_dd, paths, 0.0, snr_db, lp, kp, N, M, rng,
                est_names, delta_k_list, delta_k_sidetone, A=A,
            )
            for name in est_names:
                errors[name][trial] = ests[name] - nu_f

        metrics = {name: compute_mse_bias(errors[name]) for name in est_names}
        results[f"d_off_{d_off}"] = {
            "delay_offset": float(d_off),
            "metrics": metrics,
            "elapsed_s": time.time(),
        }
        print(f"  delay offset {d_off:>3d} | Prop MSE={metrics['proposed']['mse']:.3e} "
              f"bias={metrics['proposed']['bias']:.3e} | "
              f"MP MSE={metrics['matrix_pencil']['mse']:.3e}")

    return {"experiment": "exp7b_weak_echo", "config": {
        "k_db": k_db, "snr_db": snr_db, "mc_times": mc_times, "nu_f": nu_f,
        "delay_offsets": list(delay_offsets),
    }, "results": results}


# ============================================================
# Experiment 7c: Residual timing offset (fractional delay) sweep
# ============================================================

def run_timing_sweep(cfg=None, lambda_list=None, snr_db=10.0, mc_times=None,
                     nu_f=0.3, k_db=40.0):
    """
    Residual timing offset (fractional delay) sweep under near-LOS channel.

    lambda in {0, ±0.05, ±0.1, ±0.2} delay-tap units.
    Even for a single path, fractional delay creates cross-row leakage.

    Args:
        cfg: SimConfig
        lambda_list: fractional delay values
        snr_db: fixed global SNR
        k_db: Rician K-factor (default 40 dB ~ effectively single-path)
    """
    if cfg is None:
        cfg = DEFAULT
    if lambda_list is None:
        lambda_list = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
    if mc_times is None:
        mc_times = 1000

    M, N = cfg.M, cfg.N
    lp, kp = cfg.pilot_delay, cfg.pilot_doppler
    A = cfg.pilot_amp
    delta_k_list = cfg.delta_k_list
    delta_k_sidetone = cfg.delta_k_sidetone
    est_names = ["proposed", "sidetone_only", "matrix_pencil", "he2023"]

    print("#" * 78)
    print(f"#  Exp 7c: Residual timing offset sweep (K={k_db}dB, SNR={snr_db}dB, MC={mc_times})")
    print(f"#  lambda in {lambda_list} (delay-tap units)")
    print("#" * 78)

    results = {}
    for lam in lambda_list:
        errors = {name: np.zeros(mc_times) for name in est_names}
        for trial in range(mc_times):
            rng = np.random.default_rng(cfg.seed + trial + int(abs(lam) * 1000))

            X_dd = generate_engineering_frame_zp(M, N, lp, kp, A, rng)
            paths = rician_paths(
                M, N, K_db=k_db, P_s=1, nu_f=nu_f,
                nu_scatter_std=0.05, l_pilot=lp,
                seed=cfg.seed + trial,
            )
            # at K=40dB the scattered path is ~100x weaker; effectively LOS
            ests = _run_single_trial(
                X_dd, paths, lam, snr_db, lp, kp, N, M, rng,
                est_names, delta_k_list, delta_k_sidetone, A=A,
            )
            for name in est_names:
                errors[name][trial] = ests[name] - nu_f

        metrics = {name: compute_mse_bias(errors[name]) for name in est_names}
        results[f"lambda_{lam:+.2f}"] = {
            "lambda": float(lam),
            "metrics": metrics,
        }
        print(f"  lambda={lam:+.2f} | Prop MSE={metrics['proposed']['mse']:.3e} "
              f"bias={metrics['proposed']['bias']:.3e} | "
              f"MP MSE={metrics['matrix_pencil']['mse']:.3e}")

    return {"experiment": "exp7c_timing_sweep", "config": {
        "snr_db": snr_db, "mc_times": mc_times, "nu_f": nu_f, "k_db": k_db,
        "lambda_list": list(lambda_list),
    }, "results": results}


# ============================================================
# Experiment 7d: Residual frequency offset sweep
# ============================================================

def run_freq_sweep(cfg=None, delta_nu_list=None, snr_db=10.0, mc_times=None,
                   nu_f=0.3, k_db=40.0):
    """
    Residual frequency offset sweep: the true fractional Doppler is
    nu_true = nu_f + delta_nu; the estimators are evaluated against nu_f.

    This mimics the case where the receiver's coarse Doppler pre-compensation
    leaves a small residual frequency error delta_nu on top of the fractional
    Doppler the estimator must resolve.

    Args:
        cfg: SimConfig
        delta_nu_list: residual frequency offsets (default [-0.1, -0.05, 0, 0.05, 0.1])
        snr_db: fixed global SNR
        k_db: Rician K-factor
    """
    if cfg is None:
        cfg = DEFAULT
    if delta_nu_list is None:
        delta_nu_list = [-0.1, -0.05, 0.0, 0.05, 0.1]
    if mc_times is None:
        mc_times = 1000

    M, N = cfg.M, cfg.N
    lp, kp = cfg.pilot_delay, cfg.pilot_doppler
    A = cfg.pilot_amp
    delta_k_list = cfg.delta_k_list
    delta_k_sidetone = cfg.delta_k_sidetone
    est_names = ["proposed", "sidetone_only", "matrix_pencil", "he2023"]

    print("#" * 78)
    print(f"#  Exp 7d: Residual frequency offset sweep (K={k_db}dB, SNR={snr_db}dB, MC={mc_times})")
    print(f"#  delta_nu in {delta_nu_list}")
    print("#" * 78)

    results = {}
    for dnu in delta_nu_list:
        nu_true = nu_f + dnu
        errors = {name: np.zeros(mc_times) for name in est_names}
        for trial in range(mc_times):
            rng = np.random.default_rng(cfg.seed + trial + int(abs(dnu) * 1000))

            X_dd = generate_engineering_frame_zp(M, N, lp, kp, A, rng)
            paths = rician_paths(
                M, N, K_db=k_db, P_s=1, nu_f=nu_true,
                nu_scatter_std=0.05, l_pilot=lp,
                seed=cfg.seed + trial,
            )
            ests = _run_single_trial(
                X_dd, paths, 0.0, snr_db, lp, kp, N, M, rng,
                est_names, delta_k_list, delta_k_sidetone, A=A,
            )
            # evaluate error against the *nominal* nu_f the receiver assumes
            for name in est_names:
                errors[name][trial] = ests[name] - nu_f

        metrics = {name: compute_mse_bias(errors[name]) for name in est_names}
        results[f"dnu_{dnu:+.2f}"] = {
            "delta_nu": float(dnu),
            "nu_true": float(nu_true),
            "metrics": metrics,
        }
        print(f"  dnu={dnu:+.2f} | Prop MSE={metrics['proposed']['mse']:.3e} "
              f"bias={metrics['proposed']['bias']:.3e} | "
              f"MP MSE={metrics['matrix_pencil']['mse']:.3e}")

    return {"experiment": "exp7d_freq_sweep", "config": {
        "snr_db": snr_db, "mc_times": mc_times, "nu_f": nu_f, "k_db": k_db,
        "delta_nu_list": list(delta_nu_list),
    }, "results": results}


# ============================================================
# First-order bias analysis (analytic, for the paper)
# ============================================================

def first_order_bias_analysis(cfg=None, k_db_list=None, nu_f=0.3, p_s=1):
    """
    Analytic first-order bias of the proposed estimator vs K-factor.

    Under a weak scattered path of amplitude h_s (|h_s| << |h_los|),
    the pilot-row response is y[k] = s_los[k] + s_scat[k].
    The pair product picks up a first-order perturbation
        delta Z_dk / Z_dk ~ (s_scat+ / s_los+) + (s_scat- / s_los-),
    whose phase contribution biases the estimate. Averaged over the
    scattered-path phase, the bias scales with the NLOS/LOS amplitude
    ratio ~ 1/sqrt(K+1) (per-path) — i.e., ~ 1/K in power terms.

    This function computes the RMS pair-perturbation ratio
        E[|s_scat[kp+dk]/s_los[kp+dk]|]  vs K
    to confirm the ~1/sqrt(K+1) scaling used in the paper narrative.

    Returns:
        dict of K -> RMS ratio
    """
    if cfg is None:
        cfg = DEFAULT
    if k_db_list is None:
        k_db_list = [0.0, 5.0, 10.0, 15.0, 20.0]

    M, N = cfg.M, cfg.N
    lp, kp = cfg.pilot_delay, cfg.pilot_doppler
    A = cfg.pilot_amp
    delta_k_list = cfg.delta_k_list

    from frame.frame_gen import generate_pure_frame
    X_dd = generate_pure_frame(M, N, lp, kp, A)
    X_tf = isfft_fft(X_dd)

    results = {}
    for k_db in k_db_list:
        paths = rician_paths(M, N, K_db=k_db, P_s=p_s, nu_f=nu_f, seed=1)
        R_los = apply_rician_channel(X_tf, [paths[0]])
        R_all = apply_rician_channel(X_tf, paths)
        s_los = sfft_fft(R_los)[lp, :]
        s_all = sfft_fft(R_all)[lp, :]
        s_scat = s_all - s_los

        ratios = []
        for dk in delta_k_list:
            for kk in [(kp + dk) % N, (kp - dk) % N]:
                if abs(s_los[kk]) > 1e-12:
                    ratios.append(abs(s_scat[kk] / s_los[kk]))
        rms = float(np.sqrt(np.mean(np.array(ratios) ** 2)))
        # theoretical 1/sqrt(K+1) per-path amplitude ratio
        K_lin = 10 ** (k_db / 10.0)
        theory = np.sqrt(1.0 / (K_lin + 1.0) / p_s)
        results[f"K_{k_db:.0f}dB"] = {
            "k_db": float(k_db),
            "rms_ratio": rms,
            "theory_1_sqrt_K": float(theory),
        }
        print(f"  K={k_db:>5.1f}dB | RMS scat/LOS ratio={rms:.4f} | "
              f"1/sqrt((K+1)*P_s)={theory:.4f}")

    return results


# ============================================================
# main
# ============================================================

def run_all(cfg=None, mc_times=None, snr_db=10.0):
    """Run all four experiments + bias analysis, save results to JSON."""
    if cfg is None:
        cfg = DEFAULT
    if mc_times is None:
        mc_times = 500  # smaller default for a full sweep; can increase

    out = {"experiment": "exp7_multipath", "config": {
        "M": cfg.M, "N": cfg.N, "lp": cfg.pilot_delay, "kp": cfg.pilot_doppler,
        "A": cfg.pilot_amp, "snr_db": snr_db, "mc_times": mc_times,
        "nu_f_default": cfg.nu_f_default,
    }}

    print("\n" + "=" * 78)
    print("EXPERIMENT: Multipath / Rician + residual sync errors")
    print("=" * 78)

    # 7a: K sweep
    k_sweep = run_k_sweep(cfg, snr_db=snr_db, mc_times=mc_times)
    out["k_sweep"] = k_sweep

    # 7b: weak echo
    weak_echo = run_weak_echo(cfg, snr_db=snr_db, mc_times=mc_times)
    out["weak_echo"] = weak_echo

    # 7c: timing sweep
    timing = run_timing_sweep(cfg, snr_db=snr_db, mc_times=mc_times)
    out["timing_sweep"] = timing

    # 7d: freq sweep
    freq = run_freq_sweep(cfg, snr_db=snr_db, mc_times=mc_times)
    out["freq_sweep"] = freq

    # bias analysis
    bias = first_order_bias_analysis(cfg)
    out["bias_analysis"] = bias

    # save
    results_path = os.path.join(PROJECT_ROOT, "results", "exp7_multipath.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment: multipath / Rician channel and residual synchronization")
    parser.add_argument("--mc", type=int, default=500, help="Monte Carlo trials per point")
    parser.add_argument("--snr", type=float, default=10.0, help="Global SNR (dB)")
    parser.add_argument("--k-sweep-only", action="store_true", help="Run only 7a (K sweep)")
    args = parser.parse_args()

    if args.k_sweep_only:
        res = run_k_sweep(mc_times=args.mc, snr_db=args.snr)
        print(json.dumps(res, indent=2, default=str))
    else:
        run_all(mc_times=args.mc, snr_db=args.snr)
