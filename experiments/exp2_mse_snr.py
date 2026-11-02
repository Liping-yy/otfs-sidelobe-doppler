# -*- coding: utf-8 -*-
"""
Experiment 2：coreMSE-SNRprecisioncomparison
==========================

Corresponding experiment outlineExperiment 2（papermost criticalexperiment）。

two groupsframe × 4Curves：
  frame：Pure impulse frame / engineering frame(Kmax=10, A=3.0, QPSKdata)
  Curves：Proposed(sidelobe+Peak self-multiplication) / onlysidelobe(Ablation) / Matrix Pencil / He2023(correlationclose+quasi-Newton)

Fixed νf=0.3，SNR = -5~20dB step size5dB，MC=2000 times/SNR。
Measurements：MSE、bias、vsBaselineratio。

Output：2figure（pure frame MSE-SNR、engineering frame MSE-SNR）+ 1table（numerical summary）。

Note：Original Prony baseline replaced by He2023 due to mathematical equivalence with MP under single-path。

"""

import sys
import os
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.params import DEFAULT, get_config
from modem.otfs_modem import (
    isfft_fft, sfft_fft,
    apply_fractional_doppler_tf,
    add_awgn, add_awgn_local_snr,
)
from frame.frame_gen import generate_pure_frame, generate_engineering_frame
from estimator import get_estimator, ESTIMATOR_REGISTRY
from utils.metrics import save_results, compute_mse_bias


# ============================================================
# unit times trial：givenframe、νf、SNR → eachestimator outputν̂f
# ============================================================

def _run_single_trial_pure(X_dd, nu_f, snr_db, lp, kp, N, rng, est_names, delta_k_list, delta_k_sidetone, A=3.0):
    """pure frameunit times trial。"""
    M = X_dd.shape[0]
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
        elif name in ("matrix_pencil", "prony"):
            estimates[name] = fn(y_row, kp, N)
        elif name == "he2023":
            estimates[name] = fn(y_row, kp, N, M=M, A=A, lp=lp)
        else:
            estimates[name] = fn(y_row, kp, N, delta_k_list=delta_k_list)
    return estimates


def _run_single_trial_eng(X_dd, nu_f, snr_db, lp, kp, N, M, rng, est_names, delta_k_list, delta_k_sidetone, A=3.0):
    """engineering frameunit times trial（Global SNRpath）。

    schemeA：SNR definitionasTotal frame signal power / noise power，and pure framesame asbaseline。
    pilotboostA=3.0makesidelobevalidSNRhighatGlobal SNR~15dB，
    thisin MSE-SNRCurvesuppertableappearasengineering frameCurvesleftshift~15dB。
    """
    X_tf = isfft_fft(X_dd)
    R_tf = apply_fractional_doppler_tf(X_tf, nu_f)
    # Global SNR：useallframe averagepowerdefinition
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
# main function
# ============================================================

def run_experiment2(cfg: "SimConfig" = None, nu_f: float = None,
                    snr_list=None, mc_times: int = None,
                    frame_types=None, est_names=None):
    """
    Experiment 2main function：coreMSE-SNRprecisioncomparison。

    Args:
        cfg: SimConfigInstance
        nu_f: fractional Doppler（default0.3）
        snr_list: SNRlistdB（defaultcfg.snr_range）
        mc_times: Monte Carlo times(default default2000)
        frame_types: Frame type list（default["pure", "engineering"]）
        est_names: estimator namelist

    Returns:
        results: dict
    """
    if cfg is None:
        cfg = DEFAULT
    if nu_f is None:
        nu_f = cfg.nu_f_default
    if snr_list is None:
        snr_list = cfg.snr_range
    if mc_times is None:
        mc_times = cfg.mc_times
    if frame_types is None:
        frame_types = ["pure", "engineering"]
    if est_names is None:
        # 4curvesindependenttechniquespathpath：closed-form(versiontext) / Ablation / SVD super-resolution(MP) / correlationclose+quasi-Newton(He2023)
        # Note：PronyatSingle pathlowerandMPnumberlearnequivalent（Hankelrank1degenerate），replaced byHe2023
        est_names = ["proposed", "sidetone_only", "matrix_pencil", "he2023"]

    M, N = cfg.M, cfg.N
    lp, kp = cfg.pilot_delay, cfg.pilot_doppler
    A = cfg.pilot_amp
    Kmax = cfg.kmax_default
    delta_k_list = cfg.delta_k_list
    delta_k_sidetone = cfg.delta_k_sidetone

    print("#" * 75)
    print("#  Experiment 2: core MSE-SNR precisioncomparison")
    print(f"#  nu_f={nu_f}, SNR={list(snr_list)} dB, MC={mc_times}")
    print(f"#  frame: {frame_types}")
    print(f"#  estimator: {est_names}")
    print("#" * 75)

    all_results = {}

    for frame_type in frame_types:
        print(f"\n--- {frame_type} frame ---")
        frame_data = {}

        for snr_db in snr_list:
            rng = np.random.default_rng(cfg.seed)
            errors = {name: np.zeros(mc_times) for name in est_names}

            t_start = time.time()
            for trial in range(mc_times):
                # each timestrialusenosameseedderived
                trial_rng = np.random.default_rng(cfg.seed + trial + int(abs(snr_db) * 1000) + 1)

                if frame_type == "pure":
                    X_dd = generate_pure_frame(M, N, lp, kp, A)
                    ests = _run_single_trial_pure(
                        X_dd, nu_f, snr_db, lp, kp, N, trial_rng,
                        est_names, delta_k_list, delta_k_sidetone, A=A
                    )
                else:
                    X_dd = generate_engineering_frame(M, N, lp, kp, A, Kmax, trial_rng)
                    ests = _run_single_trial_eng(
                        X_dd, nu_f, snr_db, lp, kp, N, M, trial_rng,
                        est_names, delta_k_list, delta_k_sidetone, A=A
                    )

                for name in est_names:
                    errors[name][trial] = ests[name] - nu_f

            t_elapsed = time.time() - t_start

            # computemetrics
            metrics = {}
            for name in est_names:
                metrics[name] = compute_mse_bias(errors[name])

            frame_data[f"snr_{snr_db:.0f}dB"] = {
                "snr_db": float(snr_db),
                "metrics": metrics,
                "elapsed_s": t_elapsed,
            }

            # progressprint
            best_name = min(metrics, key=lambda k: metrics[k]["mse"])
            best_mse = metrics[best_name]["mse"]
            he_mse = metrics.get("he2023", {}).get("mse", float("nan"))
            print(f"  SNR={snr_db:>+5.0f}dB | "
                  f"Proposed={metrics['proposed']['mse']:.3e} | "
                  f"SideOnly={metrics['sidetone_only']['mse']:.3e} | "
                  f"MP={metrics['matrix_pencil']['mse']:.3e} | "
                  f"He2023={he_mse:.3e} | "
                  f"{t_elapsed:.1f}s")

        all_results[frame_type] = frame_data

    # ==================== Print summary table ====================
    for frame_type in frame_types:
        print(f"\n{'='*85}")
        print(f"  {frame_type} frame MSE-SNR Summary (nu_f={nu_f}, MC={mc_times})")
        print(f"{'='*85}")
        header = f"{'SNR(dB)':>8s}"
        for name in est_names:
            header += f" | {name:>16s} MSE"
        header += f" | {'Prop/MP':>8s}"
        print(header)
        print("-" * 85)

        for snr_db in snr_list:
            key = f"snr_{snr_db:.0f}dB"
            data = all_results[frame_type][key]
            metrics = data["metrics"]
            row = f"{snr_db:>+8.0f}"
            for name in est_names:
                row += f" | {metrics[name]['mse']:>16.4e}"
            prop_mse = metrics["proposed"]["mse"]
            mp_mse = metrics["matrix_pencil"]["mse"]
            ratio = prop_mse / mp_mse if mp_mse > 0 else float("inf")
            row += f" | {ratio:>7.2f}x"
            print(row)

        # biastable
        print(f"\n  {frame_type} frame Bias Summary")
        print("-" * 85)
        header = f"{'SNR(dB)':>8s}"
        for name in est_names:
            header += f" | {name:>16s} bias"
        print(header)
        print("-" * 85)
        for snr_db in snr_list:
            key = f"snr_{snr_db:.0f}dB"
            metrics = all_results[frame_type][key]["metrics"]
            row = f"{snr_db:>+8.0f}"
            for name in est_names:
                row += f" | {metrics[name]['bias']:>+16.4e}"
            print(row)

    print(f"{'='*85}")

    # ==================== saveresult ====================
    output = {
        "experiment": "exp2_mse_snr",
        "config": {
            "M": M, "N": N, "lp": lp, "kp": kp, "A": A,
            "Kmax": Kmax, "nu_f": nu_f,
            "snr_list": list(snr_list),
            "mc_times": mc_times,
            "delta_k_list": delta_k_list,
            "delta_k_sidetone": delta_k_sidetone,
            "frame_types": frame_types,
            "est_names": est_names,
            "seed": cfg.seed,
            "local_snr": cfg.local_snr,
        },
        "results": all_results,
    }

    results_path = os.path.join(PROJECT_ROOT, "results", "exp2_mse_snr.json")
    save_results(output, results_path)

    return output


if __name__ == "__main__":
    # defaultparameterrun（MC=2000，completeSNRrange）
    # small-scaletestcan: run_experiment2(mc_times=200, snr_list=[0, 10])
    results = run_experiment2()
