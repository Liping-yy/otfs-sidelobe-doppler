# -*- coding: utf-8 -*-
"""
exp1_theory_verify: Experiment 1 - Theoretical identity numerical verification
================================================================================

Corresponds to experiment outline Experiment 1.
Noiseless pure impulse frame, MC=1, verify 4 groups of mathematical identities:
  (a) Dirichlet kernel sign opposition: D(nuf+Dk)*D(nuf-Dk) < 0, Dk>=1
  (b) Sidelobe pair Phase-sum invariance:
      psi(Dk) = arg{-y_plus*y_minus} = 2*pi*nuf*(N-1)/N, independent of Dk
  (c) Peak self-multiplication identity:
      arg{-y0^2} - pi = 2*pi*nuf*(N-1)/N
  (d) Unified phase sum:
      psi_bar = arg{y0^2 + Sum(-y_plus*y_minus)}
      -> nu_f_hat = psi_bar * N / (2*pi*(N-1))
      Error ~ 1e-15

Output: 1 table (JSON + console output), optional 1 verification figure.
"""

import sys
import os
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.params import DEFAULT, get_config
from modem.otfs_modem import isfft_fft, sfft_fft, apply_fractional_doppler_tf
from frame.frame_gen import generate_pure_frame
from utils.dirichlet import (
    dirichlet_kernel,
    dirichlet_sign_opposite,
    theoretical_phase_sum,
    nu_f_from_phase,
)
from utils.metrics import save_results, print_comparison


def run_experiment1(cfg: "SimConfig" = None, nu_f_list=None):
    """
    Experiment 1 main function: Theoretical identity numerical verification.

    Args:
        cfg: SimConfig instance (default uses DEFAULT)
        nu_f_list: List of nuf values to test
            (default covers typical values + singularity neighborhood)

    Returns:
        results: dict containing all verification data
    """
    if cfg is None:
        cfg = DEFAULT
    if nu_f_list is None:
        nu_f_list = [-0.45, -0.3, -0.1, 0.0, 0.02, 0.1, 0.3, 0.45]

    M, N = cfg.M, cfg.N
    lp, kp = cfg.pilot_delay, cfg.pilot_doppler
    A = cfg.pilot_amp
    delta_k_list = cfg.delta_k_list  # (1,2,...,7)

    print("#" * 75)
    print("#  Experiment 1: Theoretical identity numerical verification")
    print(f"#  Parameters: M={M}, N={N}, lp={lp}, kp={kp}, A={A}, MC=1 (Noiseless)")
    print("#" * 75)

    all_results = []

    for nu_f in nu_f_list:
        # --- Generate pure frame + noiseless transmission ---
        X_dd = generate_pure_frame(M, N, lp, kp, A)
        X_tf = isfft_fft(X_dd)
        R_tf = apply_fractional_doppler_tf(X_tf, nu_f)
        Y_dd = sfft_fft(R_tf)
        y_row = Y_dd[lp, :]  # Pilot row

        psi_theory = theoretical_phase_sum(nu_f, N)

        # === (a) Dirichlet kernel sign opposition ===
        sign_results = []
        for dk in delta_k_list:
            d_plus = dirichlet_kernel(np.array([nu_f + dk]), N)[0]
            d_minus = dirichlet_kernel(np.array([nu_f - dk]), N)[0]
            sign_ok = (d_plus * d_minus) < 0
            sign_results.append({
                "delta_k": dk,
                "D_plus": float(d_plus),
                "D_minus": float(d_minus),
                "sign_opposite": bool(sign_ok),
            })

        # === (b) Sidelobe pair Phase-sum invariance ===
        sidetone_results = []
        for dk in delta_k_list:
            k_plus = (kp + dk) % N
            k_minus = (kp - dk) % N
            yp = y_row[k_plus]
            ym = y_row[k_minus]
            psi_meas = float(np.angle(-yp * ym))
            err = abs(psi_meas - psi_theory)
            sidetone_results.append({
                "delta_k": dk,
                "psi_meas": psi_meas,
                "psi_theory": float(psi_theory),
                "abs_err": float(err),
            })

        # === (c) Peak self-multiplication identity ===
        y_peak = y_row[kp]
        psi0_raw = float(np.angle(-y_peak * y_peak))
        psi0_corrected = ((psi0_raw - np.pi + np.pi) % (2 * np.pi)) - np.pi  # Compensate pi + wrap
        peak_err = abs(psi0_corrected - psi_theory)

        # === (d) Unified phase sum closed-form estimation ===
        from estimator.proposed import estimate_nu_f
        nu_f_hat = estimate_nu_f(y_row, kp, N, delta_k_list)
        unified_err = abs(nu_f_hat - nu_f)

        # === Sidelobe-only (Ablation, verify singularity) ===
        from estimator.sidetone_only import estimate_nu_f as est_sidetone
        nu_f_sidetone = est_sidetone(y_row, kp, N, delta_k_list=cfg.delta_k_sidetone)
        sidetone_err = abs(nu_f_sidetone - nu_f)

        result = {
            "nu_f": float(nu_f),
            "psi_theory": float(psi_theory),
            "(a)_dirichlet_sign": {
                "all_opposite": all(r["sign_opposite"] for r in sign_results),
                "details": sign_results,
            },
            "(b)_sidetone_invariance": {
                "max_err": float(max(r["abs_err"] for r in sidetone_results)),
                "details": sidetone_results,
            },
            "(c)_peak_self_square": {
                "psi0_raw": psi0_raw,
                "psi0_corrected": float(psi0_corrected),
                "abs_err": float(peak_err),
            },
            "(d)_unified_estimate": {
                "nu_f_hat": float(nu_f_hat),
                "abs_err": float(unified_err),
            },
            "sidetone_only_ablation": {
                "nu_f_hat": float(nu_f_sidetone),
                "abs_err": float(sidetone_err),
                "note": "nuf=0 sidelobes all zero -> singularity (expected fail)" if abs(nu_f) < 1e-10 else "",
            },
        }
        all_results.append(result)

    # ==================== Print table ====================
    print()
    print("=" * 90)
    print(f"{'nuf':>6s} | {'(a)sign_opp':>12s} | {'(b)sidetone_maxerr':>14s} | "
          f"{'(c)peak_err':>12s} | {'(d)unified_err':>12s} | {'sidetone_only_err':>12s}")
    print("-" * 90)

    for r in all_results:
        nu = r["nu_f"]
        a_ok = "PASS" if r["(a)_dirichlet_sign"]["all_opposite"] else "FAIL"
        b_err = r["(b)_sidetone_invariance"]["max_err"]
        c_err = r["(c)_peak_self_square"]["abs_err"]
        d_err = r["(d)_unified_estimate"]["abs_err"]
        s_err = r["sidetone_only_ablation"]["abs_err"]
        print(f"{nu:>+6.2f} | {a_ok:>12s} | {b_err:>14.2e} | "
              f"{c_err:>12.2e} | {d_err:>12.2e} | {s_err:>12.2e}")

    print("=" * 90)

    # ==================== Verdict ====================
    tol = 1e-12  # Machine precision tolerance
    all_pass = True
    for r in all_results:
        nu = r["nu_f"]
        # (a) Dirichlet sign opposite: only holds when |nuf|>0 and |nuf|<0.5
        #     nuf=0: D(+dk)=D(-dk), signs same (singularity, expected behavior)
        if abs(nu) > 1e-10 and abs(nu) < 0.49:
            if not r["(a)_dirichlet_sign"]["all_opposite"]:
                print(f"  [FAIL] nu_f={nu:+.2f}: Dirichlet sign opposite failed")
                all_pass = False
        # (b) Sidelobe Phase-sum invariance: nuf=0 sidelobes all zero, psi undefined, skip
        if abs(nu) > 1e-10:
            if r["(b)_sidetone_invariance"]["max_err"] > tol:
                print(f"  [FAIL] nu_f={nu:+.2f}: sidetone invariance exceeded tolerance")
                all_pass = False
        # (c) Peak self-multiplication identity: should hold for all nuf (including 0)
        if r["(c)_peak_self_square"]["abs_err"] > tol:
            print(f"  [FAIL] nu_f={nu:+.2f}: peak self-square exceeded tolerance")
            all_pass = False
        # (d) Unified phase-sum estimation: should hold for all nuf (including 0)
        if r["(d)_unified_estimate"]["abs_err"] > tol:
            print(f"  [FAIL] nu_f={nu:+.2f}: unified estimate exceeded tolerance")
            all_pass = False

    # nuf=0 sidetone-only singularity is expected behavior
    for r in all_results:
        nu = r["nu_f"]
        if abs(nu) < 1e-10:
            s_err = r["sidetone_only_ablation"]["abs_err"]
            if s_err > tol:
                print(f"  [EXPECTED] nu_f={nu:+.2f}: sidetone-only singular (sidelobes=0), unified formula OK")
            else:
                print(f"  [INFO] nu_f={nu:+.2f}: sidetone-only err=0 by coincidence (angle(0)=0=nuf_theory); will crash with noise")

    print()
    if all_pass:
        print("  [PASS] Exp1 all passed: 4 identities hold at machine precision.")
    else:
        print("  [FAIL] Exp1 has failures, please check.")
    print()

    # ==================== Save results ====================
    output = {
        "experiment": "exp1_theory_verify",
        "config": {
            "M": M, "N": N, "lp": lp, "kp": kp, "A": A,
            "delta_k_list": delta_k_list,
            "nu_f_list": list(nu_f_list),
            "mc_times": 1,
            "noise": "none",
        },
        "results": all_results,
        "tolerance": tol,
        "all_pass": all_pass,
    }

    results_path = os.path.join(PROJECT_ROOT, "results", "exp1_theory_verify.json")
    save_results(output, results_path)

    return output


if __name__ == "__main__":
    results = run_experiment1()
