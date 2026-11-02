# -*- coding: utf-8 -*-
"""
Dirichlet kernel (periodic Sinc function) utilities
================================

D_N(x) = sin(πx) / sin(πx/N)

coreproperty（derivationdocx §2.4.3, §3.2）：
  - Limit is N as x -> 0 (L'Hopital's rule)
  - D = 0 when x is an integer (not a multiple of N)
  - Symbols are opposite at Symmetric sidelobe +/- Dk（§3.2closekeyFeatures）
  - purerealnumber，butSymbolscanpositivecannegative

"""

import numpy as np


def dirichlet_kernel(x, N: int) -> np.ndarray:
    """
    computeDirichlet kernel D_N(x) = sin(πx) / sin(πx/N)。

    Args:
        x: locset（canasanalysis)
        N: periodexpectation（Dopplerbin)

    Returns:
        D_N(x): Dirichlet kernelvalue（realnumbernumbergroups）
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x, dtype=float)

    # 0/0case：x→0timepolarlimitasN，x→kNtimealsoasN
    mask_singular = np.abs(np.sin(np.pi * x / N)) < 1e-12
    mask_regular = ~mask_singular

    # ordinaryprogrammingcompute
    if np.any(mask_regular):
        xm = x[mask_regular]
        result[mask_regular] = np.sin(np.pi * xm) / np.sin(np.pi * xm / N)

    # Singularity point：x ≡ 0 (mod N) time D=N，otherintegertime D=0
    if np.any(mask_singular):
        xs = x[mask_singular]
        for i, xi in enumerate(xs):
            if abs(xi % N) < 1e-10 or abs(xi % N - N) < 1e-10:
                result_flat = result.ravel()
                idx = np.arange(len(x)).reshape(x.shape)[mask_singular][i]
                result_flat[idx] = float(N)

    return result


def dirichlet_sign_opposite(nu_f: float, delta_k: int, N: int) -> bool:
    """
    verifysymmetricsidelobeDirichlet kernelSymbolscorrelationfeedbackFeatures（§3.2）。

    forat|νf| < 0.5andΔk >= 1:
      D(νf+Δk) and D(νf-Δk) Symbolstotaliscorrelationfeedback。

    Args:
        nu_f: fractional Doppler
        delta_k: sidelobeoffset
        N: Doppler dimension size

    Returns:
        True if Symbolscorrelationfeedback
    """
    d_plus = dirichlet_kernel(np.array([nu_f + delta_k]), N)[0]
    d_minus = dirichlet_kernel(np.array([nu_f - delta_k]), N)[0]
    return d_plus * d_minus < 0


def theoretical_phase_sum(nu_f: float, N: int) -> float:
    """
    theoreticalPhase-sum ψ = 2π·νf·(N-1)/N（derivationdocx §3.3, §3.6unifiedformula）。

    forall delta_k ∈ {0, 1, ..., K}identityholds（Δk=0needπCompensation）。

    Args:
        nu_f: fractional Doppler
        N: Doppler dimension size

    Returns:
        ψ_theory: theoreticalphase sum (rad)
    """
    return 2 * np.pi * nu_f * (N - 1) / N


def theoretical_phase_diff(delta_k: int, N: int) -> float:
    """
    theoreticalphasedifference Δϕ = -2π·Δk·(N-1)/N + π（derivationdocx §3.7）。

    onlybyNsumΔkdecidetheorem，andνfirrationalclose。used forinterferenceverificationbasepath。

    Args:
        delta_k: sidelobeoffset
        N: Doppler dimension size

    Returns:
        Δϕ_theory: theoreticalphasedifference (rad)
    """
    return -2 * np.pi * delta_k * (N - 1) / N + np.pi


def nu_f_from_phase(psi: float, N: int) -> float:
    """
    fromPhase-sumfeedbackrecurrencefractional Doppler（derivationdocx §4.3）。

    ν̂f = ψ̄ · N / (2π(N-1))

    Args:
        psi: observedetectPhase-sum ψ̄
        N: Doppler dimension size

    Returns:
        nu_f_hat: fractional DopplerEstimate
    """
    return psi * N / (2 * np.pi * (N - 1))


if __name__ == "__main__":
    # quickverify
    N = 16
    nu_f = 0.3

    # Symbolscorrelationfeedbackverify
    for dk in range(1, 8):
        assert dirichlet_sign_opposite(nu_f, dk, N), f"Δk={dk}Symbolscorrelationfeedbackverifyfail"
    print("Dirichlet kernelSymbolscorrelationfeedbackFeatures: PASS")

    # Phase-sumtheoreticalvalue
    psi = theoretical_phase_sum(nu_f, N)
    print(f"ψ_theory(νf=0.3, N=16) = {psi:.6f} rad")

    # numericalverify
    d1 = dirichlet_kernel(np.array([nu_f - 1]), N)[0]
    d2 = dirichlet_kernel(np.array([nu_f + 1]), N)[0]
    print(f"D(-0.7) = {d1:.4f}, D(1.3) = {d2:.4f}, Symbolscorrelationfeedback: {d1*d2 < 0}")
