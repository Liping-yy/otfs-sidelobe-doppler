# -*- coding: utf-8 -*-
"""
Phase-sum + peak self-multiplication closed-form fractional Doppler estimator —— completeachieves
======================================================

coreimprovement：atatomicrationalsidelobeforPhase-sum(dk=1..K)basisbasisupper，addinputPeak self-multiplication(dk=0)，
willMSEfrom1.32e-3dimensionalityto2.52e-4（SNR=12dB），changegoodapproximately5x，keeppureclosed-formirrationalsearch。

sametimeachievespathpathE（peakcomplexNewtonrefinement）：Toeplitzusepeaky[kp]completecomplexinformation
（magnitude+phase）andtheoreticalDirichlet kerneltemplatedo1-2stepNewtoniterationfitting，
recoverarg()operasdiscarddiscardmagnitudeinformation，enter1stepreducingMSE。

3kindestimatoratsame asscriptmidcomparison：
  1. phase_sum_sidetone   : onlysidelobeforPhase-sum（Baseline，dk=1..K）
  2. phase_sum_with_peak  : sidelobe+Peak self-multiplication（dk=0..K），pureclosed-form
  3. peak_newton_refine   : upperdescribeinitial value + peakcomplexNewtonrefinement（pathpathE）

sametimeandMatrix Pencil(MP)Basepathdopublictranslationcontrol（sameframe、sameSNR definition、sameseed）。

dependency: common_otfs.py (samedirectory), numpy, scipy
"""

import numpy as np
from scipy.linalg import eig as scipy_eig

from common_otfs import (
    isfft, sfft,
    generate_impulse_pilot_frame,
    generate_embedded_pilot_frame,
    transmit_otfs,
    dirichlet_kernel,
)

# from test_phase_sum_qpsk_idi importFFTacceleratedversionlinksumengineeringframe generation（localsidelobepowerSNR definition）
from test_phase_sum_qpsk_idi import (
    isfft_fft,
    sfft_fft,
    generate_dd_frame,
)

# ============================================================
# estimator 1: onlysidelobeforPhase-sum（Baseline）
# ============================================================
def estimate_phase_sum_sidetone(
    y_pilot_row: np.ndarray,
    kp: int,
    N: int,
    delta_k_list=(1, 2, 3, 4),
):
    """
    Sidelobe-only phase-sum estimator（atomicscheme）。

    psi(dk) = arg(-y[kp+dk] * y[kp-dk]) = 2*pi*nu_f*(N-1)/N
    weightedrecovervectorfusion: sum_vec = sum(w_dk * exp(j*psi_dk))
    nu_f_hat = angle(sum_vec) * N / (2*pi*(N-1))

    Args:
        y_pilot_row: pilot row y[lp, :], shape (N,)
        kp: pilot Doppler column
        N: Doppler dimension
        delta_k_list: Sidelobe offset list

    Returns:
        nu_f_hat: fractional DopplerEstimate
    """
    psi_theory = 0.0  # placeholder，unwrapuse
    psis = []
    weights = []

    for dk in delta_k_list:
        if kp + dk >= N or kp - dk < 0:
            continue
        yp = y_pilot_row[kp + dk]
        ym = y_pilot_row[kp - dk]
        prod = -yp * ym
        psi_meas = np.angle(prod)
        psis.append(psi_meas)
        weights.append(np.abs(prod))

    if len(psis) == 0:
        return 0.0

    sum_vec = sum(w * np.exp(1j * p) for w, p in zip(weights, psis))
    nu_f_hat = np.angle(sum_vec) * N / (2.0 * np.pi * (N - 1))
    return nu_f_hat


# ============================================================
# estimator 2: sidelobe+Peak self-multiplication（coreimprovement）
# ============================================================
def estimate_phase_sum_with_peak(
    y_pilot_row: np.ndarray,
    kp: int,
    N: int,
    delta_k_list=(1, 2, 3, 4, 5, 6, 7),
):
    """
    sidelobefor + Peak self-multiplicationPhase-sumestimator。

    Peak self-multiplication(dk=0):
        psi_0 = arg(-y[kp]^2) = 2*pi*nu_f*(N-1)/N + pi
        needCompensationpi: psi_0_corrected = psi_0 - pi

    sidelobefor(dk>=1): andBaselinesame
        psi_dk = arg(-y[kp+dk] * y[kp-dk]) = 2*pi*nu_f*(N-1)/N

    weightedrecovervectorfusion: weightweight w = |prod| = |y_a * y_b|

    Args:
        y_pilot_row: pilot row
        kp: pilot Doppler column
        N: Doppler dimension
        delta_k_list: Sidelobe offset list（noincluding0，0byPeak self-multiplicationhandling）

    Returns:
        nu_f_hat: fractional DopplerEstimate
    """
    psis = []
    weights = []

    # --- dk=0: Peak self-multiplication ---
    y_peak = y_pilot_row[kp]
    prod0 = -y_peak * y_peak  # -y[kp]^2
    psi0 = np.angle(prod0) - np.pi  # CompensationD(nu_f)Symbolstranslaterotationindexinputpi
    # wrapto (-pi, pi]
    psi0 = ((psi0 + np.pi) % (2.0 * np.pi)) - np.pi
    psis.append(psi0)
    weights.append(np.abs(prod0))

    # --- dk>=1: sidelobefor ---
    for dk in delta_k_list:
        if kp + dk >= N or kp - dk < 0:
            continue
        yp = y_pilot_row[kp + dk]
        ym = y_pilot_row[kp - dk]
        prod = -yp * ym
        psis.append(np.angle(prod))
        weights.append(np.abs(prod))

    sum_vec = sum(w * np.exp(1j * p) for w, p in zip(weights, psis))
    nu_f_hat = np.angle(sum_vec) * N / (2.0 * np.pi * (N - 1))
    return nu_f_hat


# ============================================================
# estimator 3: peakcomplexNewtonrefinement（pathpathE）
# ============================================================
def _dirichlet_template(nu_f, N, A_known=True):
    """
    theoreticaltemplate: f(nu_f) = (1/N) * D(nu_f) * exp(j*pi*nu_f*(N-1)/N)
    itsmid D(nu_f) = sin(pi*nu_f) / sin(pi*nu_f/N) (Dirichlet kernel)

    returnrecovernumerical（includingmagnitude+phase），used forcomplex domainNewtonfitting。
    """
    # avoidremovezero
    denom = np.sin(np.pi * nu_f / N)
    if np.abs(denom) < 1e-15:
        return 1.0 + 0j  # nu_f=0timeD=N, butnormalizedback=1
    D = np.sin(np.pi * nu_f) / denom
    phase = np.exp(1j * np.pi * nu_f * (N - 1) / N)
    return D / N * phase


def _dirichlet_template_deriv(nu_f, N):
    """
    template f(nu_f)analyticalderivativenumber df/d(nu_f)。
    used forNewtoniterationstep。

    f(nu_f) = (1/N) * sin(pi*nu_f)/sin(pi*nu_f/N) * exp(j*pi*nu_f*(N-1)/N)

    let D = sin(pi*nu_f)/sin(pi*nu_f/N), phi = pi*nu_f*(N-1)/N
    f = D/N * exp(j*phi)

    dD/dnu_f = [pi*cos(pi*nu_f)*sin(pi*nu_f/N) - sin(pi*nu_f)*(pi/N)*cos(pi*nu_f/N)] / sin^2(pi*nu_f/N)
    dphi/dnu_f = pi*(N-1)/N

    df/dnu_f = (dD/dnu_f / N) * exp(j*phi) + (D/N) * j * dphi/dnu_f * exp(j*phi)
    """
    denom = np.sin(np.pi * nu_f / N)
    if np.abs(denom) < 1e-15:
        # nu_f≈0timenumericalderivativenumber
        return 1j * np.pi * (N - 1) / N  # approximation
    sin_num = np.sin(np.pi * nu_f)
    cos_num = np.cos(np.pi * nu_f)
    cos_den = np.cos(np.pi * nu_f / N)

    dD = (np.pi * cos_num * denom - sin_num * (np.pi / N) * cos_den) / (denom ** 2)
    D = sin_num / denom
    phi = np.pi * nu_f * (N - 1) / N
    dphi = np.pi * (N - 1) / N

    df = (dD / N) * np.exp(1j * phi) + (D / N) * 1j * dphi * np.exp(1j * phi)
    return df


def estimate_peak_newton_refine(
    y_pilot_row: np.ndarray,
    kp: int,
    N: int,
    delta_k_list=(1, 2, 3, 4, 5, 6, 7),
    n_newton_steps: int = 2,
):
    """
    pathpathE：peakcomplexNewtonrefinement。

    Steps:
    1. use estimate_phase_sum_with_peak giveoutputinitial value nu_f_0
    2. Newtoniteration: nu_f_{k+1} = nu_f_k - Re{residual/deriv}
       itsmid residual = y[kp] - template(nu_f_k)
       template(nu_f) = A_hat * f(nu_f), A_hatbypeakmagnitudeEstimate

    Toeplitzusepeaky[kp]completecomplexinformation（magnitude+phase），recoverarg()discarddiscardmagnitudedimension。

    Notemeaning: thisnoisgrid search，isforalreadyknownon-pathpropertyfunctionclosed-formparameterrefinement，
         eachstepO(1)computequantity（Dirichlet kernelanditsderivativenumberclosed-formcompute）。

    Args:
        y_pilot_row: pilot row
        kp: pilot Doppler column
        N: Doppler dimension
        delta_k_list: Sidelobe offset list
        n_newton_steps: Newtoniterationstepnumber

    Returns:
        nu_f_hat: refinementbackfractional DopplerEstimate
    """
    # Step 1: initial value
    nu_f = estimate_phase_sum_with_peak(y_pilot_row, kp, N, delta_k_list)

    # Step 2: peakcomplexobservedetect
    y_peak = y_pilot_row[kp]

    # Step 3: Newtoniteration
    for _ in range(n_newton_steps):
        tmpl = _dirichlet_template(nu_f, N)
        dtmpl = _dirichlet_template_deriv(nu_f, N)

        # A_hat: minimum2multipliermagnitudeEstimate (scalar)
        # residual = y_peak - A_hat * tmpl
        # let d(residual)/dA = 0 -> A_hat = Re{y_peak * conj(tmpl)} / |tmpl|^2
        A_hat = np.real(y_peak * np.conj(tmpl)) / (np.abs(tmpl) ** 2 + 1e-30)

        # fornu_fResidualderivativenumber
        # dr/d(nu_f) = -A_hat * dtmpl
        resid = y_peak - A_hat * tmpl
        dr_dnu = -A_hat * dtmpl

        # Newtonstep: nu_f_new = nu_f - Re{resid / dr_dnu}
        # takerealpartialbecausenu_fisrealparameter
        step = np.real(resid * np.conj(dr_dnu)) / (np.abs(dr_dnu) ** 2 + 1e-30)
        nu_f = nu_f - step

        # Limit toconjunctionprocessrange
        if nu_f > 0.5:
            nu_f = 0.5
        elif nu_f < -0.5:
            nu_f = -0.5

    return nu_f


# ============================================================
# MPBasepath（frombasepath_matrix_pencil.pyshiftimplant，publictranslationcontrol）
# ============================================================
def _mp_core(y_dt, n_paths=1):
    """Matrix-PencilCore solver。"""
    N = len(y_dt)
    L = max(3, N // 3)
    # Hankelmatrix
    n_rows = N - L
    n_cols = L + 1
    H = np.zeros((n_rows, n_cols), dtype=complex)
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i + j
            if idx < N:
                H[i, j] = y_dt[idx]
    H1 = H[:, :-1]
    H2 = H[:, 1:]
    A = H1.conj().T @ H2
    B = H1.conj().T @ H1
    eigvals, _ = scipy_eig(A, b=B)
    valid_mask = np.isfinite(eigvals)
    eigvals = eigvals[valid_mask]
    if len(eigvals) == 0:
        return 0.0
    angles = np.angle(eigvals)
    nu_est_all = N * angles / (2.0 * np.pi)
    nu_est_all = ((nu_est_all + 0.5) % 1.0) - 0.5
    # Select root with maximum amplitude
    Z = np.zeros((N, len(eigvals)), dtype=complex)
    for p_idx, lam in enumerate(eigvals):
        Z[:, p_idx] = lam ** np.arange(N)
    amp_est = np.linalg.pinv(Z) @ y_dt
    best_idx = np.argmax(np.abs(amp_est))
    return nu_est_all[best_idx]


def estimate_matrix_pencil(y_pilot_row, N, n_paths=1):
    """MPestimator：pilot row -> IDFT -> Hankel SVD -> Generalized eigenvalue。"""
    y_dt = np.fft.ifft(y_pilot_row)
    return _mp_core(y_dt, n_paths=n_paths)


# ============================================================
# Monte Carlosimulationmain function
# ============================================================
def mc_compare(
    M=64,
    N=16,
    lp=5,
    kp=8,
    A=1.0,
    nu_f_true=0.3,
    snr_db=12.0,
    mc_times=2000,
    seed=2026,
    frame_mode="impulse",
    Kmax=10,
    delta_k_list=(1, 2, 3, 4, 5, 6, 7),
):
    """
    4kindestimatorMonte Carlocomparison。

    Args:
        frame_mode: "impulse"=pureunitimpulsepilotframe, "guard"=engineering frame(QPSK+guard zero)
        Kmax: engineering frameguard zerolongdegree（onlyframe_mode="guard"timecreateeffect）
        delta_k_list: Sidelobe offset list
    """
    rng = np.random.default_rng(seed)

    err_sidetone = []
    err_with_peak = []
    err_newton = []
    err_mp = []

    for _ in range(mc_times):
        # --- frame generation + Transmission link ---
        if frame_mode == "impulse":
            # Pure impulse frame：use common_otfs standardlink（allframepower≈sidelobepower，irrationalSNRBias）
            X_dd = generate_impulse_pilot_frame(M, N, lp, kp, pilot_value=A)
            Y_dd = transmit_otfs(X_dd, nu_f_true, snr_db, M, N, rng=rng)
            y_pilot_row = Y_dd[lp, :]
        else:
            # engineering frame：use test_phase_sum_qpsk_idi link（localsidelobepowerdefinitionSNR）
            # reason：common_otfs.transmit_otfs useallframe averagepower(includingQPSKdata)algorithmNoise，
            # allframepower/sidelobepower≈523x=27.2dB，derivativeconsistentsideloberealinterSNRJacobisettheoremvaluelow27dB
            X_dd = generate_dd_frame(M, N, lp, kp, A, Kmax, rng)
            X_tf = isfft_fft(X_dd)
            n_arr = np.arange(N)
            phase_rot = np.exp(1j * 2 * np.pi * nu_f_true * n_arr / N)
            R_tf = X_tf * phase_rot[:, np.newaxis]
            y_clean = sfft_fft(R_tf)

            # localsidelobepowercompute sigma（and test_phase_sum_qpsk_idi completeallconsistent）
            local_powers = []
            for dk in delta_k_list[:4]:
                local_powers.append(np.abs(y_clean[lp, kp + dk]) ** 2)
                local_powers.append(np.abs(y_clean[lp, kp - dk]) ** 2)
            power_signal = np.mean(local_powers)
            sigma_noise = np.sqrt(power_signal / (10.0 ** (snr_db / 10.0)))
            noise = (rng.standard_normal(y_clean.shape) +
                     1j * rng.standard_normal(y_clean.shape)) / np.sqrt(2)
            y_noisy = y_clean + sigma_noise * noise
            y_pilot_row = y_noisy[lp, :]

        # --- estimator1: onlysidelobe ---
        nuf1 = estimate_phase_sum_sidetone(y_pilot_row, kp, N, delta_k_list[:4])
        err_sidetone.append(nuf1 - nu_f_true)

        # --- estimator2: sidelobe+Peak self-multiplication ---
        nuf2 = estimate_phase_sum_with_peak(y_pilot_row, kp, N, delta_k_list)
        err_with_peak.append(nuf2 - nu_f_true)

        # --- estimator3: peakNewtonrefinement ---
        nuf3 = estimate_peak_newton_refine(
            y_pilot_row, kp, N, delta_k_list, n_newton_steps=2,
        )
        err_newton.append(nuf3 - nu_f_true)

        # --- MPBaseline ---
        nuf_mp = estimate_matrix_pencil(y_pilot_row, N, n_paths=1)
        err_mp.append(nuf_mp - nu_f_true)

    err_sidetone = np.array(err_sidetone)
    err_with_peak = np.array(err_with_peak)
    err_newton = np.array(err_newton)
    err_mp = np.array(err_mp)

    # --- print ---
    frame_desc = "pureimpulse" if frame_mode == "impulse" else f"engineering frameKmax={Kmax}"
    print("=" * 75)
    print(f"MCcomparison | {frame_desc} | SNR={snr_db}dB | MC={mc_times} | nu_f={nu_f_true}")
    print(f"parameter: M={M}, N={N}, lp={lp}, kp={kp}, A={A}")
    print("=" * 75)
    print(f"{'approaches':<28s} {'bias':>10s} {'MSE':>12s} {'vs MP':>8s}")
    print("-" * 75)
    print(f"{'onlysidelobe(dk=1..4)':<28s} {np.mean(err_sidetone):>+10.4e} "
          f"{np.mean(err_sidetone**2):>12.4e} "
          f"{np.mean(err_sidetone**2)/np.mean(err_mp**2):>7.1f}x")
    print(f"{'sidelobe+Peak self-multiplication(dk=0..7)':<28s} {np.mean(err_with_peak):>+10.4e} "
          f"{np.mean(err_with_peak**2):>12.4e} "
          f"{np.mean(err_with_peak**2)/np.mean(err_mp**2):>7.1f}x")
    print(f"{'peakNewtonrefinement(pathpathE)':<28s} {np.mean(err_newton):>+10.4e} "
          f"{np.mean(err_newton**2):>12.4e} "
          f"{np.mean(err_newton**2)/np.mean(err_mp**2):>7.1f}x")
    print(f"{'MP (Hankel SVD)':<28s} {np.mean(err_mp):>+10.4e} "
          f"{np.mean(err_mp**2):>12.4e} {'---':>8s}")
    print("=" * 75)

    return {
        "sidetone": err_sidetone,
        "with_peak": err_with_peak,
        "newton": err_newton,
        "mp": err_mp,
    }


# ============================================================
# primaryprocesssequence
# ============================================================
if __name__ == "__main__":
    BASE = dict(
        M=64, N=16, lp=5, kp=8, A=3.0,
        mc_times=2000, seed=2026,
    )

    # ===== Pure impulse pilot frame =====
    print("\n" + "#" * 75)
    print("# pureunitimpulsepilotframe (irrationalIDI)")
    print("#" * 75)

    for snr in [0, 5, 10, 12, 15, 20]:
        mc_compare(**BASE, nu_f_true=0.3, snr_db=snr, frame_mode="impulse")

    # ===== engineering frame（QPSK + guard zero Kmax=10） =====
    print("\n" + "#" * 75)
    print("# engineering frame (QPSK + guard zero Kmax=10)")
    print("#" * 75)

    for snr in [0, 5, 10, 12, 15, 20]:
        mc_compare(**BASE, nu_f_true=0.3, snr_db=snr, frame_mode="guard", Kmax=10)

    # ===== nu_f=0.0 Singularity pointtest =====
    print("\n" + "#" * 75)
    print("# nu_f=0.0 Singularity pointtest (Pure impulse frame)")
    print("#" * 75)

    mc_compare(**BASE, nu_f_true=0.0, snr_db=12, frame_mode="impulse")

    # ===== nu_f=0.02 singularityneighborhoodtest =====
    print("\n" + "#" * 75)
    print("# nu_f=0.02 singularityneighborhoodtest (Pure impulse frame)")
    print("#" * 75)

    mc_compare(**BASE, nu_f_true=0.02, snr_db=12, frame_mode="impulse")

    print("\n" + "=" * 75)
    print("alltestdone。")
    print("coreimprovement: Phase-sum+Peak self-multiplication -> estimator2 (estimate_phase_sum_with_peak)")
    print("burstbreakdirection: peakcomplexNewtonrefinement -> estimator3 (estimate_peak_newton_refine)")
    print("=" * 75)
