"""
Monte Carlo simulation：Matrix‑Pencil(MP)super-resolutionfractional DopplerEstimateBaseline
【Approach B explanation】
1. MPBasepathuse common_otfs.py standardOTFSlink：Dopplercorrelationshiftapplyattime domainSymbols，andLiang 2024textcontributionachieveskeepconsistent；
2. versionBasepathonlyused for【processthinkpureimpulsepilot】conditionloweralgorithmmethodFeaturesperformshow；
3. ⚠️Link difference important disclaimer：
   - Matrix‑PencilBaseline：time domainapplyDoppler；
   - versiontextcommitoutputphase‑sum / twolevelmixedalgorithmmethod：Based onTF domainphaserotationlink；
   twosetlinknumberlearntheoreticalequivalent，butnumericalachievesfinenodenosame。thereforeversionBasepath**nodirectandbandIDIdataleakageengineering framedoMSEnumericalhorizontalvectorcomparison**，onlyused forshowMPversionownalgorithmmethodFeatures（irrationalsingular pts、super-resolution）。

objective：
1. standardpureimpulsepilotlowerrecoverappearMatrix‑PencilBasepathalgorithmmethodperformance
2. MPalgorithmmethodnostoreatnu_f=0phaseconvolutionwrapSingularity point，canandphase‑sumclassalgorithmmethoddoFeaturescontrol
Constraint：
1. onlymodelsimulateSingle-path LOS channel；DD domainimpulsepilot
Notemeaning：
1. dependencyouterpartialmodule common_otfs.py，samedirectorylowermuststoreat thatfile
2. dependencyscipy；pip install scipy
3. MPalgorithmmethodusecompletepilotdelayrowy[lp, :]doEstimate，norequiredsidelobeΔksampling
referencepaper：Liang et al., Matrix Pencil‑Based Channel Estimation Algorithm for OTFS, IEEE Comm. Lett., 2024
"""
import numpy as np
from scipy.linalg import eig  # scipyonlyrationalGeneralized eigenvaluebparameter！
from common_otfs import generate_impulse_pilot_frame, transmit_otfs


# ============================================================================
# Matrix‑Pencil coresubfunction
# ============================================================================
def idft_dt_domain(y_pilot_row: np.ndarray, N: int) -> np.ndarray:
    """DDpilot rowalongDoppleraxisdoIDFT，transformtransformtoDT domain"""
    return np.fft.ifft(y_pilot_row)


def build_hankel(y_dt: np.ndarray, L: int):
    N = len(y_dt)
    n_rows = N - L
    n_cols = L + 1
    H = np.zeros((n_rows, n_cols), dtype=complex)
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i + j
            if idx < N:
                H[i, j] = y_dt[idx]
    return H


def matrix_pencil_core(y_dt: np.ndarray, n_paths: int = 1):
    """
    Matrix‑PencilCore solver：originalGeneralized eigenvalueshapeformula，usescipy.linalg.eig(A,b=B)
    Args:
        y_dt: DT domain received sequence
        n_paths: Number of paths to estimate，Single pathsimulationFixedas1
    Returns:
        nu_f_hat: fractional DopplerEstimatevalue ∈[-0.5,0.5]
    """
    N = len(y_dt)
    L = max(3, N // 3)
    H = build_hankel(y_dt, L)

    H1 = H[:, :-1]
    H2 = H[:, 1:]

    # originalMPGeneralized eigenvalue A v = λ B v
    A = H1.conj().T @ H2
    B = H1.conj().T @ H1

    # scipy.linalg.eigsupportdurableGeneralized eigenvalue b=B
    eigvals, eigvecs = eig(A, b=B)

    # filternumericalabnormalinf/naneigenvalue
    valid_mask = np.isfinite(eigvals)
    eigvals = eigvals[valid_mask]
    if len(eigvals) == 0:
        return 0.0

    angles = np.angle(eigvals)
    nu_est_all = N * angles / (2 * np.pi)
    nu_est_all = ((nu_est_all + 0.5) % 1.0) - 0.5

    # Least-squares fitting，Select root with maximum amplitude
    Z = np.zeros((N, len(eigvals)), dtype=complex)
    for p_idx, lam in enumerate(eigvals):
        Z[:, p_idx] = lam ** np.arange(N)
    amp_est = np.linalg.pinv(Z) @ y_dt
    best_idx = np.argmax(np.abs(amp_est))
    nu_f_hat = nu_est_all[best_idx]

    return nu_f_hat


# ============================================================================
# Monte Carlo simulationfunction，interfaceforhomogeneous mc_phase_sum_qpsk_idi
# return err_mp: EstimateError1LZWnumpynumbergroups
# ============================================================================
def mc_matrix_pencil(
    M=64,
    N=16,
    lp=5,
    kp=8,
    A=1.0,
    nu_f_true=0.3,
    snr_db: float = 12,
    mc_times: int = 2000,
    seed=2026,
    n_paths=1,
    debug_one_frame: bool = False
):
    """
    Matrix‑PencilMonte Carlo simulation（standardpureimpulsepilot，usecommon_otfstime domainDopplerlink）
    Args:
        M:delaybinnumber
        N:time slot/Dopplerbinnumber
        lp:pilotdelayindex
        kp:pilotDoppler index
        A:pilot amplitude（generate_impulse_pilot_framepilot_value）
        nu_f_true:truerealfractional Doppler
        snr_db:signal-to-noise ratiodB
        mc_times:Monte Carlo times count
        seed:randomseed
        n_paths:pathsnumber，Single pathtake1
        debug_one_frame: Truethenprint1frameNoiselessDT domainphase，used formodeltypeverification
    Returns:
        err_mp: np.ndarray，eachframeEstimateError
    """
    rng = np.random.default_rng(seed)
    err_mp = []

    for idx_mc in range(mc_times):
        # standardimpulsepilotframe
        X_dd = generate_impulse_pilot_frame(M, N, lp, kp, pilot_value=A)
        # ✅usecommon_otfscompleteTransmission link：time domainapplyDoppler + AWGN + demodulationrollbackDD
        Y_dd = transmit_otfs(X_dd, nu_f_true, snr_db, M, N, rng=rng)

        y_pilot_row = Y_dd[lp, :]
        y_dt = idft_dt_domain(y_pilot_row, N)

        # scheduler trial：print1frameNoiselesslowerDT domainphase，verificationisnotapproximationpathproperty
        if debug_one_frame and idx_mc == 0 and snr_db > 80:
            phase_dt = np.angle(y_dt)
            print("\n[DEBUG] DT‑domain phase (should be approximately linear):")
            print(np.round(phase_dt, 3))

        nuf_est = matrix_pencil_core(y_dt, n_paths=n_paths)
        err_mp.append(nuf_est - nu_f_true)

    err_mp = np.array(err_mp)
    print("=" * 65)
    print(f"Matrix‑Pencil MC | SNR={snr_db} dB, MC={mc_times}")
    print(f"parameter: M={M},N={N},lp={lp},kp={kp},A={A}")
    print(f"nu_f_true={nu_f_true:.3f}, n_paths={n_paths}")
    print("=" * 65)
    print(f"【Matrix‑PencilEstimate】")
    print(f"Errorallvalue(bias) = {np.mean(err_mp):.4e}")
    print(f"MSE = {np.mean(err_mp**2):.4e}")
    print("=" * 65)
    return err_mp


# ============================================================================
# simpleinputshowexample，onlyused forversionaddressscheduler trial；dobatchcomparisontimedirectschedulerusemc_matrix_pencili.e.can
# ============================================================================
if __name__ == "__main__":
    COMMON = dict(
        M=64,
        N=16,
        lp=5,
        kp=8,
        A=1.0,
        nu_f_true=0.3,
        snr_db=10,
        mc_times=2000,
        seed=2026,
        n_paths=1
    )
    _ = mc_matrix_pencil(**COMMON)

    # scheduler trialmodelformula：printopenlowerfacethisrow，snr_db=100observeobserveDT domainphaseisnotpathproperty
    # _ = mc_matrix_pencil(**{**COMMON, "snr_db":100, "mc_times":1, "debug_one_frame":True})
