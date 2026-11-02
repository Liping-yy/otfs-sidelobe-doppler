"""
Monte Carlo unit test：Pure pilot + DD-domain AWGN noise
Extended from verified phase-sum invariance unit test
verify：
1. Noiseindexinputzero-meanrandom phase jitter
2. Magnitude-vector weighted fusion denoising (complex vector summation then angle; direct arithmetic average of angles prohibited)
Constraint：
1. Does not trigger kp +/- Dk mod-N circular wrapping
2. nu_f=0asalgorithmmethodSingularity point，versiontestavoidopenthis pts
3. 4weightfor‑looponlyused forunitverify，nosuitableconjunctionhypergeometriclarge-scalesimulation
Notemeaning：
1. delta_knocantriggerfirekp±Δkmodel‑Nloopwrap，notthenDirichletsymmetricconditionbreak，invarianceFails
2. nu_f=0asversionalgorithmmethodSingularity point，microsmallNoisewillderivativeconsistentMSEsignificantuphigh，versiontestavoidopenthis pts
3. onlymodelsimulateDD domainAWGNrandomNoise；dataIDIleakagebandcomesystematicbiasnoatversiontestinner，Weighted fusionnocaneliminationsystembias
4. theoreticalirrationalnoise|ψ|<π，butNoiseconditionlowerpsicancanfirecreatewrap；large-scalesimulationsuggestaugmentaddunwrap
"""
import numpy as np
from tqdm import tqdm  # progresscurvesKKT

def isfft(x: np.ndarray) -> np.ndarray:
    """
    DD -> TF ISFFT，standarddefinition
    x[l, k] : l(delay,row 0..M‑1), k(Doppler,column 0..N‑1)
    return X[n, m] : ntime slot, msubcarrier
    """
    M, N = x.shape
    X = np.zeros((N, M), dtype=complex)
    for n_idx in range(N):
        for m_idx in range(M):
            for l_idx in range(M):
                for k_idx in range(N):
                    X[n_idx, m_idx] += x[l_idx, k_idx] * \
                        np.exp(1j * 2 * np.pi * (n_idx * k_idx / N - m_idx * l_idx / M))
    return X

def sfft(Y: np.ndarray) -> np.ndarray:
    """
    TF -> DD SFFT inverse transform
    Y[n,m] time slotn，subcarrierm
    return y[l,k] ldelayrow，kDopplercolumn
    """
    N, M = Y.shape
    y = np.zeros((M, N), dtype=complex)
    for l_idx in range(M):
        for k_idx in range(N):
            for n_idx in range(N):
                for m_idx in range(M):
                    y[l_idx, k_idx] += Y[n_idx, m_idx] * \
                        np.exp(-1j * 2 * np.pi * (n_idx * k_idx / N - m_idx * l_idx / M))
    y = y / (M * N)
    return y

def mc_phase_sum_noise(
    M=64,
    N=16,
    lp=5,
    kp=8,
    A=1.0,
    nu_f_true=0.3,
    snr_db: float = 12,
    mc_times: int = 2000,
    delta_k_list=(1, 2, 3, 4),
    seed=2026
):
    """
    AWGNMonte Carlophase suminvariancetest
    Args:
        M: subcarriernumber
        N: time slotnumber
        lp: pilotdelayrow
        kp: pilot Doppler column
        A: pilotmagnitude
        nu_f_true: truerealfractional Doppler
        snr_db: DD domainequivalentSNR(dB)
        mc_times: Monte Carlo times count
        delta_k_list: usesymmetricoffsetΔkcollection，nocantriggerfiremod-Nwrap
        seed: randomseed
    Returns:
        err_single: single pairsidelobeEstimateErrornumbergroups
        err_weighted: multisidelobevectorweightedEstimateErrornumbergroups
    """
    rng = np.random.default_rng(seed)
    psi_theory = 2 * np.pi * nu_f_true * (N - 1) / N

    err_single = []
    err_weighted = []

    # ========== 【closekeyoptimization】preventcomputepuresignal，readalgorithm1 times！noneedplacetomcloopinner ==========
    x_dd = np.zeros((M, N), dtype=complex)
    x_dd[lp, kp] = A
    X_tf = isfft(x_dd)
    n_arr = np.arange(N)
    phase_rot = np.exp(1j * 2 * np.pi * nu_f_true * n_arr / N)
    R_tf = X_tf * phase_rot[:, np.newaxis]
    y_clean = sfft(R_tf)

    power_signal = np.mean(np.abs(y_clean) ** 2)
    sigma_noise = np.sqrt(power_signal / (10.0 ** (snr_db / 10.0)))

    # tqdmprogresscurves，mc_times loopouterfacesetprogresscurves
    for _ in tqdm(range(mc_times), desc="Monte Carloiteration"):
        # ---------- 2. atDD domainsuperposeaddrecoverGaussianAWGNNoise（SFFToutputNoiseaswhiteGaussian） ----------
        noise = (rng.standard_normal(y_clean.shape) +
                 1j * rng.standard_normal(y_clean.shape)) / np.sqrt(2)
        y_noisy = y_clean + sigma_noise * noise

        psi_batch = []
        weight_batch = []
        for dk in delta_k_list:
            k_plus = (kp + dk) % N
            k_minus = (kp - dk) % N
            yp = y_noisy[lp, k_plus]
            ym = y_noisy[lp, k_minus]

            prod = -yp * ym
            psi_meas = np.angle(prod)
            w = np.abs(prod)

            psi_batch.append(psi_meas)
            weight_batch.append(w)

            # unitgroupssidelobeEstimateError
            nuf_est_single = psi_meas * N / (2 * np.pi * (N - 1))
            err_single.append(nuf_est_single - nu_f_true)

        # ---------- paper4.2：magnitude‑vectorWeighted fusion：recovervectorweightedsummationthen takeangledegree ----------
        sum_vec = 0.0 + 0j
        for psi, w in zip(psi_batch, weight_batch):
            sum_vec += w * np.exp(1j * psi)
        psi_weighted = np.angle(sum_vec)
        nuf_est_w = psi_weighted * N / (2 * np.pi * (N - 1))
        err_weighted.append(nuf_est_w - nu_f_true)

    err_single = np.array(err_single)
    err_weighted = np.array(err_weighted)

    print("\n" + "=" * 60)
    print(f"Monte Carlo | SNR={snr_db} dB, MC={mc_times}, nu_f_true={nu_f_true:.3f}")
    print(f"theoretical ψ = {psi_theory:.4f} rad")
    print("=" * 60)
    print("【unitgroupssidelobe（all delta_kmixedcount）】")
    print(f"Errorallvalue = {np.mean(err_single):.4e}")
    print(f"MSE = {np.mean(err_single**2):.4e}")
    print("\n【multisidelobemagnitude‑vectorWeighted fusion】")
    print(f"Errorallvalue = {np.mean(err_weighted):.4e}")
    print(f"MSE = {np.mean(err_weighted**2):.4e}")
    gain = np.mean(err_single**2) / np.mean(err_weighted**2)
    print(f"\nMSEdimensionalitynoisegain(single pair/weighted) = {gain:.2f} x")
    print("=" * 60)

    return err_single, err_weighted


if __name__ == "__main__":
    # showexample：unitSNR ptstest
    _ = mc_phase_sum_noise(
        M=64,
        N=16,
        lp=5,
        kp=8,
        A=1.0,
        nu_f_true=0.3,
        snr_db=12,
        mc_times=2000,
        delta_k_list=[1, 2, 3, 4],
        seed=2026
    )
