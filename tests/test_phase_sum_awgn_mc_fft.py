"""
Monte Carlo unit test：Pure pilot + DD-domain AWGN noise (FFT-accelerated)
Extended from verified phase-sum invariance unit test

verify：
1. Noiseindexinputzero-meanrandom phase jitter
2. Magnitude-vector weighted fusion denoising (complex vector summation then angle; direct arithmetic average of angles prohibited)

Constraint：
1. Does not trigger kp +/- Dk mod-N circular wrapping
2. nu_f=0asalgorithmmethodSingularity point，versiontestavoidopenthis pts

Notemeaning：
1. delta_knocantriggerfirekp±Δkmodel-Nloopwrap，notthenDirichletsymmetricconditionbreak，invarianceFails
2. nu_f=0asversionalgorithmmethodSingularity point，microsmallNoisewillderivativeconsistentMSEsignificantuphigh，versiontestavoidopenthis pts
3. onlymodelsimulateDD domainAWGNrandomNoise；dataIDIleakagebandcomesystematicbiasnoatversiontestinner，Weighted fusionnocaneliminationsystembias
4. theoreticalirrationalnoise|ψ|<π，butNoiseconditionlowerpsicancanfirecreatewrap；large-scalesimulationsuggestaugmentaddunwrap
"""

import numpy as np


# ============================================================================
# FFTacceleratedversion ISFFT / SFFT（andpaperformulanumberlearnstrictequivalent）
# ============================================================================
def isfft_fft(x: np.ndarray) -> np.ndarray:
    """
    FFTacceleratedversion ISFFT

    paperformula：X[n,m] = sum_l sum_k x[l,k] * exp(1j*2pi*(n*k/N - m*l/M))

    Derivation mapping：
    - alongDopplerkaxis（axis=1，size=N）：sum_k x[l,k]*exp(j2pi*nk/N)
      = N * ifft(x, axis=1)[l,n]  （ifftnaturalband1/N，needmultiplierNCompensation）
    - alongdelaylaxis（axis=0，size=M）：sum_l f[l,n]*exp(-j2pi*ml/M)
      = fft(f, axis=0)[m,n]

    Combined：X = fft(N*ifft(x, axis=1), axis=0).T

    x[l, k] : l(delay, Mrow), k(Doppler, Ncolumn) shape(M,N)
    return X[n, m] : ntime slot(Nrow), msubcarrier(Mcolumn) shape(N,M)
    """
    M, N = x.shape
    t1 = N * np.fft.ifft(x, axis=1)   # shape(M,N): f[l,n] = sum_k x[l,k]*exp(j2pi*nk/N)
    t2 = np.fft.fft(t1, axis=0)        # shape(M,N): g[m,n] = sum_l f[l,n]*exp(-j2pi*ml/M)
    X = t2.T                           # shape(N,M): X[n,m] = g[m,n]
    return X


def sfft_fft(Y: np.ndarray) -> np.ndarray:
    """
    FFTacceleratedversion SFFTinverse transform

    paperformula：y[l,k] = 1/(M*N) sum_n sum_m Y[n,m] * exp(-1j*2pi*(n*k/N - m*l/M))

    Derivation mapping：
    - alongsubcarriermaxis（axis=1，size=M）：sum_m Y[n,m]*exp(j2pi*ml/M)
      = M * ifft(Y, axis=1)[n,l]  （ifftnaturalband1/M，needmultiplierMCompensation）
    - alongtime slotnaxis（axis=0，size=N）：sum_n g[n,l]*exp(-j2pi*nk/N)
      = fft(g, axis=0)[k,l]  （fftnaturalbandexp(-j2pi*nk/N)）

    Combined：y = fft(M*ifft(Y, axis=1), axis=0).T / (M*N)

    Y[n,m] : ntime slot(Nrow), msubcarrier(Mcolumn) shape(N,M)
    return y[l,k] : ldelay(Mrow), kDoppler(Ncolumn) shape(M,N)
    """
    N, M = Y.shape
    t1 = M * np.fft.ifft(Y, axis=1)   # shape(N,M): g[n,l] = sum_m Y[n,m]*exp(j2pi*ml/M)
    t2 = np.fft.fft(t1, axis=0)        # shape(N,M): h[k,l] = sum_n g[n,l]*exp(-j2pi*nk/N)
    y = t2.T / (M * N)                  # shape(M,N): y[l,k] = h[k,l]/(MN)
    return y


# ============================================================================
# Monte Carlotest
# ============================================================================
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
    seed=2026,
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

    for _ in range(mc_times):
        # ---------- 1. generatepureDDpilot，ISFFT-Doppler-SFFTgettoNoiselessDD ----------
        x_dd = np.zeros((M, N), dtype=complex)
        x_dd[lp, kp] = A
        X_tf = isfft_fft(x_dd)
        n_arr = np.arange(N)
        phase_rot = np.exp(1j * 2 * np.pi * nu_f_true * n_arr / N)
        R_tf = X_tf * phase_rot[:, np.newaxis]
        y_clean = sfft_fft(R_tf)

        # ---------- 2. atDD domainsuperposeaddrecoverGaussianAWGNNoise（SFFToutputNoiseaswhiteGaussian） ----------
        power_signal = np.mean(np.abs(y_clean) ** 2)
        sigma_noise = np.sqrt(power_signal / (10.0 ** (snr_db / 10.0)))
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

        # ---------- paper4.2：magnitude-vectorWeighted fusion：recovervectorweightedsummationthen takeangledegree ----------
        sum_vec = 0.0 + 0j
        for psi, w in zip(psi_batch, weight_batch):
            sum_vec += w * np.exp(1j * psi)
        psi_weighted = np.angle(sum_vec)
        nuf_est_w = psi_weighted * N / (2 * np.pi * (N - 1))
        err_weighted.append(nuf_est_w - nu_f_true)

    err_single = np.array(err_single)
    err_weighted = np.array(err_weighted)

    print("=" * 60)
    print(f"Monte Carlo [FFTacceleratedversion] | SNR={snr_db} dB, MC={mc_times}, nu_f_true={nu_f_true:.3f}")
    print(f"theoretical ψ = {psi_theory:.4f} rad")
    print("=" * 60)
    print("【unitgroupssidelobe（all delta_kmixedcount）】")
    print(f"Errorallvalue = {np.mean(err_single):.4e}")
    print(f"MSE = {np.mean(err_single**2):.4e}")
    print("\n【multisidelobemagnitude-vectorWeighted fusion】")
    print(f"Errorallvalue = {np.mean(err_weighted):.4e}")
    print(f"MSE = {np.mean(err_weighted**2):.4e}")
    gain = np.mean(err_single**2) / np.mean(err_weighted**2)
    print(f"\nMSEdimensionalitynoisegain(single pair/weighted) = {gain:.2f} x")
    print("=" * 60)

    return err_single, err_weighted


# ============================================================================
# primaryprocesssequence
# ============================================================================
if __name__ == "__main__":
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
        seed=2026,
    )
