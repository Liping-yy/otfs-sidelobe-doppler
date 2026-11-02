"""
Monte Carlo simulation：DD-domain QPSK data symbols + pilot + guard zero frame structure

verifyappearphenomenon：
1. When guard zero is sufficient, data leakage is small, estimation MSE close to pure pilot
2. When guard zero is insufficient, introduces systematic bias (non-zero mean noise), MSE significantly degrades
3. Multiple sidelobe weighted fusion can only suppress random AWGN noise, cannot eliminate systematic bias from data IDI

Constraint：
1. Does not trigger kp +/- Dk mod-N circular wrapping
2. nu_f=0asalgorithmmethodSingularity point，versiontestavoidopenthis pts
3. delta_k_listmaximum valuenocanhypergeometricpassKmax

Notemeaning：
1. rationalQPSKdatatime，each1framex_ddallrandomnosame，musteachframecompleterunISFFT-Doppler-SFFT
2. onlymodelsimulateSingle-path LOS channel；DD domaindataSymbolsbandcomeIDIDopplerbetweeninterference
3. theoreticalirrationalnoise|ψ|<π，butNoise+IDIconditionlowerpsicancanfirecreatewrap；alreadyaddinputunwraphandling
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
# generatebandguard zeroDD domainmixedframe（pilot + QPSKdata）
# ============================================================================
def generate_dd_frame(
    M: int,
    N: int,
    lp: int,
    kp: int,
    A: float,
    Kmax: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    constructDD domainmixedframe：pilot + guard zero + QPSK data，forhomogeneoustextdoc5.1nodeFrame structure

    DD-domain framedefinition：
    x[l,k] = A,           l=lp, k=kp
    x[l,k] = 0,           l=lp, k∈[kp-Kmax, kp+Kmax], k≠kp
    x[l,k] = QPSKSymbols,    otherall(l,k)

    Args:
        M: delaydimension(row)
        N: Doppler dimension(column)
        lp: pilotdelayrow
        kp: pilot Doppler column
        A: pilotrecovermagnitude(realnumber>0)
        Kmax: unitedgeguard zerolongdegree
        rng: Random generator

    Returns:
        x_dd: shape(M,N) DD-domain frame
    """
    # QPSKnormalizedconstellation，averagepower = 1
    qpsk_sym = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    x_dd = rng.choice(qpsk_sym, size=(M, N))

    # placesetpilot
    x_dd[lp, kp] = A

    # lprowsettingsguard zerowindow：kp-Kmax … kp+Kmax，k≠kpset0
    for dk in range(-Kmax, Kmax + 1):
        k_col = (kp + dk) % N
        if k_col == kp:
            continue
        x_dd[lp, k_col] = 0.0

    return x_dd

# ============================================================================
# Monte Carlotest：QPSKdata + IDIleakage
# ============================================================================
def mc_phase_sum_qpsk_idi(
    M=64,
    N=16,
    lp=5,
    kp=8,
    A=1.0,
    Kmax: int = 10,
    nu_f_true=0.3,
    snr_db: float = 12,
    mc_times: int = 2000,
    delta_k_list=(1, 2, 3, 4),
    seed=2026,
):
    """
    AWGN + QPSKdataIDIMonte CarloPhase-sum invariancetest

    Args:
        M: subcarriernumber
        N: time slotnumber
        lp: pilotdelayrow
        kp: pilot Doppler column
        A: pilotrecovermagnitude
        Kmax: unitedgeguard zerolongdegree
        nu_f_true: truerealfractional Doppler
        snr_db: DD domainequivalentSNR(dB)
        mc_times: Monte Carlo times count
        delta_k_list: usesymmetricoffsetΔkcollection
        seed: randomseed

    Returns:
        err_single: single pairsidelobeEstimateErrornumbergroups
        err_weighted: multisidelobevectorweightedEstimateErrornumbergroups
    """
    # runtimebreakspeech：sidelobeobservedetectΔknocanhypergeometricpassguard zeroKmax
    assert max(delta_k_list) <= Kmax, \
        f"sidelobeobservedetectΔk={max(delta_k_list)}hypergeometricpassguard zeroKmax={Kmax}，sample pointwillfalls atdataregion"

    rng = np.random.default_rng(seed)
    psi_theory = 2 * np.pi * nu_f_true * (N - 1) / N

    err_single = []
    err_weighted = []

    for mc_idx in range(mc_times):
        # ---------- 1. each1frameweightnewgeneraterandomDDframe：pilot+guard zero+QPSKdata ----------
        x_dd = generate_dd_frame(M, N, lp, kp, A, Kmax, rng)

        # completeOTFStransformtransformlink，each1framemustweightnewrun
        X_tf = isfft_fft(x_dd)
        n_arr = np.arange(N)
        phase_rot = np.exp(1j * 2 * np.pi * nu_f_true * n_arr / N)
        R_tf = X_tf * phase_rot[:, np.newaxis]
        y_clean = sfft_fft(R_tf)

        # ---------- 2. superposeaddDD domainAWGNNoise ----------
        # fixpositive：usesidelobesample pointlocalpowercomputesigma_noise，whilenon-allframe averagepower
        # reason：allframe averagepowerbydataSymbolsdominant，derivativeconsistentsidelobesample pointrealinterSNRfarlowatsettheoremvalue
        local_powers = []
        for dk in delta_k_list:
            local_powers.append(np.abs(y_clean[lp, kp + dk]) ** 2)
            local_powers.append(np.abs(y_clean[lp, kp - dk]) ** 2)
        power_signal = np.mean(local_powers)
        sigma_noise = np.sqrt(power_signal / (10.0 ** (snr_db / 10.0)))
        noise = (rng.standard_normal(y_clean.shape) +
                 1j * rng.standard_normal(y_clean.shape)) / np.sqrt(2)
        y_noisy = y_clean + sigma_noise * noise

        # ---------- 3. samplingsidelobe、psicompute、unwrap、Weighted fusion ----------
        psi_batch = []
        weight_batch = []
        for dk in delta_k_list:
            # wrapdetectdetectbreakspeech（Unit testnosupportdurablemodel-Nwrap）
            if (kp + dk) >= N or (kp - dk) < 0:
                raise ValueError(
                    f"delta_k={dk}triggerfirekp±Δkwrap，Unit testnosupportdurable。"
                    f"kp={kp}, N={N}, requestensurekp-Kmax>=0andkp+Kmax<N"
                )
            k_plus = kp + dk
            k_minus = kp - dk
            yp = y_noisy[lp, k_plus]
            ym = y_noisy[lp, k_minus]

            prod = -yp * ym
            psi_meas = np.angle(prod)
            # unwraphandling：viatheoreticaltruevalueasmidextendopen
            psi_meas = np.unwrap([psi_meas, psi_theory])[0]

            w = np.abs(prod)
            psi_batch.append(psi_meas)
            weight_batch.append(w)

            # unitgroupssidelobeEstimateError
            nuf_est_single = psi_meas * N / (2 * np.pi * (N - 1))
            err_single.append(nuf_est_single - nu_f_true)

        # paper4.2：magnitude-vectorWeighted fusion
        sum_vec = 0.0 + 0j
        for psi, w in zip(psi_batch, weight_batch):
            sum_vec += w * np.exp(1j * psi)
        psi_weighted = np.angle(sum_vec)
        nuf_est_w = psi_weighted * N / (2 * np.pi * (N - 1))
        err_weighted.append(nuf_est_w - nu_f_true)

    err_single = np.array(err_single)
    err_weighted = np.array(err_weighted)

    print("=" * 60)
    print(f"QPSKdata+IDIMonte Carlo | SNR={snr_db} dB, MC={mc_times}")
    print(f"parameter: M={M}, N={N}, lp={lp}, kp={kp}, A={A}, Kmax={Kmax}")
    print(f"nu_f_true={nu_f_true:.3f}, theoreticalψ={psi_theory:.4f} rad")
    print(f"delta_k_list={delta_k_list}")
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
# primaryprocesssequence：3groupscomparisonexperiment
# ============================================================================
if __name__ == "__main__":
    # publictotalparameter（noincludingdelta_k_list，eachcaseindependentsettings）
    COMMON = dict(
        M=64,
        N=16,
        lp=5,
        kp=8,
        nu_f_true=0.3,
        snr_db=12,
        mc_times=2000,
        seed=2026,
    )

    # Case 1: guard zerosufficient，pilotdataequalpower
    print("\n" + "=" * 60)
    print("experimentgroups Case 1: Kmax=10, A=1.0, delta_k=[1,2,3,4] (guard zerosufficient)")
    print("Expected：leakagesmall，performanceconnectapproximationpurepilot")
    print("=" * 60)
    _ = mc_phase_sum_qpsk_idi(**COMMON, A=1.0, Kmax=10, delta_k_list=[1, 2, 3, 4])

    # # Case 2: guard zero insufficient，pilotdataequalpower
    # # delta_k_listmust<=Kmax，soKmax=2timereadtake[1,2]
    # print("\n" + "=" * 60)
    # print("experimentgroups Case 2: Kmax=2, A=1.0, delta_k=[1,2] (guard zero insufficient)")
    # print("Expected：clearvisiblesystembias，MSEdegrades")
    # print("=" * 60)
    # _ = mc_phase_sum_qpsk_idi(**COMMON, A=1.0, Kmax=2, delta_k_list=[1, 2])
    #
    # # Case 3: guard zero insufficient，butpilot power boostB=3
    # print("\n" + "=" * 60)
    # print("experimentgroups Case 3: Kmax=2, A=3.0, delta_k=[1,2] (pilot power boostB=3)")
    # print("Expected：pilot powerraisehighcompresscontrolIDI，biasrationalthatcachedecompose")
    # print("=" * 60)
    # _ = mc_phase_sum_qpsk_idi(**COMMON, A=3.0, Kmax=2, delta_k_list=[1, 2])
