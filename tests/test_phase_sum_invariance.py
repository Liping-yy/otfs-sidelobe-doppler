"""
Unit test：Verify phase-sum invariance ψ(Δk) = arg{-y₊·y₋}

Pure pilot, noiseless, no data, strictly aligned with derivation document
Fixed delay row l=lp，Only changing Doppler column index
Verification: psi obtained with different Dk are almost exactly equal to theoretical values

Key points：
- y_plus, y_minus take DD  domainsame asdelayrowuppernosameDopplercolumnunitcomplex pts
- compute psi：np.angle(-y_plus * y_minus)，noaddtotalconjugate
- ISFFT/SFFT usematrixdirectachieves
"""

import numpy as np


def isfft(x: np.ndarray) -> np.ndarray:
    """
    DD -> TF ISFFT，standarddefinition
    x[l, k] : l(delay,row 0..M-1), k(Doppler,column 0..N-1)
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


def test_phase_sum_invariance(M=64, N=16, lp=5, kp=8, A=1.0, nu_f=0.3):
    """
    testPhase-sum invariance：ψ(Δk) shouldand Δk irrationalclose
    
    Args:
        M: Delay dimension size(subcarriernumber)
        N: Doppler dimension size(time slotnumber)
        lp: pilotdelayrow
        kp: pilot Doppler column
        A: pilotmagnitude
        nu_f: to beEstimatefractional Doppler |nu_f|<0.5
    
    Returns:
        bool: testisnotpass（std < 1e-10）
    """
    # 1. constructpureDD domainpilot：only(lp,kp)=A，otherall0
    x_dd = np.zeros((M, N), dtype=complex)
    x_dd[lp, kp] = A

    # 2. ISFFT transformtransformtoTF domain
    X_tf = isfft(x_dd)

    # 3. applyfractional Dopplerphaserotation（TF domainndimension）
    n_arr = np.arange(N)
    phase_rot = np.exp(1j * 2 * np.pi * nu_f * n_arr / N)
    R_tf = X_tf * phase_rot[:, np.newaxis]

    # 4. SFFT transformtransformrollbackDD domain
    y_dd = sfft(R_tf)

    # theoreticalpsitruevalue
    psi_theory = 2 * np.pi * nu_f * (N - 1) / N

    print("=" * 50)
    print("Unit test：purepilotNoiseless")
    print("=" * 50)
    print(f"M={M}, N={N}, nu_f={nu_f:.2f}")
    print(f"pilot position: (lp={lp}, kp={kp})")
    print(f"theoretical ψ = {psi_theory:.4f} rad")
    print()

    psi_list = []
    delta_k_set = [1, 2, 3, 4]

    for delta_k in delta_k_set:
        # symmetricsidelobe，Dopplercolumnindex，mod-Nloop
        k_plus = (kp + delta_k) % N
        k_minus = (kp - delta_k) % N

        # Fixed delay row lp，readtakeunitcomplexgrid pts
        y_plus = y_dd[lp, k_plus]
        y_minus = y_dd[lp, k_minus]

        # core：psi = arg(- y_plus * y_minus)，irrationaltotalconjugate！
        psi_meas = np.angle(-y_plus * y_minus)
        psi_list.append(psi_meas)

        diff = psi_meas - psi_theory
        status = "PASS" if abs(diff) < 1e-10 else "FAIL"
        print(f"Δk={delta_k:d} | ψ_meas={psi_meas:.4f} | "
              f"ψ_theory={psi_theory:.4f} | Δ={diff:.4e} [{status}]")

    psi_arr = np.array(psi_list)
    print()
    print("=" * 50)
    print("countresult")
    print("=" * 50)
    print(f"eachΔkdetectgetψallvalue = {np.mean(psi_arr):.4f}")
    print(f"eachΔkdetectgetψstandarddifference = {np.std(psi_arr):.4e}")

    std = np.std(psi_arr)
    passed = std < 1e-10
    if passed:
        print(f"[PASS] standarddifference {std:.2e} < 1e-10，Phase-sum invarianceholds！")
    else:
        print(f"[FAIL] standarddifference {std:.2e} >= 1e-10，Phase-sum invariancenoholds！")
    print()

    return passed


if __name__ == "__main__":
    # testnosameparameter
    test_cases = [
        {"M": 64, "N": 16, "nu_f": 0.3, "lp": 5, "kp": 8},
        {"M": 64, "N": 16, "nu_f": 0.1, "lp": 0, "kp": 0},
        {"M": 64, "N": 16, "nu_f": -0.2, "lp": 10, "kp": 7},
        {"M": 32, "N": 8, "nu_f": 0.25, "lp": 3, "kp": 4},
    ]

    all_passed = True
    for tc in test_cases:
        passed = test_phase_sum_invariance(**tc)
        all_passed = all_passed and passed
        print()

    print("=" * 50)
    if all_passed:
        print("alltestpass！Phase-sum invarianceverifybecomefunction。")
    else:
        print("partialanalysistestfail，requestcheckachieves。")
    print("=" * 50)
