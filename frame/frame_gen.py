# -*- coding: utf-8 -*-
"""
DD-domain frame generation
==========

Three frame types：
  1. Pure impulse frame：Single pilot impulse, no data——Verify theoretical upper bound
  2. engineering frame（legacy）：pilot + guard zero + QPSK data——Verify practical accuracy（exp1-6legacyuse）
  3. engineering frame（zero-padded）：Pilot row fully zero-padded + QPSK data——Used in new paper experiments

Frame structure (zero-padded version)：
  x[l_p, k_p] = A
  x[l_p, k]   = 0,   for all k != kp (entire row zero-padded, no Kmax concept)
  x[l, k]     = QPSK, for all l != lp

Author: AI Engineer
Date: 2026-09-11
"""

import numpy as np


# ============================================================
# QPSKconstellationgenerate
# ============================================================

def generate_qpsk_symbols(shape, rng: np.random.Generator) -> np.ndarray:
    """
    Generate QPSK symbol matrix (normalized power=1)。

    Gray mapping: (0,0)→(1+j)/√2, (0,1)→(1-j)/√2,
              (1,0)→(-1+j)/√2, (1,1)→(-1-j)/√2

    Args:
        shape: Output shape (M, N) or (M, N, K)
        rng: Random generator

    Returns:
        QPSK symbol matrix
    """
    bits = rng.integers(0, 2, size=(*shape, 2))
    return ((1 - 2 * bits[..., 0]) + 1j * (1 - 2 * bits[..., 1])) / np.sqrt(2)


# ============================================================
# Pure impulse frame
# ============================================================

def generate_pure_frame(
    M: int,
    N: int,
    lp: int = 5,
    kp: int = 8,
    pilot_value: float = 1.0,
) -> np.ndarray:
    """
    Generate pure impulse pilot frame (no data, no guard zero)。

    onlyat(lp, kp)handlerplacesetpilot，otherallzero。

    Args:
        M: Delay dimension size
        N: Doppler dimension size
        lp: pilotdelayrow
        kp: pilot Doppler column
        pilot_value: pilotmagnitude A

    Returns:
        X_dd: [M, N] DD-domain frame
    """
    X_dd = np.zeros((M, N), dtype=complex)
    X_dd[lp, kp] = pilot_value
    return X_dd


# ============================================================
# engineering frame（legacy：pilot + guard zero + QPSK data）
# ============================================================

def generate_engineering_frame(
    M: int,
    N: int,
    lp: int,
    kp: int,
    pilot_amp: float,
    Kmax: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate engineering frame：pilot + guard zero + QPSK data（legacy，keptcompatible）。

    Frame structure:
      x[l,k] = A,           l=lp, k=kp
      x[l,k] = 0,           l=lp, |k-kp| <= Kmax, k != kp  （guard zero）
      x[l,k] = QPSKSymbols,    otherall(l,k)                    （datadata）

    Notemeaning：thisfunctionkeptused forlegacyexperiment(exp1-6original)。newexperimentrequestuse
    generate_engineering_frame_zp()。

    Args:
        M: Delay dimension size
        N: Doppler dimension size
        lp: pilotdelayrow
        kp: pilot Doppler column
        pilot_amp: pilotmagnitude A
        Kmax: Guard zero one-sided length (legacy parameter)
        rng: Random generator

    Returns:
        X_dd: [M, N] DD-domain frame
    """
    X_dd = generate_qpsk_symbols((M, N), rng)

    # pilot row settingsguard zero
    for dk in range(-Kmax, Kmax + 1):
        k_col = (kp + dk) % N
        X_dd[lp, k_col] = 0.0

    # placesetpilot
    X_dd[lp, kp] = pilot_amp

    return X_dd


# ============================================================
# engineering frame（new version：Pilot row fully zero-padded + QPSK data）
# ============================================================

def generate_engineering_frame_zp(
    M: int,
    N: int,
    lp: int,
    kp: int,
    pilot_amp: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate engineering frame：Pilot row fully zero-padded + QPSK data（new version，No Kmax concept）。

    Frame structure（correspondingpapermodifiedback）:
      x[l_p, k_p] = A,         pilot
      x[l_p, k]   = 0,         forall k ≠ k_p  （integerrowzero-padded）
      x[l, k]     = QPSKSymbols,  for all l != lp  （Non-pilot rows fully filled with data）

    andlegacydifference：
      - legacy(generate_engineering_frame): onlyguard zeroregion(|k-kp|<=Kmax)setzero，
        pilot rowguard zeroviaoutercolumnstillcanplacesetQPSKSymbols
      - new version: pilot row fully zero-padded，allDopplercolumnallirrationaldata interference

    Args:
        M: Delay dimension size
        N: Doppler dimension size
        lp: pilotdelayrow
        kp: pilot Doppler column
        pilot_amp: pilotmagnitude A
        rng: Random generator

    Returns:
        X_dd: [M, N] DD-domain frame
    """
    # allframefillQPSKdata
    X_dd = generate_qpsk_symbols((M, N), rng)

    # pilot row fully zero-padded（No Kmax concept）
    X_dd[lp, :] = 0.0

    # placesetpilot
    X_dd[lp, kp] = pilot_amp

    return X_dd


# ============================================================
# frame typerouted
# ============================================================

def generate_frame(
    frame_type: str,
    M: int,
    N: int,
    lp: int = 5,
    kp: int = 8,
    pilot_amp: float = 3.0,
    Kmax: int = 10,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Unified frame generation entry point (routed by type)。

    Args:
        frame_type: "pure"=Pure impulse frame, "engineering"=engineering frame(legacy),
                    "engineering_zp"=engineering frame(zero-padded,new version)
        M, N: Frame dimension
        lp, kp: pilot position
        pilot_amp: pilotmagnitude
        Kmax: Guard zero length (legacy engineering frame only)
        rng: Random generator (required for engineering frame)

    Returns:
        X_dd: [M, N] DD-domain frame
    """
    if rng is None:
        rng = np.random.default_rng()

    if frame_type == "pure":
        return generate_pure_frame(M, N, lp, kp, pilot_amp)
    elif frame_type == "engineering":
        return generate_engineering_frame(M, N, lp, kp, pilot_amp, Kmax, rng)
    elif frame_type == "engineering_zp":
        return generate_engineering_frame_zp(M, N, lp, kp, pilot_amp, rng)
    else:
        raise ValueError(
            f"Unknownframe type: {frame_type}，available 'pure', 'engineering', 'engineering_zp'"
        )


# ============================================================
# self-test
# ============================================================

def self_test():
    """Verify frame generation correctness。"""
    M, N = 64, 16
    lp, kp = 5, 8
    rng = np.random.default_rng(2026)

    # Test 1: pure frame
    X_pure = generate_pure_frame(M, N, lp, kp, pilot_value=3.0)
    nonzeros = np.sum(np.abs(X_pure) > 0)
    print(f"[1] pure framenon-zero count: {nonzeros} (expected1)",
          "PASS" if nonzeros == 1 else "FAIL")
    print(f"    pilot value: {X_pure[lp, kp]} (expected3.0)")

    # Test 2: legacyengineering frameguard zero
    Kmax = 10
    X_eng = generate_engineering_frame(M, N, lp, kp, 3.0, Kmax, rng)
    guard_ok = True
    for dk in range(-Kmax, Kmax + 1):
        k_col = (kp + dk) % N
        if k_col == kp:
            continue
        if X_eng[lp, k_col] != 0:
            guard_ok = False
            break
    print(f"[2] legacyengineering frameguard zeroregionallzero: {'PASS' if guard_ok else 'FAIL'}")
    print(f"    pilot value: {X_eng[lp, kp]} (expected3.0)")

    # Test 3: legacyengineering framenon-pilot rowrationalQPSK
    other_rows_nonzero = np.sum(np.abs(X_eng[0, :]) > 0)
    print(f"[3] legacynon-pilot row(row0)QPSKSymbolsnumber: {other_rows_nonzero} (expected{N})",
          "PASS" if other_rows_nonzero == N else "FAIL")

    # Test 4: new versionzero-paddedframe
    rng2 = np.random.default_rng(2026)
    X_zp = generate_engineering_frame_zp(M, N, lp, kp, 3.0, rng2)

    # pilot row: onlykphandlernon-zero
    pilot_row_nonzero = np.sum(np.abs(X_zp[lp, :]) > 0)
    print(f"[4] new versionzero-paddedframepilot rownon-zero count: {pilot_row_nonzero} (expected1)",
          "PASS" if pilot_row_nonzero == 1 else "FAIL")
    print(f"    pilot value: {X_zp[lp, kp]} (expected3.0)")

    # pilot rowotherlocsetallzero
    pilot_row_zero = all(X_zp[lp, k] == 0 for k in range(N) if k != kp)
    print(f"[5] new versionzero-paddedframepilot rowotherallzero: {'PASS' if pilot_row_zero else 'FAIL'}")

    # non-pilot rowrationalQPSK
    other_rows_nonzero_zp = np.sum(np.abs(X_zp[0, :]) > 0)
    print(f"[6] new versionnon-pilot row(row0)QPSKSymbolsnumber: {other_rows_nonzero_zp} (expected{N})",
          "PASS" if other_rows_nonzero_zp == N else "FAIL")

    # Test 7: unifiedrouted
    X_route = generate_frame("pure", M, N, lp, kp, pilot_amp=1.0)
    print(f"[7] routedpure frame: {'PASS' if X_route[lp, kp] == 1.0 else 'FAIL'}")

    X_route2 = generate_frame("engineering_zp", M, N, lp, kp, pilot_amp=2.0, rng=rng)
    print(f"[8] routedzero-paddedframe: {'PASS' if X_route2[lp, kp] == 2.0 else 'FAIL'}")

    print("\nFrame generation self-test done。")


if __name__ == "__main__":
    self_test()
