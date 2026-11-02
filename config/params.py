# -*- coding: utf-8 -*-
"""
Global unified simulation parameters (shared across the entire paper)
=====================================================================

Corresponds to the unified system parameter table in Section 1 of the
experiment outline. All modules import via `from config.params import SimConfig`;
scattered hardcoded parameters elsewhere are prohibited.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class SimConfig:
    """Global simulation parameters for OTFS fractional Doppler estimation."""

    # ===== OTFS system parameters =====
    M: int = 64              # Delay dimension size (number of subcarriers)
    N: int = 16              # Doppler dimension size (number of time slots)
    delta_f: float = 120e3   # Subcarrier spacing (Hz), 120 kHz
    cp_len: int = 0          # CP length (single-path LOS, CP=0)

    # ===== LEO satellite channel parameters =====
    fc: float = 28e9         # Carrier frequency (Hz), Ka-band 28 GHz
    altitude: float = 600e3  # Satellite altitude (m), 600 km
    elevation_min: float = 40.0  # Minimum operating elevation angle (degrees)

    # ===== Pilot parameters =====
    pilot_amp: float = 3.0   # Pilot amplitude A (PILOT_BOOST=3.0)
    pilot_delay: int = 5     # Pilot delay row index lp
    pilot_doppler: int = 8   # Pilot Doppler column index kp
    zc_root: int = 1         # ZC sequence root index (reserved; impulse pilot currently used)

    # ===== Engineering frame parameters =====
    kmax_default: int = 10   # Default guard zero one-sided length
    qpsk_mode: str = "gray"  # QPSK mapping mode

    # ===== Simulation general configuration =====
    seed: int = 2026         # Global random seed
    mc_times: int = 2000     # Default Monte Carlo count
    snr_range: Tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
    nu_f_default: float = 0.3  # Default fractional Doppler (far from singularity)

    # ===== Estimator parameters =====
    delta_k_list: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)  # Sidelobe offset list
    delta_k_sidetone: Tuple[int, ...] = (1, 2, 3, 4)       # Sidelobe-only (ablation control)

    # ===== Engineering frame SNR definition =====
    # Approach A (current): global SNR = total frame signal power / noise power
    #   Pilot boost A=3.0 makes sidelobe effective SNR ~15 dB higher than global;
    #   the engineering frame curve is left-shifted ~15 dB in the MSE-SNR plot,
    #   intuitively demonstrating the engineering value of the pilot boost.
    local_snr: bool = False


# Default configuration instance (import directly for use)
DEFAULT = SimConfig()


def get_config(**overrides) -> SimConfig:
    """
    Create a configuration instance, allowing partial parameter override.

    Examples:
        cfg = get_config(M=32, N=8)  # Modify frame parameters
    """
    return SimConfig(**{k: v for k, v in overrides.items()
                        if k in SimConfig.__dataclass_fields__})
