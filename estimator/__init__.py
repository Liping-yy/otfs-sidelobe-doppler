# -*- coding: utf-8 -*-
"""
estimator package - Fractional Doppler estimators
==================================================

Unified calling interface:
    estimate_nu_f(y_pilot_row, kp, N, **kwargs) -> float

Five estimators:
  - proposed:       Sidelobe + peak self-multiplication closed-form (core approach)
  - sidetone_only:  Sidelobe-only (ablation control)
  - matrix_pencil:  Matrix Pencil baseline (Liang 2024 IEEE CL)
  - he2023:         He 2023 two-stage correlation + quasi-Newton baseline (IEEE CL)
  - prony:          Prony baseline (arXiv:2506.17599, kept as backup)
"""

# Unified interface registry
ESTIMATOR_REGISTRY = {}


def _register(name, module):
    """Register estimator into global dictionary."""
    ESTIMATOR_REGISTRY[name] = module.estimate_nu_f


from estimator import proposed
from estimator import sidetone_only
from estimator import matrix_pencil
from estimator import he2023
from estimator import prony

_register("proposed", proposed)
_register("sidetone_only", sidetone_only)
_register("matrix_pencil", matrix_pencil)
_register("he2023", he2023)
_register("prony", prony)


def get_estimator(name: str):
    """
    Get an estimator function.

    Args:
        name: "proposed" / "sidetone_only" / "matrix_pencil" / "prony"

    Returns:
        estimate_nu_f function
    """
    if name not in ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown estimator: {name}, available: {list(ESTIMATOR_REGISTRY.keys())}")
    return ESTIMATOR_REGISTRY[name]


__all__ = [
    "proposed",
    "sidetone_only",
    "matrix_pencil",
    "he2023",
    "prony",
    "ESTIMATOR_REGISTRY",
    "get_estimator",
]
