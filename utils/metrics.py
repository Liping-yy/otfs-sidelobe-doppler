# -*- coding: utf-8 -*-
"""
simulationmetricsUtilities
============

unifiedErrormetricssumresultclusteringconjunctionUtilities。
All experiment scripts compute performance metrics through this module, ensuring consistent definitions。

"""

import numpy as np
import json
import os
from datetime import datetime


def compute_mse(errors: np.ndarray) -> float:
    """computemean squared error MSE = mean(err²)。"""
    return float(np.mean(np.asarray(errors) ** 2))


def compute_bias(errors: np.ndarray) -> float:
    """computeBias bias = mean(err)。"""
    return float(np.mean(np.asarray(errors)))


def compute_mse_bias(errors: np.ndarray) -> dict:
    """Simultaneously compute MSE and bias。"""
    return {
        "bias": compute_bias(errors),
        "mse": compute_mse(errors),
        "std": float(np.std(np.asarray(errors))),
        "n_trials": len(errors),
    }


def mse_ratio(errors_a: np.ndarray, errors_b: np.ndarray) -> float:
    """
    Compute MSE ratio of two estimators (a/b)。
    used for "Proposed vs MP" equalcomparison。
    """
    mse_a = compute_mse(errors_a)
    mse_b = compute_mse(errors_b)
    if mse_b == 0:
        return float("inf")
    return mse_a / mse_b


def save_results(results: dict, filepath: str) -> None:
    """
    saveexperimentresulttoJSONfile。

    Args:
        results: resultdict
        filepath: outputpaths（including.jsonbacksuffix）
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # appendaddtimebetweenclick
    results["_meta"] = {
        "saved_at": datetime.now().isoformat(),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"resultalreadysave: {filepath}")


def print_comparison(name: str, results: dict, baseline_key: str = None) -> None:
    """
    printmultiestimatorcomparisontable。

    Args:
        name: experimentnamesymmetric
        results: {estimator_name: {"bias":..., "mse":..., "n_trials":...}}
        baseline_key: Baselineestimator name（computeratiouse）
    """
    print("=" * 70)
    print(f"  {name}")
    print("=" * 70)
    print(f"  {'approaches':<28s} {'bias':>10s} {'MSE':>12s} {'vsBaseline':>8s}")
    print("-" * 70)

    baseline_mse = None
    if baseline_key and baseline_key in results:
        baseline_mse = results[baseline_key].get("mse", None)

    for est_name, metrics in results.items():
        if est_name.startswith("_"):
            continue
        bias = metrics.get("bias", 0)
        mse = metrics.get("mse", 0)
        ratio_str = "---"
        if baseline_mse and baseline_mse > 0:
            ratio_str = f"{mse / baseline_mse:.1f}x"
        print(f"  {est_name:<28s} {bias:>+10.4e} {mse:>12.4e} {ratio_str:>8s}")

    print("=" * 70)
