# -*- coding: utf-8 -*-
"""
Plotting script for the multipath / Rician experiment: Rician K-factor sweep,
===============================================================================

Generates four figures from results/exp7_multipath.json:

  fig7a_k_sweep.pdf     MSE vs Rician K-factor (all estimators)
  fig7b_weak_echo.pdf   MSE vs weak-echo delay offset (proposed vs MP)
  fig7c_timing_sweep.pdf  MSE vs residual timing offset (fractional delay)
  fig7d_freq_sweep.pdf    MSE vs residual frequency offset

Output: PDF (vector) + PNG (preview) in project_files/figures/
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---- same global style as plot_all_figures.py ----
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'proposed': '#e41a1c',
    'sidetone_only': '#377eb8',
    'matrix_pencil': '#4daf4a',
    'he2023': '#984ea3',
    'ideal': '#777777',
}
MARKERS = {
    'proposed': 'o',
    'sidetone_only': 's',
    'matrix_pencil': 'D',
    'he2023': '^',
}
LABELS = {
    'proposed': 'Proposed',
    'sidetone_only': 'Sidetone Only',
    'matrix_pencil': 'Matrix Pencil',
    'he2023': 'He 2023',
}
EST_NAMES = ['proposed', 'sidetone_only', 'matrix_pencil', 'he2023']


def _save_fig(fig, name):
    pdf_path = os.path.join(FIGURES_DIR, f"{name}.pdf")
    png_path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  saved: {pdf_path}")
    print(f"  saved: {png_path}")


def _load_json(name):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def plot_exp7a_k_sweep():
    """Figure 7a: MSE vs Rician K-factor (log-y)."""
    d = _load_json("exp7_multipath")
    k_sweep = d["k_sweep"]
    cfg = k_sweep["config"]
    k_list = [k_sweep["results"][f"K_{k:.0f}dB"]["k_db"]
              for k in cfg["k_db_list"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    for est in EST_NAMES:
        mses = [k_sweep["results"][f"K_{k:.0f}dB"]["metrics"][est]["mse"]
                for k in cfg["k_db_list"]]
        ax.semilogy(k_list, mses,
                    color=COLORS[est], marker=MARKERS[est],
                    label=LABELS[est], markersize=7, linewidth=1.8)

    ax.set_xlabel(r'Rician $K_r$-factor (dB)')
    ax.set_ylabel('MSE')
    ax.set_title(rf'MSE vs Rician $K_r$-factor (SNR={cfg["snr_db"]:.0f}dB, '
                 rf'$P_s$={cfg["p_s"]}, $\nu_f$={cfg["nu_f"]}, MC={cfg["mc_times"]})')
    ax.legend(loc='upper right')
    ax.set_xticks(k_list)
    ax.yaxis.set_major_locator(LogLocator(numticks=7))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    fig.tight_layout()
    _save_fig(fig, 'fig7a_k_sweep')


def plot_exp7b_weak_echo():
    """Figure 7b: MSE vs weak-echo delay offset (proposed vs MP)."""
    d = _load_json("exp7_multipath")
    we = d["weak_echo"]
    cfg = we["config"]
    offsets = cfg["delay_offsets"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for est in ['proposed', 'matrix_pencil', 'he2023']:
        mses = [we["results"][f"d_off_{off}"]["metrics"][est]["mse"]
                for off in offsets]
        ax.semilogy(offsets, mses,
                    color=COLORS[est], marker=MARKERS[est],
                    label=LABELS[est], markersize=7, linewidth=1.8)

    ax.set_xlabel('Weak-echo delay offset (delay taps from pilot row)')
    ax.set_ylabel('MSE')
    ax.set_title(rf'Weak Second-Path Scenario ($K_r$={cfg["k_db"]:.0f}dB, '
                 rf'SNR={cfg["snr_db"]:.0f}dB, MC={cfg["mc_times"]})')
    ax.legend(loc='upper left')
    ax.set_xticks(offsets)
    ax.yaxis.set_major_locator(LogLocator(numticks=6))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    fig.tight_layout()
    _save_fig(fig, 'fig7b_weak_echo')


def plot_exp7c_timing_sweep():
    """Figure 7c: MSE vs residual timing offset (fractional delay lambda)."""
    d = _load_json("exp7_multipath")
    ts = d["timing_sweep"]
    cfg = ts["config"]
    lambdas = cfg["lambda_list"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for est in ['proposed', 'matrix_pencil', 'he2023']:
        mses = [ts["results"][f"lambda_{lam:+.2f}"]["metrics"][est]["mse"]
                for lam in lambdas]
        ax.semilogy(lambdas, mses,
                    color=COLORS[est], marker=MARKERS[est],
                    label=LABELS[est], markersize=6, linewidth=1.6)

    ax.set_xlabel(r'Residual timing offset $\lambda$ (delay-tap units)')
    ax.set_ylabel('MSE')
    ax.set_title(rf'Residual Timing-Offset Sensitivity ($K_r$={cfg["k_db"]:.0f}dB, '
                 rf'SNR={cfg["snr_db"]:.0f}dB, MC={cfg["mc_times"]})')
    ax.legend(loc='upper left')
    ax.yaxis.set_major_locator(LogLocator(numticks=7))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    fig.tight_layout()
    _save_fig(fig, 'fig7c_timing_sweep')


def plot_exp7d_freq_sweep():
    """Figure 7d: MSE vs residual frequency offset (delta_nu)."""
    d = _load_json("exp7_multipath")
    fs = d["freq_sweep"]
    cfg = fs["config"]
    dnus = cfg["delta_nu_list"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for est in ['proposed', 'matrix_pencil', 'he2023']:
        mses = [fs["results"][f"dnu_{dnu:+.2f}"]["metrics"][est]["mse"]
                for dnu in dnus]
        ax.semilogy(dnus, mses,
                    color=COLORS[est], marker=MARKERS[est],
                    label=LABELS[est], markersize=7, linewidth=1.8)

    ax.set_xlabel(r'Residual frequency offset $\delta\nu$')
    ax.set_ylabel('MSE')
    ax.set_title(rf'Residual Frequency-Offset Sensitivity ($K_r$={cfg["k_db"]:.0f}dB, '
                 rf'SNR={cfg["snr_db"]:.0f}dB, MC={cfg["mc_times"]})')
    ax.legend(loc='upper left')
    ax.set_xticks(dnus)
    ax.yaxis.set_major_locator(LogLocator(numticks=7))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    fig.tight_layout()
    _save_fig(fig, 'fig7d_freq_sweep')


def plot_all_exp7():
    print("=" * 60)
    print("  Multipath / Rician figures")
    print(f"  output directory: {FIGURES_DIR}")
    print("=" * 60)

    print("\n[1/4] fig7a: MSE vs Rician K-factor...")
    plot_exp7a_k_sweep()
    print("\n[2/4] fig7b: Weak second-path scenario...")
    plot_exp7b_weak_echo()
    print("\n[3/4] fig7c: Residual timing-offset sensitivity...")
    plot_exp7c_timing_sweep()
    print("\n[4/4] fig7d: Residual frequency-offset sensitivity...")
    plot_exp7d_freq_sweep()

    print("\n" + "=" * 60)
    print("  Experiment 7 figures generated!")
    print("=" * 60)


if __name__ == "__main__":
    plot_all_exp7()
  
