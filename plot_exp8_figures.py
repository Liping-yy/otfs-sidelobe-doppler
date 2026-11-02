# -*- coding: utf-8 -*-
"""
Plotting script for the matched-sample MP comparison experiment
==========================================================================

Generates:
  fig8_matched_mp.pdf  MSE vs SNR, two subplots (pure / engineering_zp),
                       curves: Proposed, MP-full, MP-matched, He 2023.

Output: PDF + PNG in project_files/figures/
"""

import os
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
    'matrix_pencil_matched': '#ff7f00',
    'he2023': '#984ea3',
}
MARKERS = {
    'proposed': 'o',
    'sidetone_only': 's',
    'matrix_pencil': 'D',
    'matrix_pencil_matched': 'v',
    'he2023': '^',
}
LABELS = {
    'proposed': 'Proposed',
    'sidetone_only': 'Sidetone Only',
    'matrix_pencil': 'MP (full row)',
    'matrix_pencil_matched': 'MP (matched samples)',
    'he2023': 'He 2023',
}


def _save_fig(fig, name):
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.png"))
    plt.close(fig)
    print(f"  saved: {os.path.join(FIGURES_DIR, name)}.pdf / .png")


def plot_exp8_matched_mp():
    with open(os.path.join(RESULTS_DIR, "exp8_matched_mp.json"), encoding="utf-8") as f:
        d = json.load(f)
    cfg = d["config"]
    snr_list = cfg["snr_list"]
    est_names = cfg["est_names"]

    frame_titles = {
        "pure": "Pure Frame",
        "engineering_zp": "Engineering Frame (Zero-Padded, $A=3.0$)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax_idx, frame_type in enumerate(["pure", "engineering_zp"]):
        ax = axes[ax_idx]
        frame_data = d["results"][frame_type]
        for est in est_names:
            mses = [frame_data[f"snr_{int(snr)}dB"]["metrics"][est]["mse"]
                    for snr in snr_list]
            ax.semilogy(snr_list, mses,
                        color=COLORS[est], marker=MARKERS[est],
                        label=LABELS[est], markersize=5, linewidth=1.5)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('MSE')
        ax.set_title(frame_titles[frame_type])
        ax.legend(loc='lower left', fontsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(LogLocator(numticks=6))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.set_ylim(bottom=1e-7)

    fig.suptitle(r'Matched-Sample MP Comparison ($\nu_f=0.3$, MC=2000): '
                 r'MP uses the same 15-sample window as Proposed',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "fig8_matched_mp")


if __name__ == "__main__":
    plot_exp8_matched_mp()
  
