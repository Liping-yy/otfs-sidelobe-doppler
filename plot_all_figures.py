# -*- coding: utf-8 -*-
"""
Plotting script v2: Generate all experiment figures（zero-paddedframeversion）
=============================================

from results/ lower JSON filereaddata，generatepaperusefigure:

  figure2: Experiment 2 MSE-SNRCurves (2subplot: pure + engineering_zp)
  figure3: Experiment 3 Full-rangenu_f MSESweep (Unchanged, still using pure frame)
  figure4: Experiment 4 pilot powerAsensitivity (MSE vs A, 3estimator)
  figure5: Experiment 5 BER-SNRCurves (5Curves)
  figure6: Experiment 6 complexitycomparison (bar chart)

outputformat: PDF (vector) + PNG (preview)
save location: project/figures/

"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# global style
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

# colors and markers
COLORS = {
    'proposed': '#e41a1c',
    'sidetone_only': '#377eb8',
    'matrix_pencil': '#4daf4a',
    'he2023': '#984ea3',
    'ideal': '#777777',
    'no_comp': '#ff7f00',
}
MARKERS = {
    'proposed': 'o',
    'sidetone_only': 's',
    'matrix_pencil': 'D',
    'he2023': '^',
    'ideal': '-',
    'no_comp': 'v',
}
LABELS = {
    'proposed': 'Proposed',
    'sidetone_only': 'Sidetone Only',
    'matrix_pencil': 'Matrix Pencil',
    'he2023': 'He 2023',
    'ideal': 'Ideal Sync',
    'no_comp': 'No Compensation',
}


def _save_fig(fig, name):
    """save as PDF + PNG"""
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


# ============================================================
# figure2: Experiment 2 MSE-SNRCurves (pure + engineering_zp)
# ============================================================

def plot_exp2_mse_snr():
    """Experiment 2(v2): MSE-SNRCurves (pure + engineering_zp two subplots)"""
    d = _load_json("exp2_mse_snr")
    cfg = d['config']
    snr_list = cfg['snr_list']
    est_names = ['proposed', 'he2023', 'matrix_pencil', 'sidetone_only']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    frame_labels = {
        'pure': ('Pure Pulse Frame', 'Pure Frame'),
        'engineering_zp': (r'Engineering Frame (Zero-Padded, $A=3.0$)', 'Engineering Frame'),
    }

    for ax_idx, frame_type in enumerate(['pure', 'engineering_zp']):
        ax = axes[ax_idx]
        frame_data = d['results'][frame_type]

        for est in est_names:
            mses = []
            for snr in snr_list:
                key = f"snr_{int(snr)}dB"
                mses.append(frame_data[key]['metrics'][est]['mse'])

            ax.semilogy(snr_list, mses,
                        color=COLORS[est], marker=MARKERS[est],
                        label=LABELS[est], markersize=5, linewidth=1.5)

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('MSE')
        ax.set_title(frame_labels[frame_type][0])
        ax.legend(loc='lower left')
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(LogLocator(numticks=6))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.set_ylim(bottom=1e-7)

    fig.suptitle(r'MSE vs SNR ($\nu_f=0.3$, MC=2000)', fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, 'fig2_mse_snr')


# ============================================================
# figure3: Experiment 3 Full-rangenu_f MSESweep (unchanged)
# ============================================================

def plot_exp3_nu_sweep():
    """Experiment 3: Full-rangenu_f MSESweep (Proposed vs onlysidelobe)"""
    d = _load_json("exp3_full_nu_sweep")
    cfg = d['config']

    results = d['results']
    nu_vals = sorted([float(k) for k in results.keys()])

    est_names = ['proposed', 'sidetone_only']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [2, 1]})

    # left figure: Full-range (-0.5, 0.5)
    ax_full = axes[0]
    for est in est_names:
        mses = [results[f"{nu:.4f}"]['metrics'][est]['mse'] for nu in nu_vals]
        ax_full.semilogy(nu_vals, mses,
                         color=COLORS[est], marker=MARKERS[est],
                         label=LABELS[est], markersize=3, linewidth=1.2)

    ax_full.set_xlabel(r'$\nu_f$')
    ax_full.set_ylabel('MSE')
    ax_full.set_title(rf'Full Range ($\nu_f \in (-0.5, 0.5)$, SNR={cfg["snr_db"]}dB)')
    ax_full.legend(loc='upper center')
    ax_full.set_ylim(1e-7, 1e0)

    # right figure: zoom-in (-0.1, 0.1)
    ax_zoom = axes[1]
    zoom_mask = [nu for nu in nu_vals if -0.1 <= nu <= 0.1]
    for est in est_names:
        mses_zoom = [results[f"{nu:.4f}"]['metrics'][est]['mse'] for nu in zoom_mask]
        ax_zoom.semilogy(zoom_mask, mses_zoom,
                         color=COLORS[est], marker=MARKERS[est],
                         label=LABELS[est], markersize=4, linewidth=1.2)

    ax_zoom.set_xlabel(r'$\nu_f$')
    ax_zoom.set_ylabel('MSE')
    ax_zoom.set_title(r'Zoom: $\nu_f \in (-0.1, 0.1)$')
    ax_zoom.legend(loc='upper center')
    ax_zoom.set_ylim(1e-7, 1e0)

    fig.suptitle(r'Singular Point Elimination (Pure Frame, MC=1000)', fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, 'fig3_nu_sweep')


# ============================================================
# figure4: Experiment 4 pilot powerAsensitivity (MSE vs A line plot)
# ============================================================

def plot_exp4_a_sensitivity():
    """Experiment 4(v2): pilot powerAsensitivity (MSE vs A, 3estimator)"""
    d = _load_json("exp4_a_sensitivity")
    meta = d['config']
    a_list = meta['a_list']
    est_names = ['proposed', 'matrix_pencil', 'he2023']

    fig, ax = plt.subplots(figsize=(7, 5))

    for est in est_names:
        mses = []
        for a in a_list:
            mses.append(d['results_by_a'][f"{a:.1f}"][est]['mse'])

        ax.semilogy(a_list, mses,
                    color=COLORS[est], marker=MARKERS[est],
                    label=LABELS[est], markersize=7, linewidth=1.8)

    ax.set_xlabel('Pilot Amplitude $A$')
    ax.set_ylabel('MSE')
    ax.set_title(rf'Pilot Power Sensitivity (SNR={meta["snr_db"]}dB, $\nu_f$={meta["nu_f"]}, MC={meta["mc_times"]})')
    ax.legend(loc='upper right')
    ax.set_xticks(a_list)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(LogLocator(numticks=6))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    fig.tight_layout()
    _save_fig(fig, 'fig4_a_sensitivity')


# ============================================================
# figure5: Experiment 5 BER-SNRCurves
# ============================================================

def plot_exp5_ber_snr():
    """Experiment 5(v2): BER-SNRCurves (5Curves)"""
    d = _load_json("exp5_ber_snr")
    meta = d['meta']
    snr_list = meta['snr_list']
    ber_data = d['ber']

    curve_names = ['ideal', 'proposed', 'he2023', 'matrix_pencil', 'no_comp']

    fig, ax = plt.subplots(figsize=(8, 5))

    for name in curve_names:
        bers = []
        for snr in snr_list:
            key = f"{float(snr)}"
            if key in ber_data[name]:
                bers.append(ber_data[name][key]['ber'])
            else:
                bers.append(0.0)

        bers_plot = [max(b, 1e-7) for b in bers]

        if name == 'ideal':
            ax.semilogy(snr_list, bers_plot,
                        color=COLORS[name], linestyle='--',
                        label=LABELS[name], linewidth=2)
        elif name == 'no_comp':
            ax.semilogy(snr_list, bers_plot,
                        color=COLORS[name], marker=MARKERS[name],
                        label=LABELS[name], markersize=5, linewidth=1.5)
        else:
            ax.semilogy(snr_list, bers_plot,
                        color=COLORS[name], marker=MARKERS[name],
                        label=LABELS[name], markersize=5, linewidth=1.5)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER')
    ax.set_title(rf'Link BER Performance ($\nu_f=0.3$, Zero-Padded Frame, MC={meta["mc_times"]})')
    ax.legend(loc='upper right')
    ax.set_ylim(1e-5, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(LogLocator(numticks=7))

    fig.tight_layout()
    _save_fig(fig, 'fig5_ber_snr')


# ============================================================
# figure6: Experiment 6 complexitycomparison (bar chart)
# ============================================================

def plot_exp6_complexity():
    """Experiment 6(v2): complexitycomparisonbar chart"""
    d = _load_json("exp6_complexity")
    timing = d['timing']

    names = ['proposed', 'sidetone_only', 'matrix_pencil', 'he2023']
    means = [timing[n]['mean_us'] for n in names]
    stds = [timing[n]['std_us'] for n in names]
    labels = [LABELS[n] for n in names]
    colors = [COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(range(len(names)), means, yerr=stds,
                  color=colors, edgecolor='black', linewidth=0.5,
                  capsize=4, width=0.6)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Average Time ($\mu$s)')
    ax.set_title(r'Computational Complexity (Zero-Padded Frame, SNR=10dB)')
    ax.set_yscale('log')

    # annotate valuesand ratio
    prop_mean = means[0]
    for i, (bar, mean_val) in enumerate(zip(bars, means)):
        ratio = mean_val / prop_mean
        if i == 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                    f'{mean_val:.1f}$\mu$s',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                    f'{mean_val:.1f}$\mu$s\n({ratio:.1f}x)',
                    ha='center', va='bottom', fontsize=9)

    ax.set_ylim(bottom=1e1)  # minimum10us
    fig.tight_layout()
    _save_fig(fig, 'fig6_complexity')


# ============================================================
# main function
# ============================================================

def plot_all():
    """generateallfigure table"""
    print("=" * 60)
    print("  Generate all experiment figures [v2 - Zero-Padded Frame]")
    print(f"  output directory: {FIGURES_DIR}")
    print("=" * 60)

    print("\n[1/5] figure2: MSE-SNRCurves...")
    plot_exp2_mse_snr()

    print("\n[2/5] figure3: Full-rangenu_fSweep...")
    plot_exp3_nu_sweep()

    print("\n[3/5] figure4: pilotpowerAsensitivity...")
    plot_exp4_a_sensitivity()

    print("\n[4/5] figure5: BER-SNRCurves...")
    plot_exp5_ber_snr()

    print("\n[5/5] figure6: complexitycomparison...")
    plot_exp6_complexity()

    print("\n" + "=" * 60)
    print("  allfigure generationdone!")
    print("=" * 60)


if __name__ == "__main__":
    plot_all()

