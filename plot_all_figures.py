# -*- coding: utf-8 -*-
"""
Plotting script: Generate all experiment figures
================================

Read data from JSON files under results/, generate paper figures:

  figure2: Experiment 2 MSE-SNRCurves (2subplot: pure + engineering)
  figure3: Experiment 3 Full-rangenu_f MSESweep (Proposed vs onlysidelobe, includingzoom-in)
  figure4: Experiment 4 Kmax x A heatmap (3subplot: Proposed/MP/He2023)
  figure5: Experiment 5 BER-SNRCurves (5Curves)

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

# project root (plot_all_figures.py located at project/ lower)
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
    'proposed': '#e41a1c',       # red
    'sidetone_only': '#377eb8',   # blue
    'matrix_pencil': '#4daf4a',   # green
    'he2023': '#984ea3',          # purple
    'ideal': '#777777',           # gray
    'no_comp': '#ff7f00',         # orange
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
# figure2: Experiment 2 MSE-SNRCurves
# ============================================================

def plot_exp2_mse_snr():
    """Experiment 2: MSE-SNRCurves (pure + engineering two subplots)"""
    d = _load_json("exp2_mse_snr")
    cfg = d['config']
    snr_list = cfg['snr_list']
    est_names = ['proposed', 'he2023', 'matrix_pencil', 'sidetone_only']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax_idx, frame_type in enumerate(['pure', 'engineering']):
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
        ax.set_title(f'{"Pure Frame" if frame_type=="pure" else "Engineering Frame"} (Kmax={cfg["Kmax"]}, A={cfg["A"]})')
        ax.legend(loc='lower left')
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(LogLocator(numticks=6))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.set_ylim(bottom=1e-7)

    fig.suptitle(r'MSE vs SNR ($\nu_f=0.3$, MC=2000)', fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, 'fig2_mse_snr')


# ============================================================
# figure3: Experiment 3 Full-rangenu_f MSESweep
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
# figure4: Experiment 4 Kmax×Aheatmap
# ============================================================

def plot_exp4_heatmap():
    """Experiment 4: Kmax×Aheatmap (3subplot)"""
    d = _load_json("exp4_kmax_a_heatmap")
    meta = d['meta']
    kmax_list = meta['kmax_list']
    a_list = meta['a_list']
    est_names = ['proposed', 'matrix_pencil', 'he2023']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # find globalvmin/vmax（logarithmic scale）
    all_mses = []
    for est in est_names:
        for kmax in kmax_list:
            for a in a_list:
                val = d['heatmap'][est][str(kmax)][f"{a:.1f}"]['mse']
                all_mses.append(val)
    all_mses = np.array(all_mses)
    vmin = max(all_mses.min(), 1e-7)
    vmax = all_mses.max()

    for ax_idx, est in enumerate(est_names):
        ax = axes[ax_idx]
        data = np.zeros((len(kmax_list), len(a_list)))
        for i, kmax in enumerate(kmax_list):
            for j, a in enumerate(a_list):
                data[i, j] = d['heatmap'][est][str(kmax)][f"{a:.1f}"]['mse']

        im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r',
                       norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                       origin='lower')

        ax.set_xticks(range(len(a_list)))
        ax.set_xticklabels([f'{a}' for a in a_list])
        ax.set_yticks(range(len(kmax_list)))
        ax.set_yticklabels([f'{kmax}' for kmax in kmax_list])
        ax.set_xlabel('Pilot Amplitude A')
        ax.set_ylabel('Guard-Zone Length $K_{max}$')
        ax.set_title(LABELS[est])

        # annotate values
        for i in range(len(kmax_list)):
            for j in range(len(a_list)):
                val = data[i, j]
                color = 'white' if val > vmax * 0.1 else 'black'
                ax.text(j, i, f'{val:.1e}', ha='center', va='center',
                        fontsize=8, color=color)

        # mark guard zero zero constraintline Kmax_min = 1/(rho*A)
        rho = 0.1
        for j, a in enumerate(a_list):
            kmin = 1.0 / (rho * a)
            # findkmin falls atwhichtwokmaxofbetween
            for i, kmax in enumerate(kmax_list):
                if kmax < kmin:
                    # at thatgrid pointupperequationdraw horizontal pathmark guard zero zero insufficient
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                               fill=False, edgecolor='blue',
                                               linewidth=1.5, linestyle='--'))

    # shared colorbar (placeatbelow the figure，tight layout)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        orientation='horizontal',
                        fraction=0.035, pad=0.04, aspect=35,
                        label='MSE')

    fig.suptitle(rf'Engineering Sensitivity (SNR={meta["snr_db"]}dB, $\nu_f$={meta["nu_f"]}, MC={meta["mc_times"]})'
                 + '\n(blue dashed: guard zone insufficient)',
                 fontsize=12, y=1.01)
    _save_fig(fig, 'fig4_kmax_a_heatmap')


# ============================================================
# figure5: Experiment 5 BER-SNRCurves
# ============================================================

def plot_exp5_ber_snr():
    """Experiment 5: BER-SNRCurves (5Curves)"""
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

        # 0valueirrationalmethodin logfiguredisplayed, usetiny valuealternatives
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
    ax.set_title(rf'Link BER Performance ($\nu_f=0.3$, Engineering Frame, MC={meta["mc_times"]})')
    ax.legend(loc='upper right')
    ax.set_ylim(1e-5, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(LogLocator(numticks=7))

    fig.tight_layout()
    _save_fig(fig, 'fig5_ber_snr')


# ============================================================
# main function
# ============================================================

def plot_all():
    """generateallfigure table"""
    print("=" * 60)
    print("  Generate all experiment figures")
    print(f"  output directory: {FIGURES_DIR}")
    print("=" * 60)

    print("\n[1/4] figure2: MSE-SNRCurves...")
    plot_exp2_mse_snr()

    print("\n[2/4] figure3: Full-rangenu_fSweep...")
    plot_exp3_nu_sweep()

    print("\n[3/4] figure4: Kmax×Aheatmap...")
    plot_exp4_heatmap()

    print("\n[4/4] figure5: BER-SNRCurves...")
    plot_exp5_ber_snr()

    print("\n" + "=" * 60)
    print("  allfigure generationdone!")
    print("=" * 60)


if __name__ == "__main__":
    plot_all()
