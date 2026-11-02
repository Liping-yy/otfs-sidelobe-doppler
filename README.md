# OTFS Sidelobe-Phase Closed-Form Doppler Estimator

Simulation code for the paper:

> **A Sidelobe-Phase-Exploiting Closed-Form Doppler Estimator for LEO OTFS Systems**
> *Physical Communication* (Elsevier), manuscript ID `PHYCOM-D-26-02857`

This repository reproduces **all figures and numerical results** reported in the paper.

---

## Overview

For LEO satellite OTFS links with a dominant line-of-sight path, the fractional Doppler
shift spreads the pilot energy across the whole Doppler row through a Dirichlet kernel.
This code implements a **closed-form estimator** that exploits the *phase-sum invariance*
of the symmetric sidelobe pairs, requiring only `O(K)` operations — versus `O(N³)` for
matrix-pencil (MP) subspace methods and `O(G·MN log MN)` for grid-search methods.

**Key ideas implemented here**

| Concept | Where |
|---|---|
| Dirichlet kernel `D_N(x) = sin(πx)/sin(πx/N)` and its zero structure | `utils/dirichlet.py` |
| ISFFT / SFFT round trip with Doppler phase rotation | `modem/otfs_modem.py` |
| Peak self-multiplication + symmetric sidelobe-pair complex sum | `estimator/proposed.py` |
| Implicit amplitude weighting (complex sum ≡ magnitude-weighted phase fusion) | `estimator/proposed.py` |
| Single-path LOS and finite-`K_r` Rician channels | `channel/` |
| Pure-pilot and zero-padded engineering frames | `frame/frame_gen.py` |
| Baselines: MP, Prony, He 2023, sidelobe-only ablation | `estimator/` |

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ with NumPy, SciPy, and Matplotlib.

---

## Nominal configuration

All parameters are centralised in `config/params.py` (no scattered hard-coded values):

| Parameter | Value | Meaning |
|---|---|---|
| `M` | 64 | Delay dimension (subcarriers) |
| `N` | 16 | Doppler dimension (time slots) |
| `Δf` | 120 kHz | Subcarrier spacing |
| `fc` | 28 GHz | Carrier (Ka-band) |
| altitude | 600 km | LEO satellite altitude |
| `A` | 3.0 | Pilot amplitude boosting (`A² = 9`) |
| `l_p`, `k_p` | 5, 8 | Pilot delay / Doppler indices |
| `K` | ⌊(N−1)/2⌋ = 7 | Sidelobe-pair count (complete symmetric set) |

Data symbols are unit-energy **QPSK**; the pilot row is zero-padded outside `k = k_p`.

---

## Running the experiments

Scripts are in `experiments/`. Each writes its output to `results/*.json`, and the
plotting scripts in the repository root render `figures/*.pdf` and `figures/*.png`.

| Script | Paper experiment | Produces |
|---|---|---|
| `exp1_theory_verify.py` | Exp. 1 — verification of the four key identities | Table 2 |
| `exp3_full_nu_sweep.py` | Exp. 1 — full `ν_f` sweep and zoom near 0 | **Fig. 1** |
| `exp2_mse_snr.py` / `exp2_mse_snr_v2.py` | MSE vs SNR, both frames | Fig. 1 / Table |
| `exp8_matched_mp.py` | Exp. 2 — matched-sample MP comparison | **Fig. 2** |
| `exp5_ber_snr.py` / `exp5_ber_snr_v2.py` | Exp. 4 — BER vs SNR | **Fig. 3** |
| `exp7_multipath_v2.py` | Exp. 6 — Rician `K_r` sweep, weak echo, timing/freq offsets | **Figs. 4–6** |
| `exp4_a_sensitivity_v2.py`, `exp4_kmax_a_heatmap.py` | Pilot-power sensitivity | Tables |
| `exp6_complexity.py` / `exp6_complexity_v2.py` | Exp. 5 — FLOP counts and runtime | Tables 5–6 |

Plot all figures:

```bash
python plot_exp7_figures.py
python plot_exp8_figures.py
python plot_all_figures_v2.py
```

Example — single experiment:

```bash
python -m experiments.exp8_matched_mp
```

---

## Repository layout

```
config/        Global simulation parameters (single source of truth)
modem/         OTFS modulation: ISFFT, SFFT, Doppler rotation, AWGN
channel/       Single-path LOS channel; finite-K_r Rician multipath model
frame/         Pure-impulse frame and zero-padded engineering frame
estimator/     proposed (closed form), sidetone_only, matrix_pencil, prony, he2023
utils/         Dirichlet kernel, metrics, plotting helpers
experiments/   exp1 … exp8 (one script per paper experiment)
results/       JSON outputs of the experiments
figures/       Rendered figures (PDF + PNG)
tests/         Unit tests
```

---

## Method summary

The received pilot row is modelled as

```
y[k] = (h₀A/N) · D_N(ν_f + k_p − k) · exp(jπ(ν_f + k_p − k)(N−1)/N) + w[k]
```

The estimator forms the single complex sum

```
S = y[k_p]² + Σ_{Δk=1}^{K} Z(Δk),      Z(Δk) = −y[k_p+Δk]·y[k_p−Δk]
ψ̄ = arg{S},      ν̂_f = ψ̄ · N / (2π(N−1))
```

Because `Δk` terms cancel in the phase, **every** symmetric pair contributes the same
phase `2πν_f(N−1)/N`, so the sum is phase-coherent while the magnitudes automatically
provide amplitude weighting (`|Z(Δk)|`). The peak self-multiplication removes the
singularity at `ν_f = 0` that would otherwise break the sidelobe-only variant.

---

## Notes on reproducibility

- Random seeds are fixed in `config/params.py`; Monte-Carlo counts are recorded in each
  experiment script and in the JSON outputs under `results/`.
- The engineering frame includes QPSK data on all non-pilot delay rows. Because the
  pilot row is zero-padded and cross-row interference vanishes *exactly* under the
  single-path model (Lemma 1 in the paper), the noiseless pilot row of the engineering
  frame is identical to that of the pure frame; only the noise scaling differs
  (the global-SNR convention, a 20.5 dB effective-SNR shift).
- FLOP counts follow the convention declared in the paper (complex multiply = 6,
  complex add = 2, real multiply/add = 1, `arg`/atan2 = 1 equivalent).

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{otfs_sidelobe_doppler_2026,
  title   = {A Sidelobe-Phase-Exploiting Closed-Form Doppler Estimator for {LEO} {OTFS} Systems},
  author  = {<authors>},
  journal = {Physical Communication},
  year    = {2026},
  note    = {Manuscript ID PHYCOM-D-26-02857}
}
```

---

## License

Released for academic reproducibility. Please contact the authors before commercial use.
