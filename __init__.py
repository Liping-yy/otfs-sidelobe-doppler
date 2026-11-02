# -*- coding: utf-8 -*-
"""
OTFS Fractional Doppler Closed-Form Estimation Research Project
================================================================

Closed-form fractional Doppler estimation based on Sinc sidelobe
phase-sum invariance and joint peak self-multiplication.

Project structure:
  config/       Global unified parameters
  modem/        OTFS modulation and demodulation (ISFFT/SFFT/Doppler/AWGN)
  channel/      Channel model (Single-path LOS)
  frame/        Frame generation (Pure impulse frame / Engineering frame)
  estimator/    Estimators (proposed / sidetone_only / matrix_pencil / prony)
  utils/        Utilities (Dirichlet kernel / metrics / plotting)
  experiments/  Experiment scripts (exp1~exp6)
  results/      Experimental output
"""

__version__ = "0.1.0"
