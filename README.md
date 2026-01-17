# Local Time Acceleration (LTA): Phenomenological Cosmology & The Hubble Tension

This repository contains the codebase and analysis pipeline for the paper **"Local Time Acceleration: A Phenomenological Solution to the Hubble Tension Driven by Kinematic Anomalies in the Local Void"**.

The project investigates a phenomenological extension to ΛCDM where the observed redshift includes a small, observer-anchored frequency drift accumulated along the line of sight. Using the Pantheon+SH0ES supernova dataset and consensus BAO constraints, this code performs a joint likelihood analysis with full covariance transmission to test for the presence of a "Time Viscosity" or local kinematic anomaly.

## Repository Contents

* `lta_power.py`: The main analysis script. Handles cosmology calculation, LTA redshift mapping, likelihood evaluation (SN anchored + BAO), optimization, and robustness testing (Jackknife/CV/Nulls).
* `idsurvey_to_telescope.csv`: Mapping file used for the telescope-grouped Jackknife (leave-one-out) robustness tests.
* `Pantheon+SH0ES.dat` / `Pantheon+SH0ES_STAT+SYS.cov`: The official Pantheon+ dataset files.
* `BAO_consensus_results_dM_Hz.txt` / `BAO_consensus_covtot_dM_Hz.txt`: (User must provide) The consensus BAO data files.
* `planck_chains/`: Directory for Planck 2018 MCMC chains if using correlated priors.

## Key Features

1.  **Phenomenological LTA Model:** Implements a bounded, Earth-history-modulated frequency drift $s(t_{ret})$ that modifies the observed redshift $z_{obs}$ relative to the cosmological $z_{cos}$.
2.  **Rigorous SN Likelihood:** Uses a ladder-anchored likelihood that conditions the Hubble-flow SNe on the calibrator subset using the full Schur complement of the covariance matrix.
3.  **Robustness Suite:** Includes built-in routines for:
    * **Jackknife Resampling:** Leave-one-group-out tests by telescope/survey to rule out calibration systematics.
    * **Parametric Bootstrap (Null Injections):** Monte Carlo calibration of the $\Delta\chi^2$ significance against 10,000+ $\Lambda$CDM mocks.
    * **Conditional Cross-Validation:** Tests predictive power on held-out folds accounting for long-range covariance correlations.

## Getting Started

### Prerequisites

* Python 3.8+
* `numpy`
* `scipy`
* `pandas`
* `matplotlib`

### Reproduction Command

To reproduce the primary results presented in the paper (including the global fit, Jackknife robustness table, and high-precision null calibration), run the following command:

```bash
python lta_power.py \
  --use-planck-priors \
  --sn-anchor-m-to-calibrators \
  --bao-rd-fid 147.78 \
  --do-loo \
  --loo-map idsurvey_to_telescope.csv \
  --do-cv \
  --cv-block survey \
  --cv-seed 100 \
  --do-cv-null \
  --cv-null-n 50 \
  --cv-null-seed 100 \
  --fix-g-complex 1.0 \
  --fix-g-life 1.0 \
  --do-null \
  --null-n 10000 \
  --cv-score both \
  --outdir ./output_loo_cv_10000
  ```
  
