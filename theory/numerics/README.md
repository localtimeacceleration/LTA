# OTA derivation-program numerics

Runs the paper's own likelihood machinery (`../../lta_power.py`, imported as a library — **no
edits to it or to any data file**) against the in-repo Pantheon+SH0ES + consensus-BAO data, in
the fiducial configuration of the README reproduction command (run-tag `20260112-013808`).

## Layout

| File | Purpose | Output |
|---|---|---|
| `harness.py` | fiducial setup, fits, and the I(χ)-profile injection layer (runtime rebinding of `lta_integral_I` / `lta_local_s`; SN likelihood, BAO Jacobian and inverse mapping all see the injected profile consistently) | — |
| `run_t1_reproduce.py` | validation: reproduce baseline & step-A fiducial fits; injection-layer control | `out/t1_reproduce.json` |
| `run_d4_kernel_scan.py` | D4: compact-source (physical dilution) and extended-well kernel scans | `out/d4_kernel_scan.json` |
| `run_d8_void_headtohead.py` | D8: linear-theory void head-to-head; hemispheric sky splits | `out/d8_void_sky.json` |
| `run_d6_host.py` | D6: host-mass split, residual–mass correlations, timing-channel numbers | `out/d6_host_timing.json` |
| `run_x_decompose.py` | χ² decomposition (SN/BAO/prior) per model; dipole-hemisphere profile likelihoods | `out/x_decompose.json` |
| `out/t1b_prior_sensitivity.json` | step-A Δχ² under the `omega_m_alpha` prior space (no direct H0 pull): Δχ² = 5.5 | — |
| `out/d4_d8_summary.png` | fitted I(χ) profiles + model-family Δχ² comparison | — |

## Validation status

`run_t1_reproduce.py` reproduces the paper's fiducial numbers to optimizer tolerance
(baseline χ² 332.44 vs 332.549; Δχ² 13.59 vs 13.652; s_anchor 2.445 vs 2.4493; t_anchor 0.670
vs 0.670), and the injected step-A-shape control matches the native pipeline at ΔΔχ² ≈ 0.01.

## Reproducing

```bash
pip install numpy scipy pandas matplotlib
cd theory/numerics
python3 run_t1_reproduce.py      # ~20 s
python3 run_d4_kernel_scan.py    # ~1 min
python3 run_d8_void_headtohead.py
python3 run_d6_host.py
python3 run_x_decompose.py       # ~15 min (profile likelihoods)
```

Requires the repo's data files and `planck_chains/` in place (all already in-repo).
