"""D8 numerics: mundane-alternative head-to-head and sky-split diagnostic.

(1) Linear-theory outflow from a centered underdensity (compensated profile),
    fit with the same likelihood, covariance, and dof count as step A.
    The fitted amplitude maps to the implied cumulative contrast |delta_c|.
(2) Hemispheric sky splits (celestial N/S control; CMB dipole apex axis):
    baseline and step-A refits per hemisphere (calibrators kept in all subsets).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import harness as H

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

t0 = time.time()
sn, bao, epochs, zmax_table = H.setup_fiducial(use_planck_priors=True)
res_b = H.fit_baseline(sn, bao, epochs, zmax_table)
H0_b, Om_b, alpha_b = [float(v) for v in res_b.x]
chi2_b = float(res_b.fun)
H.set_anchor_from_baseline(sn, H0_b, Om_b, zmax_table, epochs)
res_a = H.fit_stepA(sn, bao, epochs, zmax_table, H0_b, Om_b, alpha_b)
bench = H.describe_fit("stepA", res_a, chi2_b)
print(f"[d8] baseline chi2={chi2_b:.3f}  stepA dchi2={bench['dchi2_vs_baseline']:.3f}")

results = {"baseline_chi2": chi2_b, "stepA": bench, "void": [], "sky_splits": {}}

# ---------------- void head-to-head ----------------
tables_b = H.lp.build_cosmology_tables(H0=H0_b, Om=Om_b, zmax=zmax_table)
for gam in (2.0, 3.0):
    for Rv in (30.0, 50.0, 70.0, 100.0, 150.0, 200.0):
        prof = H.LinearVoidProfile(R_v_mpc=Rv, gamma=gam)
        res = H.fit_custom(sn, bao, epochs, zmax_table, H0_b, Om_b, alpha_b, prof,
                           A_max=0.02, A0=1e-3)
        d = H.describe_fit(f"void Rv={Rv} gamma={gam}", res, chi2_b)
        # implied contrast: I = |delta_c| * I_raw;  fitted I = A * I_raw / norm
        prof.grids(tables_b, epochs)  # refresh last_norm_v at baseline tables
        delta_c = d["amplitude"] / prof.last_norm_v if prof.last_norm_v > 0 else np.nan
        d.update(R_v_mpc=Rv, gamma=gam, implied_abs_delta_c=float(delta_c))
        results["void"].append(d)
        print(f"[d8] void Rv={Rv:6.1f} gam={gam}: dchi2={d['dchi2_vs_baseline']:8.3f}  "
              f"A={d['amplitude']:.3e}  |delta_c|={delta_c:.3f}")

best_v = max(results["void"], key=lambda d: d["dchi2_vs_baseline"])
print(f"[d8] best void: {best_v['tag']} dchi2={best_v['dchi2_vs_baseline']:.3f} "
      f"|delta_c|={best_v['implied_abs_delta_c']:.3f}")

# ---------------- sky splits ----------------
df = pd.read_csv(H.REPO / "Pantheon+SH0ES.dat", sep=r"\s+", comment="#")


def match_columns(colnames):
    """Map SNData rows -> raw-file rows via (CID, IDSURVEY, zHD)."""
    key = {}
    for j in range(len(df)):
        key[(str(df["CID"].iloc[j]), int(df["IDSURVEY"].iloc[j]),
             round(float(df["zHD"].iloc[j]), 6))] = j
    rows = []
    for i in range(len(sn.y)):
        rows.append(key[(str(sn.cid[i]), int(sn.idsurvey[i]), round(float(sn.zHD[i]), 6))])
    rows = np.asarray(rows, dtype=int)
    return {c: df[c].to_numpy()[rows] for c in colnames}


cols = match_columns(["RA", "DEC", "HOST_LOGMASS"])
ra = np.radians(np.asarray(cols["RA"], dtype=float))
dec = np.radians(np.asarray(cols["DEC"], dtype=float))
nvec = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])

axes = {
    "celestial_N": np.array([0.0, 0.0, 1.0]),
    # CMB dipole apex, equatorial (RA=167.94 deg, Dec=-6.94 deg)
    "cmb_dipole": np.array([
        np.cos(np.radians(-6.94)) * np.cos(np.radians(167.94)),
        np.cos(np.radians(-6.94)) * np.sin(np.radians(167.94)),
        np.sin(np.radians(-6.94)),
    ]),
}

is_cal = sn.is_calibrator if sn.is_calibrator is not None else np.zeros(len(sn.y), bool)

for axis_name, ax in axes.items():
    proj = nvec @ ax
    for side, mask_side in (("toward", proj >= 0), ("away", proj < 0)):
        idx = np.where(mask_side | is_cal)[0]
        n_hf = int(np.sum(mask_side & ~is_cal))
        sn_sub = H.lp.subset_sn(sn, idx)
        rb = H.lp.fit_baseline(sn_sub, bao, epochs, zmax_table)
        H0s, Oms, als = [float(v) for v in rb.x]
        ra_fit = H.fit_stepA(sn_sub, bao, epochs, zmax_table, H0s, Oms, als)
        d = H.describe_fit(f"{axis_name}:{side}", ra_fit, float(rb.fun))
        d.update(n_hf=n_hf, n_total=int(idx.size))
        results["sky_splits"][f"{axis_name}:{side}"] = d
        print(f"[d8] split {axis_name}:{side:>6}  N_HF={n_hf:3d}  "
              f"dchi2={d['dchi2_vs_baseline']:7.3f}  s_anchor={d['amplitude']:.3f}")

results["runtime_s"] = time.time() - t0
(OUT / "d8_void_sky.json").write_text(json.dumps(results, indent=2))
print(f"[d8] wrote out/d8_void_sky.json ({time.time()-t0:.0f}s)")
