"""Decomposition and profile-error pass.

(1) chi2 decomposition (SN / BAO / prior) at the best fit of each model family.
(2) 1D s_anchor profile likelihoods for the CMB-dipole hemisphere fits -> 1-sigma
    intervals from the dchi2=1 crossings, so the anisotropy hint gets a real error bar.
"""
import json
import time
from pathlib import Path

import numpy as np

import harness as H

OUT = Path(__file__).parent / "out"
t0 = time.time()

sn, bao, epochs, zmax_table = H.setup_fiducial(use_planck_priors=True)
res_b = H.fit_baseline(sn, bao, epochs, zmax_table)
H0_b, Om_b, alpha_b = [float(v) for v in res_b.x]
chi2_b = float(res_b.fun)
H.set_anchor_from_baseline(sn, H0_b, Om_b, zmax_table, epochs)


def decompose(x, use_lta, profile=None):
    """Evaluate SN/BAO/prior chi2 parts at parameter vector x."""
    if profile is not None:
        H.activate_profile(profile)
    try:
        H0, Om, alpha = float(x[0]), float(x[1]), float(x[2])
        tables = H.lp.build_cosmology_tables(H0=H0, Om=Om, zmax=zmax_table)
        if use_lta:
            lta = H.lp.LTAParams(float(x[4]), float(x[5]), float(x[6]))
            epochs_eff = H.lp.LTAEpochs(t_life_gyr=float(x[3]))
        else:
            lta = H.lp.LTAParams(0.0, 0.0, 0.0)
            epochs_eff = epochs
        params = {"H0": H0, "Om": Om, "alpha_rd": alpha, "s_anchor": float(x[4]) if use_lta else 0.0,
                  "g_complex": 1.0, "g_life": 1.0, "t_life_gyr": float(x[3]) if use_lta else epochs.t_life_gyr}
        c_sn, _ = H.lp.chi2_sn(params, sn, tables, epochs_eff, use_lta=use_lta, lta_override=lta)
        c_bao = H.lp.chi2_bao(params, bao, tables, epochs_eff, use_lta=use_lta, lta_override=lta)
        c_pr = H.lp.EARLY_PRIORS.chi2(H0, Om, alpha) if H.lp.EARLY_PRIORS is not None else 0.0
        return {"sn": float(c_sn), "bao": float(c_bao), "prior": float(c_pr),
                "total": float(c_sn + c_bao + c_pr)}
    finally:
        H.deactivate_profile()


out = {"baseline": decompose(np.array([H0_b, Om_b, alpha_b]), use_lta=False)}

res_a = H.fit_stepA(sn, bao, epochs, zmax_table, H0_b, Om_b, alpha_b)
out["stepA"] = decompose(np.asarray(res_a.x), use_lta=True)

for name, prof in (
    ("offset(compact)", H.CompactSourceProfile(lam_mpc=3000.0, R_reg_mpc=1.0)),
    ("well(Rw200,n2)", H.ExtendedWellProfile(R_w_mpc=200.0, n=2.0)),
    ("void(Rv100,g2)", H.LinearVoidProfile(R_v_mpc=100.0, gamma=2.0)),
):
    res = H.fit_custom(sn, bao, epochs, zmax_table, H0_b, Om_b, alpha_b, prof,
                       A_max=0.05 if "void" not in name else 0.02)
    out[name] = decompose(np.asarray(res.x), use_lta=True, profile=prof)
    out[name]["amplitude"] = float(np.asarray(res.x)[4])

for k, v in out.items():
    print(f"[dec] {k:18s} SN={v['sn']:8.3f}  BAO={v['bao']:7.3f}  prior={v['prior']:7.3f}  tot={v['total']:8.3f}")

# ---------------- hemisphere profile errors ----------------
import pandas as pd

df = pd.read_csv(H.REPO / "Pantheon+SH0ES.dat", sep=r"\s+", comment="#")
key = {}
for j in range(len(df)):
    key[(str(df["CID"].iloc[j]), int(df["IDSURVEY"].iloc[j]), round(float(df["zHD"].iloc[j]), 6))] = j
rows = np.array([key[(str(sn.cid[i]), int(sn.idsurvey[i]), round(float(sn.zHD[i]), 6))]
                 for i in range(len(sn.y))], dtype=int)
ra = np.radians(df["RA"].to_numpy(float)[rows])
dec = np.radians(df["DEC"].to_numpy(float)[rows])
nvec = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
apex = np.array([np.cos(np.radians(-6.94)) * np.cos(np.radians(167.94)),
                 np.cos(np.radians(-6.94)) * np.sin(np.radians(167.94)),
                 np.sin(np.radians(-6.94))])
is_cal = sn.is_calibrator
proj = nvec @ apex

profiles = {}
for side, mask_side in (("toward", proj >= 0), ("away", proj < 0)):
    idx = np.where(mask_side | is_cal)[0]
    sn_sub = H.lp.subset_sn(sn, idx)
    rb = H.lp.fit_baseline(sn_sub, bao, epochs, zmax_table)
    H0s, Oms, als = [float(v) for v in rb.x]
    ra_fit = H.fit_stepA(sn_sub, bao, epochs, zmax_table, H0s, Oms, als)
    xbest = np.asarray(ra_fit.x, dtype=float)
    s_best, chi2_best = float(xbest[4]), float(ra_fit.fun)
    s_grid = np.linspace(0.0, 8.0, 81)
    prof_chi2 = []
    for s in s_grid:
        xx = xbest.copy()
        xx[4] = s
        # re-minimize over (H0, Om, alpha) at fixed s
        from scipy.optimize import minimize
        r = minimize(
            lambda y: H.lp.total_chi2(np.array([y[0], y[1], y[2], 3.8, s, 1.0, 1.0]),
                                      sn_sub, bao, epochs, use_lta=True, zmax_table=zmax_table),
            xbest[:3], method="Powell",
            bounds=[(40, 100), (0.05, 0.6), (0.6, 1.4)], options={"maxiter": 120},
        )
        prof_chi2.append(float(r.fun))
    prof_chi2 = np.asarray(prof_chi2)
    d1 = prof_chi2 - prof_chi2.min()
    inside = s_grid[d1 <= 1.0]
    profiles[side] = {
        "s_best": s_best, "chi2_best": chi2_best,
        "s_lo_1sig": float(inside.min()), "s_hi_1sig": float(inside.max()),
        "s_grid": s_grid.tolist(), "dchi2_profile": d1.tolist(),
    }
    print(f"[dec] dipole {side:>6}: s = {s_best:.2f}  [{inside.min():.2f}, {inside.max():.2f}] (1sig)")

s_t, s_a = profiles["toward"]["s_best"], profiles["away"]["s_best"]
sig_t = 0.5 * (profiles["toward"]["s_hi_1sig"] - profiles["toward"]["s_lo_1sig"])
sig_a = 0.5 * (profiles["away"]["s_hi_1sig"] - profiles["away"]["s_lo_1sig"])
signif = abs(s_t - s_a) / np.hypot(sig_t, sig_a)
print(f"[dec] dipole anisotropy: ds = {s_t - s_a:.2f} +- {np.hypot(sig_t, sig_a):.2f}  ({signif:.2f} sigma)")

out["dipole_profiles"] = profiles
out["dipole_anisotropy_sigma"] = float(signif)
out["runtime_s"] = time.time() - t0
(OUT / "x_decompose.json").write_text(json.dumps(out, indent=2))
print(f"[dec] wrote out/x_decompose.json ({time.time()-t0:.0f}s)")
