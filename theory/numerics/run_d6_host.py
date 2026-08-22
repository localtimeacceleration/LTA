"""D6 numerics: host-environment endpoint test + timing-channel amplitudes.

(1) Host-mass split: refit baseline and step A on (HF & logM>=10) vs (HF & 0<logM<10),
    calibrators kept in both subsets. The localized endpoint tier predicts an
    s_anchor difference tracking host organization; the observer-well reading
    predicts the same s_anchor in both. NOTE: m_b_corr already includes the
    standardized host-mass step, so this probes environment dependence BEYOND it.
(2) Weighted correlation of anchored baseline residuals with HOST_LOGMASS (HF only).
(3) Clock/PTA amplitude numbers at the fitted step-A parameters.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from scipy.stats import pearsonr, spearmanr

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
s_anchor_full = float(np.asarray(res_a.x)[4])
print(f"[d6] full-sample stepA s_anchor={s_anchor_full:.4f} dchi2={chi2_b-float(res_a.fun):.3f}")

results = {"full": H.describe_fit("stepA-full", res_a, chi2_b), "host_split": {},
           "residual_correlation": {}, "timing_channel": {}}

# ---------------- host columns ----------------
df = pd.read_csv(H.REPO / "Pantheon+SH0ES.dat", sep=r"\s+", comment="#")
key = {}
for j in range(len(df)):
    key[(str(df["CID"].iloc[j]), int(df["IDSURVEY"].iloc[j]),
         round(float(df["zHD"].iloc[j]), 6))] = j
rows = np.array([key[(str(sn.cid[i]), int(sn.idsurvey[i]), round(float(sn.zHD[i]), 6))]
                 for i in range(len(sn.y))], dtype=int)
logmass = df["HOST_LOGMASS"].to_numpy(dtype=float)[rows]
is_cal = sn.is_calibrator if sn.is_calibrator is not None else np.zeros(len(sn.y), bool)
valid_mass = logmass > 0.0

# ---------------- (1) host-mass split ----------------
for name, mask_hf in (
    ("highmass", (~is_cal) & valid_mass & (logmass >= 10.0)),
    ("lowmass", (~is_cal) & valid_mass & (logmass < 10.0)),
):
    idx = np.where(mask_hf | is_cal)[0]
    sn_sub = H.lp.subset_sn(sn, idx)
    rb = H.lp.fit_baseline(sn_sub, bao, epochs, zmax_table)
    H0s, Oms, als = [float(v) for v in rb.x]
    ra_fit = H.fit_stepA(sn_sub, bao, epochs, zmax_table, H0s, Oms, als)
    d = H.describe_fit(f"host-{name}", ra_fit, float(rb.fun))
    d.update(n_hf=int(np.sum(mask_hf)))
    results["host_split"][name] = d
    print(f"[d6] {name:9s} N_HF={d['n_hf']:3d}  s_anchor={d['amplitude']:.3f}  "
          f"dchi2={d['dchi2_vs_baseline']:.3f}")

# ---------------- (2) residual vs mass correlation (HF, baseline anchored) ----------------
tables = H.lp.build_cosmology_tables(H0=H0_b, Om=Om_b, zmax=zmax_table)
chi_sn = tables.chi_of_z(sn.zHD)
mu_pred = 5.0 * np.log10((1.0 + sn.zHEL) * chi_sn) + 25.0
mu_ref = H.lp.sn_mu_reference(sn, mu_pred)
r0 = sn.y - mu_ref
Cinv_r0 = cho_solve(sn.cho, r0, check_finite=False)
M_best = float((sn.ones @ Cinv_r0) / sn.ones_Cinv_ones)
resid = sn.y - (mu_ref + M_best)

for zcut_name, zmask in (("all_HF", np.ones(len(sn.y), bool)), ("z<0.06", sn.zHD < 0.06)):
    m = (~is_cal) & valid_mass & zmask
    pr = pearsonr(logmass[m], resid[m])
    sr = spearmanr(logmass[m], resid[m])
    results["residual_correlation"][zcut_name] = {
        "N": int(np.sum(m)),
        "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue),
    }
    print(f"[d6] resid~logM ({zcut_name}) N={int(np.sum(m))}: "
          f"pearson r={pr.statistic:+.3f} (p={pr.pvalue:.3f})  "
          f"spearman r={sr.statistic:+.3f} (p={sr.pvalue:.3f})")

# ---------------- (3) timing-channel amplitudes ----------------
gA = float(H.lp.earth_history_g(np.array([H.lp.LTA_T_ANCHOR_GYR]),
                                H.lp.LTAParams(s_anchor_full, 1.0, 1.0), epochs)[0])
s_now = s_anchor_full / gA  # km/s/Mpc
MPC_KM = 3.0857e19
GYR_S = 3.156e16
s_now_si = s_now / MPC_KM  # 1/s
# powerlaw B=p=1, tL=3.8: gdot(0) = -(B+tL)/(tL*B) per Gyr
gdot0 = -(1.0 + 3.8) / 3.8
sdot_si = abs(s_now_si * gdot0 / GYR_S)  # 1/s^2
T_pta = 15.0 * 365.25 * 86400.0
x_resid = sdot_si * T_pta**3 / 6.0  # s, after nu/nudot absorption
results["timing_channel"] = {
    "s_now_km_s_mpc": s_now,
    "frac_freq_drift_per_yr": s_now_si * 3.156e7,
    "sdot_si_per_s2": sdot_si,
    "pta_residual_15yr_s": x_resid,
    "note": "constant drift absorbed in nu-dot fits; residual is the curvature term",
}
print(f"[d6] s_now={s_now:.3f} km/s/Mpc -> drift {s_now_si*3.156e7:.3e}/yr; "
      f"PTA curvature residual over 15 yr ~ {x_resid*1e9:.3f} ns")

results["runtime_s"] = time.time() - t0
(OUT / "d6_host_timing.json").write_text(json.dumps(results, indent=2))
print(f"[d6] wrote out/d6_host_timing.json ({time.time()-t0:.0f}s)")
