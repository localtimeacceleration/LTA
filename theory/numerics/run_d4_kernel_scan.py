"""D4 numerics: kernel-weighted forward models with physical amplitude factors.

Branch A (compact source at observer, two-leg timing, 1/chi Yukawa dilution restored):
scan (lam_mpc, R_reg_mpc).
Branch B (extended chronometric well, one-leg timing, growth-modulated):
scan (R_w_mpc, n).

Benchmark: step-A powerlaw fit in the identical likelihood configuration.
"""
import json
import time
from pathlib import Path

import numpy as np

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
print(f"[d4] baseline chi2={chi2_b:.3f}  stepA dchi2={bench['dchi2_vs_baseline']:.3f}")

results = {"baseline_chi2": chi2_b, "stepA": bench, "compact": [], "well": []}

for R_reg in (0.1, 1.0):
    for lam in (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0):
        prof = H.CompactSourceProfile(lam_mpc=lam, R_reg_mpc=R_reg)
        res = H.fit_custom(sn, bao, epochs, zmax_table, H0_b, Om_b, alpha_b, prof)
        d = H.describe_fit(f"compact lam={lam} Rreg={R_reg}", res, chi2_b)
        d.update(lam_mpc=lam, R_reg_mpc=R_reg)
        results["compact"].append(d)
        print(f"[d4] compact lam={lam:7.1f} Rreg={R_reg}: dchi2={d['dchi2_vs_baseline']:8.3f}  A={d['amplitude']:.3e}")

for n in (2.0, 4.0):
    for Rw in (20.0, 35.0, 50.0, 70.0, 100.0, 150.0, 200.0):
        prof = H.ExtendedWellProfile(R_w_mpc=Rw, n=n)
        res = H.fit_custom(sn, bao, epochs, zmax_table, H0_b, Om_b, alpha_b, prof)
        d = H.describe_fit(f"well Rw={Rw} n={n}", res, chi2_b)
        d.update(R_w_mpc=Rw, n=n)
        results["well"].append(d)
        print(f"[d4] well Rw={Rw:6.1f} n={n}: dchi2={d['dchi2_vs_baseline']:8.3f}  A={d['amplitude']:.3e}")

# I(chi) curves at family best fits, for the derivation documents
best_c = max(results["compact"], key=lambda d: d["dchi2_vs_baseline"])
best_w = max(results["well"], key=lambda d: d["dchi2_vs_baseline"])
tables = H.lp.build_cosmology_tables(H0=H0_b, Om=Om_b, zmax=zmax_table)
chi_plot = np.linspace(0.0, 600.0, 400)

lta_best = H.lp.LTAParams(
    s_anchor_km_s_per_mpc=float(np.asarray(res_a.x)[4]), g_complex=1.0, g_life=1.0
)
H.deactivate_profile()
I_stepA = H.lp.lta_integral_I(chi_plot, tables, lta_best, epochs)

curves = {"chi_mpc": chi_plot.tolist(), "I_stepA": np.asarray(I_stepA).tolist()}
for name, d, cls, keys in (
    ("I_compact_best", best_c, H.CompactSourceProfile, ("lam_mpc", "R_reg_mpc")),
    ("I_well_best", best_w, H.ExtendedWellProfile, ("R_w_mpc", "n")),
):
    prof = cls(**{k: d[k] for k in keys})
    g_chi, g_I, _ = prof.grids(tables, epochs)
    curves[name] = (d["amplitude"] * np.interp(chi_plot, g_chi, g_I)).tolist()
results["curves"] = curves
results["runtime_s"] = time.time() - t0

(OUT / "d4_kernel_scan.json").write_text(json.dumps(results, indent=2))
print(f"[d4] best compact: {best_c['tag']} dchi2={best_c['dchi2_vs_baseline']:.3f}")
print(f"[d4] best well:    {best_w['tag']} dchi2={best_w['dchi2_vs_baseline']:.3f}")
print(f"[d4] wrote out/d4_kernel_scan.json ({time.time()-t0:.0f}s)")
