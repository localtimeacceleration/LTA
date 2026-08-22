"""Task 1: validate the harness by reproducing the fiducial baseline and step-A fits.

Target (run-tag 20260112-013808):
  baseline chi2 = 332.549 (H0=68.5165, Om=0.29966, alpha_rd=0.99791)
  LTA(powerlaw) chi2 = 318.897, s_anchor=2.4493, dchi2 = 13.652, t_anchor=0.670 Gyr
Also validates the injection layer with the step-A shape control profile.
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
print(f"[t1] setup done N_SN={len(sn.y)}  ({time.time()-t0:.1f}s)")

res_b = H.fit_baseline(sn, bao, epochs, zmax_table)
H0_b, Om_b, alpha_b = [float(v) for v in res_b.x]
chi2_b = float(res_b.fun)
print(f"[t1] baseline: H0={H0_b:.4f} Om={Om_b:.5f} alpha={alpha_b:.5f} chi2={chi2_b:.3f}")

t_anchor = H.set_anchor_from_baseline(sn, H0_b, Om_b, zmax_table, epochs)
print(f"[t1] t_anchor = {t_anchor:.4f} Gyr (target 0.670)")

res_a = H.fit_stepA(sn, bao, epochs, zmax_table, H0_b, Om_b, alpha_b)
xa = np.asarray(res_a.x, dtype=float)
print(
    f"[t1] stepA: H0={xa[0]:.4f} Om={xa[1]:.5f} alpha={xa[2]:.5f} "
    f"s_anchor={xa[4]:.4f} chi2={float(res_a.fun):.3f} dchi2={chi2_b - float(res_a.fun):.3f}"
)

# injection-layer control: same shape via the custom machinery
ctrl = H.StepAShapeControl()
res_c = H.fit_custom(sn, bao, epochs, zmax_table, H0_b, Om_b, alpha_b, ctrl, A_max=0.05)
xc = np.asarray(res_c.x, dtype=float)
print(
    f"[t1] control(injected stepA shape): A={xc[4]:.5g} chi2={float(res_c.fun):.3f} "
    f"dchi2={chi2_b - float(res_c.fun):.3f}"
)

summary = {
    "t_anchor_gyr": t_anchor,
    "baseline": {"H0": H0_b, "Om": Om_b, "alpha_rd": alpha_b, "chi2": chi2_b},
    "stepA": H.describe_fit("stepA", res_a, chi2_b),
    "stepA_injected_control": H.describe_fit("stepA-control", res_c, chi2_b),
    "targets": {"chi2_baseline": 332.549, "chi2_stepA": 318.897, "dchi2": 13.652},
    "runtime_s": time.time() - t0,
}
(Path(OUT) / "t1_reproduce.json").write_text(json.dumps(summary, indent=2))
print(f"[t1] wrote out/t1_reproduce.json  ({time.time()-t0:.1f}s total)")
