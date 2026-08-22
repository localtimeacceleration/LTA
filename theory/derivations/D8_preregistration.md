# D8 — Statistical program: executed diagnostics and preregistered tests

**Goal.** Answer the ledger's G1 honestly: the full-data Δχ² is a one-parameter,
boundary-constrained amplitude detection whose conditional cross-validation is negative — so the
theory earns belief only through *new* discriminants. This document (i) reports the diagnostics
executed in this program, clearly labeled exploratory (the data have been seen), and (ii)
preregisters the forward tests with statistic, null, and threshold fixed *before* the data that
will decide them.

Numerics: `theory/numerics/run_d8_void_headtohead.py`, `run_d6_host.py`, `run_x_decompose.py`
(→ `out/d8_void_sky.json`, `out/d6_host_timing.json`, `out/x_decompose.json`; figure
`out/d4_d8_summary.png`).

---

## 1. Executed diagnostics (exploratory — data already seen)

### E1. Mundane-alternative head-to-head: linear-theory void outflow (ledger G6)

Model: centered compensated underdensity, cumulative contrast
$\bar\delta(<r) = \delta_c/[1+(r/R_v)^\gamma]$, linear outflow
$v(r) = -\tfrac13 f H_0 \bar\delta(<r)\,r$, imprint $I = v/c$; same likelihood, covariance,
priors, and free-parameter count as the step-A fit (background + one amplitude, shape scanned).
The beaming term omitted here is ≤ 0.01 mag (Theorem D6.2b) — χ²-negligible.

| Model (physical region $|\delta_c|\le1$) | best Δχ² | implied $|\delta_c|$ |
|---|---|---|
| void $R_v$=70, γ=2 | 7.47 | 0.59 |
| void $R_v$=100, γ=2 | 7.28 | 0.37 |
| void $R_v$=150, γ=2 | 7.25 | 0.26 |
| void $R_v$=200, γ=2 | 7.63 | 0.24 |
| *(unphysical corner $R_v$=30, γ=3)* | *(7.83)* | *(8.0 — excluded, δ ≥ −1)* |
| **step A** | **13.59** | — |
| pure offset mode | 10.10 | — |

**Verdict.** The kinematic void model **underperforms step A by ≈ 6 units** and even the
structureless offset by ≈ 2.6: compensation makes $v(r)$ turn over where the data want the
imprint still growing (see figure, right panel). At matched dof, the coherent-outflow reading of
the red valley is *not* an adequate account of the step-A preference. This is a genuine positive
result for LTA-type mappings — with two caveats: the required void depths (0.26–0.59) are
already 3–6× deeper than ΛCDM typical at those scales, so the kinematic alternative was
independently strained; and the offset mode is doing most of the work for every model (next).

### E2. The offset mode and its mundane candidates (from D4-F1/F2)

A structureless uniform $\ln(1+z)$ boost of the HF sample, $I_0 \approx 9.8\times10^{-4}$
(≈ 293 km/s), achieves Δχ² = 10.10 of step A's 13.59. Notes:

- It is **not** absorbable into the SN absolute calibration: the ladder-anchored likelihood
  already profiles $M$ against the calibrators, and a z-mapping offset produces a $1/z$-shaped
  μ-residual, not a constant — the 10.1 is shape information, not an $M$ degeneracy.
- As a *systematic*, it would require a ~300 km/s coherent monopole error in the HF redshift
  frame (peculiar-velocity corrections, frame transformations) — several times the plausible
  size of those corrections' uncertainties (tens of km/s), so a pure-systematic reading is
  strained but not absurd; it stays on the kill list as the leading mundane threat.
- Physically, within the theory it is the observer-endpoint constant $\epsilon\,\delta\psi_{\rm loc,obs}$
  (D3 §4) — a well too large to resolve ($R_w \to \infty$ limit).

### E3. Sky-split anisotropy (exploratory)

Step-A amplitude refit per hemisphere (calibrators retained in both; anchor fixed at full-sample
value; 1σ from profile likelihoods, `out/x_decompose.json`):

| Axis | toward | away |
|---|---|---|
| CMB dipole apex (profile-converged) | $s_{\rm anchor} = 4.6 \pm 0.75$ (N=122) | $2.2 \pm 0.7$ (N=155) |
| celestial N (control, coarse fits) | 3.01 (N=190) | 2.86 (N=87) |

The CMB-dipole split shows a real contrast: $\Delta s = 2.4 \pm 1.0$ (≈2.3σ) from the profile
likelihoods, 1.6σ using the coarser 4-parameter fits (`out/x_decompose.json`) — a *hint*, not a
detection; the control axis shows none.
Direction and sign are what a structure-linked well predicts (D5 §3.3) and align with the
CosmicFlows-era reports of excess bulk flow toward roughly the same direction. Because this axis
was chosen after seeing the paper's results, it cannot count as evidence — it graduates to a
preregistered test (T1).

### E4. Host-environment null (D6.2) and roughness consistency (D2.3)

Executed and reported in D6: no residual–host-mass correlation (p ≈ 0.35–0.51), split amplitudes
consistent; no excess low-z scatter (χ²/dof ≈ 0.91) so the η ≲ 0.35 smoothness bound is
respected. Both consistent with the observer-well identification; both remain standing cuts on
the microphysics.

## 2. Preregistered forward tests

Each specifies data not yet used (or not yet existing), statistic, null, and decision threshold.
These definitions are frozen by this commit; changes require a logged amendment.

**T1 — Dipole-aligned anisotropy (primary).**
*Data:* any independent low-z SN compilation (ZTF SN Ia DR; LSST early low-z), or Pantheon+
successor releases restricted to SNe not in Pantheon+.
*Statistic:* $\Delta s = s_{\rm anchor}^{\rm toward} - s_{\rm anchor}^{\rm away}$ about the CMB
dipole apex (axis frozen here), identical likelihood machinery.
*Null:* isotropy, $\Delta s = 0$, calibrated by hemisphere-preserving mocks.
*Decision:* ≥3σ → correlated-well promoted; <1σ with errors ≤ 0.5 km/s/Mpc → well anisotropy
bounded, observer-history readings stay dead (D5.1), offset/systematic reading strengthened.

**T2 — Velocity-survey cross-check.**
*Data:* CosmicFlows-4 (and successors) distance–redshift catalogs — independent of SN Hubble
diagram calibration.
*Statistic:* fit the same $W(\chi)$ well family to the CF4 monopole+dipole of
$cz - H_0 d$; compare implied $(\epsilon\Delta\psi, R_w)$ with the SN-fit region (D5 table).
*Null:* ΛCDM velocity statistics.
*Decision:* overlapping confidence regions → strong promotion (two independent probes, one
well); disjoint at >3σ → the well reading fails O3 and is retired.

**T3 — Conditional-CV shape test.**
*Data:* the existing Pantheon+ folds (the paper's CV protocol, unchanged, seed-frozen).
*Statistic:* conditional held-out Δχ² of well($R_w$, n=2) **against the offset model**, not
against ΛCDM — the question G1 actually poses is whether the *shape* generalizes.
*Decision:* well beats offset conditionally in ≥70% of folds → shape is real; otherwise the
defensible claim reduces to the offset mode.

**T4 — Sub-sample roughness.**
*Data:* first ZTF low-z volume with z < 0.02 coverage.
*Statistic:* excess scatter vs $\sigma_\mu(z) = 0.36\,\eta$ mag prediction (D2.3).
*Decision:* measured excess → η measured (microphysics window); none → η bound tightens.

**T5 — DESI radial BAO.**
*Data:* DESI DR2+ $H(z) r_d$ at z = 0.4–0.6.
*Statistic:* the well family's predicted few-per-mil radial distortion (D6 §5) vs measured.
*Decision:* the $(R_w, \epsilon\Delta\psi)$ region shrinks or the well is excluded at SN-required
depth; this is the test the theory cannot dodge, because D4 showed BAO already taxes the well.

**T6 — (Future lever) chronometric-vs-kinematic μ discriminant.** 6–10 mmag (D6.2b);
actionable when coherent low-z μ systematics reach the sub-10-mmag level (Rubin era), or earlier
with standard sirens + EM counterparts.

## 3. Decision tree

- T1 **and** (T2 or T3) pass → the correlated chronometric well graduates from hypothesis to
  measured structure; the D7 successor task (derivative-screened completion) becomes urgent
  rather than optional.
- T3 fails but T1 passes → anisotropic offset: revisit both well geometry and direction-dependent
  systematics.
- T1 and T2 fail → the defensible residue is the offset mode; publish as "coherent low-z
  monopole anomaly, origin unidentified," and the OTA mechanism for *this* dataset is retired.
  (The framework and this program remain; the claim does not.)

Every branch of the tree is publishable. That is the point of preregistering now.
