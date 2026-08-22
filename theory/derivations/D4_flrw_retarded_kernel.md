# D4 — The FLRW retarded kernel, and the kernel-weighted refits

**Goal.** Replace the two-leg "horizon approximation" with the actual retarded solution:
derive the FLRW retarded Green's function including its tail, quantify when null-cone timing is
adequate, restore the amplitude factors App. V.5 dropped, and re-fit the resulting forward models
against the same SN+BAO likelihood as step A. Executed numerically in
`theory/numerics/run_d4_kernel_scan.py` (results: `theory/numerics/out/d4_kernel_scan.json`,
decomposition in `out/x_decompose.json`), with the harness validated against the paper's fiducial
run in `out/t1_reproduce.json` (reproduces baseline χ²=332.44 vs paper 332.55, Δχ²=13.59 vs
13.65, s_anchor=2.445 vs 2.449, t_anchor=0.670 ✓).

---

## 1. The retarded Green's function on FLRW, with its tail

Work in conformal coordinates, $ds^2 = a^2(\eta)(-d\eta^2 + d\vec x^2)$ (units c=1 in this
section). For $(\Box - m^2)\psi = S$, substitute $\psi = u/a$:

$$
u'' - \nabla^2 u + M^2(\eta)\,u = a^3 S,
\qquad
M^2(\eta) \equiv m^2 a^2(\eta) - \frac{a''}{a}.
$$

For $M^2 = 0$ (massless, and $a''=0$, i.e. exactly conformal) the problem is flat-space and the
retarded Green's function is pure null cone, $G = \delta(\Delta\eta - r)/4\pi r$. In general:

> **Structure D4.1.** $\;G_{\rm ret}(\eta,\eta';r) = \dfrac{\delta(\Delta\eta - r)}{4\pi r}
> + T(\eta,\eta';r)\,\theta(\Delta\eta - r)$, with the tail $T$ generated entirely by
> $M^2(\eta)$. To first Born order,
> $$ T(\eta,\eta';r) = -\frac{1}{8\pi}\,\bar M^2 + O(M^4), $$
> where $\bar M^2$ is an average of $M^2$ over the interior of the cone segment. The tail's
> fractional contribution to the line-of-sight imprint accumulated to comoving radius χ scales as
> $$ \frac{\delta I_{\rm tail}}{I} \sim \tfrac12\,(m_{\rm eff}\chi)^2 + O\!\big((H_0\chi/c)^2\big). $$

**Numbers for the fit range.** The SH0ES-HF sample spans $z = 0.0234$–$0.149$
($\chi \approx 100$–$620$ Mpc; $H_0\chi/c \le 0.14$). For any $m_{\rm eff}^{-1} \gtrsim 2$ Gpc
(required anyway for the well to span the sample, §3 below), the tail correction to $I$ is
$\lesssim 1$–$2\%$ across the whole range — far below the fit's amplitude resolution
(σ(A)/A ≈ 30%).

> **Conclusion D4.2.** Null-cone *timing* is safe at the percent level in this dataset; the
> horizon approximation's real sin was never timing — it was the dropped **amplitude factors**
> ($1/\chi$, Yukawa, scale factors). Those are restored below and change the verdicts
> qualitatively.

**Limit theorems (timing).** Collapsing $G_{\rm ret}$ onto the null cone and solving for the field
on the observer's past light cone reproduces:
- *compact support at the observer* → the two-leg rule $\eta' = \eta_0 - 2\chi/c$ (App. V
  recovered — the factor 2 is the out-and-back geometry of a point source at the observer);
- *extended support* → one-leg evaluation: the field at the emission event is set by the local
  source history at that location, at lookback time $t_{\rm lb}(z)$, weighted by the spatial
  profile. **The "2χ rule" is a compact-source artifact**; comparisons of activation timescales
  between the two geometries must be made at the level of $I(\chi)$, not of $g$-parameters.

## 2. Kernel-weighted forward models

Implemented through the injection layer of `theory/numerics/harness.py` (runtime rebinding of
`lta_integral_I` / `lta_local_s`; the SN likelihood, BAO Jacobian, and inverse mapping all flow
through the injected profile consistently; validated end-to-end by refitting the step-A shape
through the injection layer: Δχ² = 13.58 vs native 13.59 ✓).

**Branch A — compact source, physics restored** (two-leg timing × dilution):
$\psi_{\rm emit}(\chi) \propto g(t_{\rm ret}(2\chi))\, e^{-\chi/\lambda}\, R_{\rm reg}/\max(\chi,R_{\rm reg})$,
$I = \epsilon[\psi(0)-\psi_{\rm emit}(\chi)]$. Scan λ ∈ {10…3000} Mpc, R_reg ∈ {0.1, 1} Mpc.

**Branch B — extended chronometric well** (one-leg timing × growth):
$\psi(\chi) \propto D(t_{\rm lb}(\chi))\,W(\chi)$, $W = [1+(\chi/R_w)^n]^{-1}$.
Scan $R_w$ ∈ {20…200} Mpc, n ∈ {2, 4}.

## 3. Results

Benchmark (identical likelihood configuration): baseline χ² = 332.444; **step A Δχ² = 13.59**.

| Model | best Δχ² | best parameters | fitted amplitude εΔψ |
|---|---|---|---|
| Compact (any λ, any R_reg) | **10.10** | shape-insensitive | 9.8×10⁻⁴ |
| Well, n=2 | **13.04** | R_w = 200 Mpc (still rising) | 4.6×10⁻³ |
| Well, n=4 (steep edge) | 10.09 | R_w = 20 Mpc | 9.8×10⁻⁴ |
| Well n=2, R_w=100 | 10.85 | — | 1.9×10⁻³ |
| Well n=2, R_w=150 | 12.11 | — | 3.2×10⁻³ |

χ² decomposition at best fits (`out/x_decompose.json`):

| Model | SN | BAO | prior | total |
|---|---|---|---|---|
| baseline | 324.36 | 3.42 | 4.66 | 332.44 |
| step A | 310.18 | 4.05 | 4.63 | 318.85 |
| offset (=compact) | 314.16 | 3.55 | 4.63 | 322.34 |
| well (200, n=2) | **309.69** | 5.08 | 4.64 | 319.41 |

### Findings

**F1 — The compact branch collapses to a pure monopole offset.** The SH0ES-HF selection has
$z_{\rm min} = 0.0234$ (χ ≈ 100 Mpc): *every* Hubble-flow SN sits beyond the region where a
compact-source profile has structure, and calibrators are mapping-independent (they are compared
to Cepheid distances). So all compact variants degenerate to a constant
$I_0 \approx 9.8\times10^{-4}$ — a uniform $\ln(1+z)$ boost of the HF sample (≈ 293 km/s) —
independent of λ and R_reg to three decimals in χ². Its Δχ² = 10.10.

**F2 — Most of the step-A preference is the offset mode.** A structureless monopole already
captures 10.1 of 13.6; the *shape* of the step-A activation contributes only ≈ 3.5. (At fixed
observed z, a constant I produces a 1/z distance-modulus valley — which is most of what the
"red valley" is.) Interpretive altitude for everything downstream: any mechanism or systematic
producing a ~10⁻³ uniform redshift offset of the HF sample relative to the calibrator-anchored
prediction — including a ~300 km/s monopole redshift-calibration error, which is what D8 must
exclude — competes for 3/4 of the signal.

**F3 — The physically-diluted compact branch is dead as an explanation of the shape, exactly as
the amplitude gate predicts** (program §3.1): its fit survives only by abandoning its shape
content entirely (F1), and carrying the fitted amplitude back down the $1/\chi$ profile to the
source requires $\ln A \sim 10^{15}$ (Earth) to $\sim 30$ (Galaxy) at the source edge — excluded.
Kill criterion of D4-branch-A: **triggered** (shape) — formalized in D5.

**F4 — The extended well is fit-viable but must be large and gentle.** Δχ² rises monotonically
with R_w through 200 Mpc and prefers the soft edge (n=2 ≫ n=4). The steep-edged Local-Void-scale
well (50–100 Mpc) is disfavored by 3–6 units relative to R_w ≥ 150. The data want the imprint to
keep accumulating across the entire HF range (out to ≥ 400–600 Mpc) — consistent with the step-A
activation's own reach ($s(t_{\rm ret})$ support out to $t_L$=3.8 Gyr ↔ χ ≈ 500 Mpc). At
R_w=200 the well fits the SN sector *better* than step A (309.7 vs 310.2) and pays the
difference in the BAO Jacobian (5.08 vs 4.05) — the first quantitative demonstration in this
program that BAO genuinely polices the mapping (paper Sec. 5.4's design intent, working as
built).

**F5 — Required well depth grows with R_w:** εΔψ = 1.9×10⁻³ (R_w=100) → 4.6×10⁻³ (R_w=200).
D5 must deliver these depths from structure whose ΛCDM density contrast at those scales is only
σ(R) ≈ 0.1–0.03 for R = 100–200 Mpc — the central quantitative tension the correlated branch
must resolve.

## 4. Outputs

- Tail-bound (D4.1–D4.2): timing systematic ≤ 2% in-range → the injected null-cone kernels are
  adequate forward models for this dataset.
- The injection layer itself (`harness.py`): a drop-in `--kernel`-mode equivalent for
  `lta_power.py`, validated at 0.01 in Δχ² against the native implementation, usable by the
  world-tunnel workstream for arbitrary $W(\vec x)$ profiles.
- F1–F5 verdicts to D5 (support selection) and D8 (offset-mode systematics on the kill list).
