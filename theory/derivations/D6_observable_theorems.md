# D6 — Observable-propagation theorems and channel amplitudes

**Goal.** Derive what every observational channel sees at the fitted amplitude — as theorems where
possible, as computed numbers otherwise. Executed numerics: `theory/numerics/run_d6_host.py`
(→ `out/d6_host_timing.json`).

Fitted reference point throughout: $I_{\rm sat} = \epsilon\Delta\psi \approx 2.7\times10^{-3}$
(step A) to $4.6\times10^{-3}$ (well $R_w$=200), $s_{\rm now} = 4.96$ km/s/Mpc.

---

## 1. Theorem: time-dilation equality (the first thing a referee checks)

Source events separated by clock time $d\tau^{\rm clk}_e = A(\psi_e)\,d\tau_e$; geometric
propagation stretches proper intervals by $(1+z_{\rm geom})$; the observer's clock reads
$d\tau^{\rm clk}_o = A(\psi_o)\,d\tau_o$. Chaining:

$$
\frac{d\tau^{\rm clk}_o}{d\tau^{\rm clk}_e}
= (1+z_{\rm geom})\,\frac{A(\psi_o)}{A(\psi_e)}
= 1+z_{\rm obs},
$$

which is the *same* factor that rescales spectroscopic frequencies (U.7).

> **Theorem D6.1.** Chronometric endpoint imprints preserve the standard stretch–redshift
> relation exactly: light-curve time dilation follows $z_{\rm obs}$, with zero residual at any
> order in $I$. SN stretch tests (Goldhaber-type), quasar variability dilation, and GRB duration
> scalings constrain nothing here. A "spectroscopic-only" variant of LTA — where lines shift but
> durations don't — would be instantly dead; the completed theory is not that variant, and this
> theorem should be boxed in the paper (it currently only gestures at the issue in Sec. 8.2).

## 2. Theorem: endpoint-only structure and distance duality

Photon energies and arrival rates are both clock-referenced, so both dilute by the full
$(1+z_{\rm obs})$: flux $= L/[4\pi\chi^2(1+z_{\rm obs})^2]$, hence

$$
d_L = (1+z_{\rm obs})\,\chi(z_{\rm cos}),
\qquad
d_L = (1+z_{\rm obs})^2 d_A
$$

— Etherington duality holds *in the observed redshift*. Two corollaries:

- **(a)** The pipeline's $d_L = (1+z_{\rm HEL})\chi(z_{\rm cos})$ is exactly right for the
  chronometric case ✓ (no missing factor).
- **(b) Kinematic discriminant.** A true peculiar velocity adds one further factor:
  $d_L^{\rm kin} = (1+z_{\rm obs})\chi(\bar z)(1+z_{\rm pec})$ (relativistic beaming of the
  source). Chronometric and kinematic explanations of the same $z$-offset therefore differ in
  distance modulus by $\Delta\mu = 5\log_{10}(1+I) \approx 2.17\,I \approx 6$–$10$ mmag at
  fitted amplitudes — a real but currently sub-systematics discriminant (Pantheon+ floor ≈ 20–30
  mmag coherent). Consequence for the D8 void fit: omitting the beaming term there biases
  nothing at present precision (≤ 0.01 mag, χ²-negligible), as assumed.
- **(c) Endpoint-only propagation.** $I$ depends only on ψ at the two endpoints, so there is no
  ISW-like accumulation along the path: the local well does **not** imprint on CMB photons in
  transit. The CMB sees only (i) the unobservable observer-endpoint monopole and (ii) ψ
  fluctuations *at last scattering*, suppressed by the order-sector growth factor
  $D_\psi(z{=}1100)/D_\psi(0)$; for ψ tracking matter growth this is a ≲percent-level
  contribution to first-peak power — flagged for the full-$C_\ell$ treatment (paper App. W.8),
  not currently constraining.

## 3. Pulsar timing and clocks (computed)

From the executed run (`out/d6_host_timing.json`):

| Quantity | Value |
|---|---|
| $s_{\rm now}$ | 4.96 km/s/Mpc $= 1.61\times10^{-19}\,{\rm s^{-1}} = 5.07\times10^{-12}\,{\rm yr^{-1}}$ |
| PTA residual from drift *curvature*, 15 yr | **0.11 ns** |
| current PTA common-signal sensitivity | ~10–100 ns |

- A *uniform* clock-rate drift is locally unobservable (all clocks co-drift; it is a unit
  redefinition — Theorem D3.2's local shadow), and in pulsar timing a constant fractional drift
  is absorbed into the fitted $\dot\nu$ — the App. P monopole degeneracy, here made quantitative:
  the first non-absorbed term (from activation curvature $\ddot g$) is 0.11 ns over 15 yr,
  **two to three orders below current PTA sensitivity.** The timing channel is safe and will
  remain so for decades. Laboratory clock *ratio* comparisons constrain only non-universal
  couplings, addressed in D7.

## 4. Host-environment endpoint tier: executed test, null result

If the localized tier ($\delta\psi_{\rm loc}$ at the emitter) mattered, SNe in more organized /
more massive hosts would carry systematically different endpoint imprints — a Hubble-residual
correlation with host properties *beyond* the standardized mass step (note: `m_b_corr` already
includes the SALT mass-step correction, so this tests the increment). Executed on Pantheon+
(`run_d6_host.py`):

| Test | Result |
|---|---|
| $s_{\rm anchor}$, hosts log M ≥ 10 (N=135) | 3.10 |
| $s_{\rm anchor}$, hosts log M < 10 (N=142) | 2.80 |
| residual–mass Pearson r (all HF, N=277) | −0.057 (p = 0.35) |
| residual–mass Spearman r (z<0.06, N=203) | −0.054 (p = 0.44) |

> **Result D6.2.** No detectable emitter-environment dependence: the split amplitudes agree
> within errors and the residual correlations are null. This *supports* the D5 identification
> (the signal is about the observer's location in a large-scale well, common to all lines of
> sight) and bounds the per-dex endpoint term at roughly
> $\epsilon\,\delta\psi_{\rm loc} \lesssim 10^{-4}$ per dex of host mass (from the residual
> correlation limit ≈ 0.01–0.02 mag at low z). It also removes the last data-side motivation for
> terrestrial-history modulation factors (`g_complex`, `g_life`) beyond their fixed-at-unity
> fiducial role.

## 5. BAO under the kernel models (executed)

The χ² decomposition (`out/x_decompose.json`) shows the radial-ruler police at work: the
$R_w$=200 well buys its SN improvement (309.7 vs baseline 324.4) at a BAO cost of +1.66
(5.08 vs 3.42), because its imprint is still growing through $z = 0.38$–$0.61$ and distorts the
Jacobian there; step A, whose imprint saturates earlier, pays only +0.63. **Prediction:** at
DESI-DR2 radial-BAO precision the well family's allowed $(R_w, \epsilon\Delta\psi)$ region
shrinks visibly; a well large enough for the SN preference must begin to show in
$H(z)\,r_d$ at the few-per-mil level at $z\simeq0.4$–0.6. This is preregistered as D8 test T5.

## 6. Channel table (at fitted amplitude)

| Channel | Prediction | Current bound | Margin / status |
|---|---|---|---|
| SN stretch vs spec-z | exact equality (Thm D6.1) | consistent | ∞ (theorem) |
| Distance duality (obs-z) | exact (Thm §2) | consistent | ∞ (theorem) |
| Chrono-vs-kinematic μ offset | 6–10 mmag | ~20–30 mmag syst. | future lever |
| PTA monopole curvature | 0.11 ns / 15 yr | 10–100 ns | ~10²–10³ |
| Terrestrial clock ratios | 0 (universal coupling) | — | D7 handles non-universal loops |
| Host-environment endpoint | ≲10⁻⁴/dex (measured null) | — | supports observer-well reading |
| CMB in-transit imprint | 0 (endpoint theorem) | — | safe |
| CMB last-scattering wells | ≲1% first-peak power | percent-level | needs App. W.8 treatment |
| Radial BAO at z≈0.5 | few-per-mil in $H r_d$ (well) | consensus BAO: +1.7 χ² | D8-T5, DESI decides |
