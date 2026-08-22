# D5 — Source-support selection: what can actually carry the step-A signal

**Goal.** Turn the program's §3 structural arguments plus the D4 refits into a closed selection
over source supports, and state — quantitatively — what the surviving branch (the correlated-tier
chronometric well) must deliver. This is the make-or-break derivation of the program; it ends
with the branch alive but carrying three named, numbered obligations, one genuine out-of-sample
cross-check, and one honest warning about the offset mode.

**Inputs.** D3 (frame theorem, transfer function), D4 (kernel refits F1–F5), numerics in
`theory/numerics/out/`.

---

## 1. The amplitude gate as a theorem (compact supports excluded)

For source support of coherence scale $R_s$ centered on the observer, the sourced field on the
past light cone carries the propagation factor $e^{-\chi/\lambda}/\chi$ (D4 kernel), so the field
at the source edge exceeds the field at χ by $\ge \chi/R_s$. Requiring the fitted imprint
(εΔψ ≈ 10⁻³ maintained across the sample, χ up to ~600 Mpc) at the far end forces, at the source
edge:

| $R_s$ | Earth (10⁷ m) | Sun (10¹¹ m) | Galaxy (10 kpc) | Local Group (1 Mpc) | ≥ Local Sheet (≥30 Mpc) |
|---|---|---|---|---|---|
| $\ln A$ at edge | ~10¹⁵ | ~10¹¹ | ~30–60 | ~0.3–0.6 | ≤ 10⁻² |

> **Theorem D5.1 (compact-source exclusion).** Any support with $R_s \lesssim$ few Mpc requires
> chronometric factors at its own edge that are excluded by elementary observations (terrestrial
> metrology; stellar and pulsar astrophysics at the Galactic center; Local Group distance/timing
> consistency). Independently, the D4 refits show the compact branch's *shape* contributes
> nothing: below the SH0ES-HF floor $z_{\rm min}=0.0234$ it has no data to shape, and it
> collapses to the monopole offset mode (D4-F1). Both amplitude and shape verdicts close the
> branch. The Earth-history reading of the activation — anything where terrestrial or biological
> history *drives the z-dependence* — is theoretically foreclosed, for any transduction
> efficiency. App. B survives only as a bound on the z-independent endpoint offset.

## 2. Background support excluded exactly

By Theorem D3.2 a homogeneous $\bar\psi(t)$ is a frame choice: its chronometric factor
reparameterizes $H(z)$ and cannot produce mapping structure at any order. Combined with D5.1:

> **Corollary D5.2.** The redshift-dependent part of the step-A signal, if physical, lives in
> $\delta\psi_{\rm corr}$ — the spatially inhomogeneous, structure-correlated tier — plus at most
> a constant endpoint offset from $\delta\psi_{\rm loc,obs}$. There is no third option inside
> this theory.

## 3. What the correlated well must deliver

### 3.1 Depth and extent (from the D4 refits)

The kernel refits (D4-F4/F5) demand a **large, soft-edged** well:

| $R_w$ (n=2) | 100 Mpc | 150 Mpc | 200 Mpc |
|---|---|---|---|
| Δχ² | 10.85 | 12.11 | 13.04 |
| required depth εΔψ | 1.9×10⁻³ | 3.2×10⁻³ | 4.6×10⁻³ |

The preference is still rising at $R_w = 200$ Mpc: the data want the imprint accumulating across
the entire HF volume (χ ≈ 100–600 Mpc). This is **larger than the Local Void (~60 Mpc) and
comparable to or larger than Laniakea (~160 Mpc)** — the well is a supervolume-scale feature, not
a neighborhood one. (Equivalently in step-A language: the fitted activation keeps $s>0$ out to
$t_{\rm ret}\sim$ Gyr ↔ χ ~ 500 Mpc.) The two-leg → one-leg timing change (D4 limit theorems)
means the well profile $W(\chi)$, not the g-parameters, is the object to compare across
workstreams; the world-tunnel solution should be expressed as $W(\vec x)$ and validated against
the table above.

### 3.2 Supply side: the two sub-channels of the transfer function

From D3.3, $\;\delta\psi_{\rm corr}(k) = \dfrac{\kappa_\sigma\,\delta\sigma_{\rm corr} +
\epsilon\,\delta T^{(m)}}{f^2(k^2/a^2 + m_{\rm eff}^2)}$, with $m_{\rm eff}^{-1}\gtrsim$ 2 Gpc
required (D4.2) so the $k^2$ term dominates in-range.

**(i) Universal-gravity channel** (conformal coupling to matter density). Using
$\Phi_k = -4\pi G a^2\delta\rho/k^2$ and $8\pi G = M_p^{-2}$:

$$
\epsilon\,\delta\psi_{\rm corr}\big|_{\rm grav} = -\,2\,\alpha^2\,\Phi ,
\qquad \alpha \equiv \frac{\epsilon M_p}{f},
$$

the standard scalar–tensor result: the scalar potential is α² times the Newtonian potential.
Large-scale structure has $\Delta\Phi/c^2 \sim (3\text{–}5)\times10^{-5}$ between typical points
separated by 100–200 Mpc (near-scale-invariant potential fluctuations). Delivering
εΔψ = 4.6×10⁻³ then requires

$$
\alpha^2 \sim \frac{4.6\times10^{-3}}{2\times(4\times10^{-5})} \approx 60
\qquad\Longrightarrow\qquad \alpha_{\rm cosmo} \approx 8 .
$$

Against the unscreened Cassini bound $\alpha \lesssim 3.4\times10^{-3}$ (D7), this channel needs
an environmental screening ratio $\alpha_{\rm cosmo}/\alpha_{\rm local} \gtrsim 2000$.
**Obligation O1** (adjudicated in D7): exhibit a screening mechanism delivering that ratio while
keeping the field's range ≥ Gpc in voids — or retire this channel.

**(ii) Dissipation channel** ($\kappa_\sigma$, the OTA-specific route). Entropy production is
strongly nonlinear in density (shocks, star formation, AGN live in collapsed structures), so
$\delta\sigma_{\rm corr}$ traces $\delta_m$ with a large effective bias; the channel can reach the
required depth with $\alpha$ small if $\kappa_\sigma$ is large enough. The price is set
elsewhere: the same $\kappa_\sigma$ must (a) respect D2's fluctuation bound η ≲ 0.35
(**Obligation O2**), because a strongly-coupled dissipative source is also a noisy one, and
(b) be *universal* — every supervolume then carries wells of the same statistics
(**Obligation O3**, next).

### 3.3 The no-tuning consistency test (O3) — and a genuine cross-check

If the well field is universal, every observer sits in one, and a chronometric well masquerades
as a peculiar-velocity field in any survey that estimates $v = cz_{\rm obs} - H_0 d$: shell
monopoles *and* dipoles of magnitude $\sim c\,\varepsilon\Delta\psi$ appear at every location.
Required depths correspond to

$$
c\,\varepsilon\Delta\psi \approx 570\ {\rm km/s}\ (R_w{=}100)
\quad\text{to}\quad 1380\ {\rm km/s}\ (R_w{=}200).
$$

For a random observer in the well field, apparent bulk flows (dipoles) of a comparable order are
generic. Observed bulk flows at 150–250 Mpc depth are ~250–400 km/s (CosmicFlows-4 era
measurements — themselves already reported as 2–3× *above* the ΛCDM expectation of ~140–150 km/s).
So:

- taken at face value, the required well amplitude overshoots measured flow statistics by a
  factor ~2–4 at $R_w=200$, and is marginal at $R_w=100$–150 — **the correlated branch is
  squeezed from above by velocity-survey statistics**; a well field tuned to be deep here and
  shallow elsewhere violates no-tuning and exits the theory;
- but the *sign* of the situation is notable: the same surveys report an unexplained flow excess
  in roughly the CMB-dipole direction, and the D8 sky split of the step-A amplitude found
  $s_{\rm anchor} = 4.6 \pm 0.75$ toward the CMB dipole apex vs $2.2 \pm 0.7$ away (≈2.3σ from
  profile likelihoods; 1.6σ under the coarser fits — `out/x_decompose.json`). A
  structure-linked chronometric well predicts exactly
  this co-alignment. This is the program's cleanest genuine out-of-sample cross-check and is
  preregistered as D8 test 2.

### 3.4 The honest warning: the offset mode

D4-F2 stands over this whole section: 10.1 of the 13.6 is a structureless monopole offset
(≈ 293 km/s uniform), which the well family reaches in its $R_w\to\infty$ limit. The shape
evidence that specifically prefers a *finite, structured* well over "offset + anything" is
ΔΔχ² ≈ 2.9 (well) to 3.5 (step A) — modest. Until D8's discriminants report, the possibility
that the signal is an offset-type systematic (e.g. a ~300 km/s low-z redshift-frame or
calibration monopole) remains live and is on the D8 kill list. The correlated-well hypothesis
earns the right to be believed by winning the sky-split and shape tests, not by the monopole.

## 4. Statement of the surviving identification

> **The specific nature of the effect (current best formulation).** If the step-A preference is
> physical, it is the past-light-cone section of a chronometric potential of the order sector,
> coherent over ≳150–200 Mpc, correlated with (and grown alongside) the local supervolume's
> structure, read out as an endpoint frequency ratio on every null path. Earth-retarded time was
> light-cone bookkeeping; the two-leg factor was a compact-source artifact; the observer's role
> is location, not history.

**Obligations carried forward:** O1 → D7 (screening window), O2 → D2/D8 (roughness), O3 → D8
(flow statistics + sky split). **Kill status:** compact and background branches closed
(D5.1, D5.2); correlated branch alive under O1–O3; offset-systematic hypothesis alive and
scheduled for execution in D8.
