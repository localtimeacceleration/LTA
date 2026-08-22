# D2 — Open-EFT derivation of the order-sector dynamics (Schwinger–Keldysh)

**Goal.** Derive — rather than posit — the sourced field equation for ψ, its retarded/causal
structure, the constitutive law `J_neq = κ_σ σ`, and the unavoidable stochastic companion with its
observational consequence. This closes gaps G3 (κ_σ posited) and G5 (retarded boundary conditions
assumed) of the program ledger and produces the first quantitative parameter-space cut.

**Inputs.** D1's separation of routes and its source object σ_s(x); the paper's Eq. (psi_kg)
as the target of the Markovian limit; standard in-in (closed-time-path, CTP) effective-action
technology.

---

## 1. Setup: what ψ is, microscopically

Let ψ(x) be the block-averaged order density of D1 §5 (per App. A.4 normalization,
ψ = φ_loc/φ*), promoted to a collective coordinate: at each coarse-graining cell, ψ is a
function of the microscopic configuration, and we seek the effective dynamics of its long-wavelength
modes after integrating out everything else (matter fields, radiation, short-wavelength modes),
collectively "the environment" Ξ.

The bare EFT action for ψ is the one already adopted in the paper (App. U.3, restricted here to
flat space; gravity returns in D3):

$$
S_\psi[\psi] = \int d^4x\left[ -\tfrac{f^2}{2}(\partial\psi)^2 - f^4 U(\psi)\right],
$$

and the system–environment coupling is taken linear in ψ at leading order in the derivative/field
expansion:

$$
S_{\rm int} = g\int d^4x\; \psi(x)\, \mathcal{O}(x),
$$

where 𝒪(x) is a scalar composite operator of the environment. The physical identification —
the *one* modeling input of D2, replacing the paper's posited constitutive law — is:

> **(C) Coupling hypothesis.** 𝒪 is the local operator whose nonequilibrium expectation value
> measures sustained organized dissipation; in a driven steady state,
> ⟨𝒪(x)⟩_neq = c_𝒪 · σ_s(x) + O(σ²), where σ_s is D1's entropy-production density and c_𝒪 is
> an environment transport coefficient (computed below, §4). In equilibrium ⟨𝒪⟩_eq = 0 (𝒪 is
> defined with its equilibrium value subtracted).

Everything else below is *derived*.

## 2. CTP effective action and the emergence of retarded structure

On the closed-time-path contour, the field is doubled (ψ₊ forward branch, ψ₋ backward), and the
influence of Ξ is contained in the influence functional
$F[\psi_+,\psi_-] = \langle T_C \exp\{ \tfrac{i}{\hbar} g\int_C \psi\,\mathcal{O}\}\rangle_\Xi$.
To second order in the cumulant expansion (Feynman–Vernon):

$$
S_{\rm IF} = g\!\int\! d^4x\, \psi_\Delta \langle \mathcal{O}\rangle
+ \frac{g^2}{2}\!\int\! d^4x\, d^4x' \left[
2\,\psi_\Delta(x)\, G_R^{\mathcal O}(x,x')\, \psi_c(x')
+ \frac{i}{\hbar}\,\psi_\Delta(x)\, N(x,x')\, \psi_\Delta(x')
\right],
$$

in Keldysh variables $\psi_c = (\psi_+ + \psi_-)/2$, $\psi_\Delta = \psi_+ - \psi_-$, with

$$
G_R^{\mathcal O}(x,x') = -\frac{i}{\hbar}\,\theta(t-t')\,\big\langle[\mathcal{O}(x),\mathcal{O}(x')]\big\rangle,
\qquad
N(x,x') = \frac{1}{2}\,\big\langle\{\delta\mathcal{O}(x),\delta\mathcal{O}(x')\}\big\rangle .
$$

Varying with respect to $\psi_\Delta$ (and Hubbard–Stratonovich-ing the noise kernel) yields the
effective Langevin equation for the physical field $\psi \equiv \psi_c$:

$$
\boxed{\;
f^2\big(\Box - m_0^2\big)\psi(x) \;-\; g^2\!\int\! d^4x'\, G_R^{\mathcal O}(x,x')\,\psi(x')
\;=\; -\,g\,\langle\mathcal{O}(x)\rangle_{\rm neq} \;-\; \xi(x),
\qquad
\langle \xi(x)\xi(x')\rangle = g^2 N(x,x') .\;}
$$

> **Theorem D2.1 (causality is derived, not imposed).** The self-energy kernel entering the
> equation of motion is *retarded* — proportional to θ(t−t′) — as an automatic consequence of the
> CTP contour ordering. The paper's "retarded boundary conditions" (Secs. psi_dynamics_eq, App. V)
> are therefore not an assumption of the completed theory; they are the only structure the in-in
> formalism permits for an open system. Gap G5 is closed.

> **Theorem D2.2 (no source without noise).** The same coupling g that lets the environment
> source ψ (through ⟨𝒪⟩ and Im G_R) forces the noise ξ with kernel g²N. A silent (noise-free)
> constitutive source is impossible. Every viable OTA parameter point therefore predicts a
> stochastic component of the imprint, quantified in §5.

## 3. Markovian limit: recovery of the paper's field equation

When the environment correlation time τ_c is short compared to ψ's dynamical times, expand
$\psi(x')$ about $x$ inside the self-energy integral:

- the even/real part of $g^2 G_R^{\mathcal O}$ renormalizes the mass,
  $m_\psi^2 = m_0^2 + \delta m^2$, and the kinetic normalization;
- the odd part produces local friction $f^2\gamma\,\dot\psi$ with
  $\gamma = -\lim_{\omega\to0}\, g^2\,{\rm Im}\,\tilde G_R^{\mathcal O}(\omega,\vec k\!\to\!0)/(f^2\omega)\;\ge 0$;
- the source term becomes, with hypothesis (C),
  $-g\langle\mathcal{O}\rangle_{\rm neq} = -\,g\,c_{\mathcal O}\,\sigma_s(x) \equiv \kappa_\sigma\,\sigma_s(x)$
  (sign conventions as in the paper).

$$
\Rightarrow\qquad
(\Box - m_\psi^2)\,\psi - \gamma\,\dot\psi \;=\; \kappa_\sigma\,\sigma_s(x)/f^2 \;+\; \xi/f^2 ,
$$

which is exactly Eq. (psi_kg) of the paper plus the two derived corrections: friction γ and noise
ξ. The paper's equation is the γ→0, ξ→0 limit — legitimate for the *mean-field* imprint, but the
corrections carry the new physics content (γ controls how the well tracks its source; ξ sets the
imprint's roughness).

## 4. The constitutive coefficient as a Kubo formula (gap G3 closed)

The coefficient c_𝒪 in hypothesis (C) is an ordinary transport coefficient of the environment:
linear response of ⟨𝒪⟩ to the thermodynamic force X that drives the steady state (affinity per
unit volume, with entropy production σ_s = X·J_X):

$$
c_{\mathcal O} = \lim_{\omega\to 0}\frac{1}{T_{\rm eff}}\int_0^\infty\! dt\, e^{i\omega t}
\int d^3x\,\big\langle \delta\mathcal{O}(\vec x, t)\; \delta \hat\jmath_X(0,0)\big\rangle_{\rm NESS},
\qquad
\kappa_\sigma = -\,g\,c_{\mathcal O}.
$$

Thus κ_σ is not a free postulate: it is g times a computable NESS correlation function. What
remains genuinely free in the EFT is the pair (g, choice of 𝒪) — as it should be: that is the
theory's coupling constant, constrained by data (D4/D5) and by consistency (D7), not by
bookkeeping.

**Equilibrium protection.** In exact equilibrium the NESS correlator of 𝒪 with any current
vanishes (⟨𝒪⟩_eq = 0 and detailed balance kills the odd correlator), so κ_σ·σ_s → 0: undriven
matter does not source ψ. This derives the paper's convention "all LTA phenomenology is sourced by
nonzero ψ in *driven* material environments" (App. A conventions) instead of assuming it.

## 5. Fluctuation–dissipation and the stochastic imprint bound

For an environment near local equilibrium at temperature T (or a NESS with effective temperature
T_eff), the kernels obey the FDT:
$\tilde N(\omega,\vec k) = \coth\!\big(\tfrac{\hbar\omega}{2k_BT_{\rm eff}}\big)\,{\rm Im}\,\tilde G_R^{\mathcal O}(\omega,\vec k)$.
In the classical limit, the stationary variance of the sourced field about its mean, per mode,
is thermal-like, and integrating modes up to the EFT cutoff gives a field variance
$\varsigma_\psi^2 \equiv \langle\delta\psi^2\rangle$ set by (γ, T_eff, m_eff, f). Rather than
committing to a cutoff, parameterize the *fractional roughness* of the well,
$\eta \equiv \varsigma_\psi / \Delta\psi_{\rm well}$, which the microphysics must deliver.

**Propagation to the Hubble diagram.** The imprint is $I = \epsilon[\psi_{\rm obs}-\psi_{\rm emit}]$.
For emitters separated by more than the field correlation length, the fluctuating parts of
$\psi_{\rm emit}$ are independent draws, so each SN acquires an independent redshift scatter
$\delta I \simeq \sqrt2\,\epsilon\,\varsigma_\psi = \sqrt2\,\eta\,(\epsilon\Delta\psi_{\rm well})
= \sqrt2\,\eta\, I_{\rm sat}$. At fixed observed z this maps to a distance-modulus scatter

$$
\sigma_\mu(z) \simeq \frac{5}{\ln 10}\,\frac{(1+z)^2}{H(z)\,d_L(z)}\; c\,\delta I
\;\xrightarrow[z\ll1]{}\; \frac{5}{\ln 10}\,\frac{\delta I}{z}.
$$

At the fiducial $I_{\rm sat}=2.67\times10^{-3}$ and the SH0ES-HF lower edge $z\simeq0.023$:

$$
\sigma_\mu(0.023) \simeq \frac{5}{\ln10}\cdot\frac{\sqrt2\,\eta\cdot 2.67\times10^{-3}}{0.023}
\simeq 0.36\,\eta\ \ {\rm mag}.
$$

The Pantheon+ low-z error budget leaves at most ~0.10–0.15 mag of unmodeled scatter, and the
fiducial fit shows no excess ($\chi^2/{\rm dof} \approx 318.9/350 \approx 0.91$ — see
`numerics/out/t1_reproduce.json`). Hence:

> **Bound D2.3 (well smoothness).** $\eta \lesssim 0.3$–0.4: the order-sector well sampled by the
> SN endpoints must be smooth at the ≳3:1 signal-to-roughness level on the field's correlation
> scale. Any microphysical parameter point (g, γ, T_eff, m_eff, f) whose FDT variance violates
> this is excluded — the kill criterion of D2, now operational. Conversely this disfavors any
> "turbulent/patchy order field" variant of OTA at the fitted amplitude.

## 6. Outputs handed onward

| Object | Where used |
|---|---|
| Retarded kernel $G_R$ with θ-function causality (Theorem D2.1) | D4 (FLRW kernel) |
| Constitutive law $\kappa_\sigma = -g\,c_{\mathcal O}$, Kubo form (§4) | D5 (amplitude accounting) |
| Friction γ (well lags source on timescale γ⁻¹) | D5 (growth-history mapping) |
| Noise ξ, FDT, roughness bound η ≲ 0.3–0.4 (Bound D2.3) | D8 test 4; D7 parameter cuts |
| Equilibrium protection (§4) | D7 (EP discussion: undriven matter is inert) |

**Validation.** The Markovian, noise-free limit is exactly the paper's Eq. (psi_kg) ✓. The
sourced retarded solution used in App. V is the mean-field sector of the boxed equation ✓.

**Status of the kill criterion.** Not triggered at the level of available data (no excess low-z
scatter observed; η-bound is a constraint, not a contradiction). It becomes sharper with any
future dataset extending below z≈0.02, where σ_μ(z) ∝ 1/z amplifies the roughness signal.
