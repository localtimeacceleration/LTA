# D3 — Covariant embedding and the exact three-tier decomposition

**Goal.** Embed D2's open dynamics covariantly, verify stress-energy bookkeeping (Bianchi
consistency and the exchange vector), and upgrade the program's three-tier language — background /
correlated / localized — from decomposition-by-fiat to theorem-level structure. Two exact results
do most of the work: the *frame-redefinition theorem* (which makes the §3.2 background exclusion
exact) and the *tier superposition theorem* with its correlated-tier transfer function (the formal
interface for the world-tunnel workstream).

**Inputs.** The paper's action (Eq. U_action / JF_action) and field equations (U.4); D2's derived
source, friction, and noise; standard scalar–tensor bookkeeping.

---

## 1. Action, currents, and conservation

Jordan frame (matter minimally coupled to $g^{\rm J}$):

$$
\mathcal S = \int d^4x \sqrt{-g_{\rm J}}\left[\frac{M_p^2}{2}F(\psi)R_{\rm J}
- \frac{f^2}{2}(\nabla\psi)^2 - f^4 U(\psi)\right]
+ \mathcal S_m[g_{\rm J},\Phi] + \mathcal S_{\rm neq}[g_{\rm J},\psi,\Xi].
$$

Define the nonequilibrium-sector currents by variation:

$$
T^{(\rm neq)}_{\mu\nu} \equiv -\frac{2}{\sqrt{-g}}\frac{\delta \mathcal S_{\rm neq}}{\delta g^{\mu\nu}},
\qquad
J_{\rm neq} \equiv -\frac{1}{\sqrt{-g}}\frac{\delta \mathcal S_{\rm neq}}{\delta\psi},
\qquad
E_\Xi \equiv \frac{\delta \mathcal S_{\rm neq}}{\delta \Xi}.
$$

**Generalized conservation.** Diffeomorphism invariance of $\mathcal S_{\rm neq}$ alone gives the
off-shell identity

$$
\nabla^\mu T^{(\rm neq)}_{\mu\nu} = -\,J_{\rm neq}\,\nabla_\nu\psi \;+\; E_\Xi\cdot\pounds_\nu\Xi ,
$$

i.e. the neq sector exchanges momentum with ψ (first term) and with its own microscopic carriers
(second term, vanishing on the Ξ shell). Meanwhile $\mathcal S_m$ is separately diffeo-invariant
and minimally coupled, so $\nabla^\mu T^{(m)}_{\mu\nu}=0$ *identically in the Jordan frame* —
matter test bodies follow $g_{\rm J}$ geodesics, which is what protects the equivalence principle
at tree level (used in D7). Taking the divergence of the Einstein equation (U.4) and using the
scalar equation

$$
f^2\Box\psi - f^4U' + \frac{M_p^2}{2}F'R = J_{\rm neq}
$$

one verifies that all remaining divergences cancel: the $\nabla^\mu(\nabla_\mu\nabla_\nu F -
g_{\mu\nu}\Box F)$ terms combine with $\tfrac{M_p^2}{2}F'R\,\nabla_\nu\psi$ and the ψ-kinetic
divergence to reproduce $-J_{\rm neq}\nabla_\nu\psi$, matching the neq identity above.

> **Result D3.1 (Bianchi consistency).** The system (Einstein + scalar + neq + matter) is
> consistent for *any* $\mathcal S_{\rm neq}$; no on-shell tuning is needed. What the choice of
> $\mathcal S_{\rm neq}$ does control is the exchange vector
> $Q_\nu \equiv -J_{\rm neq}\nabla_\nu\psi$ — energy pumped between the order sector and the
> dissipative matter sector. On FLRW, $Q_0 = -J_{\rm neq}\dot{\bar\psi}$: the Phase-1 background
> continuously extracts (or deposits) energy at this rate, which is exactly the term the paper's
> App. W.4 tracks. D2's friction γ and noise ξ sit inside $J_{\rm neq}$ as its dynamical parts.

**Kill-criterion check (from the program):** the exchange $Q_0$ modifies the background only
through the already-present W.4 scalings, so the Phase-1 fit is not destabilized by the D2
completion. Not triggered.

## 2. The frame-redefinition theorem (exact background exclusion)

The chronometric factor is fixed by the action: $A(\psi)=F^{-1/2}(\psi)$, and realized clock time
obeys $d\tau_{\rm clk} = A(\psi)\,d\tau$. Consider the *homogeneous* piece $\bar\psi(t)$ and
define along comoving worldlines

$$
d\tilde t \equiv A(\bar\psi(t))\,dt,
\qquad
\tilde a(\tilde t) \equiv a(t)\,A(\bar\psi(t)).
$$

For comoving emitter and observer, the clock-measured redshift is
$1+z_{\rm obs} = \dfrac{a(t_0)A(\bar\psi_0)}{a(t_e)A(\bar\psi_e)} = \dfrac{\tilde a(\tilde t_0)}{\tilde a(\tilde t_e)}$,
and radial null rays satisfy $c\,dt/a = c\,d\tilde t/\tilde a$, so comoving distances, conformal
structure, and hence *every* distance–redshift and clock–rate observable of the homogeneous
sector are those of an FRW model with scale factor $\tilde a(\tilde t)$.

> **Theorem D3.2 (homogeneous chronometric factor is pure frame).** A spatially uniform
> $A(\bar\psi(t))$ is exactly absorbed by the redefinition $(a,t)\to(\tilde a,\tilde t)$: it is a
> reparameterization of the expansion history, not an observable imprint. The homogeneous tier
> affects observations **only** through its stress-energy backreaction on $\tilde H(z)$ (the
> Phase-1 channel). Consequently the program's §3.2 background exclusion is exact, not
> approximate: no choice of $\bar\psi$ history can generate the low-z mapping signal, and the
> "background-growth" reading of step A is closed permanently. Only *spatial inhomogeneity* of ψ
> is observable in the mapping channel.

(This also cleans up the paper's Sec. "Universal law vs. observer-local limit": the universal
homogeneous part was never testable in the mapping; the paper's choice to fit only the local
imprint was forced, not merely convenient.)

## 3. Tier decomposition as a theorem

Decompose the D2 source by support and statistics:

$$
\sigma_s(x) = \bar\sigma(t) \;+\; \delta\sigma_{\rm corr}(x) \;+\; \delta\sigma_{\rm loc}(x),
$$

- $\bar\sigma$: ensemble mean (homogeneous);
- $\delta\sigma_{\rm corr}$: the component correlated with large-scale structure, coherent on
  10–10² Mpc, vanishing ensemble mean;
- $\delta\sigma_{\rm loc}$: sub-Mpc environments (individual galaxies, hosts, the observer's own
  neighborhood), uncorrelated between distinct endpoints.

Because the D2 equation of motion is linear in ψ at EFT order (nonlinearities enter through
$U''(\bar\psi)$ only as the tier-dependent effective mass), the retarded solution superposes:

> **Theorem D3.3 (tier superposition).** $\psi = \bar\psi + \delta\psi_{\rm corr} + \delta\psi_{\rm loc}$
> with each tier sourced by its own $\sigma$-component through the same retarded kernel, and
> cross-terms entering only at $O(U'''\,\delta\psi^2)$ — bounded and negligible for
> $\epsilon\,\delta\psi \sim 10^{-3}$ unless $U$ is pathological. The sharpened statement's
> "retarded superposition of background, correlated, and localized responses" is thereby the
> literal mathematical structure of the solution, not an analogy.

**Tier equations.**

1. **Background:** the paper's Phase-1 system (JF_friedmann / JF_scalar) with source $\bar\sigma$
   — unchanged, plus Theorem D3.2 governing what it can and cannot imprint.
2. **Correlated (the world-tunnel object):** linearizing about the background in the quasi-static
   subhorizon regime, with the conformal coupling bringing in matter density perturbations:

   $$
   \left(\frac{\nabla^2}{a^2} - m_{\rm eff}^2\right)\delta\psi_{\rm corr}
   = -\frac{\kappa_\sigma}{f^2}\,\delta\sigma_{\rm corr}
     \;-\;\frac{\epsilon}{f^2}\,\delta T^{(m)} ,
   \qquad m_{\rm eff}^2 = f^2 U''(\bar\psi),
   $$

   giving the **transfer function**

   $$
   \delta\psi_{\rm corr}(\vec k, t)
   = \frac{\kappa_\sigma\,\delta\sigma_{\rm corr}(\vec k,t) + \epsilon\,\delta T^{(m)}(\vec k,t)}
          {f^2\left(k^2/a^2 + m_{\rm eff}^2\right)} .
   $$

   Two consequences worth flagging now: (i) even with κ_σ→0, the conformal coupling *alone*
   makes ψ trace the density field — a chronometric well is a generic consequence of living in a
   large-scale density fluctuation in this theory; (ii) the D2 friction γ makes
   $\delta\psi_{\rm corr}$ lag its source on timescale γ⁻¹, which is the microscopic origin of the
   "growth history" $D_\psi(t)$ in the D5 factorization $\delta\psi_{\rm corr} \approx W(\vec x)\,D_\psi(t)$.
3. **Localized:** the Yukawa endpoint solution of App. U.6, restricted to sub-Mpc support. Its
   observational weight is an endpoint term uncorrelated across the sample — tested and found
   absent at current sensitivity by the D6 host-split (see `numerics/out/d6_host_timing.json`),
   which bounds this tier rather than the theory.

## 4. The imprint, tier by tier

With $\mathcal C = \ln A = \epsilon\psi$ (paper convention, U.7):

$$
I = \underbrace{\epsilon[\bar\psi(t_0)-\bar\psi(t_e)]}_{\text{pure frame (Thm D3.2): absorbed in } \tilde H(z)}
+ \underbrace{\epsilon[\delta\psi_{\rm corr}(x_{\rm obs})-\delta\psi_{\rm corr}(x_{\rm emit})]}_{I_{\rm corr}(\chi,\hat n):\ \text{the step-A signal candidate}}
+ \underbrace{\epsilon[\delta\psi_{\rm loc,obs}-\delta\psi_{\rm loc,emit}]}_{I_{\rm loc}:\ \text{endpoint offset + per-object scatter}}
$$

- $I_{\rm corr}$ inherits direction dependence from $W(\vec x)$ — the D8 sky-split channel.
- The observer half of $I_{\rm loc}$ is a constant across the sample. Note from the D4 numerics
  that a pure constant-I mode already captures Δχ² ≈ 10.1 of the step-A 13.6 — so a large share
  of the fitted preference is attributable to *any* mechanism (or systematic) that produces a
  uniform ln(1+z) offset ≈ 10⁻³ of the Hubble-flow sample relative to the calibrator-anchored
  prediction. This altitude matters when weighing mundane alternatives (D8).
- The emitter half of $I_{\rm loc}$ is per-object scatter, bounded by D2's η-limit and by the
  D6 host-correlation null.

## 5. Outputs and validation

| Output | Consumer |
|---|---|
| Exchange vector $Q_\nu$ and Bianchi consistency (D3.1) | Phase-1 stability; App. W.4 |
| Frame-redefinition theorem (D3.2) | closes the background branch exactly; paper Sec. 5 cleanup |
| Tier superposition + transfer function (D3.3) | D5 (well profile), world-tunnel interface |
| Tier-resolved imprint (§4) | D4/D5 forward models; D8 test design |

**Validation.** Phase-1 equations and the anomaly prior are untouched ✓. U.6's Yukawa limit is
tier 3 ✓. The transfer function reproduces Eq. (U_perturbation_eq) in the appropriate limit ✓.

**Kill criterion.** None triggered: the exchange-vector structure is exactly the one the paper's
W.4 already carries. The load-bearing consequence is Theorem D3.2 — from here on, every viable
reading of step A lives in $\delta\psi_{\rm corr}$ (plus a possible constant offset), and D5's
job is to decide whether local large-scale structure can supply it.
