# D7 — Local constraints, the readout–force lock, and the forced-screening window

**Goal.** Intersect the fitted imprint with local gravity tests, honestly. The centerpiece is a
lock theorem showing that in the current OTA action the chronometric readout and the fifth-force
coupling are the same parameter — after which the constraint chain closes every branch of the
simple action and states exactly what a viable completion must add.

**Inputs.** Action (U.3), D4/D5 fitted amplitudes ($I \equiv \epsilon\Delta\psi \approx
2.7$–$4.6\times10^{-3}$ over $L \approx 150$–200 Mpc), D3 conservation results.

---

## 1. Canonical normalization and the lock theorem

Canonical field $\varphi_c = f\psi$; chronometric factor $A = F^{-1/2} = e^{\epsilon\psi}$;
scalar–matter coupling in the standard scalar–tensor sense

$$
\alpha \equiv M_p\,\frac{d\ln A}{d\varphi_c} = \frac{\epsilon M_p}{f}.
$$

The observable imprint is $I = \Delta\ln A = \epsilon\,\Delta\psi = \alpha\,\dfrac{\Delta\varphi_c}{M_p}$.

> **Theorem D7.1 (readout–force lock).** In the action (U.3), the parameter that converts a field
> excursion into a redshift imprint is the *same* α that sets the scalar-mediated fifth force and
> the PPN deviation ($\gamma - 1 = -2\alpha^2/(1+\alpha^2)$). There is no "clock-only" coupling:
> $A$ and the matter–scalar force both descend from the single function $F(\psi)$. Any fitted
> $I \ne 0$ therefore implies both a force coupling α and a canonical excursion
> $\Delta\varphi_c = I\,M_p/\alpha$. One cannot weaken the force by shrinking α without
> proportionally inflating the required field excursion — which is what the energy argument
> below monetizes.

## 2. The unscreened no-go (both channels, all α)

Two independent requirements collide:

- **Cassini:** $|\gamma-1| \le 2.3\times10^{-5} \Rightarrow \alpha \le 3.4\times10^{-3}$ for an
  unscreened field — and the well needs range $m_{\rm eff}^{-1} \gtrsim 2$ Gpc (D4.2), so there
  is no Yukawa relief at AU scales: the bound applies in full.
- **Energy budget of the well:** the imprint's gradient energy is
  $\rho_{\rm grad} \simeq \left(\dfrac{\Delta\varphi_c}{L}\right)^2 = \left(\dfrac{I\,M_p}{\alpha L}\right)^2$.
  Demanding merely $\rho_{\rm grad} \le \rho_{\rm crit}$ (the well must not outweigh the
  universe) gives
  $$
  \alpha \;\ge\; \frac{I\,M_p}{L\,\sqrt{\rho_{\rm crit}}} \;\approx\; 0.045\text{–}0.06
  $$
  for $(I, L)$ = (2.7×10⁻³, 150 Mpc) – (4.6×10⁻³, 200 Mpc).

> **Theorem D7.2 (unscreened closure).** $\alpha \le 3.4\times10^{-3}$ (Cassini) and
> $\alpha \gtrsim 0.05$ (energy) are contradictory: the unscreened conformal implementation is
> closed for **every** sourcing channel — gravity-sourced *and* dissipation-sourced alike,
> because the lock theorem makes the energy argument channel-blind. At the Cassini-saturating
> α the well would carry $\sim(0.05/0.0034)^2 \approx 2\times10^2\,\rho_{\rm crit}$ in gradient
> energy — excluded by over two orders of magnitude. (This formalizes program §3.4 with the
> D4-fitted numbers.)

Corollary: even in a *screened* completion, the void-scale coupling must satisfy
$\alpha_{\rm cosmo} \gtrsim 0.05$, so the well's gradient energy is at least a non-negligible
fraction of $\rho_{\rm crit}$ over its volume. That energy gravitates: a viable completion must
check the well's own contribution to local dynamics ($H_{\rm local}$, large-scale flows). Flagged
as a required follow-up calculation in the successor task list; potentially a feature (the order
sector is *supposed* to carry cosmological energy in Phase 1), but it must be computed, not
assumed benign.

## 3. Screening-class survey (what can and cannot deliver the ratio)

Required: $\alpha_{\rm local}^{\rm eff} \le 3.4\times10^{-3}$ in the solar neighborhood while
$\alpha_{\rm cosmo} \gtrsim 0.05$ (dissipation channel) or $\approx 8$ (gravity channel, D5-O1),
with force range ≥ 2 Gpc in voids.

- **Chameleon / symmetron / density-dependent-mass class: closed.** The Wang–Hui–Khoury theorem
  bounds the Compton wavelength of any chameleon-like field at cosmological density to
  ≲ 1 Mpc — three orders short of the required Gpc range. The mechanism that screens the solar
  system by raising $m_{\rm eff}(\rho)$ necessarily kills the long-range well. No parameter
  tuning escapes this; it is structural.
- **Vainshtein / derivative-screening class: the only open door.** Kinetic self-interactions
  (braiding/galileon-type operators added to the action) screen locally by derivative
  suppression while leaving cosmological-range effects intact — the right *shape* of
  environment dependence. The door is narrow: GW170817 ($c_T = c$) eliminates most of the
  quartic/quintic structure; Lunar Laser Ranging constrains residual Vainshtein leakage; ISW
  cross-correlations have already killed the self-accelerating cubic galileon. A concrete
  candidate operator set clearing all three is the **defined successor model-building task**;
  until it exists, the theory's viability is conjectural at exactly this point.
- **"Clock-only coupling" escape: closed by Theorem D7.1** within the present action. Evading
  the lock requires coupling $A(\psi)$ to a sector that defines clock rates but carries no
  stress — not available in a diffeomorphism-invariant action where clocks are made of matter.

## 4. Secondary constraints (for the eventual screened candidate)

- **$\dot G_{\rm eff}/G$:** $G_{\rm eff} \propto F^{-1} = A^2 \Rightarrow \dot G/G =
  2\epsilon\dot{\bar\psi}$. A Phase-1 roll of $\epsilon\Delta\bar\psi$ per Hubble time gives
  $\dot G/G \sim 1.4\times10^{-10}\,(\epsilon\Delta\bar\psi)/{\rm yr}$; LLR's
  $|\dot G/G| \lesssim 10^{-13}/{\rm yr}$ then demands $\epsilon\Delta\bar\psi \lesssim 10^{-3}$
  per Hubble time in unscreened form — the same order as the fitted spatial imprint, so the
  background roll and the well depth are jointly squeezed; a screened candidate defers this to
  its screened value, which must be computed.
- **Equivalence principle:** tree-level safe — matter couples minimally to $g_{\rm J}$, so free
  fall is composition-blind regardless of how state-dependent the *sourcing* is (D2's
  equilibrium protection lemma covers the source side; the force side is pure α). MICROSCOPE's
  $\eta < 10^{-15}$ constrains composition-dependent couplings arising at loop level — the
  standard dilaton problem, inherited unchanged; a universal-coupling ansatz must be protected
  by a symmetry in the eventual completion.
- **Laboratory clock ratios:** null by universality (all clocks share $A$); only the loop-level
  non-universalities above could surface here, at or below their MICROSCOPE-constrained size.

## 5. Verdict

| Branch | Status | Reason |
|---|---|---|
| Unscreened conformal (any sourcing) | **closed** | Theorem D7.2 (Cassini × energy) |
| Chameleon-class screening | **closed** | range no-go (≤ Mpc at cosmic density) |
| Vainshtein-class kinetic completion | **open, conjectural** | must clear $c_T$, LLR, ISW; not yet constructed |
| Clock-only readout | **closed** | Theorem D7.1 (readout–force lock) |

The honest summary for the paper: *the current OTA action cannot host its own best-fit signal;
the signal, if physical, points to a derivative-screened extension of the order sector, and
constructing (or excluding) that extension is now the theory's critical path.* This is the
program's kill-criterion machinery working as designed — it did not kill the phenomenon; it
killed the simplest theory of it and named the unique surviving direction.
