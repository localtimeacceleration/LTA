# OTA Derivation Program: from constrained quantum states to the LTA observables

**Status:** working theory document (not paper text).
**Evidence anchor:** repo state at branch point from `main` (run-tag `20260112-013808`; paper `lta_paper.tex` as of commit `0349692`).
**Purpose:** sharpen the OTA conceptual statement and lay out the ordered series of microscopic
derivations needed to replace the observer-retarded ("horizon") approximation of step A with a
derived mechanism, with each derivation tied to a validation hook in the existing pipeline and an
explicit kill criterion.

---

## 0. Scope, provenance, and naming

**Naming convention used here.** *LTA* refers to the phenomenological redshift mapping and its
SN+BAO inference (step A): `1+z_obs = (1+z_cos) exp[I(χ)]` with `s(χ) = c dI/dχ` evaluated through
the Earth-retarded activation `g(t_ret)`. *OTA* (order → time acceleration) refers to the theory
layer beneath it: the order sector `ψ`, its sources, its dynamics, and the chronometric readout.
Step A is an LTA result; this document is the OTA program.

**What this document is anchored to.** Everything below builds on what is actually in this
repository:

- Fiducial step-A result (run-tag `20260112-013808`): `Δχ²_tot = 13.652`
  (χ² 332.549 → 318.897), `p_MC ≈ 2×10⁻³` against 10,000 parametric-bootstrap nulls,
  AIC/BIC favor LTA at k=3→4.
- Best fit: `s_anchor = 2.449 km/s/Mpc`, `t_A = 0.670 Gyr`, `g(t_A) = 0.493`,
  `s_now ≈ 4.97 km/s/Mpc`, `I_sat ≈ 2.67×10⁻³`; background `(H₀, Ω_m, α_rd)` essentially
  unshifted between ΛCDM and ΛCDM+LTA.
- Robustness: telescope-group jackknife `Δχ²₍₋g₎ ∈ [12.27, 21.37]` (median 13.73; removing CfA
  *increases* the preference); improvement concentrated at `z ≲ 0.03` (the "red valley").
- The negative result: conditional CV is `−7.78` (marginal CV `+113.72` is dominated by
  correlated-covariance bookkeeping, not predictive skill). No out-of-sample discriminant has yet
  been demonstrated.

**Not in this repository:** the runs from the last few weeks, including the configuration quoted at
`Δχ² ≈ 10.8`. The last pushed commit is from March. Nothing below depends on whether the reference
number is 13.65 or 10.8 — both are one-parameter amplitude detections of a single coherent low-z
mode under a fixed activation shape — but the ledger in §2 has explicit slots that should be filled
when those runs are pushed. **Action item: push the recent run outputs (or at least the config +
summary tables) so the program can be re-anchored to the current conservative configuration.**

---

## 1. The framing sentence, sharpened

### 1.1 Current statement and its three defects

> "Ordered configurations have constrained quantum states which change the additional causal
> stress response carried by the order sector. The universe's geometry and every observed signal
> reflect the retarded superposition of the resulting background, correlated, and localized
> responses across cosmic history."

This is close to right in architecture but not yet a statement one can derive from. Three specific
problems:

1. **Circularity / category slip in sentence 1.** "Constrained quantum states *change the
   additional* causal stress response" presupposes the response it is trying to introduce, and it
   conflates two distinct physical routes that the derivations must keep separate:
   (a) a constrained (relative-entropy-displaced) state necessarily carries **modular energy** —
   by the exact identity `ΔK_ref = S_rel + ΔS` — and modular energy gravitates through the ordinary
   semiclassical channel `⟨T_μν⟩`; this is not a new force and needs no new sector.
   (b) the **order sector** `ψ` is a distinct, emergent collective field whose *state* is driven by
   organized dissipation and which couples to matter clocks conformally. Route (b) is where all LTA
   phenomenology lives. Relative entropy is *not itself a source* in semiclassical gravity; if `ψ`
   is sourced by `S_rel`-density, that is a **constitutive claim about open-system dynamics** and
   must be derived as such (D2), not postulated as a new charge.

2. **"Carried by the order sector" is ambiguous.** It can mean (i) the order sector's own
   stress-energy `T^(ψ)_μν + T^(neq)_μν` (which backreacts on geometry — Phase 1), or (ii) the
   chronometric readout `A(ψ)` that biases clock rates without touching null geodesics (Phase 2).
   These have completely different observational signatures and constraints. The sharpened
   statement must name both.

3. **The "correlated" tier has no object in the current formalism.** The paper's decomposition is
   two-tier: `ψ = ψ̄(t) + δψ(x)` with `δψ` observer-local. The framing sentence correctly demands
   three tiers — background, correlated, localized — and §3 below shows the microscopic analysis
   *forces* the correlated tier to carry the low-z signal. Naming it forces the derivation (D3).

### 1.2 Sharpened statement (proposed replacement)

> **An ordered configuration is one whose local reduced state is displaced from its
> maximum-entropy reference; the displacement is measured by relative entropy and, through the
> first law of entanglement, necessarily carries modular energy. The order sector ψ is the
> coarse-grained collective field conjugate to this displacement: maintaining the displacement
> requires dissipation, and ψ responds to the resulting entropy-production density through a
> retarded, dissipative response kernel derived from its open-system dynamics — causality is a
> consequence of the in-in structure of that derivation, not an assumption. The order sector acts
> on observables through two channels: its stress-energy, which backreacts on geometry, and its
> conformal coupling to matter, which rescales realized clock time. Every observable is then the
> same retarded superposition of this response over the source history, in three tiers: the
> homogeneous tier sets the background expansion (Λ-replacement, Phase 1); the correlated tier,
> sourced by the ordered growth of large-scale structure, sets spatially coherent chronometric
> wells; and the localized tier, sourced by each observer's and emitter's immediate environment,
> sets endpoint terms. The step-A LTA profile is the past-light-cone section of the correlated
> tier's local well, with Earth-retarded time acting as light-cone bookkeeping — not evidence that
> the source is the observer.**

The last clause is the substantive sharpening, and it is not aesthetic — it is forced by the three
structural results in §3.

### 1.3 Dictionary

| Phrase in statement | Formal object | Where derived | Paper anchor |
|---|---|---|---|
| ordered configuration | `ρ_A` with `S_rel(ρ_A‖ρ_ref) > 0` on a causal diamond | D1 | Eq. (U_Srel_def), App. A.1 |
| carries modular energy | `ΔK_ref = S_rel + ΔS`; `δ⟨K⟩ = δS` linearized | D1 | new |
| order sector ψ | coarse-grained collective field, `ψ = φ_loc/φ*` | D1→D2 | Eq. (app_psi_def), U.1 |
| retarded dissipative response | in-in effective equation `(□−m²_eff)ψ + Σ_R ∗ ψ = κ_σ σ + ξ` | D2 | replaces Eq. (psi_kg) |
| stress channel | `T^(ψ)_μν`, `T^(neq)_μν`, exchange `Q^ν` | D3 | U.4, W.4 |
| clock channel | `A(ψ) = F(ψ)^{−1/2}`, `dτ_clk = A dτ` | D3 (fixed by U.3) | U.3, Lemma 4 |
| retarded superposition | `ψ = G_ret ∗ (κ_σ σ)` with full FLRW kernel | D4 | upgrades App. V |
| three tiers | `σ = σ̄(t) + δσ_corr(x) + δσ_loc(x)`, mirrored in ψ | D3, D5 | upgrades Eq. (Jneq_split_main) |
| light-cone bookkeeping | degeneracy theorem: `t_ret ↔ χ` locked on the cone | §3.3, D5 | App. V.4 |

---

## 2. Evidence ledger: what the existing runs establish, and what they do not

### 2.1 Established (in-repo)

| # | Result | Where |
|---|---|---|
| E1 | One-parameter LTA amplitude improves the joint SN+BAO fit by `Δχ² ≈ 13.7`, calibrated `p_MC ≈ 2×10⁻³` on 10⁴ full-covariance mocks | §9, §10.2 |
| E2 | The preference is a coherent low-z structure (`z ≲ 0.03` red valley), not spread thin | §9.4 |
| E3 | Not driven by any single telescope group; CfA removal strengthens it | §10.1 |
| E4 | Background `(H₀, Ω_m, α_rd)` unshifted → the effect is in the *mapping*, not the expansion history | §9.3 |
| E5 | BAO Jacobian consistency holds at the fitted amplitude (radial ruler not violated) | §9.6 |
| E6 | Second-law envelope: `I_sat ~ 10⁻³` does not exceed Earth-system throughput accounting *given a free transduction parameter* `ε_tc ~ O(0.1–1)` | App. B |
| E7 | Phase-1 anomaly prior `Ω_ψ = |δ_tot|/(6α_tot)` links the background tier to the trilogy inputs | App. A.0 |

### 2.2 Not established / negative / by-hand (each one is an obligation on the program)

| # | Gap | Consequence for the program |
|---|---|---|
| G1 | **Conditional CV is negative (−7.78).** No out-of-sample predictive win. | The program must produce *new preregistered discriminants* (D8), not refits of the same mode. A full-data Δχ² of this size for one boundary-constrained parameter will not persuade anyone on its own — correctly so. |
| G2 | **Amplitude factors of the Green's function are dropped** ("folded into g(t)", App. V.5 item 4). | This is not a fold; restoring `1/χ`, Yukawa, and scale factors changes the χ-shape qualitatively. §3.1 shows restoring them *excludes* the compact-source branch outright. |
| G3 | `κ_σ` (constitutive law `J_neq = κ_σ σ`) is posited, not derived. | D2 (Kubo formula). |
| G4 | `ε_tc ~ O(1)` in the Earth-budget closure is put in by hand. | Superseded: §3.1 shows the literal-Earth branch is dead regardless of `ε_tc`; App. B survives only as a consistency check of the *localized endpoint* tier, not of the z-shape. |
| G5 | Retarded boundary conditions are assumed. | D2 derives them (in-in contour); bonus: a fluctuation–dissipation noise companion with testable consequences. |
| G6 | The mundane alternative (coherent local peculiar-velocity / local-void structure under ΛCDM) has not been fit head-to-head with the same dof. | D8, mandatory. If a linear-theory bulk-flow/void model with one amplitude parameter matches E1–E2, the LTA-specific claim collapses. |
| G7 | The `Δχ² ≈ 10.8` conservative configuration and all post-March runs are not in the repo. | Push them; fill this table's slots; re-anchor benchmarks in D4/D8. |
| G8 | Time-dilation channel asserted as a "falsification channel" but the model's prediction is not yet derived. | D6 theorem 1 (it comes out *safe*, see below — worth having as a theorem, since it is the first thing a referee checks). |

---

## 3. Three structural results that reorganize the program

These are short derivations, done here at order-of-magnitude rigor; D5 and D7 formalize them. They
are the reason the sharpened statement in §1.2 relocates the effect to the correlated tier.

### 3.1 Compact-source exclusion (the amplitude gate)

The step-A profile assigns the χ-dependence of `I` to the observer-source history via two-leg
timing while dropping the propagation amplitude. Restore it. For a source of coherence scale
`R_s` centered on the observer, the sourced field at the emission event obeys (Eq. U_green_retarded)

    δψ_emit(χ) ≈ (κ_σ Σ(t_ret) / 4π f² χ) · e^{−m_eff χ} ,   Σ ≡ ∫ σ d³x,

so the field at the source edge exceeds the field at χ by `~ χ/R_s`. The fitted imprint needs
`ε δψ ~ I_sat ~ 2.7×10⁻³` maintained out to `χ ~ 100 Mpc` (z ≈ 0.023, the red-valley scale). Then
the chronometric potential at the source edge, `ln A = ε δψ(R_s)`, is:

| Source hypothesis | `R_s` | `χ/R_s` at 100 Mpc | implied `ln A` at source edge | verdict |
|---|---|---|---|---|
| Earth / biosphere | ~10⁷ m | ~3×10¹⁷ | ~10¹⁵ | absurd; excluded by everything local |
| Sun / heliosphere | ~10¹¹ m | ~3×10¹³ | ~10¹¹ | excluded |
| Galaxy | ~10 kpc | ~10⁴ | ~30 | clock rates e³⁰ faster at the Galactic center; excluded by every pulsar |
| Local Group | ~1 Mpc | ~10² | ~0.3 | 30% clock-rate offsets inside the LG; excluded by local distance-ladder/timing consistency |
| Local Sheet / Void walls | ~30–100 Mpc | ~1–3 | ~10⁻³ | **only survivor** |

Additionally, a `1/χ` profile produces the wrong *shape* (`I ~ const − 1/χ`, steepest at small χ,
long flat tail), incompatible with the fitted bounded activation regardless of amplitude.

**Conclusion.** No compact source can generate the observed χ-dependence by propagation. The
χ-dependence of `I` must come from spatially **extended support** on the scale of the signal
itself (tens of Mpc). The literal-Earth-source branch — including any reading where terrestrial
history (life, complexity) *drives the z-shape* — is theoretically foreclosed, independent of the
transduction efficiency. (The fiducial choice `g_complex = g_life = 1` was already the right call;
this makes it permanent.) App. B's budget closure survives only as a bound on the *localized
endpoint* term `ε δψ_loc`, which is z-independent and degenerate with calibration.

### 3.2 Background-mode exclusion (why Phase 1 cannot produce the red valley)

If instead the χ-dependence comes from the homogeneous tier, `I(χ) = ε[ψ̄(t₀) − ψ̄(t_e)]`, then at
low z, `I ≈ ε ψ̄̇ · (χ/c)`, i.e. `s(χ) = ε ψ̄̇ = const`. A constant s is degenerate with a shift of
`H₀` inside `z ≲ 0.05` — it cannot produce the *decay* of s across the red valley. Producing an
O(1) decay of s within `z ≲ 0.05` from ψ̄ alone requires ψ̄ structure on ~300 Myr timescales; but
Phase 1 requires ψ̄ to act as Λ (overdamped, w ≈ −1, Hubble-timescale evolution), and an
oscillating ψ̄ (which a mass `m_eff ~ (100 Mpc)⁻¹` would naively allow) behaves as dark matter,
not Λ, violating the Phase-1 fit. **The background tier renormalizes H₀ and replaces Λ; it cannot
carry the low-z signal.**

### 3.3 Light-cone degeneracy, and the two-leg factor as a compact-source artifact

On the past light cone, comoving distance and lookback time are locked (`χ ↔ t_lb`), so a
*temporal* observer-history profile `ψ_⊕(t_ret(χ))` and a *spatial* local-structure well
`ψ_well(χ)` produce identical monotone `I(χ)` in the isotropic monopole. Step A therefore cannot
distinguish them — and by §3.1–3.2 only the spatial reading survives. Note also that the two-leg
rule `t_ret = t_lb(z_e), χ(z_e) = 2χ` is specific to point-source-at-observer geometry (out-leg +
in-leg); for extended support the timing collapses to **one-leg** (`t_e = t_lb(z)` at the emission
event, weighted by the local growth history). So when the recent runs are re-fit under the D4
kernel, the activation timescale parameters `(B, p, t_L)` will reparameterize; comparisons across
the two timings must be done at the level of `I(χ)`, not of g-parameters.

**Consequence for the world-tunnel workstream.** The world-tunnel solution is not an alternative
model to step A — under this program it is *the* surviving identification of the step-A signal:
the correlated-tier well `δψ_corr(x)` around our position, grown with local structure. Its
required interface objects are defined in D5.

### 3.4 The coupling-strength chain (screening is forced, not optional)

Canonically normalize: `φ_c = f ψ`, matter couples through `A = e^{εψ}`, so the scalar–tensor
coupling is `α ≡ M_p d ln A/dφ_c = ε M_p/f`. Then:

1. Cassini: `|γ−1| ≈ 2α² < 2.3×10⁻⁵ ⇒ α ≲ 3.4×10⁻³` for an unscreened field with range
   ≳ AU — and the well needs range ≳ 100 Mpc, so unscreened means unscreened *everywhere*.
2. The well needs `ε Δψ ≳ I_sat ≈ 2.7×10⁻³`, i.e. canonical excursion
   `Δφ_c ≳ (I_sat/α) M_p ≈ 0.8 M_p` across ~100 Mpc.
3. Gradient energy of that excursion: `ρ_grad ~ (Δφ_c/L)² ~ 4×10² ρ_crit`. Excluded by ~2.5
   orders of magnitude even before finer tests.

So the unscreened conformal implementation is self-inconsistent at the Cassini-saturating corner:
**either an environmental screening mechanism makes `α_local ≪ α_cosmic` (chameleon-direction:
screened in the dense solar neighborhood, unscreened at void scale — the correct sign of
environment-dependence for this model), or the conformal-coupling implementation is dead.** D7's
job is to derive the actual window, not to decide whether one is needed.

---

## 4. The derivation series

Each derivation lists: **Goal / Input / Derive / Output / Validation / Kill criterion.** The output
of each is the input of the next; validation hooks name the concrete pipeline artifact.

### D1 — Ordered configurations: microscopic definition and stress response

- **Goal:** make sentence 1 of §1.2 exact; separate the modular-energy route from the
  order-sector route.
- **Input:** causal-diamond reduced state `ρ_A`, reference `ρ_ref` (vacuum/local KMS/MaxEnt);
  App. A.1, U.1; trilogy capacity results.
- **Derive:** (i) exact identity `ΔK_ref = S_rel + ΔS` and linearized first law `δS = δ⟨K⟩`;
  (ii) for ball-shaped regions, express the modular-energy displacement as weighted moments of
  `δ⟨T_μν⟩`, giving the precise sense in which a constrained state "carries stress response";
  (iii) show `S_rel` is *not* an independent gravitational charge — semiclassical gravity sees
  only `⟨T_μν⟩` — so any ψ-sourcing by organization must be constitutive/dissipative (sets up D2);
  (iv) commuting limit → KL inventory of Lemma 1 (consistency with App. A.1).
- **Output:** `σ_ord(x)`: the coarse-grained relative-entropy/modular displacement density, with
  its exact relation to local stress-energy moments; a clean statement of what is and is not new.
- **Validation:** reduces to Lemmas 1–3 chain; Phase-1 anomaly prior (E7) untouched.
- **Kill criterion:** none (definitional), but it eliminates by construction any variant where
  "information gravitates directly" — keeping the theory out of a class of known-dead ideas.

### D2 — Open-EFT derivation of the order-sector dynamics (Schwinger–Keldysh)

- **Goal:** derive, rather than posit, the sourced field equation, the retarded structure, and the
  constitutive coupling `κ_σ`.
- **Input:** D1's `σ_ord`; a coarse-grained collective coordinate ψ for the organization density;
  environment = everything integrated out.
- **Derive:** in-in effective action → influence functional →
  `(□ − m²_eff) ψ + ∫ Σ_R ψ = κ_σ σ + ξ`, with (i) retarded self-energy `Σ_R` — causality is
  *derived* from the contour, closing G5; (ii) `κ_σ` as a Kubo-type transport coefficient of the
  environment (closing G3); (iii) noise `ξ` obeying a fluctuation–dissipation relation.
- **Output:** the causal response kernel and, unavoidably, a **stochastic companion**: a predicted
  line-of-sight variance `σ_I(z)` in the imprint.
- **Validation:** Markovian/local limit reproduces Eq. (psi_kg); `σ_I(z)` is computable in
  `lta_power.py` today as an extra diagonal (then correlated) term against Pantheon+ residual
  scatter.
- **Kill criterion:** if the FDT noise floor implied by the `(κ_σ, dissipation)` values needed for
  the well amplitude exceeds the observed SN Hubble-residual scatter budget, that parameter region
  is dead. (This is the first place the program can *self-terminate* quantitatively.)

### D3 — Covariant embedding and the exact three-tier split

- **Goal:** make sentence 2 of §1.2 theorem-level; give the "correlated" tier its object.
- **Input:** action Eq. (U_action) with `S_neq`; D2 dynamics.
- **Derive:** (i) `T^(neq)_μν` and exchange vector `Q^ν` from diffeomorphism invariance of
  `S_neq`; Bianchi consistency with the `F(ψ)R` term (extends U.4, W.4);
  (ii) unique decomposition `σ = σ̄(t) + δσ_corr(x) + δσ_loc(x)` (homogeneous mean /
  structure-tracing fluctuation field / sub-Mpc endpoint environments), mirrored tier-wise in
  `ψ = ψ̄ + δψ_corr + δψ_loc`, each with its own equation: Friedmann-level (Phase 1), a transfer
  function on LSS scales (new), and quasi-static Yukawa endpoints (U.6);
  (iii) confirm the clock channel `A = F^{−1/2}` is fixed by the action (no new parameter beyond
  ε ≡ −½ d ln F/dψ).
- **Output:** the three-tier field content; the `δψ_corr` sector as the formal interface to the
  world-tunnel workstream.
- **Validation:** Phase-1 equations unchanged; W.4 energy-exchange scalings recovered; §3.2
  exclusion re-derived exactly.
- **Kill criterion:** if Bianchi consistency forces `Q^ν` terms that spoil the Phase-1 fit
  (energy exchange visible in `H(z)` beyond allowed), the specific `S_neq` closure is rejected
  (iterate the closure, not the framework).

### D4 — Exact retarded kernel in FLRW; retire the two-leg proxy

- **Goal:** replace the horizon approximation with the full retarded solution, amplitude factors
  included; produce a drop-in forward model.
- **Input:** D2 kernel; App. V.1–V.5.
- **Derive:** (i) `G_ret` on FLRW for the massive, generally non-conformally-coupled case,
  including the tail (conformal-trick decomposition: null-cone piece + curvature/mass tail);
  (ii) the line-of-sight kernel `K(χ; η′)` with all factors (`1/χ`, `a(η′)`, `e^{−m_eff χ}`,
  tail integral); (iii) limit theorems: two-leg timing re-emerges for compact support
  (reproduces App. V), one-leg growth-weighted timing for extended support (§3.3);
  (iv) the kernel-weighted profile `s_K(χ)` for arbitrary source support `(R_s, history)`.
- **Output:** a `--kernel` forward mode for `lta_power.py`: `s(χ)` computed from
  `(m_eff, R_s, source history)` instead of `g(t_ret(2χ))`.
- **Validation:** benchmark = step-A Δχ² (13.65 fiducial; re-anchor to the ~10.8 configuration
  when pushed, G7). The kernel model must reach comparable Δχ² *with physical amplitude factors
  in place*.
- **Kill criterion (per branch):** a support hypothesis whose best kernel-weighted fit falls well
  short of the phenomenological benchmark for all `(m_eff, R_s)` is dead. §3.1 predicts this
  outcome for all compact branches — D4 makes it quantitative and final.

### D5 — Source-support selection: the correlated well, made predictive

- **Goal:** formalize §3.1–§3.3; identify the surviving source model and make it over-constrained.
- **Input:** D3's `δψ_corr` equation; D4 kernel; external local-structure reconstructions
  (2M++/CosmicFlows-class density and velocity fields).
- **Derive:** (i) the amplitude gate as a theorem (compact exclusion; background exclusion);
  (ii) `δψ_corr(x, t) ≈ W(x) · D_ψ(t)` from the growth of the local density/dissipation field,
  with `D_ψ` derived from D2's response to the structure-formation history of `σ_corr`
  (the ordered, virialized, dissipating fraction — *not* raw density);
  (iii) the map between the fitted activation `(B, p, t_L)` and the well profile `W(χ)` along the
  light cone (with one-leg timing);
  (iv) the required well depth `ε Δψ_corr ≈ 2.7×10⁻³` translated into a required contrast of the
  organization field given a *universal* `κ_σ` — no our-well-only tuning permitted.
- **Output:** the specific nature of the effect, stated: **a chronometric potential well of the
  order sector, grown with and correlated to local large-scale structure, read out at the
  endpoints of every null path; step A measured its past-light-cone section.** Interface contract
  for the world-tunnel solution: it must supply `W(x)` and `D_ψ(t)` satisfying (iii)–(iv).
- **Validation:** the fitted powerlaw shape must be reproduced by a `W(χ)` consistent with the
  reconstructed local density field at the D2-derived `κ_σ`; sky-anisotropy forecast follows for
  free from the anisotropy of `W(x)` (App. V.5 item 3).
- **Kill criterion:** if reconstructed local structure cannot produce the fitted shape/amplitude
  with a universal coupling — i.e., our well must be special beyond its measured density — the
  correlated branch fails too, and with §3.1–3.2 the whole mechanism is falsified. This is the
  program's central make-or-break derivation.

### D6 — Observable-propagation theorems (every observed signal, one superposition)

- **Goal:** derive, not assert, what every channel sees at the fitted amplitude.
- **Input:** D3 channels, D5 source model.
- **Derive:**
  1. **Time-dilation equality theorem** (closes G8): with `dτ_clk = A dτ` at both endpoints,
     transient durations obey `Δt_obs = (1+z_obs) Δt_emit` with the *same* `z_obs` as
     spectroscopy — chronometric endpoint effects preserve the stretch–redshift relation exactly.
     (The model is safe on SN light-curve stretch; a "spectroscopic-only" variant would already be
     dead. Worth a boxed theorem in the paper.)
  2. BAO Jacobian re-derivation under the D4 kernel (upgrade of §7.3).
  3. Pulsar timing: universal drift `s_now ≈ 5 km/s/Mpc ≈ 5×10⁻¹² yr⁻¹` enters as an
     Earth-term monopole absorbed in `ν̇` fits (App. P), but the *curvature* of the activation
     (`g̈`) sources a monopole-correlated red process in PTAs — compute its amplitude at fitted
     parameters against current PTA monopole limits.
  4. Endpoint tier: predicted correlation of SN Hubble residuals with **host-organization
     proxies** (mass/SFR as proxies for `δψ_loc` at the emitter). Pantheon+ host data supports
     this test now; relation to the known mass step must be derived, since the mass step is
     currently absorbed by SALT standardization — a distinctive redshift-independence signature
     separates the two.
  5. CMB dipole/aberration and Tolman/surface-brightness consistency at fitted amplitude.
- **Output:** a prediction table: channel → observable → amplitude at fitted parameters → current
  bound → margin.
- **Validation:** all currently-passing channels must stay passed with margins stated.
- **Kill criterion:** any derived amplitude exceeding a current bound (most dangerous: PTA
  monopole and the host-correlation sign/shape) falsifies the fitted parameter point directly.

### D7 — Local-constraint window and forced screening

- **Goal:** formalize §3.4 into an allowed region or a no-go.
- **Input:** canonical normalization `α = ε M_p/f`; D5's required well depth; Cassini, LLR,
  `Ġ_eff/G` (from Phase-1 `ψ̄̇` via `G_eff ∝ 1/F`), Galileo-satellite redshift tests,
  MICROSCOPE.
- **Derive:** (i) EP status: conformal `A(ψ)` is composition-blind at tree level even though the
  *sourcing* `σ` is state-dependent — sourcing-side state dependence does not violate EP; make
  this a lemma (it is the model's best defense and should be stated precisely);
  (ii) the unscreened no-go (§3.4) with the gradient-energy bound done properly;
  (iii) the screening window: environment-dependent `α_env` (chameleon/symmetron-direction) with
  the *inverted* requirement — screened at solar density, active at void scale — including
  whether the same screening that saves Cassini kills the 100-Mpc well (the classic tension; it
  must be computed, not hand-waved);
  (iv) `Ġ_eff/G` from the Phase-1 background at the anomaly-prior amplitude vs pulsar bounds.
- **Output:** the allowed `(ε or α_env(ρ), f, m_eff, κ_σ)` region, jointly with D2's noise floor
  and D5's depth requirement.
- **Kill criterion:** empty window ⇒ the conformal-coupling implementation of OTA is dead as a
  whole. (A clock-sector-only non-conformal coupling would be a *different theory* with a worse
  EP problem; if reached, that decision goes back to the humans, not into a derivation.)

### D8 — Preregistered statistical program (how the theory earns G1 back)

- **Goal:** convert D2–D7 outputs into out-of-sample tests, and kill the mundane alternative or
  lose to it.
- **Input:** all previous outputs; pipeline (`lta_power.py`) extensions.
- **Derive/spec:**
  1. **Mundane-alternative head-to-head (closes G6):** fit a linear-theory local void/bulk-flow
     ΛCDM model with matched dof to the same data, same covariance treatment, same null
     calibration. If it matches E1–E2, the LTA-specific interpretation is not supported —
     publish that honestly.
  2. **Sky-split anisotropy:** D5's `W(x)` predicts the low-z improvement concentrates in sky
     regions aligned with the local structure gradient; observer-history models predict isotropy.
     Preregister the split before looking.
  3. **Kernel-shape discrimination:** D4 kernel vs generic low-z offset families on held-out
     folds — the conditional-CV protocol already built (§CV) is the right instrument; the model
     must beat it *conditionally*, this time.
  4. **Noise floor** (D2) vs measured residual scatter as a parameter-space cut.
  5. **Host-organization residual correlation** (D6.4) on Pantheon+ hosts.
  6. **PTA monopole** (D6.3) forecast filed against upcoming data releases.
- **Output:** a preregistration document per test: statistic, data, null, decision threshold —
  before the runs.
- **Kill criterion:** the program's own: if the correlated-well model cannot beat the mundane
  alternative conditionally on any preregistered channel, OTA reverts to "phenomenological
  anomaly, cause unidentified" — which is also a publishable, honest endpoint.

---

## 5. Execution order and interfaces

```
D1 ──► D2 ──► D3 ──────────► D5 ──► D6 ──► D8
              │              ▲ ▲
              └─► D4 ────────┘ │      D7 runs now (no-go part) ──► feeds D5/D8 priors
                               │
        world-tunnel W(x),D_ψ(t) supplies/consumes D5 contract
```

- **Now, in parallel:** D4 (self-contained mathematics + pipeline mode), D7's no-go chain
  (§3.4 formalization — it only needs the action), and G7 (push the recent runs).
- **Serial spine:** D1 → D2 → D3 → D5. D1 and D3 are mostly assembly from existing appendix
  material; D2 is the genuinely new derivation and the program's intellectual core.
- **World-tunnel interface:** the other workstream should target the D5 contract — deliver
  `W(x)` (well profile from local structure) and `D_ψ(t)` (growth factor of the organization
  field), and consume D4's kernel for its forward predictions. Its Earth-retarded activation
  should be re-expressed in one-leg timing (§3.3) before comparing shapes.
- **Paper impact when done:** App. V is subsumed by D4; App. B is demoted to an endpoint-tier
  consistency check (§3.1); §5's "observer-local dominant limit" language is replaced by the
  three-tier statement; the framing paragraph is replaced by §1.2.

## 6. Notation crosswalk

| Symbol | Meaning | Defined |
|---|---|---|
| `S_rel`, `φ`, `φ_loc`, `φ*` | relative-entropy order measures (quantum / classical / density / normalization) | U.1, A.1, A.4 |
| `ψ = ψ̄ + δψ_corr + δψ_loc` | order field, three tiers (this program) | D3; upgrades Eq. (psi_split_main) |
| `σ`, `σ_ord` | nonequilibrium throughput density; modular displacement density | A.4; D1 |
| `J_neq = κ_σ σ` | constitutive source; `κ_σ` from Kubo formula | U.2; D2 |
| `F(ψ)`, `A = F^{−1/2}`, `ε` | conformal coupling, chronometric factor, its slope | U.3 |
| `f`, `φ_c = fψ`, `α = εM_p/f` | kinetic normalization, canonical field, scalar–tensor coupling | U.3; §3.4 |
| `G_ret`, `K(χ;η′)`, `Σ_R`, `ξ` | retarded Green's fn, LOS kernel, self-energy, FDT noise | V; D2, D4 |
| `I(χ)`, `s(χ)`, `g(t)`, `s_anchor` | imprint, drift field, activation, amplitude | §§5–7 |
| `W(x)`, `D_ψ(t)` | correlated-tier well profile and growth factor | D5 (new) |
