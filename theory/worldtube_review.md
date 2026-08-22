# Review: the world-tube causal-order checkpoint vs the D5 contract

**Object under review:**
`experiments/ota_state_dependent_response_20260821/OTA_CAUSAL_ORDER_DERIVATION_CHECKPOINT_20260822.md`
(the other workstream's gate-status checkpoint through B4b, dated 2026-08-22), read together with
its referenced implementations (`ota_twofield_local_observables_20260821/worldtube_path_gate.py`,
`closed_worldtube_pulse.py`) and the matching contract
`STEPA_KGB_CHI_MATCHING_CONTRACT_V1.md` in the same directory.

**Standard applied:** the D5 interface contract (`theory/derivations/D5_source_support.md` §3–4,
built on D4-F1…F5), i.e. what any world-tube deliverable must supply to carry the step-A signal:
a well/tube profile of extent ≳150–200 Mpc with a soft edge, depth εΔψ ≈ 2–5×10⁻³, one-leg
retarded timing, no reliance on a homogeneous piece (Theorem D3.2) or a compact source
(Theorem D5.1).

**Harness validation at review time (2026-08-22):** `run_t1_reproduce.py` re-run before this
review — baseline χ² = 332.444, step-A Δχ² = 13.593 (s_anchor 2.4454, t_anchor 0.670),
injected-shape control Δχ² = 13.582. The benchmarks quoted below are live.

---

## Verdict in one paragraph

The checkpoint is **not yet a candidate for the D5 contract, and — to its credit — does not claim
to be**: it supplies no W(x), no I(χ), no amplitude, and explicitly rules "no Step-A or CLASS
matching is authorized." So the conditional part of this review (implement the profile, fit it,
report Δχ² against 13.59 / 10.10 / 13.04) does not trigger; there is nothing to fit. On the
contract's structural items the checkpoint is convergent with this program in every place where
it makes contact — one-leg retarded timing, refusal of homogeneous rescues, population-of-
observers statistics, joint cross-probe accounting — and it independently derives two results
this program also needs (static structures accumulate no redshift; a closed conservative pulse
leaves no permanent scar). Its one load-bearing divergence is architectural: it is a strictly
**one-metric** theory ("one metric for clocks, matter, and photons"), with no conformal clock
channel. That choice is a coherent response to the D7.1 readout–force lock, but it makes the D5
depth requirement roughly **two orders of magnitude harder** than in the two-channel OTA action,
because the well must then be a genuine metric potential. That number — not any of the remaining
gate engineering — is the cliff the world-tube program is walking toward, and it should be
confronted before B4c/B4d are built out.

---

## 1. Item-by-item against the contract

### (a) Profile scale and edge: **not supplied**

The checkpoint contains no cosmological-scale profile. Its material world-tubes are compact
objects by construction (the motivating table is crystals, molecules, stars, climates, life —
sub-pc to ~10 kpc). The only explicit spatial profile anywhere in its orbit is the B4a optical
control in `worldtube_path_gate.py`: a *compensated* (zero-mean) compact core/annulus with
support at 1.5–5.5 Mpc along the canary ray, in a synthetic metric the checkpoint itself
disclaims as "explicitly not action derived" and "imposed optical controls, not a stress-derived
or retarded-source solution." The surviving hypothesis it names — "a source- or
selection-conditioned population of time-asymmetric retarded wakes" — *could* coarse-grain to an
extended structure-correlated field, but no scale, edge softness, or profile shape is derived or
even parameterized.

**No fit was run, and none would be informative.** For completeness, the outcome of pushing the
B4a control through the injection layer is predictable without running it: the SH0ES-HF sample
starts at z_min = 0.0234 (χ ≈ 100 Mpc), so a compact profile with support ≤ 6 Mpc has no data to
shape. Compensated (zero mean), it contributes I(χ) ≈ 0 for every HF supernova → Δχ² ≈ 0.
Uncompensated, it degenerates to the constant-offset mode → Δχ² = 10.10 with zero shape content,
exactly D4-F1. Fitting a numerical test fixture the authors have already disclaimed would only
manufacture a straw man.

### (b) Depth εΔψ ≈ 2–5×10⁻³: **not supplied — and structurally expensive in this architecture**

The checkpoint says plainly that it has "not yet supplied … a physical source-to-metric
amplitude, or a data fit," so there is no number to check. But the depth requirement interacts
with the checkpoint's one-metric commitment in a way that deserves to be priced now:

- In the two-channel OTA action, the well depth is a *chronometric* quantity, εΔψ, which D7
  then taxes through the readout–force lock (α ≥ 0.05 needed vs Cassini α ≤ 3.4×10⁻³ —
  Theorems D7.1–D7.2).
- In a one-metric theory there is no separate chronometric channel at all: an endpoint redshift
  of the required size **is** a metric potential difference ΔΦ/c² ≈ 1.9–4.6×10⁻³ over
  100–200 Mpc (D4-F5 depth table). ΛCDM large-scale structure supplies ΔΦ/c² ~ 3–5×10⁻⁵ on
  those scales — a shortfall of a factor **~60–150**. The gap must be closed by the mediator's
  own stress deepening the physical potential, and that stress gravitates: it buys back, in
  velocities, lensing, and ISW, everything the two-channel model paid in fifth-force constraints
  (the D7.2 corollary, now with no screening escape, because the observable channel and the
  force channel are literally the same metric).
- The velocity-statistics squeeze of D5 §3.3 therefore applies at full strength: required depths
  correspond to 570–1380 km/s equivalent flow amplitudes against measured bulk flows of
  ~250–400 km/s at those depths.

The checkpoint has, implicitly, already accepted this bill: its own B4d requirement 8 demands "a
conditional mean large enough to matter without excessive scatter, anisotropy, lensing,
velocity, or ISW signatures." This review's contribution is to attach the number: **the
conditional mean must reach ~10⁻³ in ΔΦ/c² while the unconditioned LSS budget is ~10⁻⁵.** Any
B4c/B4d normalization exercise should be checked against this ratio first, before geometry or
population machinery is refined further.

### (c) Timing: **pass (structurally)**

No two-leg 2χ rule appears anywhere in the checkpoint or its gate code. B4b's mediator support
is retarded from the source events; the population picture evaluates fields at emission events
from the local source history — one-leg lookback, exactly the D4 limit-theorem prescription for
extended support. Two further convergences worth recording:

- B4a's result that "a static endpoint-free tube has zero accumulated frequency shift while
  lensing and Shapiro delay can remain" is the one-metric counterpart of the D5 requirement that
  the well be *grown* (the D_ψ(t) factor): static structure does not redshift; evolving
  structure and endpoints do. Both programs independently landed on the same theorem.
- The checkpoint's demotion of the Results-IV 2.523% AP displacement ("may not be inserted as
  δH(z)") and its ban on "path-only H(z)" match D4's finding that the BAO Jacobian genuinely
  polices the mapping (D4-F4); its rule that "no branch may use a redshift window chosen from
  distance or BAO residuals" is the same discipline as D8's preregistration freeze.

### (d) Traps: **both avoided in the hypothesis; one inherited obligation each**

**Homogeneous piece.** Not relied on — the opposite. The B3 cosmological canary honestly reports
the homogeneous relic fails as dark energy (Δρ ≃ Δp ∝ −a⁻⁶, fading as a⁻³ against matter), and
the checkpoint forbids post-solve mean subtraction, requiring any emergent k = 0 component to be
owned as homogeneous stress. Note the two programs' homogeneous exclusions are *different
theorems that do not conflict*: D3.2 says a homogeneous chronometric factor is pure frame
(unobservable in the mapping); the checkpoint's homogeneous component is a stress component,
observable through H(z), and is excluded by its canary dynamics instead. Inherited obligation:
any population monopole that survives to the light-cone calculation lands exactly on the
**offset mode** (10.10 of the 13.59, D4-F2) and is degenerate with the ~293 km/s calibration
systematic on the D8 kill list — an emergent wake-field monopole would need to *win against*
that mundane alternative, not merely reproduce the number.

**Compact source.** Avoided in the stated hypothesis: no single observer-centered tube is
claimed to drive the z-shape (B4a itself proves a static tube cannot), and the surviving object
is a *population* of wakes. Inherited obligation: Theorem D5.1's dilution gate applies
tube-by-tube — each wake carries the 1/r propagation factor from its own tube, so the
coarse-grained field is dominated by the large-scale modulation of tube *density*, i.e. the
population must assemble into precisely the extended structure-correlated field δψ_corr of
D5 §3.2(ii) (the dissipation channel, with tubes as the biased tracers of δσ_corr) to produce
any in-range shape at all. When a normalization exists, the D4 kernel scan already says what the
assembled field must look like: R_w ≳ 150–200 Mpc, soft edge (n ≈ 2), Δχ² still rising at
200 Mpc.

## 2. Scope note: the replacement target is a different (harder) claim

The checkpoint's B4 comparator — "GR plus a listed ordinary matter/radiation inventory, no
cosmological constant, and no homogeneous OTA source," reproducing the **absolute** SN, BAO,
CMB, clock, and structure observables through the inhomogeneous solution — is a strict
Λ-replacement claim, strictly stronger than the D5 contract (a residual-level well on top of a
standard background). The harness benchmarks only test the residual-level object; they cannot
adjudicate the replacement claim. The interface still stands, however, because the checkpoint
itself allows that "Step-A could reappear only as an effective ensemble light-cone compression
derived after the world-tube and ray calculation" — and that ensemble compression, expressed
along our light cone, *is* an I(χ). That is the object to hand over.

## 3. Interface obligations going forward (what B4c/B4d/B4e should deliver here)

1. **The deliverable is I(χ) (or W(x) + D_ψ(t)), not gate receipts.** When the population
   forward model exists, express the ensemble mean light-cone compression for an observer at our
   location as I(χ) and run it through the injection layer: subclass `IShapeProfile` in
   `theory/numerics/harness.py` (pattern: `ExtendedWellProfile`), fit with `fit_custom()`,
   report Δχ² against step A 13.59, offset 10.10, well(R_w=200, n=2) 13.04 — quoting the prior
   configuration (13.59 is the (H0, Ωm, α_rd) chain-prior number; it is 5.5 in omega_m_alpha
   space).
2. **Confront the one-metric depth ratio first.** Before further gate engineering: state the
   candidate mechanism's maximum ΔΦ/c² over 100–200 Mpc and compare to the required
   1.9–4.6×10⁻³ and the LSS budget of 3–5×10⁻⁵ (§1b above). If the answer is "~10⁻⁵," the
   one-metric route cannot reach step-A amplitude and the comparison should be re-scoped to
   whatever residual it *can* produce.
3. **The anisotropy cross-check comes for free.** A wake population conditioned on structure
   inherits the local structure dipole; the preregistered test T1 (dipole-aligned sky split,
   axis frozen; current exploratory hint s = 4.6 ± 0.75 toward vs 2.2 ± 0.7 away) applies to it
   unchanged, as does T2 (velocity-survey cross-check). **No new test and no D8 amendment are
   required by this review**; the frozen definitions already cover the world-tube hypothesis.
4. **Own the monopole.** Per the checkpoint's own rule, no mean subtraction: report the
   population's emergent k = 0 compression alongside its shape, so the offset-mode degeneracy
   (D4-F2, D8 kill list) can be scored honestly.

## 4. Status line for the program ledger

World-tube checkpoint (B4b, 2026-08-22): **compatible-in-structure, silent-in-substance** with
respect to the D5 contract. Items (c) and (d): pass. Items (a) and (b): not supplied, by the
checkpoint's own discipline ("no Step-A or CLASS matching is authorized"). No profile to fit;
benchmarks unchanged; harness revalidated (Δχ² = 13.593). The binding question handed back to
the world-tube workstream is the one-metric depth ratio of §1b.
