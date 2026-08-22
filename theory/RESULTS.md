# OTA Derivation Program — Results of the executed pass (D1–D8)

**What this is.** The derivation program defined in `theory/ota_derivation_program.md`, executed:
eight derivation documents in `theory/derivations/`, and a numerics suite in `theory/numerics/`
that runs the paper's own likelihood machinery (`lta_power.py` imported as a library, zero edits)
against the in-repo Pantheon+SH0ES + consensus-BAO data. Harness validation: reproduces the
fiducial run to optimizer tolerance (baseline χ² 332.44 vs paper 332.55; Δχ² 13.59 vs 13.652;
s_anchor 2.445 vs 2.449; t_anchor 0.670 ✓ — `numerics/out/t1_reproduce.json`).

**One-paragraph verdict.** The microscopic pass *relocates* the theory rather than confirming or
killing it. Ordered states as such cannot gravitate their way to the signal (D1); the order
sector must be an open, dissipatively driven field whose causality is derived and which cannot
source without noise (D2); a homogeneous chronometric factor is exactly pure frame, so the signal
— if physical — lives in a spatially inhomogeneous, structure-correlated chronometric well (D3).
Restoring the Green's-function amplitude factors the paper's App. V dropped shows the fitted
signal is ¾ monopole offset + ¼ shape, kills every compact source twice over, and demands a well
of ≳150–200 Mpc extent (D4, D5). The observable channels are safe where they must be
(time dilation: exact theorem; PTA: 0.11 ns; host environment: measured null) and taxed where
they should be (BAO already polices the well) (D6). The constraint chain then closes the current
action outright — readout strength and fifth-force coupling are the same parameter, and Cassini ×
the well's own gradient energy leave no unscreened or chameleon-screened window — naming a
derivative-screened (Vainshtein-class) extension as the unique surviving direction (D7). The
kinematic void alternative *underperforms* the mapping by Δχ² ≈ 6 at matched dof, and the
forward discriminants are preregistered (D8).

---

## Scorecard against the program's kill criteria

| Derivation | Kill criterion | Outcome |
|---|---|---|
| D1 | (definitional) | **Route closed:** "information gravitates directly" variants dead (Lemma D1.4: ~30 orders short even with the inventory overcounted). |
| D2 | FDT noise vs SN scatter | **Standing cut:** η ≲ 0.35 well-smoothness bound; not violated by current data (χ²/dof 0.91). |
| D3 | Exchange spoils Phase 1 | **Not triggered**; bonus theorem: homogeneous chronometric factor is exactly unobservable (frame). |
| D4 | Kernel fit falls short per branch | **Compact branch: triggered** (collapses to offset; shape content nil). **Well branch: viable** (Δχ² 13.04 vs 13.59 at R_w=200, n=2). |
| D5 | Well needs our-location tuning | **Alive under 3 obligations** (screening ratio; roughness; flow statistics — required depth ↔ 570–1380 km/s equivalent, sits factor ~2–4 above measured bulk flows, adjacent to the reported CF4 excess). |
| D6 | Any channel over budget | **Not triggered**; time-dilation equality and endpoint-only propagation proved as theorems; PTA margin 10²–10³. |
| D7 | Empty coupling window | **Triggered for the current action:** unscreened closed (α ≤ 3.4×10⁻³ vs α ≥ 0.05 needed), chameleon-class closed (range no-go), clock-only closed (lock theorem D7.1). Sole open door: Vainshtein-class kinetic completion — not yet constructed. |
| D8 | Mundane alternative matches | **Not matched:** physical void ≤ 7.6 vs step A 13.59. Leading residual threat: the ~293 km/s monopole offset mode (10.1 of 13.59), on the preregistered kill list. |

## The three headline numbers

1. **10.10 / 13.59** — the pure-offset share of the step-A preference. Three-quarters of the
   signal is a uniform ln(1+z) offset of the Hubble-flow sample (≈ 293 km/s); the activation
   *shape* contributes ≈ 3.5. Every interpretation — theirs, ours, a skeptic's — has to start
   from this decomposition. (It is not an M-calibration degeneracy: the anchored likelihood
   already profiles M.)
2. **R_w ≳ 150–200 Mpc** — the well the shape actually wants (still rising at 200), with depth
   εΔψ = 3.2–4.6×10⁻³. Not the Local Void; a supervolume-scale feature co-directional with the
   CMB dipole axis (Δs = 2.4 ± 1.0, ≈2.3σ from profile likelihoods — exploratory, promoted to
   preregistered test T1).
3. **α ≥ 0.05 vs α ≤ 3.4×10⁻³** — the readout–force lock plus the well's own gradient energy
   versus Cassini. The current OTA action cannot host its own best fit; the theory's critical
   path is now a concrete model-building task (derivative screening), not more fitting.

## What changed relative to the paper's framing

- Earth-retarded time survives only as light-cone bookkeeping; the two-leg factor is a
  compact-source artifact (one-leg for extended support). Terrestrial/biological history as a
  *driver of the z-shape* is foreclosed at theorem level (D5.1) — the fiducial `g_complex =
  g_life = 1` choice is now permanent, and App. B is demoted to a bound on the z-independent
  endpoint offset.
- "Ordered configurations … change the causal stress response" is replaced by the two-route
  statement (modular energy: guaranteed, negligible; constitutive order sector: hypothesized,
  falsifiable) — sharpened framing in `ota_derivation_program.md` §1.2 stands as written.
- App. V is superseded by D4 (tail bounded at ≤2% in-range; amplitude factors restored and
  decisive). The injection layer in `numerics/harness.py` is a working `--kernel` mode for
  arbitrary profiles, validated at 0.01 in Δχ² against the native pipeline — the world-tunnel
  workstream should express its solution as $W(\vec x)$ and run it through this.

## Prior-configuration sensitivity (ledger G7 addendum)

The step-A preference depends on the Planck-prior configuration: with the paper's correlated
(H0, Ωm, α_rd) chain prior, Δχ² = 13.59; with the code's current-default (ω_m, α_rd) space —
no direct H0 pull — the baseline relaxes to H0 = 70.4 and Δχ² drops to **5.5**
(`out/t1b_prior_sensitivity.json`). The quoted preference is, in large part, the H0 tension
itself expressed through the prior. Any headline number (13.65, or the recent 10.8) should be
reported alongside its prior configuration; recommend the paper state this explicitly.

## Files

```
theory/
  ota_derivation_program.md      program (framing, ledger, D1–D8 specs)
  RESULTS.md                     this file
  derivations/D1…D8_*.md         executed derivations
  numerics/
    harness.py                   pipeline import + profile-injection layer (validated)
    run_t1_reproduce.py          fiducial reproduction        → out/t1_reproduce.json
    run_d4_kernel_scan.py        kernel scans                 → out/d4_kernel_scan.json
    run_d8_void_headtohead.py    void + sky splits            → out/d8_void_sky.json
    run_d6_host.py               host split + timing numbers  → out/d6_host_timing.json
    run_x_decompose.py           χ² decomposition + profile σ → out/x_decompose.json
    out/d4_d8_summary.png        summary figure
```

## Recommended next actions (in order)

1. Push the recent (Δχ² ≈ 10.8) run configuration into the repo and re-anchor the D4/D8 tables
   (one command each; the harness reads any configuration `lta_power.py` accepts).
2. Point the world-tunnel workstream at the D5 §3.1 table and the injection layer: its
   deliverable is $W(\vec x)$ with $R_w \gtrsim 150$ Mpc, εΔψ per the table, tested against T2.
3. Begin the D7 successor task: a $c_T$=1-safe derivative-screened extension of the order
   sector, or a proof that none exists — this now gates the theory.
4. Execute T3 (conditional CV against the *offset*, existing data, existing protocol) — the
   cheapest decisive test on the books.
