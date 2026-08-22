# D1 — Ordered configurations: microscopic definition and stress response

**Goal.** Make the first sentence of the sharpened OTA statement exact: define an "ordered
configuration" at the level of quantum states, derive precisely what stress-energy response such a
configuration necessarily carries, and separate the two routes by which order can touch
observables — the modular-energy route (fixed by known physics, and quantitatively negligible) and
the constitutive order-sector route (everything OTA needs, derived in D2).

**Inputs.** Reduced density matrices on causal diamonds; the paper's App. A.1/U.1 definitions;
standard results: the entanglement first law, the Casini form of the Bekenstein bound, and the
Casini–Huerta–Myers (CHM) modular Hamiltonian for ball-shaped regions.

---

## 1. Definition: ordered configuration

Fix a coarse-graining scale and, at spacetime event $x$, a ball-shaped spatial region
$B(x,R)$ (equivalently its causal diamond $D(B)$). Let $\rho_B$ be the reduced state of the actual
matter configuration on $B$ and $\rho_{\rm ref}$ a reference reduced state on the same algebra:
the vacuum reduction, a local KMS (thermal) state at the environment temperature, or the MaxEnt
state under the macroscopic constraints *not* counted as order (this is the quantum version of the
paper's fixed-reference convention, App. A conventions paragraph).

> **Definition D1.1 (ordered configuration).** The configuration at $(x,R)$ is *ordered* iff
> $$ S_{\rm rel}(x,R) \equiv S(\rho_B \,\|\, \rho_{\rm ref}) = \mathrm{Tr}\!\left[\rho_B(\ln\rho_B - \ln\rho_{\rm ref})\right] > 0 . $$

$S_{\rm rel}\ge 0$ with equality iff $\rho_B=\rho_{\rm ref}$; it is monotone under CPTP
coarse-graining (data-processing inequality), which is the quantum ancestor of the paper's
Lemma 1b. In the commuting limit ($\rho_B,\rho_{\rm ref}$ co-diagonal) it reduces to the classical
KL inventory $D_{\rm KL}(p\|\pi)$ of App. A.1 — this is the paper's Eq. (U_rel_to_KL), so D1 is a
strict refinement, not a replacement, of the existing Lemma-1 chain.

## 2. The exact modular-energy identity

Let $K_{\rm ref} \equiv -\ln \rho_{\rm ref}$ (modular Hamiltonian of the reference). Writing
$\langle X\rangle_\rho = \mathrm{Tr}(\rho X)$ and $S(\rho)=-\mathrm{Tr}\rho\ln\rho$:

$$
S(\rho_B\|\rho_{\rm ref})
= \mathrm{Tr}(\rho_B\ln\rho_B) + \langle K_{\rm ref}\rangle_{\rho_B}
= -S(\rho_B) + \langle K_{\rm ref}\rangle_{\rho_B}.
$$

Since $\langle K_{\rm ref}\rangle_{\rho_{\rm ref}} = S(\rho_{\rm ref})$, subtracting and adding it:

> **Identity D1.2.**
> $$ \Delta\langle K_{\rm ref}\rangle = S_{\rm rel} + \Delta S, $$
> with $\Delta\langle K_{\rm ref}\rangle \equiv \langle K_{\rm ref}\rangle_{\rho_B} -
> \langle K_{\rm ref}\rangle_{\rho_{\rm ref}}$ and $\Delta S \equiv S(\rho_B)-S(\rho_{\rm ref})$.

Three standard corollaries, stated in OTA language:

1. **(Entanglement first law.)** To first order in $\delta\rho = \rho_B - \rho_{\rm ref}$,
   $S_{\rm rel} = O(\delta\rho^2)$, hence $\delta S = \delta\langle K_{\rm ref}\rangle$: small
   displacements from the reference exchange entropy and modular energy one-for-one.
2. **(Casini–Bekenstein bound.)** $S_{\rm rel}\ge 0 \iff \Delta S \le \Delta\langle K\rangle$:
   the entropy a region can hold above reference is paid for in modular energy.
3. **(Ordered states carry modular energy.)** If the configuration is ordered
   ($S_{\rm rel}>0$) and does not *reduce* the region's entropy ($\Delta S \ge 0$), then
   $\Delta\langle K\rangle \ge S_{\rm rel} > 0$ strictly. If order is achieved by entropy
   reduction ($\Delta S<0$, the typical biological case), the modular-energy cost is
   correspondingly smaller but the identity still fixes it exactly.

## 3. What "stress response" means, exactly

For a ball $B(x,R)$ in a QFT near its vacuum (CHM; exact for CFTs, leading-order for
near-vacuum states of massive theories), the reference modular Hamiltonian is a weighted integral
of the energy density:

$$
K_{\rm ref} = \frac{2\pi}{\hbar c}\int_{B} d^3x'\;\frac{R^2 - |\vec x' - \vec x|^2}{2R}\;
T_{00}(\vec x') \;+\; \text{const}.
$$

Combining with Identity D1.2:

> **Proposition D1.3 (stress response of an ordered configuration).** An ordered configuration's
> deviation from the reference obeys
> $$
> \frac{2\pi}{\hbar c}\int_{B} d^3x'\,\frac{R^2-r'^2}{2R}\,\delta\langle T_{00}\rangle(\vec x')
> \;=\; S_{\rm rel} + \Delta S .
> $$
> This weighted first moment of $\delta\langle T_{00}\rangle$ — not any independent "information
> charge" — is the entire sense in which a constrained quantum state *necessarily* changes the
> local stress expectation. Semiclassical gravity responds to $\delta\langle T_{\mu\nu}\rangle$
> and to nothing else.

This makes the first clause of the sharpened statement precise and simultaneously bounds it.

## 4. Lemma: no direct information gravity (the route-(a) no-go)

How large is the gravitational/chronometric effect that Proposition D1.3 can support? The
modular weight assigns one nat of relative entropy an energy scale set by the *modular
temperature* of the ball,

$$
E_{\rm 1\,nat}(R) \sim k_B T_{\rm mod}(R) = \frac{\hbar c}{2\pi R}
\approx 5\times10^{-25}\,\mathrm{J}\ \left(\frac{1\,\mathrm{cm}}{R}\right).
$$

Take the most aggressive inventory available for the biosphere — the paper's App. B *cumulative
throughput*, $\dot S_{\rm bio}\simeq4.6\times10^{11}$ W/K over 4 Gyr, i.e.
$\sim6\times10^{28}$ J/K $\approx 4\times10^{51}$ nats, all counted as standing order (a gross
overestimate of the inventory). Attached to modular energy at planetary radius
$R_\oplus\sim6.4\times10^6$ m ($\hbar c/2\pi R_\oplus \approx 8\times10^{-34}$ J/nat):

$$
\Delta E_{\rm mod} \lesssim 4\times10^{51}\times 8\times10^{-34}\,\mathrm{J}
\approx 3\times10^{18}\,\mathrm{J},
\qquad
\frac{G\,\Delta E_{\rm mod}}{c^4 R_\oplus} \approx 4\times10^{-33},
$$

i.e. a metric/chronometric potential thirty orders of magnitude below the fitted
$I_{\rm sat}\sim3\times10^{-3}$ — and this with the inventory overcounted by many orders.

> **Lemma D1.4 (no direct information gravity).** The modular-energy route from order to geometry
> — the only route guaranteed by first principles — falls short of the LTA signal by ~30 orders
> of magnitude even under maximal overcounting. Any viable
> OTA mechanism must therefore be *constitutive*: an emergent collective field $\psi$ whose state
> is driven by the organized/dissipative sector with an independent coupling strength
> ($\kappa_\sigma$, $\epsilon$), to be derived as open-system dynamics (D2), and whose stress and
> clock couplings enter through the action (D3). "Information gravitates directly" variants of
> OTA are foreclosed.

This is a feature, not a defect: it cleanly separates what is *guaranteed* (D1.2–D1.3, tiny) from
what is *hypothesized* (the $\psi$ sector, falsifiable), and it is why the framing sentence must
not say that constrained states "change the stress response" simpliciter.

## 5. The coarse-grained source objects handed to D2

Define, at coarse-graining volume $V_c$:

- **order density** $\;\varrho_{\rm ord}(x) \equiv S_{\rm rel}(x,R_c)/V_c\;$ (nats/m³), the
  quantum upgrade of App. A.4's $\varphi_{\rm loc}/k_B$; the dimensionless order field of the
  paper is $\psi = k_B\,\varrho_{\rm ord}\,V_c/\varphi_\ast$ per cell (Eq. app_psi_def);
- **maintenance throughput** $\;\sigma_s(x)\;$: the entropy-production density of the processes
  that hold $\rho_B$ away from $\rho_{\rm ref}$. Lemma 3 of the paper bounds
  $\partial_t \varrho_{\rm ord} \le \sigma_s$ (in matching units): order inventory can grow no
  faster than dissipation pays for.

D2 takes $\sigma_s(x)$ — not $S_{\rm rel}$ itself — as the operator source the order sector
couples to, because (i) Lemma D1.4 removes any direct gravitational role for $S_{\rm rel}$, and
(ii) only the *driven, time-irreversible* part of the environment can do work on a field
(equilibrium fluctuations source nothing on average; this becomes exact in D2's Kubo form).

## 6. Validation and status

- Commuting limit reproduces App. A.1 (Lemma 1) exactly; the Lemma 1–3 chain is untouched.
- Phase-1 anomaly prior (App. A.0) is untouched: it enters through the trilogy's entanglement
  coefficients, not through this section.
- Kill criterion: none (definitional), but Lemma D1.4 permanently retires a class of variants and
  fixes the burden of proof where it belongs: on the constitutive sector of D2.
