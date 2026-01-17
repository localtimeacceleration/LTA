#!/usr/bin/env python3
"""
Hybrid ΛCDM + LTA (Earth-sourced time-drag) fit to Pantheon+SH0ES + BAO.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Tuple, Optional
from pathlib import Path

import re
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import betainc
import matplotlib.pyplot as plt
import contextlib
import io
import copy
import itertools


# ----------------------------
# Physical constants
# ----------------------------
C_KM_S = 299_792.458  # speed of light in km/s
MPC_IN_KM = 3.0856775814913673e19
SEC_PER_GYR = 1e9 * 365.25 * 24.0 * 3600.0

# ----------------------------
# Optional early-universe (Planck) Gaussian priors
# ----------------------------
from typing import Optional

@dataclass(frozen=True)
class GaussianPrior:
    mean: float
    sigma: float
    label: str = ""

    def chi2(self, x: float) -> float:
        s = float(self.sigma)
        if (not np.isfinite(s)) or (s <= 0.0):
            raise ValueError(f"Prior sigma must be positive for {self.label!r}, got {self.sigma}")
        dx = float(x) - float(self.mean)
        return float((dx / s) ** 2)

from dataclasses import dataclass, field
from scipy.linalg import cho_factor, cho_solve

@dataclass(frozen=True)
class MVGaussianPrior:
    mean: np.ndarray          # shape (d,)
    cov: np.ndarray           # shape (d,d)
    labels: tuple[str, ...] = ()
    label: str = ""
    _cho: tuple = field(init=False, repr=False)

    def __post_init__(self):
        m = np.asarray(self.mean, dtype=float).ravel()
        C = np.asarray(self.cov, dtype=float)
        if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] != m.size:
            raise ValueError(f"Bad MV prior shapes: mean {m.shape}, cov {C.shape}")
        if not np.all(np.isfinite(C)) or not np.all(np.isfinite(m)):
            raise ValueError("Non-finite mean/cov in MVGaussianPrior")
        cho = cho_factor(C, lower=True, check_finite=False)
        object.__setattr__(self, "mean", m)
        object.__setattr__(self, "cov", C)
        object.__setattr__(self, "_cho", cho)

    def chi2(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.mean.size:
            raise ValueError(f"MV prior expected dim {self.mean.size}, got {x.size}")
        dx = x - self.mean
        sol = cho_solve(self._cho, dx, check_finite=False)
        return float(dx @ sol)

@dataclass(frozen=True)
class EarlyPriors:
    """
    Early-universe (Planck) priors.

    If mv is set, it overrides the diagonal priors below.

    mv_idx specifies which components of (H0, Om, alpha_rd) the MV prior applies to:
      0 -> H0
      1 -> Om
      2 -> alpha_rd
    """
    mv: Optional[MVGaussianPrior] = None
    mv_idx: tuple[int, ...] = (0, 1, 2)

    h0: Optional[GaussianPrior] = None
    om: Optional[GaussianPrior] = None
    alpha_rd: Optional[GaussianPrior] = None

    def chi2(self, H0: float, Om: float, alpha_rd: float) -> float:
        if self.mv is not None:
            vec3 = np.array([H0, Om, alpha_rd], dtype=float)
            idx = tuple(int(i) for i in self.mv_idx)

            # Backward-safe fallback: if mv_idx length doesn't match, assume leading params.
            if len(idx) != int(self.mv.mean.size):
                idx = tuple(range(int(self.mv.mean.size)))

            return self.mv.chi2(vec3[list(idx)])

        c = 0.0
        if self.h0 is not None:
            c += self.h0.chi2(H0)
        if self.om is not None:
            c += self.om.chi2(Om)
        if self.alpha_rd is not None:
            c += self.alpha_rd.chi2(alpha_rd)
        return float(c)

    def n_constraints(self) -> int:
        if self.mv is not None:
            return int(self.mv.mean.size)
        return int(self.h0 is not None) + int(self.om is not None) + int(self.alpha_rd is not None)

# Global priors object (set in main())
EARLY_PRIORS: Optional[EarlyPriors] = None


# ----------------------------
# LTA model selector (global)
# ----------------------------
LTA_FORM = "powerlaw"  # will be set per-form in main() loop

# Anchor time (Earth retarded lookback) where the fitted amplitude equals s(t_anchor).
# If None or <=0, we fall back to the old behavior (amplitude at t=0).
LTA_T_ANCHOR_GYR: Optional[float] = None

EXP_CLIP = 80.0  # exp(80) ~ 5e34, huge but finite; avoids inf/NaN during exploration

G_ANCHOR_MIN = 1e-3   # prevents s(t)/g_anchor blowups in exploration

def _anchor_g(lta, epochs: "LTAEpochs") -> float:
    """
    Returns g(t_anchor) used for amplitude anchoring.
    If anchor is unset, return 1 so s(t)=s_anchor*g(t) reduces to the old meaning.
    """
    tA = LTA_T_ANCHOR_GYR
    if (tA is None) or (tA <= 0.0):
        return 1.0
    gA = float(earth_history_g(np.array([tA], dtype=float), lta, epochs)[0])
    return float(max(gA, np.finfo(float).tiny))


def aic(chi2: float, k: int) -> float:
    return chi2 + 2*k

def bic(chi2: float, k: int, n: int) -> float:
    return chi2 + k*np.log(n)

# ----------------------------
# I/O helpers
# ----------------------------
def load_covariance_matrix(path: str, n_expected: Optional[int] = None) -> np.ndarray:
    """
    Robust covariance loader for common cosmology ASCII formats:
    - N x N matrix stored plainly
    - flattened N*N list
    - optionally leading integer N
    - lower-triangular packed (N*(N+1)/2), optionally leading N
    """
    arr = np.fromfile(path, sep=" ")
    if arr.size == 0:
        raise ValueError(f"Covariance file appears empty: {path}")

    def looks_like_int(x: float) -> bool:
        return np.isfinite(x) and abs(x - int(round(x))) < 1e-9

    # If first entry is an integer N and remaining size matches patterns
    if looks_like_int(arr[0]):
        N0 = int(round(arr[0]))
        rest = arr[1:]
        if rest.size == N0 * N0:
            return rest.reshape((N0, N0))
        if rest.size == N0 * (N0 + 1) // 2:
            cov = np.zeros((N0, N0), dtype=float)
            tri = np.tril_indices(N0)
            cov[tri] = rest
            cov[(tri[1], tri[0])] = rest
            return cov

    # If n_expected provided, try to match that
    if n_expected is not None:
        N = n_expected
        if arr.size == N * N:
            return arr.reshape((N, N))
        if arr.size == 1 + N * N and looks_like_int(arr[0]) and int(round(arr[0])) == N:
            return arr[1:].reshape((N, N))
        if arr.size == N * (N + 1) // 2:
            cov = np.zeros((N, N), dtype=float)
            tri = np.tril_indices(N)
            cov[tri] = arr
            cov[(tri[1], tri[0])] = arr
            return cov
        if arr.size == 1 + N * (N + 1) // 2 and looks_like_int(arr[0]) and int(round(arr[0])) == N:
            rest = arr[1:]
            cov = np.zeros((N, N), dtype=float)
            tri = np.tril_indices(N)
            cov[tri] = rest
            cov[(tri[1], tri[0])] = rest
            return cov

    # Infer N from perfect square
    sq = int(round(np.sqrt(arr.size)))
    if sq * sq == arr.size:
        return arr.reshape((sq, sq))

    raise ValueError(
        f"Unrecognized covariance format in {path}. "
        f"Token count={arr.size}, n_expected={n_expected}."
    )

# ----------------------------
# Cosmology + LTA model
# ----------------------------
@dataclass(frozen=True)
class LTAEpochs:
    """
    Epoch boundaries in PROPER lookback time (Gyr), measured from 'now' backwards.
    These are user-chosen and can be edited.

    Ordering should be: t_digital < t_complex < t_life
    """
    t_digital_gyr: float = 0.00001   # 10,000 years = 1e-5 Gyr (adjust as desired)
    t_complex_gyr: float = 0.6       # ~600 Myr
    t_life_gyr: float = 3.8          # ~3.8 Gyr

@dataclass
class CosmologyTables:
    """
    Interpolation tables for:
      - χ(z)  : comoving distance [Mpc]
      - H(z)  : expansion rate [km/s/Mpc]
      - t_lb(z): proper lookback time [Gyr]
      - L(z)  : dimensionless conformal lookback ∫0^z dz'/E(z')
    """
    zgrid: np.ndarray
    chi_mpc: np.ndarray
    H_km_s_mpc: np.ndarray
    tlb_gyr: np.ndarray
    conf_int: np.ndarray         # L(z) = ∫0^z dz'/E(z')
    chi_of_z: callable
    H_of_z: callable
    z_of_tlb: callable
    conf_of_z: callable          # interpolation for L(z)
    tlb_of_z: callable          # t_lb(z) [Gyr]
    z_of_chi: callable          # inverse of chi_of_z


def build_cosmology_tables(H0: float, Om: float, zmax: float, nz: int = 20000) -> CosmologyTables:
    """
    Build interpolation tables for:
    - χ(z) in Mpc
    - H(z) in km/s/Mpc
    - proper lookback time t_lb(z) in Gyr
    """
    if not (0.0 < Om < 1.0):
        raise ValueError("Om must be between 0 and 1 for flat ΛCDM here.")
    if H0 <= 0:
        raise ValueError("H0 must be positive.")
    if zmax <= 0:
        raise ValueError("zmax must be positive.")
    if nz < 1000:
        nz = 1000

    zgrid = np.linspace(0.0, zmax, nz)
    E = np.sqrt(Om * (1.0 + zgrid) ** 3 + (1.0 - Om))
    H = H0 * E

    # χ(z) = (c/H0) ∫ dz/E
    invE = 1.0 / E
    int_chi = cumulative_trapezoid(invE, zgrid, initial=0.0)
    chi = (C_KM_S / H0) * int_chi  # Mpc

    # t_lb(z) = (1/H0) ∫ dz / ((1+z)E)  converted to Gyr
    int_t = cumulative_trapezoid(1.0 / ((1.0 + zgrid) * E), zgrid, initial=0.0)
    tlb_seconds = (MPC_IN_KM / H0) * int_t
    tlb_gyr = tlb_seconds / SEC_PER_GYR

    # Dimensionless conformal integral L(z) = ∫0^z dz'/E(z')
    conf_int = int_chi.copy()  # already ∫ dz/E

    chi_of_z = interp1d(zgrid, chi, kind="cubic", bounds_error=False, fill_value="extrapolate")
    H_of_z = interp1d(zgrid, H, kind="cubic", bounds_error=False, fill_value="extrapolate")
    z_of_tlb = interp1d(tlb_gyr, zgrid, kind="linear", bounds_error=False, fill_value=(zgrid[0], zgrid[-1]))
    conf_of_z = interp1d(zgrid, conf_int, kind="cubic", bounds_error=False, fill_value="extrapolate")
    tlb_of_z = interp1d(zgrid, tlb_gyr, kind="cubic",
                        bounds_error=False, fill_value="extrapolate")

    # chi is monotonic increasing, so we can invert it safely
    z_of_chi = interp1d(chi, zgrid, kind="linear",
                        bounds_error=False, fill_value=(zgrid[0], zgrid[-1]))

    return CosmologyTables(
        zgrid=zgrid,
        chi_mpc=chi,
        H_km_s_mpc=H,
        tlb_gyr=tlb_gyr,
        conf_int=conf_int,
        chi_of_z=chi_of_z,
        H_of_z=H_of_z,
        z_of_tlb=z_of_tlb,
        conf_of_z=conf_of_z,
        tlb_of_z=tlb_of_z,
        z_of_chi=z_of_chi,
    )

def earth_retarded_lookback_gyr(chi_mpc: np.ndarray, tables: CosmologyTables) -> np.ndarray:
    """
    For a point along the incoming photon path at comoving radius χ,
    the intersecting Earth emission time is at conformal lookback distance 2χ.

    We map:
        2χ  ->  z_e via chi(z_e)=2χ
        z_e ->  t_lb(z_e) in Gyr
    """
    chi = np.asarray(chi_mpc, dtype=float)
    chi_emit = 2.0 * chi
    z_emit = tables.z_of_chi(chi_emit)
    return tables.tlb_of_z(z_emit)

def earth_history_g(tlb_gyr: np.ndarray, lta: LTAParams, epochs: LTAEpochs) -> np.ndarray:
    """
    Dimensionless Earth-history factor g(t_lb) in [0,1], where t_lb is Earth proper lookback time.

    Convention:
      - t_lb = 0 now
      - life begins at t_lb = epochs.t_life_gyr
      - g = 0 for t_lb > t_life (before life existed)
      - g = 1 at t_lb = 0 (today)

    For exp/logistic/powerlaw we use u = (t_life - t_lb) = time-since-life-start [Gyr].
    """
    t = np.asarray(tlb_gyr, dtype=float)
    tL = float(epochs.t_life_gyr)

    g = np.zeros_like(t, dtype=float)

    alive = (t >= 0.0) & (t <= tL)
    if not np.any(alive):
        return g
    if LTA_FORM == "powerlaw":
        # Normalized inverse-power decay in lookback time t (0=now, t=tL life start)
        # Parameters:
        #   B = g_complex > 0  [Gyr]  (softening timescale)
        #   p = g_life   >= 0  [-]    (exponent; allow p=0 limit)
        B = max(float(lta.g_complex), 1e-10)
        p = max(float(lta.g_life), 0.0)

        tt = np.clip(t[alive], 0.0, tL)

        # Special stable closed form for p == 1:
        # g(t) = B*(tL - t) / (tL*(B + t))
        if abs(p - 1.0) < 1e-12:
            denom = tL * (B + tt)
            denom = np.maximum(denom, np.finfo(float).tiny)
            g_alive = (B * (tL - tt)) / denom

        # p -> 0 limit is logarithmic:
        # g(t) = 1 - ln(1+t/B)/ln(1+tL/B)
        elif p < 1e-12:
            denom = np.log1p(tL / B)
            denom = float(max(denom, np.finfo(float).tiny))
            g_alive = 1.0 - (np.log1p(tt / B) / denom)

        else:
            # Use stable exp/log form:
            # r(t)  = exp(-p ln(1+t/B))
            # r(tL) = exp(-p ln(1+tL/B))
            log_r  = -p * np.log1p(tt / B)
            log_rL = -p * np.log1p(tL / B)

            r  = np.exp(np.clip(log_r,  -700.0, 700.0))
            rL = float(np.exp(np.clip(log_rL, -700.0, 700.0)))

            # denom = 1 - rL (stable when rL ~ 1)
            denom = -np.expm1(np.clip(log_rL, -700.0, 0.0))
            denom = float(max(denom, np.finfo(float).tiny))

            g_alive = (r - rL) / denom


    else:
        raise ValueError(f"Unknown LTA_FORM for earth_history_g: {LTA_FORM}")

    g[alive] = g_alive
    return np.clip(g, 0.0, 1.0)

def build_inverse_zmap(
    tables: CosmologyTables,
    lta: LTAParams,
    epochs: LTAEpochs,
    zmax: float,
    nz: int = 4000,
):
    zcos_grid = np.linspace(0.0, zmax, nz)
    zobs_grid = zobs_from_zcos(zcos_grid, tables, lta, epochs)

    # Only enforce monotonicity if it is actually violated.
    # If we always run maximum.accumulate, tiny numerical wiggles can get flattened
    # into long plateaus -> inverse map becomes locally insensitive -> ~zero gradients.
    if np.any(np.diff(zobs_grid) <= 0.0):
        zobs_grid = np.maximum.accumulate(zobs_grid)

    # Make it *strictly* increasing to keep interp1d well-conditioned and sensitivity nonzero
    # (tiny epsilon is negligible physically but avoids duplicate-x pathology).
    zobs_grid = zobs_grid + (1e-12 * np.arange(zobs_grid.size, dtype=float))

    inv = interp1d(
        zobs_grid, zcos_grid,
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, zcos_grid[-1]),
    )
    return inv


@dataclass(frozen=True)
class LTAParams:
    """
    LTA parameters.

    s_anchor_km_s_per_mpc:
        The LTA local strength evaluated at Earth retarded lookback time t_anchor:
            s_anchor = s(t_anchor)

        The model then uses:
            s(t) = s_anchor * g(t) / g(t_anchor)

        (t_anchor is set globally as LTA_T_ANCHOR_GYR in main()).

    g_complex, g_life:
        Shape parameters (meaning depends on LTA_FORM).
    """
    s_anchor_km_s_per_mpc: float
    g_complex: float
    g_life: float


_LTA_I_CACHE = {"key": None, "chi_grid": None, "I_grid": None}

def lta_integral_I(chi_mpc: np.ndarray, tables: CosmologyTables, lta,
                   epochs: LTAEpochs) -> np.ndarray:
    """
    I(χ) = (1/c) ∫_0^χ s(χ') dχ'   (dimensionless)

    Stability notes:
      - integrate on a FIXED χ-grid (not a grid tied to the current query χ),
        so Newton steps / finite-diff probes don't reshuffle the integration grid.
      - cache the last (tables, epochs, lta) -> (χ_grid, I_grid) so repeated calls
        inside a single objective evaluation are consistent and fast.
    """
    chi = np.asarray(chi_mpc, dtype=float)
    if chi.size == 0:
        return chi

    chi_abs = np.maximum(chi, 0.0)

    # Fast path: if this is a parametric model with s_anchor==0, imprint is identically zero.
    if hasattr(lta, "s_anchor_km_s_per_mpc"):
        try:
            if float(getattr(lta, "s_anchor_km_s_per_mpc")) == 0.0:
                return np.zeros_like(chi_abs)
        except Exception:
            pass

    global _LTA_I_CACHE
    # Keyed so the cache is valid across repeated calls during Newton solves
    # for a fixed (tables, epochs, lta) triple.
    key = (
        int(id(tables.chi_mpc)),
        float(epochs.t_life_gyr),
        float(getattr(lta, "s_anchor_km_s_per_mpc")),
        float(getattr(lta, "g_complex")),
        float(getattr(lta, "g_life")),
        float(LTA_T_ANCHOR_GYR or 0.0),
        str(LTA_FORM),
    )


    cached = _LTA_I_CACHE
    if cached["key"] != key:
        chi_max_grid = float(tables.chi_mpc[-1])
        if chi_max_grid <= 0.0:
            return np.zeros_like(chi_abs)

        # Cluster points near the observer (important for steep near-now powerlaws)
        n_base = 4000
        u = np.linspace(0.0, 1.0, n_base, dtype=float)
        chi_grid = chi_max_grid * (u * u)

        s_grid = lta_local_s(chi_grid, tables, lta, epochs)
        dv = cumulative_trapezoid(s_grid, chi_grid, initial=0.0)  # km/s
        I_grid = dv / C_KM_S

        cached["key"] = key
        cached["chi_grid"] = chi_grid
        cached["I_grid"] = I_grid

    chi_grid = cached["chi_grid"]
    I_grid = cached["I_grid"]
    return np.interp(chi_abs, chi_grid, I_grid, left=0.0, right=float(I_grid[-1]))


def lta_local_s(chi_mpc: np.ndarray, tables: CosmologyTables, lta,
                epochs: LTAEpochs) -> np.ndarray:
    """
    Local drift rate along the photon path at comoving radius χ.

    Supports:
      - LTAParams : s(χ) = s_anchor * g(t_ret(χ))  (parametric)
    """
    chi = np.asarray(chi_mpc, dtype=float)
    chi_abs = np.maximum(chi, 0.0)
    # Fast path: if this is a parametric object with s_anchor==0, there is no LTA
    # regardless of the global LTA_FORM
    if hasattr(lta, "s_anchor_km_s_per_mpc"):
        try:
            if float(getattr(lta, "s_anchor_km_s_per_mpc")) == 0.0:
                return np.zeros_like(chi)
        except Exception:
            pass

    sA = float(lta.s_anchor_km_s_per_mpc)
    if sA == 0.0:
        return np.zeros_like(chi)

    t_ret = earth_retarded_lookback_gyr(chi, tables)   # Gyr
    g = earth_history_g(t_ret, lta, epochs)            # in [0,1]

    gA = _anchor_g(lta, epochs)                        # g(t_anchor)
    return sA * (g / gA)


def zobs_from_zcos(
    zcos: np.ndarray,
    tables: CosmologyTables,
    lta: LTAParams,
    epochs: LTAEpochs,
) -> np.ndarray:
    z = np.asarray(zcos, dtype=float)
    chi = tables.chi_of_z(z)

    I = lta_integral_I(chi, tables, lta, epochs)
    expI = np.exp(np.clip(I, -EXP_CLIP, EXP_CLIP))
    return (1.0 + z) * expI - 1.0


def dzobs_dzcos(
    zcos: np.ndarray,
    tables: CosmologyTables,
    lta: LTAParams,
    epochs: LTAEpochs,
) -> np.ndarray:
    z = np.asarray(zcos, dtype=float)
    chi = tables.chi_of_z(z)

    I = lta_integral_I(chi, tables, lta, epochs)
    expI = np.exp(np.clip(I, -EXP_CLIP, EXP_CLIP))

    Hcos = tables.H_of_z(z)
    s_loc = lta_local_s(chi, tables, lta, epochs)

    dI_dz = np.zeros_like(z)
    good = Hcos > 0
    dI_dz[good] = s_loc[good] / Hcos[good]

    return expI * (1.0 + (1.0 + z) * dI_dz)

def invert_zobs_to_zcos(
    zobs: np.ndarray,
    tables: CosmologyTables,
    lta: LTAParams,
    epochs: LTAEpochs,
    tol: float = 1e-12,
    maxiter: int = 30,
) -> np.ndarray:
    zobs = np.asarray(zobs, dtype=float)
    z = np.clip(zobs.copy(), 0.0, None)

    zero = zobs <= 0.0
    z[zero] = 0.0
    if np.all(zero):
        return z

    for _ in range(maxiter):
        zpred = zobs_from_zcos(z, tables, lta, epochs)
        deriv = dzobs_dzcos(z, tables, lta, epochs)

        deriv = np.where(deriv == 0.0, 1e-30, deriv)
        if (not np.all(np.isfinite(zpred))) or (not np.all(np.isfinite(deriv))):
            return np.full_like(zobs, np.nan)

        step = (zpred - zobs) / deriv
        z_new = z - step

        # physical bracket for positive drag
        z_new = np.clip(z_new, 0.0, zobs)

        if np.nanmax(np.abs(z_new - z)) < tol:
            z = z_new
            break
        z = z_new

    return z

@dataclass
class SNAnchor:
    """
    Precomputed objects for the ladder-anchored SN likelihood (m_b_corr only):

    We partition SNe into calibrators (C) and Hubble-flow / non-calibrators (H).

      y_C = mu_ceph + M + eps_C
      y_H = mu_pred(theta) + M + eps_H
      [eps_C, eps_H] ~ N(0, C_full) with cross-cov terms included.

    We:
      1) infer posterior for M from calibrators only:
           M_cal,  sigma_M
      2) evaluate HF likelihood conditional on calibrators and marginalized over M:
           y_H | y_C, theta  ~ N( mu_pred_H + mu_shift + M_cal*v,  C_cond + sigma_M^2 v v^T )

         where:
           mu_shift = C_HC C_CC^{-1} (y_C - mu_ceph)
           v        = 1_H - C_HC C_CC^{-1} 1_C
           C_cond   = C_HH - C_HC C_CC^{-1} C_CH     (Schur complement)
    """
    idx_cal: np.ndarray
    idx_hf: np.ndarray
    M_cal: float
    sig_M: float
    mu_shift_hf: np.ndarray   # length nH
    v_hf: np.ndarray          # length nH
    cov_hf_eff: np.ndarray    # nH x nH
    cho_hf_eff: tuple         # cho_factor(cov_hf_eff)
    chi2_cal_const: float     # r_cal^T C_CC^{-1} r_cal (constant wrt cosmology)


def build_sn_anchor(sn: "SNData") -> Optional[SNAnchor]:
    """
    Build the ladder-anchored HF|calibrators Gaussian for THIS sn object (current y/cov/selection).
    Returns None if not applicable.
    """
    if sn.y_is_mu:
        return None
    if sn.is_calibrator is None or (not np.any(sn.is_calibrator)):
        return None

    idx_cal = np.where(sn.is_calibrator)[0]
    idx_hf  = np.where(~sn.is_calibrator)[0]
    if idx_cal.size == 0 or idx_hf.size == 0:
        return None

    # Extract covariance blocks
    C = sn.cov
    C_CC = C[np.ix_(idx_cal, idx_cal)]
    C_HH = C[np.ix_(idx_hf,  idx_hf)]
    C_HC = C[np.ix_(idx_hf,  idx_cal)]   # (nH x nC)

    # Factorize calibrator block
    cho_CC = cho_factor(C_CC, lower=True, check_finite=False)

    ones_C = np.ones(idx_cal.size, dtype=float)
    ones_H = np.ones(idx_hf.size, dtype=float)

    # delta_C = y_C - mu_ceph  (calibrator mean is mu_ceph + M)
    delta_C = sn.y[idx_cal] - sn.mu_ceph[idx_cal]
    if not np.all(np.isfinite(delta_C)):
        raise ValueError("Non-finite delta_C encountered while building SNAnchor. Check CEPH_DIST for calibrators.")

    Cinv_delta_C = cho_solve(cho_CC, delta_C, check_finite=False)
    Cinv_ones_C  = cho_solve(cho_CC, ones_C,  check_finite=False)

    denom = float(ones_C @ Cinv_ones_C)
    if (not np.isfinite(denom)) or (denom <= 0.0):
        raise ValueError("Invalid calibrator-block normalization (ones^T C^-1 ones <= 0).")

    # Calibrator-only posterior for M
    M_cal = float((ones_C @ Cinv_delta_C) / denom)
    sig_M = float(np.sqrt(1.0 / denom))

    # X = C_CC^{-1} C_CH, where C_CH = C_HC^T
    # shape: (nC x nH)
    X = cho_solve(cho_CC, C_HC.T, check_finite=False)

    # Schur complement: HF covariance conditional on calibrators
    C_cond = C_HH - (C_HC @ X)
    C_cond = 0.5 * (C_cond + C_cond.T)

    # v = 1_H - C_HC C_CC^{-1} 1_C
    v = ones_H - (C_HC @ Cinv_ones_C)

    # mu_shift = C_HC C_CC^{-1} (y_C - mu_ceph)
    mu_shift = C_HC @ Cinv_delta_C

    # Add uncertainty from M posterior
    C_eff = C_cond + (sig_M * sig_M) * np.outer(v, v)
    C_eff = 0.5 * (C_eff + C_eff.T)

    # Factorize effective HF covariance
    cho_eff = cho_factor(C_eff, lower=True, check_finite=False)

    # Constant calibrator chi2 at M_cal (does not depend on cosmology/LTA)
    r_cal = delta_C - M_cal * ones_C
    chi2_cal = float(r_cal @ cho_solve(cho_CC, r_cal, check_finite=False))

    return SNAnchor(
        idx_cal=idx_cal,
        idx_hf=idx_hf,
        M_cal=M_cal,
        sig_M=sig_M,
        mu_shift_hf=mu_shift,
        v_hf=v,
        cov_hf_eff=C_eff,
        cho_hf_eff=cho_eff,
        chi2_cal_const=chi2_cal,
    )

# ----------------------------
# Likelihood / Chi2
# ----------------------------
@dataclass
class SNData:
    zHD: np.ndarray
    zHEL: np.ndarray
    idsurvey: np.ndarray
    cid: np.ndarray
    y: np.ndarray
    cov: np.ndarray
    cho: tuple
    y_is_mu: bool
    ones: np.ndarray
    Cinv_ones: np.ndarray
    ones_Cinv_ones: float
    is_calibrator: np.ndarray
    mu_ceph: np.ndarray
    anchor: Optional[SNAnchor] = None
    group: Optional[np.ndarray] = None

def sn_mu_reference(sn: SNData, mu_pred: np.ndarray) -> np.ndarray:
    """
    Returns μ_ref used in the m_b_corr likelihood:

      - non-calibrator rows: μ_ref = μ_pred(z)
      - calibrator rows:     μ_ref = μ_Ceph (= CEPH_DIST)

    Note: Pantheon+SH0ES does not provide μ_Ceph uncertainties as a column;
    they are incorporated in the full covariance matrix.
    """
    mu_pred = np.asarray(mu_pred, dtype=float)
    if sn.y_is_mu:
        return mu_pred
    if sn.is_calibrator is None or (not np.any(sn.is_calibrator)):
        return mu_pred

    mu_ref = mu_pred.copy()
    mu_ref[sn.is_calibrator] = sn.mu_ceph[sn.is_calibrator]
    return mu_ref


@dataclass
class BAOData:
    z: np.ndarray
    DM: np.ndarray
    Hz: np.ndarray
    cov: np.ndarray
    cho: tuple
    data_vector: np.ndarray


def build_sn_data(
    path_dat: str,
    path_cov: Optional[str],
    ycol: Optional[str] = "m_b_corr",
    zhd_col: str = "zHD",
    zhel_col: str = "zHEL",
    zmin: float = 0.0,
    zmax: Optional[float] = None,
    cov_mode: str = "file",
    errcol: Optional[str] = None,
    sample: str = "all",
    filter_query: Optional[str] = None,
    anchor_m_to_calibrators: bool = False,
) -> SNData:
    """
    Build SN dataset.

    Critical correctness rule:
      - The shipped Pantheon+SH0ES_STAT+SYS.cov is appropriate for m_b_corr-like observables.
      - MU_SH0ES is SH0ES-calibrated distance modulus and generally does NOT share that covariance.
        If you use MU_SH0ES, default to diagonal covariance from MU_SH0ES_ERR_DIAG.

    cov_mode:
      - "file": use full covariance matrix from path_cov
      - "diag": build diagonal covariance from errcol
    """
    df = pd.read_csv(path_dat, sep=r"\s+", comment="#")
    n = len(df)

    if zhd_col not in df.columns:
        raise KeyError(f"Missing column '{zhd_col}' in {path_dat}. Available: {list(df.columns)}")
    if zhel_col not in df.columns:
        raise KeyError(f"Missing column '{zhel_col}' in {path_dat}. Available: {list(df.columns)}")

    # Survey ID (Pantheon+SH0ES uses IDSURVEY)
    if "IDSURVEY" in df.columns:
        # robust numeric parse (handles strings / floats safely)
        idsurvey_full = pd.to_numeric(df["IDSURVEY"], errors="coerce").fillna(-1).astype(int).to_numpy()
    else:
        idsurvey_full = np.full(n, -1, dtype=int)

    if "CID" in df.columns:
        cid_full = df["CID"].astype(str).to_numpy()
    else:
        cid_full = np.array([str(i) for i in range(n)], dtype=object)

    # Optional human-readable grouping label (telescope/survey/etc.)
    # We try common column names; if none exist, fall back to IDSURVEY as a string.
    group_full = None
    for col in (
        "TELESCOPE", "telescope",
        "SURVEY", "survey",
        "SUBSURVEY", "subsurvey",
        "DATASET", "dataset",
        "SAMPLE", "sample",
    ):
        if col in df.columns:
            group_full = df[col].astype(str).to_numpy()
            break

    if group_full is None:
        group_full = idsurvey_full.astype(str)


    # Choose ycol
    if ycol is None:
        # Prefer the cosmology-friendly observable by default.
        if "m_b_corr" in df.columns:
            ycol = "m_b_corr"
        elif "MU_SH0ES" in df.columns:
            ycol = "MU_SH0ES"
        elif "mu" in df.columns:
            ycol = "mu"
        else:
            raise KeyError(
                "Could not infer SN observable column. Please pass --sn-ycol.\n"
                f"Columns: {list(df.columns)}"
            )
    if ycol not in df.columns:
        raise KeyError(f"Missing SN observable column '{ycol}'. Columns: {list(df.columns)}")

    cov_mode = str(cov_mode).strip().lower()
    if cov_mode not in ("file", "diag"):
        raise ValueError(f"--sn-cov-mode must be 'file' or 'diag', got {cov_mode!r}")

    # Infer errcol when diag
    if cov_mode == "diag" and errcol is None:
        if str(ycol).upper() == "MU_SH0ES":
            errcol = "MU_SH0ES_ERR_DIAG"
        elif str(ycol) == "m_b_corr":
            errcol = "m_b_corr_err_DIAG"
        else:
            raise ValueError(
                f"--sn-cov-mode diag requires --sn-errcol for ycol={ycol!r} "
                f"(could not infer a default)."
            )

    # Prevent the most common misuse: MU_SH0ES with the m_b_corr covariance file.
    if cov_mode == "file" and str(ycol).upper() == "MU_SH0ES":
        raise ValueError(
            "You selected --sn-ycol MU_SH0ES with --sn-cov-mode file.\n"
            "This is almost certainly wrong because Pantheon+SH0ES_STAT+SYS.cov corresponds to m_b_corr.\n"
            "Use: --sn-cov-mode diag --sn-errcol MU_SH0ES_ERR_DIAG\n"
            "and usually also: --sn-sample shoes_hf"
        )

    zHD_full = df[zhd_col].to_numpy(dtype=float)
    zHEL_full = df[zhel_col].to_numpy(dtype=float)
    y_full = df[ycol].to_numpy(dtype=float)

    # Identify whether we're using a μ-type observable (MU_SH0ES) or an m_b-type observable (m_b_corr)
    y_is_mu = str(ycol).lower().startswith("mu")

    # Pantheon+SH0ES ladder columns (exact names from your file)
    if "IS_CALIBRATOR" in df.columns:
        is_calib_full = (df["IS_CALIBRATOR"].to_numpy(dtype=float) > 0.5)
    else:
        is_calib_full = np.zeros(n, dtype=bool)

    if "CEPH_DIST" in df.columns:
        mu_ceph_full = df["CEPH_DIST"].to_numpy(dtype=float)
    else:
        mu_ceph_full = np.full(n, np.nan, dtype=float)

    # CEPH_DIST is only meaningful for calibrators; non-cal rows are typically -9 in the release
    mu_ceph_full = np.where(is_calib_full, mu_ceph_full, np.nan)

    mask = np.ones(n, dtype=bool)

    # Sample selection
    if sample != "all":
        if sample == "shoes_hf":
            if "USED_IN_SH0ES_HF" not in df.columns:
                raise KeyError("Column USED_IN_SH0ES_HF not found; cannot use --sn-sample shoes_hf.")
            mask &= (df["USED_IN_SH0ES_HF"].to_numpy(dtype=float) > 0.5)
        elif sample == "calibrators":
            if "IS_CALIBRATOR" not in df.columns:
                raise KeyError("Column IS_CALIBRATOR not found; cannot use --sn-sample calibrators.")
            mask &= (df["IS_CALIBRATOR"].to_numpy(dtype=float) > 0.5)
        elif sample == "shoes_global":
            if "USED_IN_SH0ES_HF" not in df.columns:
                raise KeyError("Column USED_IN_SH0ES_HF not found; cannot use --sn-sample shoes_global.")
            if "IS_CALIBRATOR" not in df.columns:
                raise KeyError("Column IS_CALIBRATOR not found; cannot use --sn-sample shoes_global.")
            mask &= (
                (df["USED_IN_SH0ES_HF"].to_numpy(dtype=float) > 0.5)
                | (df["IS_CALIBRATOR"].to_numpy(dtype=float) > 0.5)
            )
        elif sample == "noncalibrators":
            if "IS_CALIBRATOR" not in df.columns:
                raise KeyError("Column IS_CALIBRATOR not found; cannot use --sn-sample noncalibrators.")
            mask &= (df["IS_CALIBRATOR"].to_numpy(dtype=float) < 0.5)
        else:
            raise ValueError(f"Unknown --sn-sample {sample!r}")

    # Optional user filter
    if filter_query is not None and str(filter_query).strip():
        try:
            extra = df.eval(str(filter_query), engine="python")
        except Exception as e:
            raise ValueError(f"Failed to eval --sn-filter expression: {filter_query!r}\nError: {e}")
        if extra.dtype != bool:
            extra = extra.astype(bool)
        mask &= extra.to_numpy()

    # z cuts
    mask &= np.isfinite(zHD_full) & np.isfinite(zHEL_full) & np.isfinite(y_full)
    mask &= (zHD_full >= float(zmin))
    if zmax is not None:
        mask &= (zHD_full <= float(zmax))

    # diag error support
    err_full = None
    if cov_mode == "diag":
        if errcol not in df.columns:
            raise KeyError(f"Missing error column '{errcol}' for diag SN cov mode.")
        err_full = df[errcol].to_numpy(dtype=float)
        mask &= np.isfinite(err_full) & (err_full > 0.0)

    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError("SN selection produced zero rows. Check --sn-zmin/--sn-zmax/--sn-sample/--sn-filter.")

    zHD = zHD_full[idx]
    zHEL = zHEL_full[idx]
    y = y_full[idx]
    idsurvey = idsurvey_full[idx]
    cid = cid_full[idx]
    group = group_full[idx] if group_full is not None else None

    is_calib = is_calib_full[idx]
    mu_ceph = mu_ceph_full[idx]

    # If calibrators are in the selected sample and we're fitting m_b_corr,
    # require full covariance mode (μ_Ceph uncertainty is in the cov matrix, not a column).
    if (not y_is_mu) and np.any(is_calib):
        bad = ~np.isfinite(mu_ceph[is_calib])
        if np.any(bad):
            raise ValueError(
                "Selected SN sample includes calibrators (IS_CALIBRATOR==1) but CEPH_DIST is non-finite "
                "for at least one calibrator row."
            )
        if cov_mode != "file":
            raise ValueError(
                "Selected SN sample includes calibrators with m_b_corr, but --sn-cov-mode is 'diag'. "
                "In Pantheon+SH0ES, μ_Ceph uncertainty is incorporated in the full covariance matrix; "
                "use --sn-cov-mode file."
            )

    # Build covariance for the selected subset
    if cov_mode == "file":
        if path_cov is None:
            raise ValueError("--sn-cov-mode file requires --sn-cov to be provided.")
        cov_full = load_covariance_matrix(path_cov, n_expected=n)
        cov = cov_full[np.ix_(idx, idx)]
    else:
        err = err_full[idx]
        cov = np.diag(err * err)

    cho = cho_factor(cov, lower=True, check_finite=False)
    ones = np.ones(idx.size, dtype=float)
    Cinv_ones = cho_solve(cho, ones, check_finite=False)
    ones_Cinv_ones = float(ones @ Cinv_ones)

    sn_out = SNData(
        zHD=zHD,
        zHEL=zHEL,
        idsurvey=idsurvey,
        cid=cid,
        y=y,
        cov=cov,
        cho=cho,
        y_is_mu=y_is_mu,
        ones=ones,
        Cinv_ones=Cinv_ones,
        ones_Cinv_ones=ones_Cinv_ones,
        is_calibrator=is_calib,
        mu_ceph=mu_ceph,
        anchor=None,
        group=group,
    )

    # Build ladder anchor (HF|calibrators) if requested and applicable
    if bool(anchor_m_to_calibrators) and (not sn_out.y_is_mu) and np.any(sn_out.is_calibrator):
        sn_out.anchor = build_sn_anchor(sn_out)
        if sn_out.anchor is not None:
            a = sn_out.anchor
            print("[SN] HF anchored by calibrators enabled (HF|calibrators, full-cov, M-marg via calibrator posterior).")
            print(f"[SN]   M_cal = {a.M_cal:.6f}  ± {a.sig_M:.6f}  (1σ from calibrators only)")
            print(f"[SN]   N_cal={int(a.idx_cal.size)}   N_HF={int(a.idx_hf.size)}")
        else:
            print("[SN] Requested --sn-anchor-m-to-calibrators but anchor not applicable (need both calibrators and HF rows).")

    return sn_out




def load_bao_results(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load BAO consensus results in either:
      A) 3-column format: z  DM  H
      B) stacked 2-column format (your file): z value with alternating DM and H rows.

    Returns (z_unique, DM, H).
    """
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if df.shape[1] < 2:
        raise ValueError(f"BAO results file must have at least 2 columns. Got {df.shape[1]} columns.")

    if df.shape[1] >= 3:
        # Format A: z DM H
        z = df.iloc[:, 0].to_numpy(float)
        DM = df.iloc[:, 1].to_numpy(float)
        Hz = df.iloc[:, 2].to_numpy(float)
        return z, DM, Hz

    # Format B: stacked 2-column
    z_all = df.iloc[:, 0].to_numpy(float)
    val_all = df.iloc[:, 1].to_numpy(float)

    if len(z_all) % 2 != 0:
        raise ValueError("Stacked BAO format must have even number of rows (DM/H pairs).")

    z_unique = []
    DM = []
    Hz = []

    for i in range(0, len(z_all), 2):
        z1, v1 = z_all[i], val_all[i]
        z2, v2 = z_all[i + 1], val_all[i + 1]
        if abs(z1 - z2) > 1e-12:
            raise ValueError(f"Mismatched z in stacked pair at rows {i},{i+1}: {z1} vs {z2}")

        z_unique.append(z1)
        DM.append(v1)   # first in pair is D_M
        Hz.append(v2)   # second in pair is H

    return np.array(z_unique), np.array(DM), np.array(Hz)


def build_bao_data(path_res: str, path_cov: str) -> BAOData:
    z, DM, Hz = load_bao_results(path_res)
    cov = load_covariance_matrix(path_cov)

    # Data vector must match covariance ordering: [DM1, H1, DM2, H2, ...]
    data_vec = np.column_stack([DM, Hz]).reshape(-1)

    if cov.shape != (data_vec.size, data_vec.size):
        raise ValueError(
            f"BAO cov shape {cov.shape} does not match data vector length {data_vec.size}.\n"
            f"Expected {data_vec.size}x{data_vec.size}."
        )

    cho = cho_factor(cov, lower=True, check_finite=False)
    return BAOData(z=z, DM=DM, Hz=Hz, cov=cov, cho=cho, data_vector=data_vec)

def load_getdist_margestats(path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Parse GetDist .margestats.
    Expected data lines like:
      paramName   mean    stdev   ...
    """
    names: list[str] = []
    means: list[float] = []
    sigmas: list[float] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            toks = line.split()
            if len(toks) < 3:
                continue
            name = toks[0]
            try:
                mu = float(toks[1])
                sd = float(toks[2])
            except Exception:
                continue
            names.append(name)
            means.append(mu)
            sigmas.append(sd)

    if not names:
        raise ValueError(f"Failed to parse any parameters from margestats: {path}")

    return names, np.array(means, dtype=float), np.array(sigmas, dtype=float)


def load_getdist_covmat(path: str, n_expected: Optional[int] = None) -> np.ndarray:
    """
    Robust GetDist .covmat reader:
    - ignores non-numeric tokens (safe if headers creep in)
    - supports leading integer N, full NxN, or packed lower-triangular
    """
    toks: list[float] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            for t in line.split():
                try:
                    toks.append(float(t))
                except Exception:
                    pass

    arr = np.array(toks, dtype=float)
    if arr.size == 0:
        raise ValueError(f"Covmat file appears empty/unreadable: {path}")

    # Reuse your logic by temporarily mimicking the interface
    def looks_like_int(x: float) -> bool:
        return np.isfinite(x) and abs(x - int(round(x))) < 1e-9

    if looks_like_int(arr[0]):
        N0 = int(round(arr[0]))
        rest = arr[1:]
        if rest.size == N0 * N0:
            return rest.reshape((N0, N0))
        if rest.size == N0 * (N0 + 1) // 2:
            cov = np.zeros((N0, N0), dtype=float)
            tri = np.tril_indices(N0)
            cov[tri] = rest
            cov[(tri[1], tri[0])] = rest
            return cov

    if n_expected is not None:
        N = int(n_expected)
        if arr.size == N * N:
            return arr.reshape((N, N))
        if arr.size == N * (N + 1) // 2:
            cov = np.zeros((N, N), dtype=float)
            tri = np.tril_indices(N)
            cov[tri] = arr
            cov[(tri[1], tri[0])] = arr
            return cov

    sq = int(round(np.sqrt(arr.size)))
    if sq * sq == arr.size:
        return arr.reshape((sq, sq))

    raise ValueError(f"Unrecognized covmat format in {path} (n_expected={n_expected}, n_tokens={arr.size}).")


def _norm_name(s: str) -> str:
    s = str(s).strip()
    if s.endswith("*"):
        s = s[:-1]
    s = s.lower().replace("_", "").replace("-", "")
    return s

def _load_paramnames(path: str) -> list[str]:
    names = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # paramnames format: "name   label..."
            name = line.split()[0]
            names.append(name)
    if not names:
        raise ValueError(f"No paramnames read from {path}")
    return names

def _find_idx(names: list[str], candidates: list[str]) -> Optional[int]:
    nmap = {_norm_name(n): i for i, n in enumerate(names)}
    for c in candidates:
        key = _norm_name(c)
        if key in nmap:
            return nmap[key]
    return None

def _find_chain_files(chain_root: str) -> list[str]:
    # Expect CosmoMC/GetDist style: root_1.txt, root_2.txt, ...
    root = Path(chain_root)
    parent = root.parent
    stem = root.name

    # glob all root_*.txt and keep those ending in _<int>.txt
    files = []
    for p in parent.glob(stem + "_*.txt"):
        m = re.match(rf"^{re.escape(stem)}_(\d+)\.txt$", p.name)
        if m:
            files.append((int(m.group(1)), str(p)))
    files.sort()
    return [f for _, f in files]

def build_planck_prior_from_chain(
    chain_root: str,
    rd_fid: float,
    chunksize: int = 200_000,
) -> MVGaussianPrior:
    """
    Build correlated Gaussian prior on (H0, Omegam, alpha_rd) from Planck MCMC chains.

    Uses:
      chain_root.paramnames
      chain_root_1.txt, chain_root_2.txt, ...
    """
    chain_root = str(chain_root)
    pn = chain_root + ".paramnames"
    if not Path(pn).exists():
        raise FileNotFoundError(f"Missing paramnames file: {pn}")

    chain_files = _find_chain_files(chain_root)
    if not chain_files:
        raise FileNotFoundError(
            f"No chain files found for root {chain_root}. Expected e.g. {chain_root}_1.txt"
        )

    names = _load_paramnames(pn)

    # Prefer direct derived columns if present; else compute from fallbacks.
    idx_H0     = _find_idx(names, ["H0", "H0*"])
    idx_h      = _find_idx(names, ["h", "H0/100"])
    idx_omegam = _find_idx(names, ["omegam", "Omega_m", "omegam*", "Omega_m*"])
    idx_om_h2  = _find_idx(names, ["omegamh2", "omegam*h2", "omegamh2*"])
    idx_rdrag  = _find_idx(names, ["rdrag", "r_drag", "rdrag*", "r_drag*"])
    idx_rdragh = _find_idx(names, ["rdragh", "rdrag*h", "rdragh*"])

    # Decide what raw columns we must read
    needed_param_indices = set()

    # H0
    if idx_H0 is not None:
        needed_param_indices.add(idx_H0)
        use_H0_mode = "H0"
    elif idx_h is not None:
        needed_param_indices.add(idx_h)
        use_H0_mode = "h"
    else:
        raise ValueError("Chain does not contain H0 or h; cannot build Planck prior on H0.")

    # Omegam
    if idx_omegam is not None:
        needed_param_indices.add(idx_omegam)
        use_om_mode = "omegam"
    elif idx_om_h2 is not None:
        needed_param_indices.add(idx_om_h2)
        # will also need h/H0
        if use_H0_mode == "H0" and idx_H0 is not None:
            needed_param_indices.add(idx_H0)
        elif use_H0_mode == "h" and idx_h is not None:
            needed_param_indices.add(idx_h)
        use_om_mode = "omegamh2_over_h2"
    else:
        raise ValueError("Chain does not contain omegam or omegamh2; cannot build Planck prior on Omega_m.")

    # rdrag
    if idx_rdrag is not None:
        needed_param_indices.add(idx_rdrag)
        use_rd_mode = "rdrag"
    elif idx_rdragh is not None:
        needed_param_indices.add(idx_rdragh)
        # will also need h/H0
        if use_H0_mode == "H0" and idx_H0 is not None:
            needed_param_indices.add(idx_H0)
        elif use_H0_mode == "h" and idx_h is not None:
            needed_param_indices.add(idx_h)
        use_rd_mode = "rdragh_over_h"
    else:
        raise ValueError("Chain does not contain rdrag or rdragh; cannot build Planck prior on r_drag.")

    needed_param_indices = sorted(needed_param_indices)

    # Chain format: col0 = weight, col1 = -lnlike, then params in paramnames order
    usecols = [0] + [2 + i for i in needed_param_indices]

    sum_w = 0.0
    sum_wx = np.zeros(3, dtype=float)
    sum_wxx = np.zeros((3, 3), dtype=float)

    for cf in chain_files:
        for chunk in pd.read_csv(
            cf, sep=r"\s+", header=None, comment="#",
            usecols=usecols, chunksize=int(chunksize)
        ):
            w = chunk.iloc[:, 0].to_numpy(dtype=float)
            if not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ValueError(f"Non-finite or negative weights encountered in {cf}")

            # Helper to pull a param by its paramnames index
            def col(param_idx: int) -> np.ndarray:
                return chunk[2 + param_idx].to_numpy(dtype=float)

            # Build H0
            if use_H0_mode == "H0":
                H0 = col(idx_H0)
            else:
                H0 = 100.0 * col(idx_h)

            h = H0 / 100.0

            # Build Omegam
            if use_om_mode == "omegam":
                Om = col(idx_omegam)
            else:
                Om = col(idx_om_h2) / (h * h)

            # Build rdrag
            if use_rd_mode == "rdrag":
                rdrag = col(idx_rdrag)
            else:
                rdrag = col(idx_rdragh) / h

            X = np.column_stack([H0, Om, rdrag]).astype(float)

            if not np.all(np.isfinite(X)):
                raise ValueError(f"Non-finite derived values encountered while reading {cf}")

            # Weighted accumulators
            sw = float(np.sum(w))
            sum_w += sw
            sum_wx += np.sum(w[:, None] * X, axis=0)
            sum_wxx += X.T @ (w[:, None] * X)

    if sum_w <= 0.0:
        raise ValueError("Total chain weight is zero; cannot build prior.")

    mean = sum_wx / sum_w
    cov = (sum_wxx / sum_w) - np.outer(mean, mean)
    cov = 0.5 * (cov + cov.T)  # symmetrize

    # Convert rdrag -> alpha = rdrag / rd_fid
    if (not np.isfinite(rd_fid)) or (rd_fid <= 0.0):
        raise ValueError(f"rd_fid must be positive, got {rd_fid}")

    J = np.diag([1.0, 1.0, 1.0 / float(rd_fid)])
    mean_a = J @ mean
    cov_a = J @ cov @ J.T

    return MVGaussianPrior(
        mean=mean_a,
        cov=cov_a,
        labels=("H0", "Omega_m", "alpha_rd"),
        label=f"Planck chain prior (root={Path(chain_root).name})",
    )

def chi2_sn(params: dict, sn: SNData, tables: CosmologyTables, epochs: LTAEpochs,
            use_lta: bool, lta_override=None, invmap=None) -> Tuple[float, dict]:
    """
    SN chi2. Supports two SN observable modes:
    - If y_is_mu: compare mu_obs - mu_pred directly.
    - Else (m_b_corr): solve analytically for best-fit additive magnitude intercept M.
    """
    # LTA config
    if use_lta:
        if lta_override is not None:
            lta = lta_override
        else:
            lta = LTAParams(
                s_anchor_km_s_per_mpc=params["s_anchor"],
                g_complex=params["g_complex"],
                g_life=params["g_life"],
            )
    else:
        lta = LTAParams(s_anchor_km_s_per_mpc=0.0, g_complex=0.0, g_life=0.0)

    # If there is no LTA effect, mapping is identity: zcos = zobs.
    if float(getattr(lta, "s_anchor_km_s_per_mpc", 0.0)) == 0.0:
        zcos = sn.zHD.copy()   # in chi2_sn
    else:
        if invmap is None:
            zcos = invert_zobs_to_zcos(sn.zHD, tables, lta, epochs)
        else:
            zcos = invmap(sn.zHD)

    if not np.all(np.isfinite(zcos)):
        return 1e80, {"invalid": "nonfinite zcos"}

    chi = tables.chi_of_z(zcos)

    # Pantheon-style luminosity distance:
    dL_mpc = (1.0 + sn.zHEL) * chi
    mu_pred = 5.0 * np.log10(dL_mpc) + 25.0

    # For m_b_corr + calibrators, use μ_ref (μ_pred for HF, μ_Ceph for calibrators)
    mu_ref = sn_mu_reference(sn, mu_pred)

    if not np.all(np.isfinite(mu_ref)):
        return 1e80, {"invalid": "nonfinite mu_ref"}

    resid_info = {}

    if sn.y_is_mu:
        resid = sn.y - mu_ref
        chi2 = float(resid @ cho_solve(sn.cho, resid, check_finite=False))
        return chi2, resid_info

    # ---- m_b_corr branch ----
    # If ladder-anchor is enabled: evaluate HF|calibrators likelihood (full covariance), with M fixed by calibrators.
    if sn.anchor is not None:
        a = sn.anchor

        # HF residual in the conditional/marginalized model:
        # mean_H = mu_pred_H + mu_shift + M_cal * v
        y_hf  = sn.y[a.idx_hf]
        mu_hf = mu_pred[a.idx_hf]

        mean_hf = mu_hf + a.mu_shift_hf + a.M_cal * a.v_hf
        r_hf = y_hf - mean_hf

        chi2_hf = float(r_hf @ cho_solve(a.cho_hf_eff, r_hf, check_finite=False))

        # Add calibrator constant for reporting consistency (does not affect optimization/deltas)
        chi2 = float(chi2_hf + a.chi2_cal_const)

        resid_info["M_cal"] = float(a.M_cal)
        resid_info["sig_M_cal"] = float(a.sig_M)
        resid_info["chi2_hf"] = float(chi2_hf)
        resid_info["chi2_cal_const"] = float(a.chi2_cal_const)
        return chi2, resid_info

    # Otherwise: original Pantheon-style intercept profiling using all selected SNe
    r0 = sn.y - mu_ref
    Cinv_r0 = cho_solve(sn.cho, r0, check_finite=False)
    M_best = float((sn.ones @ Cinv_r0) / sn.ones_Cinv_ones)
    resid = r0 - M_best
    chi2 = float(resid @ cho_solve(sn.cho, resid, check_finite=False))
    resid_info["M_best"] = M_best
    return chi2, resid_info


def chi2_bao(params: dict, bao: BAOData, tables: CosmologyTables, epochs: LTAEpochs,
             use_lta: bool, lta_override=None, invmap=None) -> float:
    # LTA config
    if use_lta:
        if lta_override is not None:
            lta = lta_override
        else:
            lta = LTAParams(
                s_anchor_km_s_per_mpc=params["s_anchor"],
                g_complex=params["g_complex"],
                g_life=params["g_life"],
            )
    else:
        lta = LTAParams(s_anchor_km_s_per_mpc=0.0, g_complex=0.0, g_life=0.0)

    if float(getattr(lta, "s_anchor_km_s_per_mpc", 0.0)) == 0.0:
        zcos = bao.z.copy()    # in chi2_bao
    else:
        if invmap is None:
            zcos = invert_zobs_to_zcos(bao.z, tables, lta, epochs)
        else:
            zcos = invmap(bao.z)

    if not np.all(np.isfinite(zcos)):
        return 1e80

    chi = tables.chi_of_z(zcos)

    alpha_rd = params["alpha_rd"]

    # Transverse BAO observable
    I = lta_integral_I(chi, tables, lta, epochs)  # dimensionless

    DM_pred = chi / alpha_rd

    # Radial BAO observable ~ H_obs * (r_d / r_d,fid)
    Hcos = tables.H_of_z(zcos)
    jac = dzobs_dzcos(zcos, tables, lta, epochs)
    Hz_pred = (Hcos * jac) * alpha_rd
    
    if (not np.all(np.isfinite(DM_pred))) or (not np.all(np.isfinite(Hz_pred))):
        return 1e80

    pred_vec = np.column_stack([DM_pred, Hz_pred]).reshape(-1)

    resid = bao.data_vector - pred_vec
    return float(resid @ cho_solve(bao.cho, resid, check_finite=False))


def total_chi2(x: np.ndarray, sn: SNData, bao: BAOData, epochs: LTAEpochs,
               use_lta: bool, zmax_table: float) -> float:
    """
    Parameter vector conventions:
    - baseline (use_lta=False): x = [H0, Om, alpha_rd]
    - lta (use_lta=True): x = [H0, Om, alpha_rd, t_life_gyr, s_anchor, g_complex, g_life]

      where t_life_gyr is the proper lookback time of the Earth-life event in Gyr.
    """
    if use_lta:

        if LTA_FORM == "powerlaw":
            H0, Om, alpha_rd, tL, s_anchor, g_c, g_l = x

            if (H0 <= 0) or (Om <= 0) or (Om >= 1) or (alpha_rd <= 0) or (s_anchor < 0):
                return np.inf
            if (tL <= 1.0) or (tL >= 8.0):
                return np.inf

            if (s_anchor > 0.0) and ((g_c <= 0.0) or (g_l < 0.0)):
                return np.inf

            params = {
                "H0": H0,
                "Om": Om,
                "alpha_rd": alpha_rd,
                "s_anchor": s_anchor,
                "g_complex": g_c,
                "g_life": g_l,
                "t_life_gyr": tL,
            }

            epochs_eff = LTAEpochs(
                t_digital_gyr=epochs.t_digital_gyr,
                t_complex_gyr=epochs.t_complex_gyr,
                t_life_gyr=tL,
            )

            lta_obj = LTAParams(s_anchor_km_s_per_mpc=s_anchor, g_complex=g_c, g_life=g_l)
            reg_penalty = 0.0
            # Guard against pathological anchor normalization (g(t_anchor) -> 0)
            if (LTA_T_ANCHOR_GYR is not None) and (s_anchor > 0.0):
                gA = float(earth_history_g(np.array([LTA_T_ANCHOR_GYR], dtype=float), lta_obj, epochs_eff)[0])

                # If gA is nonpositive or nonfinite, this parameter set is unusable.
                if (not np.isfinite(gA)) or (gA <= 0.0):
                    return 1e80

                # If gA is "too small", the anchored parameterization becomes numerically ill-conditioned:
                # s_now = s_anchor / gA explodes and the optimizer can get stuck in inf-land.
                if gA <= G_ANCHOR_MIN:
                    # Return a HUGE FINITE penalty (NOT np.inf) so Powell can recover.
                    # Use a log barrier to avoid overflow even if gA is extremely tiny.
                    lr = np.log(G_ANCHOR_MIN / max(gA, 1e-300))
                    return 1e80 * (1.0 + lr * lr)



        else:
            return np.inf

    else:
        H0, Om, alpha_rd = x
        if (H0 <= 0) or (Om <= 0) or (Om >= 1) or (alpha_rd <= 0):
            return np.inf

        params = {
            "H0": H0,
            "Om": Om,
            "alpha_rd": alpha_rd,
            "s_anchor": 0.0,
            "g_complex": 0.0,
            "g_life": 0.0,
            "t_life_gyr": epochs.t_life_gyr,
        }
        epochs_eff = epochs
        lta_obj = LTAParams(s_anchor_km_s_per_mpc=0.0, g_complex=0.0, g_life=0.0)
        reg_penalty = 0.0
    
    tables = build_cosmology_tables(H0=H0, Om=Om, zmax=zmax_table)
    invmap = None

    chi2_sn_val, _ = chi2_sn(
        params, sn, tables, epochs_eff,
        use_lta=use_lta,
        lta_override=lta_obj,
        invmap=invmap,
    )
    chi2_bao_val = chi2_bao(
        params, bao, tables, epochs_eff,
        use_lta=use_lta,
        lta_override=lta_obj,
        invmap=invmap,
    )
    chi2_prior = 0.0
    global EARLY_PRIORS
    if EARLY_PRIORS is not None:
        chi2_prior = EARLY_PRIORS.chi2(H0, Om, alpha_rd)

    return (chi2_sn_val + chi2_bao_val + reg_penalty + chi2_prior)


def count_free_params(bounds: list[tuple[float, float]], tol: float = 1e-12) -> int:
    k = 0
    for lo, hi in bounds:
        if (hi - lo) > tol:
            k += 1
    return k

@dataclass(frozen=True)
class FormSetup:
    x0: np.ndarray
    bounds: list[tuple[float, float]]
    options: dict
    x0_list: Optional[list[np.ndarray]] = None

@dataclass(frozen=True)
class FitBundle:
    form: str
    res: object
    x0_used: np.ndarray
    bounds: list[tuple[float, float]]
    H0: float
    Om: float
    alpha_rd: float
    tL: float
    s_anchor: float
    gC: float
    gL: float
    lta_best_obj: object
    epochs_best: LTAEpochs


def _default_eps_vec(n: int) -> np.ndarray:
    """
    Reasonable finite-difference step sizes for L-BFGS-B when scipy uses approx gradients.
    Must match x0 length exactly.
    """
    n = int(n)
    eps = np.empty(n, dtype=float)
    if n >= 4:
        eps[0] = 0.05   # H0-ish
        eps[1] = 0.002  # Om-ish
        eps[2] = 0.002  # alpha-ish
        eps[3] = 0.01   # t_life-ish
        if n > 4:
            eps[4:] = 0.05
            eps[4] = 0.10
    else:
        eps[:] = 1e-3
    return eps


def _coerce_eps_option(opts_in: dict, bounds: list[tuple[float, float]], tol: float = 1e-12) -> dict:
    """
    Ensure opts['eps'] is compatible with SciPy's fixed-variable removal.

    SciPy's minimize() removes variables whose bounds are exactly fixed (lo==hi)
    when finite-difference gradients are needed (e.g., L-BFGS-B with jac=None).
    If we pass a full-length eps vector, SciPy will later finite-difference on the
    reduced x, and you get the (27,) vs (28,) broadcast error.

    This function:
      - keeps scalar eps as-is (always safe)
      - accepts eps vectors of length n_full OR n_free
      - if bounds contain fixed vars and eps is full-length, drops fixed entries
      - if eps length is wrong, rebuilds a reasonable default then applies the same rule
    """
    opts = dict(opts_in) if opts_in is not None else {}
    eps = opts.get("eps", None)

    # None or scalar => always OK
    if eps is None or np.isscalar(eps):
        return opts

    eps_vec = np.asarray(eps, dtype=float).ravel()
    if eps_vec.size == 1:
        opts["eps"] = float(eps_vec[0])
        return opts

    n_full = int(len(bounds))
    fixed = np.array([abs(hi - lo) <= tol for (lo, hi) in bounds], dtype=bool)
    n_free = n_full - int(np.sum(fixed))

    # Accept either full-length or already-reduced eps vectors.
    if eps_vec.size not in (n_full, n_free):
        eps_vec = _default_eps_vec(n_full)

    # If SciPy will remove fixed vars, eps must match the reduced x dimension.
    if fixed.any():
        if n_free <= 0:
            # everything fixed; eps won't be used meaningfully
            opts["eps"] = 1e-4
            return opts

        if eps_vec.size == n_full:
            eps_vec = eps_vec[~fixed]  # drop fixed coords

        # If only 1 free var remains, scalar is acceptable
        if eps_vec.size == 1:
            opts["eps"] = float(eps_vec[0])
        else:
            opts["eps"] = eps_vec
        return opts

    # No fixed variables: ensure full-length vector
    if eps_vec.size != n_full:
        eps_vec = _default_eps_vec(n_full)

    opts["eps"] = eps_vec
    return opts

def Heff_of_zobs(zobs, H0, Om, zmax_table, lta_obj, epochs):
    zobs = np.asarray(zobs, dtype=float)
    tables = build_cosmology_tables(H0=H0, Om=Om, zmax=zmax_table)

    # Use Newton inversion here (smooth, no staircasing)
    zcos = invert_zobs_to_zcos(zobs, tables, lta_obj, epochs)

    jac = dzobs_dzcos(zcos, tables, lta_obj, epochs)   # dz_obs/dz_cos
    Hcos = tables.H_of_z(zcos)

    return Hcos * jac

def fit_baseline(
    sn: SNData,
    bao: BAOData,
    epochs: LTAEpochs,
    zmax_table: float,
    x0: Optional[np.ndarray] = None,
    bounds: Optional[list[tuple[float, float]]] = None,
    maxiter: int = 250,
) -> object:
    if x0 is None:
        x0 = np.array([70.0, 0.30, 1.0], dtype=float)
    if bounds is None:
        bounds = [(40.0, 100.0), (0.05, 0.60), (0.6, 1.4)]

    res = minimize(
        lambda x: total_chi2(x, sn, bao, epochs, use_lta=False, zmax_table=zmax_table),
        np.array(x0, dtype=float),
        method="Powell",
        bounds=bounds,
        options={"maxiter": int(maxiter)},
    )
    return res


def make_form_setup(
    form: str,
    H0_b: float,
    Om_b: float,
    alpha_b: float,
    epochs: LTAEpochs,
    args: argparse.Namespace,
) -> FormSetup:
    tL0 = float(epochs.t_life_gyr)
    if form == "powerlaw":
        x0 = np.array([H0_b, Om_b, alpha_b,
                    epochs.t_life_gyr,
                    6.0,
                    0.3,   # B
                    1.0])  # p
        bounds = [
            (40.0, 100.0),
            (0.05, 0.60),
            (0.6, 1.4),
            (tL0, tL0),
            (0.0, 50.0),
            (1e-4, 1.0),   # B
            (1, 50.0),   # p
        ]
        options = {
            "maxiter": 400,
        }
        return FormSetup(x0=x0, bounds=bounds, options=options)

    raise ValueError(f"Unknown LTA form '{form}'")


def fit_form(
    form: str,
    sn: SNData,
    bao: BAOData,
    epochs: LTAEpochs,
    zmax_table: float,
    H0_b: float,
    Om_b: float,
    alpha_b: float,
    args: argparse.Namespace,
) -> FitBundle:
    global LTA_FORM
    LTA_FORM = form

    setup = make_form_setup(form, H0_b=H0_b, Om_b=Om_b, alpha_b=alpha_b, epochs=epochs, args=args)
    obj = lambda x: total_chi2(x, sn, bao, epochs, use_lta=True, zmax_table=zmax_table)

    if (
        (getattr(args, "fix_h0", None) is not None)
        or (getattr(args, "fix_om", None) is not None)
        or (getattr(args, "fix_alpha_rd", None) is not None)
        or (getattr(args, "fix_g_complex", None) is not None)
        or (getattr(args, "fix_g_life", None) is not None)
    ):
        bounds_fixed = list(setup.bounds)
        x0_fixed = np.array(setup.x0, dtype=float).copy()

        x0_list_fixed = None
        if setup.x0_list is not None:
            x0_list_fixed = []
            for x in setup.x0_list:
                x0_list_fixed.append(np.array(x, dtype=float).copy())

        def _apply_fix(idx: int, val: float) -> None:
            bounds_fixed[idx] = (float(val), float(val))
            x0_fixed[idx] = float(val)
            if x0_list_fixed is not None:
                for xx in x0_list_fixed:
                    xx[idx] = float(val)

        if args.fix_g_complex is not None:
            _apply_fix(5, float(args.fix_g_complex))
        if args.fix_g_life is not None:
            _apply_fix(6, float(args.fix_g_life))

        if getattr(args, "fix_h0", None) is not None:
            h0_fix = float(args.fix_h0)
            if (not np.isfinite(h0_fix)) or (h0_fix <= 0.0):
                raise ValueError(f"--fix-h0 must be a positive finite number, got {args.fix_h0}")
            _apply_fix(0, h0_fix)

        if getattr(args, "fix_alpha_rd", None) is not None:
            a_fix = float(args.fix_alpha_rd)
            if (not np.isfinite(a_fix)) or (a_fix <= 0.0):
                raise ValueError(f"--fix-alpha-rd must be a positive finite number, got {args.fix_alpha_rd}")
            _apply_fix(2, a_fix)

        if getattr(args, "fix_om", None) is not None:
            om_fix = float(args.fix_om)
            if (not np.isfinite(om_fix)) or (om_fix <= 0.0) or (om_fix >= 1.0):
                raise ValueError(f"--fix-om must satisfy 0 < Om < 1, got {args.fix_om}")
            _apply_fix(1, om_fix)

        setup = FormSetup(x0=x0_fixed, bounds=bounds_fixed, options=setup.options)
    
    x0_used = setup.x0.copy()

    opts = _coerce_eps_option(setup.options, setup.bounds)
    res = minimize(obj, x0_used, method="Powell", bounds=setup.bounds, options=opts)
    
    # ------------------------------------------------------------------
    # Nested-safe guard:
    # LTA model contains baseline exactly at s_anchor=0, with (H0,Om,alpha) = baseline fit.
    # If the optimizer returns a worse point, replace it with the nested baseline point.
    # ------------------------------------------------------------------
    tL0 = float(setup.x0[3])
    B0  = float(setup.x0[5])
    p0  = float(setup.x0[6])
    x_nested = np.array([float(H0_b), float(Om_b), float(alpha_b), tL0, 0.0, B0, p0], dtype=float)
    chi2_nested = float(obj(x_nested))

    if np.isfinite(chi2_nested) and (chi2_nested < float(res.fun)):
        res.x = x_nested
        res.fun = chi2_nested
        res.success = True
        res.message = "Nested-safe: used baseline (s_anchor=0) point because it was better than optimizer result."

    # Build best-fit LTA object + epochs_best
    xbest = np.asarray(res.x, dtype=float)
    chi2 = float(res.fun)

    # Parametric forms: x = [H0, Om, alpha, tL, s_anchor, gC, gL]
    H0_l, Om_l, alpha_l, tL_l, sA_l, gC_l, gL_l = xbest
    lta_best_obj = LTAParams(s_anchor_km_s_per_mpc=float(sA_l), g_complex=float(gC_l), g_life=float(gL_l))
    epochs_best = LTAEpochs(
        t_digital_gyr=epochs.t_digital_gyr,
        t_complex_gyr=epochs.t_complex_gyr,
        t_life_gyr=float(tL_l),
    )
    return FitBundle(
        form=form,
        res=res,
        x0_used=x0_used,
        bounds=setup.bounds,
        H0=float(H0_l),
        Om=float(Om_l),
        alpha_rd=float(alpha_l),
        tL=float(tL_l),
        s_anchor=float(sA_l),
        gC=float(gC_l),
        gL=float(gL_l),
        lta_best_obj=lta_best_obj,
        epochs_best=epochs_best,
    )

def clone_sn_with_y(sn: SNData, y_new: np.ndarray) -> SNData:
    sn_new = SNData(
        zHD=sn.zHD,
        zHEL=sn.zHEL,
        idsurvey=sn.idsurvey,
        cid=sn.cid,
        y=np.asarray(y_new, dtype=float),
        cov=sn.cov,
        cho=sn.cho,
        y_is_mu=sn.y_is_mu,
        ones=sn.ones,
        Cinv_ones=sn.Cinv_ones,
        ones_Cinv_ones=sn.ones_Cinv_ones,
        is_calibrator=sn.is_calibrator,
        mu_ceph=sn.mu_ceph,
        anchor=None,
        group=sn.group,
    )
    if sn.anchor is not None:
        sn_new.anchor = build_sn_anchor(sn_new)
    return sn_new

def clone_bao_with_vec(bao: BAOData, vec_new: np.ndarray) -> BAOData:
    vec_new = np.asarray(vec_new, dtype=float)
    return BAOData(
        z=bao.z,
        DM=vec_new[0::2].copy(),
        Hz=vec_new[1::2].copy(),
        cov=bao.cov,
        cho=bao.cho,
        data_vector=vec_new.copy(),
    )


def predict_sn_mu(
    sn: SNData,
    H0: float,
    Om: float,
    epochs: LTAEpochs,
    zmax_table: float,
    lta_obj: object,
    invmap_n: Optional[int] = 4000,
) -> np.ndarray:
    """
    If invmap_n is None: use Newton inversion (matches fit-time behavior).
    Else: use a precomputed inverse map with invmap_n grid points (faster, approximate).
    """
    tables = build_cosmology_tables(H0=H0, Om=Om, zmax=zmax_table)

    if invmap_n is None:
        zcos = invert_zobs_to_zcos(sn.zHD, tables, lta_obj, epochs)
    else:
        invmap = build_inverse_zmap(tables, lta_obj, epochs, zmax=zmax_table, nz=int(invmap_n))
        zcos = invmap(sn.zHD)

    chi = tables.chi_of_z(zcos)
    dL_mpc = (1.0 + sn.zHEL) * chi
    return 5.0 * np.log10(dL_mpc) + 25.0


def predict_bao_DM_Hz(
    bao: BAOData,
    H0: float,
    Om: float,
    alpha_rd: float,
    epochs: LTAEpochs,
    zmax_table: float,
    lta_obj: object,
    invmap_n: Optional[int] = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    If invmap_n is None: use Newton inversion (matches fit-time behavior).
    Else: use a precomputed inverse map (faster, approximate).
    """
    tables = build_cosmology_tables(H0=H0, Om=Om, zmax=zmax_table)

    if invmap_n is None:
        zcos = invert_zobs_to_zcos(bao.z, tables, lta_obj, epochs)
    else:
        invmap = build_inverse_zmap(tables, lta_obj, epochs, zmax=zmax_table, nz=int(invmap_n))
        zcos = invmap(bao.z)

    chi = tables.chi_of_z(zcos)
    DM = chi / float(alpha_rd)

    Hcos = tables.H_of_z(zcos)
    jac = dzobs_dzcos(zcos, tables, lta_obj, epochs)
    Hz = (Hcos * jac) * float(alpha_rd)
    return DM, Hz

def sn_bestfit_M_for_indices(sn: SNData, mu_pred: np.ndarray, idx_fit: Optional[np.ndarray]) -> float:
    if sn.y_is_mu:
        return 0.0
    if idx_fit is None:
        idx = np.arange(sn.y.size, dtype=int)
    else:
        idx = np.asarray(idx_fit, dtype=int)
    C = sn.cov[np.ix_(idx, idx)]
    cho = cho_factor(C, lower=True, check_finite=False)
    mu_ref = sn_mu_reference(sn, mu_pred)
    r0 = sn.y[idx] - mu_ref[idx]
    ones = np.ones(idx.size, dtype=float)
    Cinv_r0 = cho_solve(cho, r0, check_finite=False)
    Cinv_ones = cho_solve(cho, ones, check_finite=False)
    denom = float(ones @ Cinv_ones)
    if denom == 0.0:
        raise ZeroDivisionError("ones^T C^{-1} ones is zero; covariance invalid?")
    return float((ones @ Cinv_r0) / denom)

def simulate_mock_dataset(
    sn: SNData,
    bao: BAOData,
    mu_model: np.ndarray,
    DM_model: np.ndarray,
    Hz_model: np.ndarray,
    rng: np.random.Generator,
    M_intercept: float = 0.0,
) -> tuple[SNData, BAOData]:
    """
    Generate a mock SN+BAO dataset drawn from a multivariate Gaussian with the *same* covariances,
    centered on the provided model predictions.

    For SN:
      - If sn.y_is_mu: y_mean = mu_model
      - Else:          y_mean = mu_model + M_intercept   (Pantheon/SH0ES-style intercept)

    For BAO:
      - data vector ordering is [DM1, H1, DM2, H2, ...] and cov matches that ordering.
    """
    mu_model = np.asarray(mu_model, dtype=float)
    DM_model = np.asarray(DM_model, dtype=float)
    Hz_model = np.asarray(Hz_model, dtype=float)

    # SN mean in data space
    if sn.y_is_mu:
        y_mean = mu_model
    else:
        mu_ref = sn_mu_reference(sn, mu_model)
        y_mean = mu_ref + float(M_intercept)


    # Draw correlated SN noise
    Lsn_raw, lower = sn.cho
    Lsn = np.tril(Lsn_raw) if lower else np.triu(Lsn_raw)
    y_mock = y_mean + (Lsn @ rng.standard_normal(y_mean.size))

    sn_mock = clone_sn_with_y(sn, y_mock)

    # BAO mean vector
    vec_mean = np.column_stack([DM_model, Hz_model]).reshape(-1)

    # Draw correlated BAO noise
    Lbao_raw, lower = bao.cho
    Lbao = np.tril(Lbao_raw) if lower else np.triu(Lbao_raw)
    vec_mock = vec_mean + (Lbao @ rng.standard_normal(vec_mean.size))

    bao_mock = clone_bao_with_vec(bao, vec_mock)

    return sn_mock, bao_mock

def _logdet_from_cho(cho: tuple) -> float:
    """log(det(C)) from cho_factor(C, lower=True)."""
    L, lower = cho
    d = np.diag(L).astype(float)
    d = np.maximum(d, np.finfo(float).tiny)
    return float(2.0 * np.sum(np.log(d)))

def gaussian_nll0(r: np.ndarray, C: np.ndarray) -> tuple[float, float, float, tuple]:
    """
    Return (nll, quad, logdet, cho) for N(0, C) up to an additive constant.
    nll = r^T C^{-1} r + log det(C)
    """
    r = np.asarray(r, dtype=float)
    C = np.asarray(C, dtype=float)
    cho = cho_factor(C, lower=True, check_finite=False)
    quad = float(r @ cho_solve(cho, r, check_finite=False))
    logdet = _logdet_from_cho(cho)
    return float(quad + logdet), quad, logdet, cho


def _effective_fd_steps(x: np.ndarray, bounds: list[tuple[float, float]], eps: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Symmetric finite-difference steps that stay inside bounds.
    If a variable is fixed (hi==lo) or too close to a bound, step -> 0.
    """
    x = np.asarray(x, dtype=float).ravel()
    eps = np.asarray(eps, dtype=float).ravel()
    h = eps.copy()
    for j, (lo, hi) in enumerate(bounds):
        lo = float(lo); hi = float(hi)
        if abs(hi - lo) <= tol:
            h[j] = 0.0
            continue
        # symmetric room on both sides
        room = min(x[j] - lo, hi - x[j])
        if (not np.isfinite(room)) or (room <= 0.0):
            h[j] = 0.0
            continue
        h[j] = min(h[j], room)
        if (not np.isfinite(h[j])) or (h[j] <= 0.0):
            h[j] = 0.0
    return h


def fd_hessian_chi2(fun_chi2, x0: np.ndarray, bounds: list[tuple[float, float]], eps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finite-difference Hessian of chi2 at x0 using symmetric steps inside bounds.
    Returns (free_idx, h_eff, H_free) where H_free is Hessian in the free subspace.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    h = _effective_fd_steps(x0, bounds, eps)
    free_idx = np.where(h > 0.0)[0]
    if free_idx.size == 0:
        return free_idx, h, np.zeros((0, 0), dtype=float)

    f0 = float(fun_chi2(x0))

    m = int(free_idx.size)
    H = np.zeros((m, m), dtype=float)

    # Diagonal
    for a, i in enumerate(free_idx):
        hi = float(h[i])
        xp = x0.copy(); xp[i] += hi
        xm = x0.copy(); xm[i] -= hi
        fp = float(fun_chi2(xp))
        fm = float(fun_chi2(xm))
        H[a, a] = (fp - 2.0 * f0 + fm) / (hi * hi)

    # Off-diagonal
    for a in range(m):
        i = int(free_idx[a]); hi = float(h[i])
        for b in range(a + 1, m):
            j = int(free_idx[b]); hj = float(h[j])

            xpp = x0.copy(); xpp[i] += hi; xpp[j] += hj
            xpm = x0.copy(); xpm[i] += hi; xpm[j] -= hj
            xmp = x0.copy(); xmp[i] -= hi; xmp[j] += hj
            xmm = x0.copy(); xmm[i] -= hi; xmm[j] -= hj

            fpp = float(fun_chi2(xpp))
            fpm = float(fun_chi2(xpm))
            fmp = float(fun_chi2(xmp))
            fmm = float(fun_chi2(xmm))

            Hij = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj)
            H[a, b] = Hij
            H[b, a] = Hij

    H = 0.5 * (H + H.T)
    return free_idx, h, H


def laplace_posterior_cov(fun_chi2, xhat: np.ndarray, bounds: list[tuple[float, float]], eps: np.ndarray, ridge_rel: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Laplace approx: p(theta|train) ~ N(xhat, Sigma), using chi2 Hessian.
    Sigma_free ≈ 2 * inv(H_chi2_free). Returns (free_idx, h_eff, Sigma_free).
    """
    free_idx, h_eff, H = fd_hessian_chi2(fun_chi2, xhat, bounds, eps)

    if H.size == 0:
        return free_idx, h_eff, np.zeros((0, 0), dtype=float)

    # Eigen-regularize (robust if H is not PD due to numerical noise)
    w, V = np.linalg.eigh(H)
    w = np.asarray(w, dtype=float)

    pos = w[w > 0]
    scale = float(np.median(pos)) if pos.size else 1.0
    scale = float(scale if np.isfinite(scale) and scale > 0 else 1.0)

    floor = float(max(ridge_rel * scale, np.finfo(float).tiny))
    w_reg = np.maximum(w, floor)

    # Sigma = 2 * V diag(1/w_reg) V^T
    inv_w = 1.0 / w_reg
    Sigma = 2.0 * (V @ (inv_w[:, None] * V.T))
    Sigma = 0.5 * (Sigma + Sigma.T)
    return free_idx, h_eff, Sigma


def fd_jacobian_vec(fun_vec, x0: np.ndarray, bounds: list[tuple[float, float]], eps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Central finite-difference Jacobian for a vector-valued function.
    Returns (free_idx, h_eff, J) with J shape (len(vec), n_free).
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    y0 = np.asarray(fun_vec(x0), dtype=float).ravel()

    h = _effective_fd_steps(x0, bounds, eps)
    free_idx = np.where(h > 0.0)[0]
    if free_idx.size == 0:
        return free_idx, h, np.zeros((y0.size, 0), dtype=float)

    J = np.zeros((y0.size, free_idx.size), dtype=float)

    for k, i in enumerate(free_idx):
        hi = float(h[i])
        xp = x0.copy(); xp[i] += hi
        xm = x0.copy(); xm[i] -= hi
        yp = np.asarray(fun_vec(xp), dtype=float).ravel()
        ym = np.asarray(fun_vec(xm), dtype=float).ravel()
        J[:, k] = (yp - ym) / (2.0 * hi)

    return free_idx, h, J


def posterior_predictive_nll(
    r_hat: np.ndarray,
    C_base: np.ndarray,
    J: np.ndarray,
    Sigma_theta: np.ndarray,
    jitter_frac: float = 1e-12,
) -> dict:
    """
    Posterior predictive for residual vector r:
      r ~ N(r_hat, C_base + J Sigma_theta J^T)
    Returns dict with nll/quad/logdet and jitter used.
    """
    r_hat = np.asarray(r_hat, dtype=float).ravel()
    C_base = np.asarray(C_base, dtype=float)
    J = np.asarray(J, dtype=float)
    Sigma_theta = np.asarray(Sigma_theta, dtype=float)

    C_pred = C_base.copy()
    if (J.size > 0) and (Sigma_theta.size > 0):
        C_pred = C_pred + (J @ Sigma_theta @ J.T)

    C_pred = 0.5 * (C_pred + C_pred.T)

    jitter_used = 0.0
    try:
        nll, quad, logdet, cho = gaussian_nll0(r_hat, C_pred)
    except np.linalg.LinAlgError:
        diag = np.diag(C_pred).astype(float)
        scale = float(np.median(diag[diag > 0])) if np.any(diag > 0) else 1.0
        jitter_used = float(jitter_frac * scale)
        C_pred = C_pred + jitter_used * np.eye(C_pred.shape[0], dtype=float)
        C_pred = 0.5 * (C_pred + C_pred.T)
        nll, quad, logdet, cho = gaussian_nll0(r_hat, C_pred)

    return dict(
        nll=float(nll),
        quad=float(quad),
        logdet=float(logdet),
        jitter_used=float(jitter_used),
    )


def chi2_gaussian_diag(r: np.ndarray, C: np.ndarray) -> float:
    """χ² using ONLY diagonal variances of C (sanity check / robustness)."""
    r = np.asarray(r, dtype=float)
    v = np.diag(np.asarray(C, dtype=float)).astype(float)
    v = np.maximum(v, np.finfo(float).tiny)
    return float(np.sum((r * r) / v))


def conditional_precompute(
    C_full: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    jitter_frac: float = 1e-12,
    *,
    cross_mode: str = "full",            # "full" | "full_quotient" | "zero" | "within_survey" | "survey_avg"
    groups: Optional[np.ndarray] = None, # IDSURVEY aligned with C_full indexing
) -> dict:
    """
    Precompute Schur-complement objects for conditional χ²_test|train.

    cross_mode:
      - full:         use the covariance as-is
      - full_quotient: use FULL C_VT pattern, but rescale cross-survey VT entries by a
                    deterministic quotient q in [0,1] computed from the covariance:
                        q = min(1, med(|corr| train–train cross-survey) / med(|corr| test–train cross-survey))
                    This preserves FULL structure but prevents “inflated VT” outlier folds.
                    Same-survey VT entries (if any) are NOT scaled.
      - zero:         force C_VT=0 (test independent of train)
      - within_survey:keep only C_VT entries where group(test)==group(train)
      - survey_avg:   IMPUTE C_VT using TRAIN-derived survey-average cross correlations:
                      rho_s = mean corr between survey s and other TRAIN surveys
                      C_VT[i,j] = rho_{survey(train j)} * sqrt(varV_i * varT_j)
    """
    idx_train = np.asarray(idx_train, dtype=int)
    idx_test  = np.asarray(idx_test, dtype=int)
    C_full = np.asarray(C_full, dtype=float)

    cross_mode = str(cross_mode).strip().lower()
    if cross_mode not in ("full", "full_quotient", "zero", "within_survey", "survey_avg"):
        raise ValueError(f"Unknown cross_mode={cross_mode!r}")

    # Diagnostics/bookkeeping for cross regularization
    cross_scale = 1.0
    cross_stat_train_med_abs_corr = float("nan")
    cross_stat_vt_med_abs_corr = float("nan")

    # Extract blocks
    C_TT = C_full[np.ix_(idx_train, idx_train)]
    C_VV = C_full[np.ix_(idx_test,  idx_test)]
    C_TV = C_full[np.ix_(idx_train, idx_test)]
    C_VT = C_TV.T

    # Need groups for within_survey / survey_avg / full_quotient
    if cross_mode in ("within_survey", "survey_avg", "full_quotient"):
        if groups is None:
            raise ValueError(f"cross_mode={cross_mode!r} requires groups=... aligned with C_full indices.")
        groups = np.asarray(groups, dtype=int)
        gT = groups[idx_train]
        gV = groups[idx_test]

    # Apply cross-mode
    if cross_mode == "zero":
        C_TV = np.zeros_like(C_TV)
        C_VT = np.zeros_like(C_VT)
    
    elif cross_mode == "full_quotient":
        # FULL pattern cross-covariance, but rescale cross-survey VT entries by a deterministic
        # quotient q in [0,1] so that test–train cross-survey coupling does not exceed the
        # typical train–train cross-survey coupling (robust median |corr|).
        dT = np.maximum(np.diag(C_TT).astype(float), np.finfo(float).tiny)
        dV = np.maximum(np.diag(C_VV).astype(float), np.finfo(float).tiny)

        corr_TT = C_TT / np.sqrt(np.outer(dT, dT))
        corr_VT_full = C_VT / np.sqrt(np.outer(dV, dT))

        # Typical cross-survey |corr| inside TRAIN
        m_train_cross = (gT[:, None] != gT[None, :])
        vals_train = np.abs(corr_TT[m_train_cross]).ravel()
        vals_train = vals_train[np.isfinite(vals_train)]
        cross_stat_train_med_abs_corr = float(np.median(vals_train)) if vals_train.size else 0.0

        # Cross-survey |corr| between TEST and TRAIN (only for pairs with different IDSURVEY)
        m_vt_cross = (gV[:, None] != gT[None, :])
        vals_vt = np.abs(corr_VT_full[m_vt_cross]).ravel()
        vals_vt = vals_vt[np.isfinite(vals_vt)]
        cross_stat_vt_med_abs_corr = float(np.median(vals_vt)) if vals_vt.size else 0.0

        # Compute quotient (never amplify; only shrink)
        if cross_stat_vt_med_abs_corr <= 0.0:
            # If VT cross-survey is essentially zero, don't touch it.
            cross_scale = 1.0
        else:
            cross_scale = float(min(1.0, cross_stat_train_med_abs_corr / cross_stat_vt_med_abs_corr))

        # Apply scaling ONLY to cross-survey VT entries (same-survey VT entries remain FULL)
        if cross_scale < 1.0 and np.any(m_vt_cross):
            C_VT = C_VT.copy()
            C_VT[m_vt_cross] *= cross_scale
            C_TV = C_VT.T

    elif cross_mode == "within_survey":
        # keep only same-group cross entries
        m = (gV[:, None] == gT[None, :])
        C_VT = C_VT * m
        C_TV = C_VT.T

    elif cross_mode == "survey_avg":
        # Build TRAIN-derived rho_s for each TRAIN survey s:
        # rho_s = mean_{i in s, j not in s} corr(i,j) using TRAIN covariance only
        dT = np.maximum(np.diag(C_TT).astype(float), np.finfo(float).tiny)
        sT = np.sqrt(dT)

        # Correlation between TRAIN points
        denom_TT = np.sqrt(np.outer(dT, dT))
        corr_TT = C_TT / denom_TT

        uniq = np.unique(gT)

        rho_by_sid: dict[int, float] = {}
        for sid in uniq:
            m_in = (gT == sid)

            per_other = []
            for tid in uniq:
                if tid == sid:
                    continue
                m_out = (gT == tid)
                if not np.any(m_out):
                    continue

                block = corr_TT[np.ix_(m_in, m_out)].ravel()
                block = block[np.isfinite(block)]
                if block.size == 0:
                    continue

                # Robust within-pair summary (median is parameter-free)
                per_other.append(float(np.median(block)))

            # Survey-balanced mean across other surveys (each tid counts equally)
            rho_by_sid[int(sid)] = float(np.mean(per_other)) if per_other else 0.0

        rho_train = np.array([rho_by_sid[int(s)] for s in gT], dtype=float)

        # Impute C_VT from rho_train and diagonals of TT/VV
        dV = np.maximum(np.diag(C_VV).astype(float), np.finfo(float).tiny)
        sV = np.sqrt(dV)

        # correlation(test i, train j) = rho_train[j]
        C_VT = np.outer(sV, sT * rho_train)
        C_TV = C_VT.T

    # Factorize TT once
    cho_TT = cho_factor(C_TT, lower=True, check_finite=False)
    TT_inv_CTV = cho_solve(cho_TT, C_TV, check_finite=False)  # (nT x nV)

    # Conditional covariance
    C_cond = C_VV - (C_VT @ TT_inv_CTV)
    C_cond = 0.5 * (C_cond + C_cond.T)

    # Robust Cholesky (jitter if needed)
    jitter_used = 0.0
    try:
        cho_cond = cho_factor(C_cond, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        diag = np.diag(C_cond).astype(float)
        scale = float(np.median(diag[diag > 0])) if np.any(diag > 0) else 1.0
        jitter_used = float(jitter_frac * scale)
        C_cond = C_cond + jitter_used * np.eye(C_cond.shape[0], dtype=float)
        cho_cond = cho_factor(C_cond, lower=True, check_finite=False)

    # Diagnostics
    tr_VV = float(np.trace(C_VV))
    tr_cond = float(np.trace(C_cond))
    var_shrink = (1.0 - tr_cond / tr_VV) if tr_VV > 0 else float("nan")

    dV = np.maximum(np.diag(C_VV).astype(float), np.finfo(float).tiny)
    dT = np.maximum(np.diag(C_TT).astype(float), np.finfo(float).tiny)
    denom = np.sqrt(np.outer(dV, dT))
    corr_VT = C_VT / denom
    max_abs_corr_VT = float(np.max(np.abs(corr_VT))) if corr_VT.size else 0.0
    mean_abs_corr_VT = float(np.mean(np.abs(corr_VT))) if corr_VT.size else 0.0

    if C_VV.shape[0] > 1:
        denomVV = np.sqrt(np.outer(dV, dV))
        corr_VV = C_VV / denomVV
        off = corr_VV - np.eye(C_VV.shape[0], dtype=float)
        mean_abs_corr_VV_offdiag = float(np.mean(np.abs(off)))
        max_abs_corr_VV_offdiag = float(np.max(np.abs(off)))
    else:
        mean_abs_corr_VV_offdiag = 0.0
        max_abs_corr_VV_offdiag = 0.0

    logdet_Ccond = _logdet_from_cho(cho_cond)

    return dict(
        idx_train=idx_train,
        idx_test=idx_test,
        cho_TT=cho_TT,
        C_VT=C_VT,
        cho_cond=cho_cond,
        C_cond=C_cond,
        jitter_used=jitter_used,
        var_shrink=var_shrink,
        logdet_Ccond=logdet_Ccond,
        max_abs_corr_VT=max_abs_corr_VT,
        mean_abs_corr_VT=mean_abs_corr_VT,
        mean_abs_corr_VV_offdiag=mean_abs_corr_VV_offdiag,
        max_abs_corr_VV_offdiag=max_abs_corr_VV_offdiag,
        cross_scale=float(cross_scale),
        cross_stat_train_med_abs_corr=float(cross_stat_train_med_abs_corr) if np.isfinite(cross_stat_train_med_abs_corr) else float("nan"),
        cross_stat_vt_med_abs_corr=float(cross_stat_vt_med_abs_corr) if np.isfinite(cross_stat_vt_med_abs_corr) else float("nan"),
        cross_mode=cross_mode,
        )

def _blockdiag_from_blocks(C: np.ndarray, blocks: list[np.ndarray]) -> np.ndarray:
    """
    Build a block-diagonal matrix using within-block entries of C and zero elsewhere.
    blocks: list of 1D integer index arrays (disjoint, covering the data vector).
    """
    B = np.zeros_like(C)
    for idx in blocks:
        idx = np.asarray(idx, dtype=int)
        B[np.ix_(idx, idx)] = C[np.ix_(idx, idx)]
    return B


def shrink_offdiag_by_blocks(C: np.ndarray, blocks: list[np.ndarray], gamma: float) -> np.ndarray:
    """
    Regularize cross-block covariance only:
      - within-block entries unchanged
      - cross-block entries multiplied by (1 - gamma)
    gamma=0 -> FULL
    gamma=1 -> block-diagonal by blocks
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0,1], got {gamma}")

    B = _blockdiag_from_blocks(C, blocks)
    # Equivalent to: within stays, offdiag shrinks
    C_reg = (1.0 - gamma) * C + gamma * B

    # Symmetrize (numerical safety)
    C_reg = 0.5 * (C_reg + C_reg.T)
    return C_reg


def _kfold_splits(n: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, val


def _gaussian_loglike_zero_mean(X: np.ndarray, C: np.ndarray) -> float:
    """
    Sum log-likelihoods under N(0, C) for rows of X.
    Returns sum over samples (up to an additive constant).
    """
    # Cholesky for stability
    L = np.linalg.cholesky(C)
    # Solve L Z = X^T  => Z = L^{-1} X^T
    Z = np.linalg.solve(L, X.T)
    quad = float(np.sum(Z * Z))
    logdet = float(2.0 * np.sum(np.log(np.diag(L))))
    m = X.shape[0]
    return -0.5 * (m * logdet + quad)


def estimate_gamma_offdiag_cv(
    mocks: np.ndarray,
    blocks: list[np.ndarray],
    k: int = 5,
    gamma_grid: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[float, dict]:
    """
    Choose gamma by K-fold CV on mocks:
      - build sample covariance on train mocks
      - regularize cross-block covariance with gamma
      - score by Gaussian predictive log-likelihood on val mocks
    """
    if gamma_grid is None:
        gamma_grid = np.linspace(0.0, 1.0, 41)

    X = np.asarray(mocks, dtype=float)
    if X.ndim != 2:
        raise ValueError("mocks must be 2D array: (n_mocks, n_data)")

    # mean-subtract mocks
    X = X - np.mean(X, axis=0, keepdims=True)

    scores = np.zeros_like(gamma_grid, dtype=float)

    for tr_idx, va_idx in _kfold_splits(X.shape[0], k=k, seed=seed):
        Xtr = X[tr_idx]
        Xva = X[va_idx]

        Ctr = np.cov(Xtr, rowvar=False, ddof=1)

        # Blockdiag target is constructed from Ctr itself (preserve within-block exactly)
        Btr = _blockdiag_from_blocks(Ctr, blocks)

        for i, g in enumerate(gamma_grid):
            Cg = (1.0 - g) * Ctr + g * Btr
            Cg = 0.5 * (Cg + Cg.T)
            scores[i] += _gaussian_loglike_zero_mean(Xva, Cg)

    best_i = int(np.argmax(scores))
    best_gamma = float(gamma_grid[best_i])

    diag = {
        "gamma_grid": gamma_grid,
        "scores": scores,
        "best_index": best_i,
        "best_gamma": best_gamma,
    }
    return best_gamma, diag


def conditional_chi2_from_precompute(r_full: np.ndarray, pre: dict) -> float:
    """Compute χ²_test|train using a precomputed Schur complement."""
    r_full = np.asarray(r_full, dtype=float)
    idx_train = pre["idx_train"]
    idx_test  = pre["idx_test"]

    r_T = r_full[idx_train]
    r_V = r_full[idx_test]

    TT_inv_rT = cho_solve(pre["cho_TT"], r_T, check_finite=False)
    r_cond = r_V - (pre["C_VT"] @ TT_inv_rT)

    return float(r_cond @ cho_solve(pre["cho_cond"], r_cond, check_finite=False))

def conditional_chi2_gaussian(resid: np.ndarray, C: np.ndarray, test_idx: np.ndarray) -> float:
    """
    Compute chi^2 for x_test | x_train under a joint Gaussian with covariance C,
    where resid = (data - model) in the SAME ordering as C.
    """
    resid = np.asarray(resid, dtype=float)
    test_idx = np.asarray(test_idx, dtype=int)

    n = C.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    train_idx = np.nonzero(mask)[0]

    r_t = resid[test_idx]
    r_T = resid[train_idx]

    C_tt = C[np.ix_(test_idx, test_idx)]
    C_tT = C[np.ix_(test_idx, train_idx)]
    C_Tt = C_tT.T
    C_TT = C[np.ix_(train_idx, train_idx)]

    # Solve C_TT^{-1} r_T and C_TT^{-1} C_Tt via Cholesky (avoid explicit inverse)
    L = np.linalg.cholesky(C_TT)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, r_T))      # C_TT^{-1} r_T
    M = np.linalg.solve(L.T, np.linalg.solve(L, C_Tt))         # C_TT^{-1} C_Tt

    # conditional mean shift
    mu = C_tT @ alpha
    r_cond = r_t - mu

    # conditional covariance (Schur complement)
    C_cond = C_tt - C_tT @ M
    C_cond = 0.5 * (C_cond + C_cond.T)

    # chi^2 with Cholesky
    Lc = np.linalg.cholesky(C_cond)
    z = np.linalg.solve(Lc, r_cond)
    return float(z.T @ z)

def chi2_gaussian(r: np.ndarray, C: np.ndarray) -> float:
    """
    Quadratic form χ² = rᵀ C⁻¹ r using a Cholesky solve (stable).
    """
    r = np.asarray(r, dtype=float)
    C = np.asarray(C, dtype=float)
    cho = cho_factor(C, lower=True, check_finite=False)
    return float(r @ cho_solve(cho, r, check_finite=False))

def chi2_sn_marginalized_M(sn: SNData, r0: np.ndarray) -> float:
    """
    Marginalize (profile out) the global additive SN intercept M for m_b_corr-like observables.

    r0 MUST be: r0 = y - mu_ref   (i.e. NO intercept subtraction).

    This matches the algebra already used in chi2_sn() when sn.y_is_mu is False,
    but is convenient for CV where we want conditional scores via:
        chi2_cond(test|train) = chi2_marg(full) - chi2_marg(train)
    """
    r0 = np.asarray(r0, dtype=float)
    Cinv_r0 = cho_solve(sn.cho, r0, check_finite=False)
    numer = float(sn.ones @ Cinv_r0)
    return float(r0 @ Cinv_r0 - (numer * numer) / sn.ones_Cinv_ones)


def make_folds_zstratified(z: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """
    z-stratified k-fold split with seed-controlled randomization.

    We sort by z, then shuffle *within small consecutive blocks* to keep each fold's
    z-distribution similar, while allowing seed to change the membership.
    """
    z = np.asarray(z, dtype=float)
    k = int(k)
    if k < 2:
        raise ValueError("k must be >= 2")

    order = np.argsort(z, kind="mergesort")  # stable sort
    rng = np.random.default_rng(int(seed))

    # Shuffle within blocks ~k to preserve stratification
    block = max(k, 2)
    order2 = order.copy()
    for s in range(0, order2.size, block):
        e = min(s + block, order2.size)
        blk = order2[s:e].copy()
        rng.shuffle(blk)
        order2[s:e] = blk

    folds = [[] for _ in range(k)]
    for i, idx in enumerate(order2):
        folds[i % k].append(int(idx))

    return [np.array(f, dtype=int) for f in folds]

def make_folds_leave_one_survey_out(idsurvey: np.ndarray, idx: np.ndarray, seed: int) -> list[np.ndarray]:
    """
    Leave-one-survey-out folds using IDSURVEY.
    Returns a list of folds, each fold is an array of GLOBAL indices to hold out.

    NOTE: fold membership is deterministic given IDSURVEY; seed only shuffles the order of surveys.
    """
    idx = np.asarray(idx, dtype=int)
    sid = np.asarray(idsurvey, dtype=int)
    if idx.size == 0:
        return []

    s = sid[idx]
    surveys = np.unique(s)

    rng = np.random.default_rng(int(seed))
    surveys = surveys.copy()
    rng.shuffle(surveys)

    folds = []
    for sv in surveys:
        m = (s == sv)
        if np.any(m):
            folds.append(idx[m])

    return folds


def subset_sn(sn: SNData, idx: np.ndarray) -> SNData:
    idx = np.asarray(idx, dtype=int)
    cov = sn.cov[np.ix_(idx, idx)]
    cho = cho_factor(cov, lower=True, check_finite=False)
    ones = np.ones(idx.size, dtype=float)
    Cinv_ones = cho_solve(cho, ones, check_finite=False)
    ones_Cinv_ones = float(ones @ Cinv_ones)

    sn_sub = SNData(
        zHD=sn.zHD[idx],
        zHEL=sn.zHEL[idx],
        idsurvey=sn.idsurvey[idx],
        cid=sn.cid[idx],
        y=sn.y[idx],
        cov=cov,
        cho=cho,
        y_is_mu=sn.y_is_mu,
        ones=ones,
        Cinv_ones=Cinv_ones,
        ones_Cinv_ones=ones_Cinv_ones,
        is_calibrator=sn.is_calibrator[idx],
        mu_ceph=sn.mu_ceph[idx],
        anchor=None,
        group=(sn.group[idx] if sn.group is not None else None),
    )
    if sn.anchor is not None:
        sn_sub.anchor = build_sn_anchor(sn_sub)
    return sn_sub

# ----------------------------
# Leave-out (jackknife) refit analysis
# ----------------------------

def _load_idsurvey_group_map(path: str) -> dict[int, str]:
    """
    Load IDSURVEY -> group label mapping (e.g. telescope/instrument).

    Supports:
      - JSON: {"14": "PS1", "15": "PS1", ...}  (keys may be str or int)
      - CSV/TSV/whitespace with columns (case-insensitive):
          IDSURVEY, GROUP
        or:
          IDSURVEY, TELESCOPE
        or:
          IDSURVEY, INSTRUMENT
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"Group map file not found: {path}")

    if path.lower().endswith(".json"):
        import json
        with open(path, "r") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("JSON group map must be a dict: {IDSURVEY: 'LABEL', ...}")
        out: dict[int, str] = {}
        for k, v in obj.items():
            out[int(k)] = str(v)
        return out

    # CSV/TSV/whitespace (auto-detect delimiter)
    df = pd.read_csv(path, sep=None, engine="python", comment="#")
    cols = {c.lower().strip(): c for c in df.columns}

    sid_col = None
    for cand in ("idsurvey", "id_survey", "survey_id"):
        if cand in cols:
            sid_col = cols[cand]
            break
    if sid_col is None:
        raise ValueError(f"Group map file must have an IDSURVEY column. Columns: {list(df.columns)}")

    grp_col = None
    for cand in ("group", "telescope", "instrument"):
        if cand in cols:
            grp_col = cols[cand]
            break
    if grp_col is None:
        raise ValueError(
            f"Group map file must have a GROUP/TELESCOPE/INSTRUMENT column. Columns: {list(df.columns)}"
        )

    sid = pd.to_numeric(df[sid_col], errors="coerce").astype("Int64")
    grp = df[grp_col].astype(str)

    m = sid.notna()
    sid = sid[m].astype(int).to_numpy()
    grp = grp[m].to_numpy()

    out = {int(s): str(g) for s, g in zip(sid, grp)}
    if not out:
        raise ValueError("Group map file produced an empty mapping.")
    return out


def _group_labels_for_leaveout(idsurvey: np.ndarray, mode: str, map_path: Optional[str]) -> np.ndarray:
    """
    Return a 1D array of group labels aligned with the provided idsurvey vector.

    mode:
      - "idsurvey": groups are numeric IDSURVEY values
      - "telescope": groups are labels from --loo-map (IDSURVEY -> label)
    """
    mode = str(mode).strip().lower()
    sid = np.asarray(idsurvey, dtype=int)

    if mode == "idsurvey":
        return sid.copy()

    if mode == "telescope":
        if not map_path:
            raise ValueError("--loo-group telescope requires --loo-map <path-to-csv-or-json>")
        mp = _load_idsurvey_group_map(map_path)
        labels = np.array([mp.get(int(s), f"UNMAPPED_{int(s)}") for s in sid], dtype=object)

        # hard warning if unmapped exist (you usually want a complete map)
        n_unmapped = int(np.sum(np.char.startswith(labels.astype(str), "UNMAPPED_")))
        if n_unmapped > 0:
            print(f"[LOO][warn] {n_unmapped}/{labels.size} rows have IDSURVEY not present in --loo-map (labeled UNMAPPED_x).")
        return labels

    raise ValueError(f"Unknown --loo-group {mode!r} (expected 'idsurvey' or 'telescope').")


def _compute_t_anchor_from_sn_sample(
    sn_sub: SNData,
    H0_b: float,
    Om_b: float,
    epochs: LTAEpochs,
    zmax_table: float,
    q: float = 0.05,
) -> float:
    """
    Recompute LTA_T_ANCHOR_GYR using the SAME rule as main():
      t_anchor = q-quantile of Earth-retarded lookback at SN endpoints under baseline approx.
    Use non-calibrators only when calibrators are present with m_b_corr-like observable.
    """
    tables = build_cosmology_tables(H0=float(H0_b), Om=float(Om_b), zmax=zmax_table)
    chi_end = tables.chi_of_z(sn_sub.zHD)
    tret_end = earth_retarded_lookback_gyr(chi_end, tables)

    tret_for_anchor = tret_end
    if (sn_sub.is_calibrator is not None) and np.any(sn_sub.is_calibrator) and (not sn_sub.y_is_mu):
        noncal = ~sn_sub.is_calibrator
        if np.any(noncal):
            tret_for_anchor = tret_end[noncal]

    t_anchor = float(np.quantile(tret_for_anchor, float(q)))
    t_anchor = float(np.clip(t_anchor, 1e-6, float(epochs.t_life_gyr) - 1e-6))
    return t_anchor


def run_leaveout_refits(
    *,
    form: str,
    sn: SNData,
    bao: BAOData,
    epochs: LTAEpochs,
    zmax_table: float,
    bounds_base: list[tuple[float, float]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """
    Leave-out refits ("jackknife" style): remove group(s), refit baseline and LTA on the remaining data.

    This is NOT cross-validation scoring. It answers:
      "Does the global Δχ² improvement persist when we remove survey/telescope groups?"

    Controls:
      --loo-group           idsurvey | telescope
      --loo-map             mapping IDSURVEY -> telescope label (for telescope mode)
      --loo-max-combo-size  remove 1 group (default) or up to m groups
      --loo-max-combos      cap number of combos evaluated (0 = no cap)
      --loo-seed            shuffles group order (useful if capped)
    """
    group_mode = str(getattr(args, "loo_group", "idsurvey")).strip().lower()
    map_path = getattr(args, "loo_map", None)
    max_size = int(getattr(args, "loo_max_combo_size", 1))
    max_combos = int(getattr(args, "loo_max_combos", 0))
    seed = int(getattr(args, "loo_seed", 0))

    if max_size < 1:
        raise ValueError("--loo-max-combo-size must be >= 1")

    N = int(sn.y.size)
    all_idx = np.arange(N, dtype=int)

    # If ladder-anchor is active (calibrators present), emulate your CV choice:
    # keep calibrators always included, and only remove from HF pool.
    anchored = (sn.anchor is not None)
    has_cal = (sn.is_calibrator is not None) and np.any(sn.is_calibrator) and (not sn.y_is_mu)

    if anchored and has_cal:
        idx_pool = all_idx[~sn.is_calibrator]   # eligible for removal
        n_cal = int(np.sum(sn.is_calibrator))
        print(f"[LOO] Anchored SN likelihood active: keeping calibrators fixed in all refits (N_cal={n_cal}); removing only from HF pool (N_pool={idx_pool.size}).")
    else:
        idx_pool = all_idx
        print(f"[LOO] Removing from full SN selection (N_pool={idx_pool.size}).")

    # Build group labels on the removal pool only
    group_labels_pool = _group_labels_for_leaveout(sn.idsurvey[idx_pool], group_mode, map_path)
    uniq_groups = np.unique(group_labels_pool)

    # Shuffle group order deterministically by seed (matters if you cap combos)
    rng = np.random.default_rng(seed)
    uniq_groups = uniq_groups.copy()
    rng.shuffle(uniq_groups)

    # Combo generator: all combinations of size 1..max_size over uniq_groups
    def _combo_iter():
        for r in range(1, max_size + 1):
            for combo in itertools.combinations(list(uniq_groups), r):
                yield combo

    # Warm-start baseline
    x0b = np.array([70.0, 0.30, 1.0], dtype=float)
    # Respect fixed bounds if user fixed parameters
    if abs(bounds_base[0][1] - bounds_base[0][0]) < 1e-12:
        x0b[0] = float(bounds_base[0][0])
    if abs(bounds_base[1][1] - bounds_base[1][0]) < 1e-12:
        x0b[1] = float(bounds_base[1][0])
    if abs(bounds_base[2][1] - bounds_base[2][0]) < 1e-12:
        x0b[2] = float(bounds_base[2][0])

    rows: list[dict] = []

    print("\n" + "=" * 80)
    print(f"LEAVE-OUT REFITS: form={form}  group={group_mode}  max_combo_size={max_size}  max_combos={max_combos}  seed={seed}")
    print("=" * 80)

    global LTA_T_ANCHOR_GYR

    n_done = 0
    for j, combo in enumerate(_combo_iter(), start=1):
        if max_combos > 0 and n_done >= max_combos:
            break

        # Which SN indices are removed?
        remove_mask_pool = np.isin(group_labels_pool, np.array(combo, dtype=object))
        idx_remove = idx_pool[remove_mask_pool]
        idx_keep = np.setdiff1d(all_idx, idx_remove, assume_unique=False)

        if idx_keep.size < 10:
            # too small to be meaningful / stable
            continue

        sn_sub = subset_sn(sn, idx_keep)

        # Baseline fit on reduced SN + full BAO
        res_b = fit_baseline(
            sn_sub, bao,
            epochs=epochs,
            zmax_table=zmax_table,
            x0=x0b,
            bounds=bounds_base,
            maxiter=250,
        )
        H0_b, Om_b, alpha_b = map(float, res_b.x)
        chi2_b = float(res_b.fun)
        x0b = np.array([H0_b, Om_b, alpha_b], dtype=float)

        # Re-anchor amplitude for THIS subset (same rule as main)
        LTA_T_ANCHOR_GYR = _compute_t_anchor_from_sn_sample(
            sn_sub, H0_b=H0_b, Om_b=Om_b, epochs=epochs, zmax_table=zmax_table, q=0.05
        )

        # LTA fit on reduced SN + full BAO
        fit_lta = fit_form(
            form=form,
            sn=sn_sub,
            bao=bao,
            epochs=epochs,
            zmax_table=zmax_table,
            H0_b=H0_b,
            Om_b=Om_b,
            alpha_b=alpha_b,
            args=args,
        )
        chi2_l = float(fit_lta.res.fun)

        dchi2 = float(chi2_b - chi2_l)

        # Pretty label for removed groups
        def _gstr(x):
            try:
                if isinstance(x, (np.integer, int)):
                    return str(int(x))
            except Exception:
                pass
            return str(x)

        removed = "|".join(_gstr(x) for x in combo)

        rows.append(dict(
            combo_index=int(n_done + 1),
            removed_groups=str(removed),
            n_groups_removed=int(len(combo)),
            N_removed=int(idx_remove.size),
            N_keep=int(idx_keep.size),
            group_mode=str(group_mode),
            t_anchor_gyr=float(LTA_T_ANCHOR_GYR),

            chi2_baseline=float(chi2_b),
            chi2_lta=float(chi2_l),
            dchi2=float(dchi2),

            H0_baseline=float(H0_b),
            Om_baseline=float(Om_b),
            alpha_baseline=float(alpha_b),

            H0_lta=float(fit_lta.H0),
            Om_lta=float(fit_lta.Om),
            alpha_lta=float(fit_lta.alpha_rd),
            s_anchor=float(fit_lta.s_anchor),
            gC=float(fit_lta.gC),
            gL=float(fit_lta.gL),
        ))

        n_done += 1

        if (n_done % 10) == 0 or n_done == 1:
            print(f"[LOO] done {n_done}  last_removed={removed}  Δχ²={dchi2:+.3f}  N_keep={idx_keep.size}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("[LOO] No leave-out refits were executed (empty table). Check grouping and selection.")
        return df

    # Summary
    d = df["dchi2"].to_numpy(float)
    print("-" * 80)
    print(f"[LOO] Δχ² summary over {len(df)} refits:")
    print(f"[LOO]   min={float(np.min(d)):+.3f}  median={float(np.median(d)):+.3f}  max={float(np.max(d)):+.3f}")
    print(f"[LOO]   positive={int(np.sum(d > 0))}/{int(d.size)}")

    # Save
    if getattr(args, "outdir", None):
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        tag = str(getattr(args, "run_tag", "")).strip() or "run"
        csv_path = outdir / f"leaveout_refits_{form}_group{group_mode}_m{max_size}_seed{seed}_{tag}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[LOO] Saved leave-out refit table: {csv_path}")

        # Simple plot: Δχ² vs refit index (optional)
        fig = plt.figure()
        plt.plot(np.arange(1, len(df) + 1), d, marker="o")
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("leave-out refit index")
        plt.ylabel("Δχ² (baseline − LTA)")
        plt.title(f"Leave-out refits: form={form}, group={group_mode}, max_combo_size={max_size}")
        plt.tight_layout()
        # reuse your save_fig convention if possible
        figpath = outdir / f"leaveout_refits_dchi2_{form}_group{group_mode}_m{max_size}_seed{seed}_{tag}.png"
        fig.savefig(figpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[LOO] Saved plot: {figpath}")

    else:
        print("[LOO] Table head:")
        print(df.head(12).to_string(index=False))

    return df


def run_null_injections(
    form: str,
    sn: SNData,
    bao: BAOData,
    epochs: LTAEpochs,
    zmax_table: float,
    H0_base: float,
    Om_base: float,
    alpha_base: float,
    chi2_base_real: float,
    chi2_lta_real: float,
    bounds_base: list[tuple[float, float]],
    args: argparse.Namespace,
    save_fig,
) -> None:
    rng = np.random.default_rng(int(args.null_seed))

    dchi2_obs = float(chi2_base_real - chi2_lta_real)
    dchi2_sims = []
    n_fail = 0

    # Null-generating model: baseline ΛCDM best-fit
    lta0 = LTAParams(s_anchor_km_s_per_mpc=0.0, g_complex=0.0, g_life=0.0)
    mu0 = predict_sn_mu(sn, H0=H0_base, Om=Om_base, epochs=epochs, zmax_table=zmax_table, lta_obj=lta0)
    DM0, Hz0 = predict_bao_DM_Hz(bao, H0=H0_base, Om=Om_base, alpha_rd=alpha_base, epochs=epochs, zmax_table=zmax_table, lta_obj=lta0)
    if sn.anchor is not None:
        M_intercept0 = float(sn.anchor.M_cal)
    else:
        M_intercept0 = sn_bestfit_M_for_indices(sn, mu0, idx_fit=None)


    dchi2_obs = float(chi2_base_real - chi2_lta_real)
    dchi2_sims = []

    print("\n" + "=" * 80)
    print(f"NULL INJECTIONS: form={form}  N={int(args.null_n)}  seed={int(args.null_seed)}")
    print(f"Observed Δχ² (baseline - {form}) = {dchi2_obs:.6f}")
    print("=" * 80)

    # Warm starts
    x0b = np.array([H0_base, Om_base, alpha_base], dtype=float)

    for i in range(int(args.null_n)):
        sn_mock, bao_mock = simulate_mock_dataset(
            sn, bao, mu0, DM0, Hz0, rng,
            M_intercept=M_intercept0,
        )

        res_b = fit_baseline(
            sn_mock, bao_mock,
            epochs=epochs,
            zmax_table=zmax_table,
            x0=x0b,
            bounds=bounds_base,
            maxiter=250
        )
        H0_bm, Om_bm, alpha_bm = map(float, res_b.x)
        chi2_bm = float(res_b.fun)

        global LTA_T_ANCHOR_GYR
        tables_anchor = build_cosmology_tables(H0=float(H0_bm), Om=float(Om_bm), zmax=zmax_table)
        chi_end = tables_anchor.chi_of_z(sn_mock.zHD)
        tret_end = earth_retarded_lookback_gyr(chi_end, tables_anchor)

        tret_for_anchor = tret_end
        if (sn_mock.is_calibrator is not None) and np.any(sn_mock.is_calibrator) and (not sn_mock.y_is_mu):
            noncal = ~sn_mock.is_calibrator
            if np.any(noncal):
                tret_for_anchor = tret_end[noncal]

        t_anchor_m = float(np.quantile(tret_for_anchor, 0.05))
        t_anchor_m = float(np.clip(t_anchor_m, 1e-6, float(epochs.t_life_gyr) - 1e-6))
        LTA_T_ANCHOR_GYR = t_anchor_m

        fit_lta = fit_form(
            form=form,
            sn=sn_mock,
            bao=bao_mock,
            epochs=epochs,
            zmax_table=zmax_table,
            H0_b=H0_bm,
            Om_b=Om_bm,
            alpha_b=alpha_bm,
            args=args,
        )
        
        chi2_lm = float(fit_lta.res.fun)
        
        # Nested-safe: LTA should never be worse than baseline for the same mock
        if chi2_lm > chi2_bm:
            chi2_lm = chi2_bm

        if (not np.isfinite(chi2_bm)) or (not np.isfinite(chi2_lm)):
            n_fail += 1
            continue

        dchi2_sims.append(chi2_bm - chi2_lm)

        x0b = np.array([H0_bm, Om_bm, alpha_bm], dtype=float)  # warm-start next baseline

    dchi2_sims = np.asarray(dchi2_sims, dtype=float)
    dchi2_sims = dchi2_sims[np.isfinite(dchi2_sims)]

    if dchi2_sims.size == 0:
        print(f"[null] All null fits failed/nonfinite (n_fail={n_fail}). Try loosening guards or increasing maxiter.")
        return

    p = (float(np.sum(dchi2_sims >= dchi2_obs)) + 1.0) / (float(dchi2_sims.size) + 1.0)

    print(f"Null mean Δχ² = {float(np.mean(dchi2_sims)):.6f}   std = {float(np.std(dchi2_sims)):.6f}")
    print(f"Used {dchi2_sims.size}/{int(args.null_n)} successful null draws; failures={n_fail}")
    print(f"Monte Carlo p-value P(Δχ²_null >= Δχ²_obs) ≈ {p:.6g}")

    fig = plt.figure()
    plt.hist(dchi2_sims, bins=25)
    plt.axvline(dchi2_obs, linestyle="--", linewidth=2)
    plt.xlabel("Δχ² (baseline − LTA) under null mocks")
    plt.ylabel("count")
    plt.title(f"Null injections for {form}: observed Δχ²={dchi2_obs:.3f}, p≈{p:.3g}")
    plt.tight_layout()
    save_fig(fig, f"null_injections_{form}")
    
    if getattr(args, "outdir", None):
        out = Path(args.outdir) / f"null_dchi2_{form}_{args.run_tag}.npy"
        np.save(out, dchi2_sims)
        print(f"[null] Saved Δχ² samples to {out}")

def run_cv_null_injections(
    form: str,
    sn: SNData,
    bao: BAOData,
    epochs: LTAEpochs,
    zmax_table: float,
    H0_base: float,
    Om_base: float,
    alpha_base: float,
    bounds_base: list[tuple[float, float]],
    args: argparse.Namespace,
    save_fig,
) -> None:
    """
    CV-null injections (parametric bootstrap on the *CV* statistic):

    - Null-generating model: baseline ΛCDM best-fit (s_anchor=0).
    - For each mock: run the SAME CV procedure you report on real data.
    - Store the total CV Δχ² (baseline − LTA) across folds.

    This directly answers: under baseline truth, how often do we get CV Δχ² as large as observed?
    """
    rng = np.random.default_rng(int(getattr(args, "cv_null_seed", 0)))
    n_draw = int(getattr(args, "cv_null_n", 50))

    # Null-generating model predictions (baseline truth)
    lta0 = LTAParams(s_anchor_km_s_per_mpc=0.0, g_complex=0.0, g_life=0.0)
    mu0 = predict_sn_mu(
        sn,
        H0=float(H0_base), Om=float(Om_base),
        epochs=epochs, zmax_table=zmax_table,
        lta_obj=lta0,
        invmap_n=None,
    )
    DM0, Hz0 = predict_bao_DM_Hz(
        bao,
        H0=float(H0_base), Om=float(Om_base), alpha_rd=float(alpha_base),
        epochs=epochs, zmax_table=zmax_table,
        lta_obj=lta0,
        invmap_n=None,
    )

    # SN intercept used to generate y in data space
    if sn.anchor is not None:
        M_intercept0 = float(sn.anchor.M_cal)
    else:
        M_intercept0 = float(sn_bestfit_M_for_indices(sn, mu0, idx_fit=None))

    # We need to run CV many times; silence fold-by-fold output & prevent per-mock file writes
    args_cv = copy.copy(args)
    args_cv.outdir = None  # disable per-mock CSV writes inside run_cross_validation

    def _save_fig_noop(fig: plt.Figure, stem: str) -> None:
        plt.close(fig)

    # --- Observed CV Δχ² on real data ---
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        df_obs = run_cross_validation(
            form=form,
            sn=sn,
            bao=bao,
            epochs=epochs,
            zmax_table=zmax_table,
            bounds_base=bounds_base,
            args=args_cv,
            save_fig=_save_fig_noop,
        )

    if df_obs is None or len(df_obs) == 0:
        print("[cv-null] ERROR: observed CV produced an empty fold table; cannot run CV-null.")
        return

    pred_mode = str(getattr(args, "cv_predictive", "plugin")).strip().lower()
    col_m = "dnll_test_marg" if pred_mode == "laplace" else "dchi2_test_marg"
    col_c = "dnll_test_cond" if pred_mode == "laplace" else "dchi2_test_cond"

    dchi2_obs_m = float(np.nansum(df_obs[col_m].to_numpy(float)))
    dchi2_obs_c = float(np.nansum(df_obs[col_c].to_numpy(float)))

    dchi2_sims_m: list[float] = []
    dchi2_sims_c: list[float] = []
    n_fail = 0

    print("\n" + "=" * 80)
    print(f"CV-NULL INJECTIONS: form={form}  N={n_draw}  seed={int(getattr(args, 'cv_null_seed', 0))}")
    print(f"Observed CV Δχ²_marg (baseline − {form}) = {dchi2_obs_m:.6f}")
    print(f"Observed CV Δχ²_cond (baseline − {form}) = {dchi2_obs_c:.6f}")
    print("=" * 80)

    for i in range(n_draw):
        sn_mock, bao_mock = simulate_mock_dataset(
            sn, bao,
            mu_model=mu0,
            DM_model=DM0,
            Hz_model=Hz0,
            rng=rng,
            M_intercept=float(M_intercept0),
        )

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = run_cross_validation(
                form=form,
                sn=sn_mock,
                bao=bao_mock,
                epochs=epochs,
                zmax_table=zmax_table,
                bounds_base=bounds_base,
                args=args_cv,
                save_fig=_save_fig_noop,
            )

        if df is None or len(df) == 0:
            n_fail += 1
            continue

        dchi2_sims_m.append(float(np.sum(df["dchi2_test_marg"].to_numpy(float))))
        dchi2_sims_c.append(float(np.sum(df["dchi2_test_cond"].to_numpy(float))))

    sims_m = np.asarray(dchi2_sims_m, dtype=float)
    sims_c = np.asarray(dchi2_sims_c, dtype=float)
    sims_m = sims_m[np.isfinite(sims_m)]
    sims_c = sims_c[np.isfinite(sims_c)]

    if sims_m.size == 0:
        print(f"[cv-null] All CV-null runs failed/nonfinite (failures={n_fail}/{n_draw}).")
        return

    p_m = (float(np.sum(sims_m >= dchi2_obs_m)) + 1.0) / (float(sims_m.size) + 1.0)
    p_c = (float(np.sum(sims_c >= dchi2_obs_c)) + 1.0) / (float(sims_c.size) + 1.0) if sims_c.size else float("nan")

    print(f"[cv-null] Used {int(sims_m.size)}/{n_draw} successful CV-null draws; failures={n_fail}")
    print(f"[cv-null] Null mean Δχ²_marg = {float(np.mean(sims_m)):.6f}   std = {float(np.std(sims_m)):.6f}")
    print(f"[cv-null] Monte Carlo p-value P(CV Δχ²_marg_null >= observed) ≈ {p_m:.6g}")

    if sims_c.size:
        print(f"[cv-null] Null mean Δχ²_cond = {float(np.mean(sims_c)):.6f}   std = {float(np.std(sims_c)):.6f}")
        print(f"[cv-null] Monte Carlo p-value P(CV Δχ²_cond_null >= observed) ≈ {p_c:.6g}")

    # Plot marginal
    fig = plt.figure()
    plt.hist(sims_m, bins=25)
    plt.axvline(dchi2_obs_m, linestyle="--", linewidth=2)
    plt.xlabel("Total CV Δχ²_marg (baseline − LTA) under baseline-null mocks")
    plt.ylabel("count")
    plt.title(f"CV-null (marg): observed={dchi2_obs_m:.3f}, p≈{p_m:.3g}, N={int(sims_m.size)}")
    plt.tight_layout()
    save_fig(fig, f"cv_null_injections_{form}_marginal")

    # Plot conditional
    if sims_c.size:
        fig2 = plt.figure()
        plt.hist(sims_c, bins=25)
        plt.axvline(dchi2_obs_c, linestyle="--", linewidth=2)
        plt.xlabel("Total CV Δχ²_cond (baseline − LTA) under baseline-null mocks")
        plt.ylabel("count")
        plt.title(f"CV-null (cond): observed={dchi2_obs_c:.3f}, p≈{p_c:.3g}, N={int(sims_c.size)}")
        plt.tight_layout()
        save_fig(fig2, f"cv_null_injections_{form}_conditional")

    # Save arrays
    if getattr(args, "outdir", None):
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        npy_m = outdir / f"cv_null_dchi2_marg_{form}_{args.run_tag}.npy"
        npy_c = outdir / f"cv_null_dchi2_cond_{form}_{args.run_tag}.npy"
        np.save(npy_m, sims_m)
        np.save(npy_c, sims_c)
        print(f"[cv-null] Saved marginal CV-null Δχ² samples to {npy_m}")
        print(f"[cv-null] Saved conditional CV-null Δχ² samples to {npy_c}")

def conditional_rhat_from_precompute(r_full: np.ndarray, pre: dict) -> np.ndarray:
    """Return r_hat_V = C_VT C_TT^{-1} r_T (the conditional mean shift)."""
    r_full = np.asarray(r_full, dtype=float)
    r_T = r_full[pre["idx_train"]]
    TT_inv_rT = cho_solve(pre["cho_TT"], r_T, check_finite=False)
    return pre["C_VT"] @ TT_inv_rT


def run_cross_validation(
    form: str,
    sn: SNData,
    bao: BAOData,
    epochs: LTAEpochs,
    zmax_table: float,
    bounds_base: list[tuple[float, float]],
    args: argparse.Namespace,
    save_fig,
    sid_to_label: Optional[dict[int, str]] = None,
) -> pd.DataFrame:
    k = int(args.cv_k)
    N = int(sn.y.size)
    all_idx = np.arange(N, dtype=int)

    score_mode = str(getattr(args, "cv_score", "marginal")).strip().lower()
    if score_mode not in ("marginal", "conditional", "both"):
        raise ValueError(f"Unknown --cv-score {score_mode!r}")

    global LTA_T_ANCHOR_GYR

    def _sid_label(sid: int) -> str:
        sid = int(sid)
        if sid_to_label is not None and sid in sid_to_label:
            return str(sid_to_label[sid])
        return str(sid)

    cv_block = str(getattr(args, "cv_block", "none")).strip().lower()
    if cv_block not in ("none", "survey"):
        raise ValueError(f"Unknown --cv-block {cv_block!r}")

    cond_cross = str(getattr(args, "cv_conditional_cross", "full")).strip().lower()
    if cond_cross not in ("auto", "full", "full_quotient", "zero", "within_survey", "survey_avg"):
        raise ValueError(f"Unknown --cv-conditional-cross {cond_cross!r}")

    pred_mode = str(getattr(args, "cv_predictive", "plugin")).strip().lower()
    if pred_mode not in ("plugin", "laplace"):
        raise ValueError(f"Unknown --cv-predictive {pred_mode!r}")

    lap_eps_scale = float(getattr(args, "cv_laplace_eps_scale", 1.0))
    lap_ridge = float(getattr(args, "cv_laplace_ridge", 1e-6))
    lap_jitter = float(getattr(args, "cv_laplace_jitter_frac", 1e-12))


    # Backwards-compatible behavior:
    # Previously, 'auto' forced 'zero' when --cv-block survey.
    # For full Gaussian conditional CV, we want 'auto' -> 'full'.
    if cond_cross == "auto":
        if cv_block == "survey":
            print(
                "[CV] --cv-conditional-cross auto: using FULL cross-covariance (Gaussian conditional). "
                "For the old independent-block behavior, pass --cv-conditional-cross zero."
            )
        cond_cross = "full"

    within_survey_cross = (cond_cross == "within_survey")
    zero_cross_cond = (cond_cross == "zero")
    cond_cross_effective = "zero" if zero_cross_cond else "full"

    has_cal = (sn.is_calibrator is not None) and np.any(sn.is_calibrator) and (not sn.y_is_mu)

    # Base index set that is allowed to be held out
    if has_cal:
        hf_idx = all_idx[~sn.is_calibrator]
        fold_base_idx = hf_idx
        print(f"[CV] Keeping {int(np.sum(sn.is_calibrator))} calibrators always in TRAIN; folds built on {hf_idx.size} HF SNe.")
    else:
        fold_base_idx = all_idx

    # Build folds
    folds: list[np.ndarray]

    if cv_block == "survey":
        # If IDSURVEY not present or degenerate, fall back
        sids = getattr(sn, "idsurvey", None)
        if sids is None:
            print("[CV] --cv-block survey requested but SN dataset has no idsurvey field; falling back to z-stratified k-fold.")
            cv_block = "none"
        else:
            uniq = np.unique(sn.idsurvey[fold_base_idx])
            if uniq.size < 2:
                print("[CV] --cv-block survey requested but only one unique IDSURVEY in selection; falling back to z-stratified k-fold.")
                cv_block = "none"
            else:
                folds = make_folds_leave_one_survey_out(sn.idsurvey, fold_base_idx, seed=int(args.cv_seed))
                k = len(folds)  # override --cv-k
                sizes = np.array([f.size for f in folds], dtype=int)
                print(f"[CV] Using leave-one-survey-out blocked CV over IDSURVEY: n_surveys={uniq.size} => k={k} folds.")
                print(f"[CV] survey fold sizes: min={int(sizes.min())}  median={float(np.median(sizes)):.1f}  max={int(sizes.max())}")

    if cv_block == "none":
        folds_local = make_folds_zstratified(sn.zHD[fold_base_idx], k=k, seed=int(args.cv_seed))
        folds = [fold_base_idx[f] for f in folds_local]

    
    # --- DEBUG: verify that cv_seed actually changes fold membership ---
    import hashlib

    def _fold_sig(arr: np.ndarray) -> str:
        a = np.sort(np.asarray(arr, dtype=np.int64))
        return hashlib.sha1(a.tobytes()).hexdigest()[:10]

    sigs = [_fold_sig(f) for f in folds]
    print(f"[CV] cv_seed={int(args.cv_seed)} fold_sigs={sigs}")

    # Sanity: check union size (should equal HF size if calibrators kept in train)
    all_test = np.unique(np.concatenate([np.asarray(f, dtype=int) for f in folds]))

    expected_union = int(fold_base_idx.size)
    print(f"[CV] unique test indices across folds: {all_test.size} (expected {expected_union})")


    # We will always compute BOTH scores; totals/plot will use:
    #   - conditional if score_mode == "conditional"
    #   - marginal otherwise (including "both")
    chi2_base_m = []
    chi2_lta_m  = []
    chi2_base_c = []
    chi2_lta_c  = []

    fold_rows: list[dict] = []

    print("\n" + "=" * 80)
    print(f"CROSS-VALIDATION: form={form}  k={k}  seed={int(args.cv_seed)}  score={score_mode}  block={cv_block}  cond_cross={cond_cross}")

    print("=" * 80)

    # Warm start for baseline per fold
    x0b = np.array([70.0, 0.30, 1.0], dtype=float)
    if abs(bounds_base[0][1] - bounds_base[0][0]) < 1e-12:
        x0b[0] = float(bounds_base[0][0])
    if abs(bounds_base[1][1] - bounds_base[1][0]) < 1e-12:
        x0b[1] = float(bounds_base[1][0])
    if abs(bounds_base[2][1] - bounds_base[2][0]) < 1e-12:
        x0b[2] = float(bounds_base[2][0])

    # Full-data anchor object (safe: depends on calibrators only + covariance)
    a_full = sn.anchor if sn.anchor is not None else None

    for i, idx_test in enumerate(folds):
        idx_test = np.asarray(idx_test, dtype=int)
        idx_train = np.setdiff1d(all_idx, idx_test, assume_unique=False)

        # --- fold metadata for logging/table ---
        test_sids = np.unique(sn.idsurvey[idx_test])
        if (cv_block == "survey") and (test_sids.size == 1):
            test_block_label = _sid_label(test_sids[0])
        else:
            test_block_label = ",".join(_sid_label(x) for x in test_sids[:10])
            if test_sids.size > 10:
                test_block_label += ",..."


        zmin_test = float(np.min(sn.zHD[idx_test])) if idx_test.size else np.nan
        zmax_test = float(np.max(sn.zHD[idx_test])) if idx_test.size else np.nan

        sn_train = subset_sn(sn, idx_train)

        # Fit baseline on TRAIN + full BAO
        res_b = fit_baseline(
            sn_train, bao,
            epochs=epochs, zmax_table=zmax_table,
            x0=x0b, bounds=bounds_base, maxiter=250
        )
        H0_b, Om_b, alpha_b = map(float, res_b.x)
        x0b = np.array([H0_b, Om_b, alpha_b], dtype=float)

        # Recompute amplitude anchor from TRAIN only (avoid leakage).
        # Use SAME rule as main(): 5th percentile of t_ret endpoints,
        # excluding calibrators for m_b_corr.
        tables_anchor = build_cosmology_tables(H0=float(H0_b), Om=float(Om_b), zmax=zmax_table)
        chi_end = tables_anchor.chi_of_z(sn_train.zHD)
        tret_end = earth_retarded_lookback_gyr(chi_end, tables_anchor)

        tret_for_anchor = tret_end
        if (sn_train.is_calibrator is not None) and np.any(sn_train.is_calibrator) and (not sn_train.y_is_mu):
            noncal = ~sn_train.is_calibrator
            if np.any(noncal):
                tret_for_anchor = tret_end[noncal]

        t_anchor_fold = float(np.quantile(tret_for_anchor, 0.05))
        t_anchor_fold = float(np.clip(t_anchor_fold, 1e-6, float(epochs.t_life_gyr) - 1e-6))
        LTA_T_ANCHOR_GYR = t_anchor_fold
        print(f"[CV] Fold {i+1}/{k}: t_anchor(train)={LTA_T_ANCHOR_GYR:.3f} Gyr  test_IDSURVEY={test_block_label}  N_test={idx_test.size}  zHD∈[{zmin_test:.4f},{zmax_test:.4f}]")

        # Fit LTA on TRAIN + full BAO
        fit_lta = fit_form(
            form=form,
            sn=sn_train,
            bao=bao,
            epochs=epochs,
            zmax_table=zmax_table,
            H0_b=H0_b,
            Om_b=Om_b,
            alpha_b=alpha_b,
            args=args,
        )

        print(f"[CV]   train baseline: H0={H0_b:.3f} Om={Om_b:.4f} alpha={alpha_b:.5f}")
        print(f"[CV]   train LTA     : H0={fit_lta.H0:.3f} Om={fit_lta.Om:.4f} alpha={fit_lta.alpha_rd:.5f} sA={fit_lta.s_anchor:.3f}")

        # Predictive (Laplace) scores (optional)
        nll_b_m = np.nan; nll_l_m = np.nan
        nll_b_c = np.nan; nll_l_c = np.nan

        # =========================
        # SCORING ON HELD-OUT FOLD
        # =========================

        # ---- Case A: ladder-anchored likelihood is enabled (your current run) ----
        if a_full is not None:
            hf_idx_full = a_full.idx_hf
            nH = int(hf_idx_full.size)

            # Map global SN indices -> HF-local indices [0..nH)
            pos = np.full(N, -1, dtype=int)
            pos[hf_idx_full] = np.arange(nH, dtype=int)

            idx_train_hf_global = np.setdiff1d(hf_idx_full, idx_test, assume_unique=False)
            idx_train_loc = pos[idx_train_hf_global]
            idx_test_loc  = pos[idx_test]

            if np.any(idx_test_loc < 0) or np.any(idx_train_loc < 0):
                raise RuntimeError("CV fold contained non-HF indices while using anchored likelihood.")

            C_full = a_full.cov_hf_eff
            C_test = C_full[np.ix_(idx_test_loc, idx_test_loc)]

            # Baseline residuals in HF space (conditional on calibrators, M marginalized via SNAnchor)
            lta0 = LTAParams(s_anchor_km_s_per_mpc=0.0, g_complex=0.0, g_life=0.0)
            mu_full_b = predict_sn_mu(
                sn,
                H0=H0_b, Om=Om_b,
                epochs=epochs,
                zmax_table=zmax_table,
                lta_obj=lta0,
                invmap_n=None,
            )
            r_hf_full_b = sn.y[hf_idx_full] - (
                mu_full_b[hf_idx_full] + a_full.mu_shift_hf + a_full.M_cal * a_full.v_hf
            )

            # Precompute conditional matrices ONCE per fold (reused for baseline + LTA)
            group_ids_hf = sn.idsurvey[hf_idx_full]  # length nH, aligned with a_full.cov_hf_eff indexing

            groups_hf = sn.idsurvey[hf_idx_full]  # aligned with HF-local indexing

            pre = conditional_precompute(
                C_full=C_full,
                idx_train=idx_train_loc,
                idx_test=idx_test_loc,
                cross_mode=cond_cross,
                groups=groups_hf,
            )

            r_test_b = r_hf_full_b[idx_test_loc]
            chi2_b_m = chi2_gaussian(r_test_b, C_test)

            # marginal validation: diag-only version of the SAME test score
            chi2_b_m_diag = chi2_gaussian_diag(r_test_b, C_test)

            chi2_b_c = conditional_chi2_from_precompute(r_hf_full_b, pre)

            # --- Identity check: chi2(full) ≈ chi2(train) + chi2(test|train) ---
            # Full χ² in HF space
            cho_full = cho_factor(C_full, lower=True, check_finite=False)
            chi2_full_b = float(r_hf_full_b @ cho_solve(cho_full, r_hf_full_b, check_finite=False))

            # Train χ² (same TT block used by the conditional precompute)
            rT_b = r_hf_full_b[idx_train_loc]
            chi2_train_b = float(rT_b @ cho_solve(pre["cho_TT"], rT_b, check_finite=False))

            # Conditional χ² (already computed)
            chi2_cond_b = float(chi2_b_c)

            # --- Identity checks ---
            # The Schur complement identities only hold (vs the FULL covariance) when we are using FULL cross-covariance,
            # and when we did not add jitter to C_cond.
            if (str(pre.get("cross_mode", "")).lower() == "full") and (float(pre["jitter_used"]) == 0.0):
                # Quadratic-form identity:
                #   r^T C^{-1} r  ==  r_T^T C_TT^{-1} r_T  +  r_{V|T}^T C_{V|T}^{-1} r_{V|T}
                err = chi2_full_b - (chi2_train_b + chi2_cond_b)
                if abs(err) > 1e-8 * (1.0 + abs(chi2_full_b)):
                    print(
                        f"[CV][warn] Gaussian identity mismatch (baseline, FULL cross): "
                        f"full - (train+cond) = {err:+.3e}  "
                        f"(full={chi2_full_b:.6f}, train={chi2_train_b:.6f}, cond={chi2_cond_b:.6f})"
                    )

                # Log-determinant identity:
                #   log det(C) == log det(C_TT) + log det(C_{V|T})
                logdet_full = _logdet_from_cho(cho_full)
                logdet_TT = _logdet_from_cho(pre["cho_TT"])
                logdet_cond = float(pre["logdet_Ccond"])
                err_ld = logdet_full - (logdet_TT + logdet_cond)
                if abs(err_ld) > 1e-8 * (1.0 + abs(logdet_full)):
                    print(
                        f"[CV][warn] logdet identity mismatch (baseline, FULL cross): "
                        f"full - (TT+cond) = {err_ld:+.3e}  "
                        f"(full={logdet_full:.6f}, TT={logdet_TT:.6f}, cond={logdet_cond:.6f})"
                    )

            elif str(pre.get("cross_mode", "")).lower() == "zero":
                # In 'zero' mode we intentionally ignore train–test cross-covariance but chi2_full_b uses the full cov,
                # so full != train + cond is expected. Do not warn.
                pass

            # LTA residuals in HF space
            mu_full_l = predict_sn_mu(
                sn,
                H0=fit_lta.H0, Om=fit_lta.Om,
                epochs=fit_lta.epochs_best,
                zmax_table=zmax_table,
                lta_obj=fit_lta.lta_best_obj,
                invmap_n=None,
            )
            r_hf_full_l = sn.y[hf_idx_full] - (
                mu_full_l[hf_idx_full] + a_full.mu_shift_hf + a_full.M_cal * a_full.v_hf
            )

            # conditional mean-shift induced by train residuals
            rhat_b = conditional_rhat_from_precompute(r_hf_full_b, pre)
            rhat_l = conditional_rhat_from_precompute(r_hf_full_l, pre)

            # simple scalars to log (Euclidean; you can also do C_cond-metric if desired)
            rhat_rms_b = float(np.sqrt(np.mean(rhat_b * rhat_b))) if rhat_b.size else 0.0
            rhat_rms_l = float(np.sqrt(np.mean(rhat_l * rhat_l))) if rhat_l.size else 0.0

            r_test_l = r_hf_full_l[idx_test_loc]
            chi2_l_m = chi2_gaussian(r_test_l, C_test)
            chi2_l_m_diag = chi2_gaussian_diag(r_test_l, C_test)

            chi2_l_c = conditional_chi2_from_precompute(r_hf_full_l, pre)
            d_c = float(chi2_b_c - chi2_l_c)

            d_m_local = float(chi2_b_m - chi2_l_m)

            gain_b = float(chi2_b_m - chi2_b_c)
            gain_l = float(chi2_l_m - chi2_l_c)
            delta_gain = float(gain_b - gain_l)

            if pred_mode == "laplace":
                # --- Laplace posterior on TRAIN parameters ---
                # Baseline posterior
                xhat_b = np.array([H0_b, Om_b, alpha_b], dtype=float)
                eps_b = _default_eps_vec(xhat_b.size) * lap_eps_scale
                fun_b = lambda x: total_chi2(x, sn_train, bao, epochs, use_lta=False, zmax_table=zmax_table)
                free_b, _, Sig_b = laplace_posterior_cov(fun_b, xhat_b, bounds_base, eps_b, ridge_rel=lap_ridge)

                # LTA posterior
                xhat_l = np.asarray(fit_lta.res.x, dtype=float).ravel()
                eps_l = _default_eps_vec(xhat_l.size) * lap_eps_scale
                fun_l = lambda x: total_chi2(x, sn_train, bao, epochs, use_lta=True, zmax_table=zmax_table)
                free_l, _, Sig_l = laplace_posterior_cov(fun_l, xhat_l, fit_lta.bounds, eps_l, ridge_rel=lap_ridge)

                # --- Residual-vector builders in HF-local ordering ---
                def resid_hf_full_baseline(x: np.ndarray) -> np.ndarray:
                    H0x, Omx, alphax = map(float, np.asarray(x, dtype=float).ravel())
                    lta0x = LTAParams(s_anchor_km_s_per_mpc=0.0, g_complex=0.0, g_life=0.0)
                    mu = predict_sn_mu(sn, H0=H0x, Om=Omx, epochs=epochs, zmax_table=zmax_table, lta_obj=lta0x, invmap_n=None)
                    return sn.y[hf_idx_full] - (mu[hf_idx_full] + a_full.mu_shift_hf + a_full.M_cal * a_full.v_hf)

                def resid_hf_full_lta(x: np.ndarray) -> np.ndarray:
                    x = np.asarray(x, dtype=float).ravel()
                    H0x, Omx, alphax, tLx, sAx, gCx, gLx = map(float, x)
                    epochs_x = LTAEpochs(t_digital_gyr=epochs.t_digital_gyr, t_complex_gyr=epochs.t_complex_gyr, t_life_gyr=tLx)
                    lta_x = LTAParams(s_anchor_km_s_per_mpc=sAx, g_complex=gCx, g_life=gLx)
                    mu = predict_sn_mu(sn, H0=H0x, Om=Omx, epochs=epochs_x, zmax_table=zmax_table, lta_obj=lta_x, invmap_n=None)
                    return sn.y[hf_idx_full] - (mu[hf_idx_full] + a_full.mu_shift_hf + a_full.M_cal * a_full.v_hf)

                # --- Vector functions for marginal test residual and conditional test|train residual ---
                def r_test_b_vec(x):  # length nV
                    return resid_hf_full_baseline(x)[idx_test_loc]

                def r_cond_b_vec(x):  # length nV
                    r = resid_hf_full_baseline(x)
                    rT = r[idx_train_loc]
                    TT_inv_rT = cho_solve(pre["cho_TT"], rT, check_finite=False)
                    return r[idx_test_loc] - (pre["C_VT"] @ TT_inv_rT)

                def r_test_l_vec(x):
                    return resid_hf_full_lta(x)[idx_test_loc]

                def r_cond_l_vec(x):
                    r = resid_hf_full_lta(x)
                    rT = r[idx_train_loc]
                    TT_inv_rT = cho_solve(pre["cho_TT"], rT, check_finite=False)
                    return r[idx_test_loc] - (pre["C_VT"] @ TT_inv_rT)

                # --- Jacobians (finite diff) ---
                freeJ_b_m, _, J_b_m = fd_jacobian_vec(r_test_b_vec, xhat_b, bounds_base, eps_b)
                freeJ_b_c, _, J_b_c = fd_jacobian_vec(r_cond_b_vec, xhat_b, bounds_base, eps_b)

                freeJ_l_m, _, J_l_m = fd_jacobian_vec(r_test_l_vec, xhat_l, fit_lta.bounds, eps_l)
                freeJ_l_c, _, J_l_c = fd_jacobian_vec(r_cond_l_vec, xhat_l, fit_lta.bounds, eps_l)

                # Ensure Jacobian free sets match Sigma free sets; if not, intersect
                def _align(J_free_idx, Sig_free_idx, J, Sig):
                    J_free_idx = np.asarray(J_free_idx, dtype=int)
                    Sig_free_idx = np.asarray(Sig_free_idx, dtype=int)
                    if J_free_idx.size == 0 or Sig_free_idx.size == 0:
                        return np.zeros((J.shape[0], 0), float), np.zeros((0, 0), float)
                    # indices within parameter vector
                    common = np.intersect1d(J_free_idx, Sig_free_idx)
                    if common.size == 0:
                        return np.zeros((J.shape[0], 0), float), np.zeros((0, 0), float)
                    # map to columns
                    jcols = np.array([np.where(J_free_idx == c)[0][0] for c in common], dtype=int)
                    scols = np.array([np.where(Sig_free_idx == c)[0][0] for c in common], dtype=int)
                    return J[:, jcols], Sig[np.ix_(scols, scols)]

                J_b_m_a, Sig_b_m_a = _align(freeJ_b_m, free_b, J_b_m, Sig_b)
                J_b_c_a, Sig_b_c_a = _align(freeJ_b_c, free_b, J_b_c, Sig_b)

                J_l_m_a, Sig_l_m_a = _align(freeJ_l_m, free_l, J_l_m, Sig_l)
                J_l_c_a, Sig_l_c_a = _align(freeJ_l_c, free_l, J_l_c, Sig_l)

                # --- Posterior predictive NLLs ---
                # Marginal test: base covariance is C_test
                out_bm = posterior_predictive_nll(r_test_b, C_test, J_b_m_a, Sig_b_m_a, jitter_frac=lap_jitter)
                out_lm = posterior_predictive_nll(r_test_l, C_test, J_l_m_a, Sig_l_m_a, jitter_frac=lap_jitter)
                nll_b_m = out_bm["nll"]
                nll_l_m = out_lm["nll"]

                # Conditional test|train: base covariance is C_cond (Schur complement)
                C_cond = pre["C_cond"]
                rcond_b = r_cond_b_vec(xhat_b)
                rcond_l = r_cond_l_vec(xhat_l)

                out_bc = posterior_predictive_nll(rcond_b, C_cond, J_b_c_a, Sig_b_c_a, jitter_frac=lap_jitter)
                out_lc = posterior_predictive_nll(rcond_l, C_cond, J_l_c_a, Sig_l_c_a, jitter_frac=lap_jitter)
                nll_b_c = out_bc["nll"]
                nll_l_c = out_lc["nll"]

            if d_c < 0:
                print(f"[CV]   rhat_rms: baseline={rhat_rms_b:.4g}  LTA={rhat_rms_l:.4g}  (test mean-shift from train)")
                print(f"[CV]   cond-gain: baseline={gain_b:+.3f}  LTA={gain_l:+.3f}  (gain_b - gain_l = {delta_gain:+.3f})")
                print(f"[CV]   check: Δ_cond ≈ Δ_marg - (gain_b-gain_l)  =>  {d_m_local:+.3f} - {delta_gain:+.3f} = {d_m_local-delta_gain:+.3f}")

                # r_V and rhat in test space
                rV_b = r_hf_full_b[idx_test_loc]
                rV_l = r_hf_full_l[idx_test_loc]

                rhat_b = conditional_rhat_from_precompute(r_hf_full_b, pre)
                rhat_l = conditional_rhat_from_precompute(r_hf_full_l, pre)

                # Q = C_cond^{-1} via its Cholesky
                cho_cond = pre["cho_cond"]

                def ipQ(a, b):
                    a = np.asarray(a, float); b = np.asarray(b, float)
                    return float(a @ cho_solve(cho_cond, b, check_finite=False))

                def normQ(a):
                    return np.sqrt(max(ipQ(a, a), 0.0))

                # cosine in Q-metric ([-1, +1])
                cos_b = ipQ(rV_b, rhat_b) / max(normQ(rV_b) * normQ(rhat_b), 1e-300)
                cos_l = ipQ(rV_l, rhat_l) / max(normQ(rV_l) * normQ(rhat_l), 1e-300)

                print(f"[CV]   Q-align: cos(rV,rhat) baseline={cos_b:+.3f}  LTA={cos_l:+.3f}")

        # ---- Case B: non-anchored SN likelihood (no calibrator anchoring) ----
        else:
            # Baseline prediction for TRAIN (to get M_train without leakage)
            lta0 = LTAParams(s_anchor_km_s_per_mpc=0.0, g_complex=0.0, g_life=0.0)
            mu_train_b = predict_sn_mu(
                sn_train,
                H0=H0_b, Om=Om_b,
                epochs=epochs,
                zmax_table=zmax_table,
                lta_obj=lta0,
                invmap_n=None,
            )

            M_train_b = 0.0
            if not sn.y_is_mu:
                M_train_b = sn_bestfit_M_for_indices(sn_train, mu_train_b, idx_fit=None)

            # Baseline prediction for FULL (to score held-out)
            mu_full_b = predict_sn_mu(
                sn,
                H0=H0_b, Om=Om_b,
                epochs=epochs,
                zmax_table=zmax_table,
                lta_obj=lta0,
                invmap_n=None,
            )
            mu_ref_full_b = sn_mu_reference(sn, mu_full_b)
            r_full_b = sn.y - (mu_ref_full_b + float(M_train_b))

            C_test = sn.cov[np.ix_(idx_test, idx_test)]

            # Precompute conditional matrices ONCE per fold (reused baseline + LTA)
            pre = conditional_precompute(
                C_full=sn.cov,
                idx_train=idx_train,
                idx_test=idx_test,
                cross_mode=cond_cross,
                groups=sn.idsurvey,
            )

            r_test_b = r_full_b[idx_test]
            chi2_b_m = chi2_gaussian(r_test_b, C_test)
            chi2_b_m_diag = chi2_gaussian_diag(r_test_b, C_test)

            chi2_b_c = conditional_chi2_from_precompute(r_full_b, pre)

            # LTA prediction for TRAIN (to get M_train without leakage)
            mu_train_l = predict_sn_mu(
                sn_train,
                H0=fit_lta.H0, Om=fit_lta.Om,
                epochs=fit_lta.epochs_best,
                zmax_table=zmax_table,
                lta_obj=fit_lta.lta_best_obj,
                invmap_n=None,
            )
            M_train_l = 0.0
            if not sn.y_is_mu:
                M_train_l = sn_bestfit_M_for_indices(sn_train, mu_train_l, idx_fit=None)

            mu_full_l = predict_sn_mu(
                sn,
                H0=fit_lta.H0, Om=fit_lta.Om,
                epochs=fit_lta.epochs_best,
                zmax_table=zmax_table,
                lta_obj=fit_lta.lta_best_obj,
                invmap_n=None,
            )
            mu_ref_full_l = sn_mu_reference(sn, mu_full_l)
            r_full_l = sn.y - (mu_ref_full_l + float(M_train_l))

            r_test_l = r_full_l[idx_test]
            chi2_l_m = chi2_gaussian(r_test_l, C_test)
            chi2_l_m_diag = chi2_gaussian_diag(r_test_l, C_test)

            chi2_l_c = conditional_chi2_from_precompute(r_full_l, pre)
            gain_b = float(chi2_b_m - chi2_b_c)
            gain_l = float(chi2_l_m - chi2_l_c)
            delta_gain = float(gain_b - gain_l)

        # Store per-fold scores
        chi2_base_m.append(float(chi2_b_m))
        chi2_lta_m.append(float(chi2_l_m))
        chi2_base_c.append(float(chi2_b_c))
        chi2_lta_c.append(float(chi2_l_c))

        # Print per-fold summary
        d_m = float(chi2_b_m - chi2_l_m)
        d_c = float(chi2_b_c - chi2_l_c)

        if d_c < 0:
            print(f"[CV]   cond-diag: var_shrink={pre['var_shrink']:+.3f}  "
                f"q={pre.get('cross_scale', 1.0):.3f}  "
                f"med|corr|_TTcross={pre.get('cross_stat_train_med_abs_corr', float('nan')):.4g}  "
                f"med|corr|_VTcross={pre.get('cross_stat_vt_med_abs_corr', float('nan')):.4g}  "
                f"max|corr_VT|={pre['max_abs_corr_VT']:.3f}  "
                f"mean|corr_VT|={pre['mean_abs_corr_VT']:.3f}  "
                f"mean|corr_VV_off|={pre['mean_abs_corr_VV_offdiag']:.3f}  "
                f"jitter={pre['jitter_used']:.2e}")

        # Fold table row (CSV)
        row = dict(
            fold=int(i + 1),
            seed=int(args.cv_seed),
            block=str(cv_block),
            cond_cross=str(cond_cross),
            test_IDSURVEY=str(test_block_label),
            N_test=int(idx_test.size),
            N_train=int(idx_train.size),
            zHD_min_test=float(zmin_test),
            zHD_max_test=float(zmax_test),
            t_anchor_train_gyr=float(LTA_T_ANCHOR_GYR),

            # Train-fit params
            H0_baseline=float(H0_b),
            Om_baseline=float(Om_b),
            alpha_baseline=float(alpha_b),

            H0_lta=float(fit_lta.H0),
            Om_lta=float(fit_lta.Om),
            alpha_lta=float(fit_lta.alpha_rd),
            s_anchor=float(fit_lta.s_anchor),
            gC=float(fit_lta.gC),
            gL=float(fit_lta.gL),

            # Scores
            chi2_test_marg_baseline=float(chi2_b_m),
            chi2_test_marg_lta=float(chi2_l_m),
            dchi2_test_marg=float(d_m),

            chi2_test_margdiag_baseline=float(chi2_b_m_diag),
            chi2_test_margdiag_lta=float(chi2_l_m_diag),
            dchi2_test_margdiag=float(chi2_b_m_diag - chi2_l_m_diag),

            chi2_test_cond_baseline=float(chi2_b_c),
            chi2_test_cond_lta=float(chi2_l_c),
            dchi2_test_cond=float(d_c),

            # Conditional diagnostics (why conditional can differ from marginal)
            cond_var_shrink=float(pre["var_shrink"]),
            cond_logdet=float(pre["logdet_Ccond"]),
            cond_jitter_used=float(pre["jitter_used"]),
            corr_max_abs_VT=float(pre["max_abs_corr_VT"]),
            corr_mean_abs_VT=float(pre["mean_abs_corr_VT"]),
            corr_mean_abs_VV_offdiag=float(pre["mean_abs_corr_VV_offdiag"]),
            cond_gain_baseline = float(gain_b),
            cond_gain_lta = float(gain_l),
            cond_gain_diff = float(delta_gain),
            cond_cross_scale=float(pre.get("cross_scale", np.nan)),
            cond_med_abs_corr_train_cross=float(pre.get("cross_stat_train_med_abs_corr", np.nan)),
            cond_med_abs_corr_vt_cross=float(pre.get("cross_stat_vt_med_abs_corr", np.nan)),
            # Posterior predictive (Laplace) scores (NaN if --cv-predictive plugin)
            nll_test_marg_baseline=float(nll_b_m),
            nll_test_marg_lta=float(nll_l_m),
            dnll_test_marg=float(nll_b_m - nll_l_m) if np.isfinite(nll_b_m) and np.isfinite(nll_l_m) else float("nan"),

            nll_test_cond_baseline=float(nll_b_c),
            nll_test_cond_lta=float(nll_l_c),
            dnll_test_cond=float(nll_b_c - nll_l_c) if np.isfinite(nll_b_c) and np.isfinite(nll_l_c) else float("nan"),

        )
        fold_rows.append(row)

        if score_mode == "conditional":
            print(f"Fold {i+1}/{k}:  χ²_test|train baseline={chi2_b_c:.3f}   {form}={chi2_l_c:.3f}   Δ={d_c:+.3f}")
        elif score_mode == "marginal":
            print(f"Fold {i+1}/{k}:  χ²_test baseline={chi2_b_m:.3f}   {form}={chi2_l_m:.3f}   Δ={d_m:+.3f}")
        else:
            print(f"Fold {i+1}/{k}:  χ²_test (marg)      baseline={chi2_b_m:.3f}   {form}={chi2_l_m:.3f}   Δ={d_m:+.3f}")
            print(f"            χ²_test|train (cond) baseline={chi2_b_c:.3f}   {form}={chi2_l_c:.3f}   Δ={d_c:+.3f}")

    # Convert to arrays
    chi2_base_m = np.asarray(chi2_base_m, dtype=float)
    chi2_lta_m  = np.asarray(chi2_lta_m, dtype=float)
    chi2_base_c = np.asarray(chi2_base_c, dtype=float)
    chi2_lta_c  = np.asarray(chi2_lta_c, dtype=float)

    d_m = chi2_base_m - chi2_lta_m
    d_c = chi2_base_c - chi2_lta_c

    print("-" * 80)

    # =========================
    # PLOTTING / SAVING FIGURES
    # =========================
    seed = int(args.cv_seed)

    def _plot_and_save(dvals: np.ndarray, ylab: str, score_tag: str) -> None:
        fig = plt.figure()
        plt.plot(np.arange(1, k + 1), dvals, marker="o")
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("fold")
        plt.ylabel(ylab)
        plt.title(f"Cross-validation improvement per fold ({form}), score={score_tag}, block={cv_block}, seed={seed}")
        plt.tight_layout()
        save_fig(fig, f"cv_dchi2_per_fold_{form}_{score_tag}_block{cv_block}_k{k}_seed{seed}")

        plt.close(fig)

    if score_mode == "both":
        _plot_and_save(d_m, "Δχ²_test (baseline − LTA)", "marginal")
        _plot_and_save(d_c, "Δχ²_test|train (baseline − LTA)", "conditional")
    elif score_mode == "conditional":
        _plot_and_save(d_c, "Δχ²_test|train (baseline − LTA)", "conditional")
    else:
        _plot_and_save(d_m, "Δχ²_test (baseline − LTA)", "marginal")
    
    # =========================
    # SAVE FOLD TABLE (CSV)
    # =========================
    if fold_rows:
        df_folds = pd.DataFrame(fold_rows)

        # Print quick summaries
        dmd = df_folds["dchi2_test_margdiag"].to_numpy(float)

        if pred_mode == "laplace":
            dm = df_folds["dnll_test_marg"].to_numpy(float)
            dc = df_folds["dnll_test_cond"].to_numpy(float)
            print(f"[CV] Predictive ΔNLL (marg): sum={np.nansum(dm):+.3f}  mean={np.nanmean(dm):+.3f}  pos={(dm>0).sum()}/{dm.size}")
            print(f"[CV] Predictive ΔNLL (cond): sum={np.nansum(dc):+.3f}  mean={np.nanmean(dc):+.3f}  pos={(dc>0).sum()}/{dc.size}")
        else:
            dm = df_folds["dchi2_test_marg"].to_numpy(float)
            dc = df_folds["dchi2_test_cond"].to_numpy(float)
            print(f"[CV] Marginal Δχ²:    sum={dm.sum():+.3f}  mean={dm.mean():+.3f}  pos={(dm>0).sum()}/{dm.size}")
            print(f"[CV] Diag-marg Δχ²:   sum={dmd.sum():+.3f}  mean={dmd.mean():+.3f}  pos={(dmd>0).sum()}/{dmd.size}")
            print(f"[CV] Conditional Δχ²: sum={dc.sum():+.3f}  mean={dc.mean():+.3f}  pos={(dc>0).sum()}/{dc.size}")

        # Save if outdir is set
        if getattr(args, "outdir", None):
            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            tag = str(getattr(args, "run_tag", "")).strip() or "cv"
            csv_path = outdir / f"cv_fold_table_{form}_block{cv_block}_k{k}_seed{int(args.cv_seed)}_{tag}.csv"
            df_folds.to_csv(csv_path, index=False)
            print(f"[CV] Saved fold table CSV: {csv_path}")
        else:
            print("[CV] Fold table (head):")
            print(df_folds.head(10).to_string(index=False))
        
        return df_folds

    return pd.DataFrame()

def _top_redshift_gaps(z: np.ndarray, top: int = 5) -> list[tuple[float, float, float]]:
    """
    Return top gaps as (z_lo, z_hi, dz) in the sorted z array.
    """
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 2:
        return []
    zs = np.sort(z)
    dz = np.diff(zs)
    if dz.size == 0:
        return []
    k = int(min(top, dz.size))
    idx = np.argsort(dz)[-k:][::-1]
    out = []
    for i in idx:
        out.append((float(zs[i]), float(zs[i + 1]), float(dz[i])))
    return out


def gap_selection_sanity(sn: SNData, args: argparse.Namespace, save_fig=None) -> None:
    """
    Prints:
      - largest gaps in selected zHD
      - binned counts in selected sample
      - binned counts in RAW file (no selection) to see if the gap is intrinsic or selection-induced
    Optionally saves a histogram plot via save_fig(fig, stem).
    """
    z_sel = np.asarray(sn.zHD, dtype=float)
    z_sel = z_sel[np.isfinite(z_sel)]

    print("\n" + "-" * 80)
    print("[sanity] Redshift-gap / selection sanity checks")
    print(f"[sanity] selected N={int(z_sel.size)}  zHD∈[{float(np.min(z_sel)):.5g},{float(np.max(z_sel)):.5g}]")
    print(f"[sanity] args: sn_sample={getattr(args,'sn_sample',None)!r}  sn_filter={getattr(args,'sn_filter',None)!r}  "
          f"sn_zmin={float(getattr(args,'sn_zmin',0.0))}  sn_zmax={getattr(args,'sn_zmax',None)}")

    # Top gaps in selected sample
    gaps = _top_redshift_gaps(z_sel, top=8)
    if gaps:
        print("[sanity] largest gaps in SELECTED zHD:")
        for (z0, z1, dz) in gaps:
            print(f"  gap: {z0:.6g} -> {z1:.6g}   Δz={dz:.6g}")
    else:
        print("[sanity] not enough selected zHD values to compute gaps.")

    # Bin counts in selected sample (log-relevant and the ranges you mentioned)
    bins = np.array([0.0, 0.005, 0.01, 0.015, 0.02, 0.023, 0.03, 0.05, 0.10, 0.15, 0.25, 0.50, 1.0, 2.5, 10.0], dtype=float)
    counts_sel, _ = np.histogram(z_sel, bins=bins)
    print("[sanity] SELECTED counts by z bins:")
    for i in range(len(bins) - 1):
        print(f"  [{bins[i]:.3g},{bins[i+1]:.3g}): {int(counts_sel[i])}")

    # Compare to RAW (unselected) file counts to see if the gap is selection-induced
    try:
        df_raw = pd.read_csv(args.sn_dat, sep=r"\s+", comment="#")
        if "zHD" in df_raw.columns:
            z_raw = pd.to_numeric(df_raw["zHD"], errors="coerce").to_numpy(dtype=float)
            z_raw = z_raw[np.isfinite(z_raw)]
            counts_raw, _ = np.histogram(z_raw, bins=bins)

            print("[sanity] RAW file counts by z bins (no selection):")
            for i in range(len(bins) - 1):
                print(f"  [{bins[i]:.3g},{bins[i+1]:.3g}): {int(counts_raw[i])}")

            # Simple gap attribution heuristic:
            # if RAW has >0 but SELECTED has 0 in a bin, your selection/cuts created the gap in that interval.
            induced = []
            intrinsic = []
            for i in range(len(bins) - 1):
                if counts_raw[i] > 0 and counts_sel[i] == 0:
                    induced.append((bins[i], bins[i+1], int(counts_raw[i])))
                if counts_raw[i] == 0 and counts_sel[i] == 0:
                    intrinsic.append((bins[i], bins[i+1]))

            if induced:
                print("[sanity][warn] bins that exist in RAW but are EMPTY in SELECTED (gap likely selection-induced):")
                for (a, b, nraw) in induced:
                    print(f"  [{a:.3g},{b:.3g}) RAW={nraw}  SELECTED=0")

        else:
            print("[sanity][warn] RAW file has no 'zHD' column; cannot compare raw-vs-selected gap structure.")
    except Exception as e:
        print(f"[sanity][warn] could not load RAW file for gap comparison: {e}")

    # Optional plot
    if save_fig is not None and z_sel.size > 0:
        fig = plt.figure()
        plt.hist(z_sel, bins=60)
        plt.xlabel("zHD (selected sample)")
        plt.ylabel("count")
        plt.title("Selected zHD histogram (gap sanity)")
        plt.tight_layout()
        save_fig(fig, "sanity_zHD_histogram")

def _ids_from_group_labels(labels: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    """
    Map arbitrary labels (strings/objects) -> integer ids [0..G-1] and return (ids, id->label dict).
    Stable under np.unique sorting.
    """
    labels = np.asarray(labels)
    uniq, inv = np.unique(labels.astype(str), return_inverse=True)
    sid_to_label = {int(i): str(uniq[i]) for i in range(int(uniq.size))}
    return inv.astype(int), sid_to_label

# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sn-dat", default="./Pantheon+SH0ES.dat")
    ap.add_argument("--sn-cov", default="./Pantheon+SH0ES_STAT+SYS.cov")
    ap.add_argument("--bao-res", default="./BAO_consensus_results_dM_Hz.txt")
    ap.add_argument("--bao-cov", default="./BAO_consensus_covtot_dM_Hz.txt")
    ap.add_argument(
        "--sn-ycol",
        default="m_b_corr",
        help=(
            "SN observable column.\n"
            "Recommended:\n"
            "  - Cosmology (Pantheon-style): m_b_corr  (uses full covariance file)\n"
            "  - SH0ES-calibrated distances: MU_SH0ES (use --sn-cov-mode diag)\n"
        ),
    )

    ap.add_argument(
        "--sn-cov-mode",
        default="file",
        choices=["file", "diag"],
        help=(
            "How to build the SN covariance.\n"
            "  - file: use --sn-cov as a full covariance matrix (recommended for m_b_corr)\n"
            "  - diag: build diagonal covariance from an error column (recommended for MU_SH0ES)\n"
        ),
    )

    ap.add_argument(
        "--sn-errcol",
        default="MU_SH0ES_ERR_DIAG",
        help=(
            "Error column to use when --sn-cov-mode diag.\n"
            "If omitted, inferred from --sn-ycol:\n"
            "  MU_SH0ES -> MU_SH0ES_ERR_DIAG\n"
            "  m_b_corr  -> m_b_corr_err_DIAG\n"
        ),
    )

    ap.add_argument(
        "--sn-sample",
        default="shoes_global",
        choices=["all", "shoes_hf", "shoes_global", "calibrators", "noncalibrators"],
        help=(
            "Which subset of Pantheon+SH0ES SNe to use:\n"
            "  - all:            everything passing z cuts\n"
            "  - shoes_hf:       USED_IN_SH0ES_HF == 1 (SH0ES Hubble-flow sample)\n"
            "  - calibrators:    IS_CALIBRATOR == 1\n"
            "  - noncalibrators: IS_CALIBRATOR == 0\n"
        ),
    )

    ap.add_argument(
        "--sn-filter",
        default=None,
        help=(
            "Optional pandas-eval boolean expression applied as an extra SN mask.\n"
            "Example: \"(USED_IN_SH0ES_HF == 1) & (IS_CALIBRATOR == 0)\""
        ),
    )
    
    ap.add_argument("--sn-zmin", type=float, default=0.0, help="Minimum zHD to include.")
    ap.add_argument("--sn-zmax", type=float, default=None, help="Maximum zHD to include.")

    ap.add_argument("--t-digital-gyr", type=float, default=LTAEpochs.t_digital_gyr)
    ap.add_argument("--t-complex-gyr", type=float, default=LTAEpochs.t_complex_gyr)
    ap.add_argument("--t-life-gyr", type=float, default=LTAEpochs.t_life_gyr)

    ap.add_argument("--zmax-table", type=float, default=2.5, help="Max redshift for interpolation tables.")
    ap.add_argument(
        "--lta-forms",
        default="powerlaw",
        help=(
            "Comma-separated list of LTA forms to fit in one run. "
            "Choices: powerlaw,"
        ),
    )
    ap.add_argument("--outdir", default=None,
                    help="If set, save plots into this directory (e.g. ./plots).")
    ap.add_argument("--plot-formats", default="png",
                    help="Comma-separated formats, e.g. 'png' or 'png,pdf'.")
    ap.add_argument("--no-show", action="store_true",
                    help="Do not call plt.show() (useful for servers/headless runs).")
    ap.add_argument("--do-null", action="store_true", help="Run null injections (parametric bootstrap).")
    ap.add_argument("--null-n", type=int, default=50, help="Number of null realizations.")
    ap.add_argument("--null-seed", type=int, default=0, help="RNG seed for null injections.")
    ap.add_argument("--null-form", default="powerlaw", help="Which LTA form to test under null.")
    ap.add_argument(
        "--do-cv-null",
        action="store_true",
        help="Run CV-null injections: simulate baseline-only mocks and compute a null distribution for the *cross-validation* Δχ² statistic.",
    )
    ap.add_argument("--cv-null-n", type=int, default=50, help="Number of CV-null realizations.")
    ap.add_argument("--cv-null-seed", type=int, default=0, help="RNG seed for CV-null injections.")
    ap.add_argument(
        "--cv-null-form",
        default=None,
        help="Which LTA form to test in CV-null. Defaults to --cv-form.",
    )


    ap.add_argument("--do-cv", action="store_true", help="Run k-fold cross-validation on SNe.")
    ap.add_argument("--cv-k", type=int, default=10, help="Number of folds for SN cross-validation.")
    ap.add_argument("--cv-seed", type=int, default=50, help="Seed for fold construction.")
    ap.add_argument("--cv-form", default="powerlaw", help="Which LTA form to cross-validate.")
    ap.add_argument(
        "--cv-score",
        default="both",
        choices=["marginal", "conditional", "both"],
        help=(
            "How to score the held-out SN fold when the SN covariance is correlated.\n"
            "  - marginal:   χ²_test = r_test^T C_test^{-1} r_test   (standard k-fold CV)\n"
            "  - conditional:χ²_test|train via Schur complement (uses train residuals)\n"
            "  - both: compute both; totals/plot use marginal."
        ),
    )
    ap.add_argument(
        "--cv-block",
        default="none",
        choices=["none", "survey"],
        help=(
            "Fold construction strategy for SN CV.\n"
            "  - none:   z-stratified k-fold (uses --cv-k)\n"
            "  - survey: leave-one-survey-out using IDSURVEY "
            "(calibrators always kept in TRAIN when ladder-anchored likelihood is active)."
        ),
    )

    ap.add_argument(
        "--cv-conditional-cross",
        default="full",
        choices=["auto", "full", "full_quotient", "zero", "within_survey", "survey_avg"],
        help=(
            "For conditional CV scoring, how to treat train–test cross-covariance C_VT.\n"
            "  - full: use C_VT from the provided covariance (true Gaussian conditional p(test|train))\n"
            "  - full_quotient: use FULL C_VT pattern but rescale cross-survey VT entries by a deterministic\n"
            "                   quotient q computed from robust |corr| stats inside TRAIN (parameter-free).\n"
            "  - zero: force C_VT=0 so test is independent of train (conditional reduces to marginal)\n"
            "  - within_survey: keep only C_VT entries where IDSURVEY(test)==IDSURVEY(train)\n"
            "                   (for leave-one-survey-out, this zeroes C_VT by construction)\n"
            "  - auto: backwards-compatible alias (currently treated as 'full'; use 'zero' explicitly for old behavior)"
            "  - survey_avg: replace train–test C_VT with an 'expected' cross-survey coupling computed\n"
            "                from TRAIN: for each train survey, use its mean cross-survey correlation\n"
            "                to other TRAIN surveys, then set C_VT[i,j]=rho_survey(j)*sqrt(var_i var_j)\n"
            "                (nonzero even for leave-one-survey-out, no tunable parameter)\n"

        ),
    )

    ap.add_argument(
        "--run-tag",
        default="",
        help="Optional string appended to every saved plot filename. "
            "If omitted, a timestamp is used (prevents stale plot confusion).",
    )
    ap.add_argument(
        "--fix-h0",
        type=float,
        default=None,
        help="If set, fix H0 to this value (km/s/Mpc) in ALL fits (baseline, LTA, CV, null).",
    )
    ap.add_argument(
        "--fix-alpha-rd",
        type=float,
        default=None,
        help=(
            "If set, fix alpha_rd (the BAO sound-horizon scaling parameter) to this value "
            "in ALL fits (baseline, LTA, CV, null)."
        ),
    )
    ap.add_argument(
        "--fix-om",
        type=float,
        default=None,
        help="If set, fix Omega_m in ALL fits (baseline, LTA, CV, null). Must satisfy 0<Om<1.",
    )
    ap.add_argument(
        "--t-anchor-gyr",
        type=float,
        default=None,
        help=(
            "Earth retarded lookback time t_anchor [Gyr] at which the LTA amplitude parameter equals s(t_anchor). "
            "If omitted, it is set automatically from the SN sample (5th percentile of endpoint t_ret under baseline approx)."
        ),
    )
    ap.add_argument(
        "--use-planck-priors",
        action="store_true",
        help=(
            "Add Gaussian priors corresponding to Planck 2018 base ΛCDM "
            "(default numbers are for base_plikHM_TTTEEE_lowl_lowE_lensing). "
            "These priors are added as χ² penalties in total_chi2."
        ),
    )

    ap.add_argument(
        "--sn-anchor-m-to-calibrators",
        action="store_true",
        help=(
            "Enable ladder-anchored SN likelihood for m_b_corr when calibrators are present:\n"
            "  - infer M from calibrators only\n"
            "  - evaluate HF likelihood conditional on calibrators (full covariance via Schur complement)\n"
            "This makes HF a true absolute-distance probe of H0 (SH0ES-style)."
        ),
    )

    # Planck 2018 defaults (68% limits) for base_plikHM_TTTEEE_lowl_lowE_lensing:
    # H0=67.36±0.54, Ωm=0.3153±0.0073, rdrag=147.09±0.26
    ap.add_argument("--planck-h0-mean", type=float, default=67.36)
    ap.add_argument("--planck-h0-sigma", type=float, default=0.54)
    ap.add_argument("--planck-om-mean", type=float, default=0.3153)
    ap.add_argument("--planck-om-sigma", type=float, default=0.0073)
    ap.add_argument("--planck-rdrag-mean", type=float, default=147.09)
    ap.add_argument("--planck-rdrag-sigma", type=float, default=0.26)

    ap.add_argument(
        "--bao-rd-fid",
        type=float,
        default=147.78,
        help=(
            "Fiducial sound horizon r_d,fid [Mpc] used to scale the BAO consensus results. "
            "Used to convert Planck r_drag prior into alpha_rd prior via alpha_rd=r_drag/r_d,fid. "
            "Set this to match your BAO compilation."
        ),
    )

    ap.add_argument(
        "--planck-prior-no-alpha",
        action="store_true",
        help="If set, do NOT apply a Planck prior on alpha_rd.",
    )
    ap.add_argument(
        "--planck-prior-no-h0",
        action="store_true",
        help="If set, do NOT apply a Planck prior on H0.",
    )
    ap.add_argument(
        "--planck-prior-no-om",
        action="store_true",
        help="If set, do NOT apply a Planck prior on Omega_m.",
    )

    ap.add_argument(
        "--planck-prior-mode",
        default="chain",
        choices=["diag", "chain"],
        help="Planck prior type: 'diag' uses (mean±sigma) uncorrelated; 'chain' uses correlated covariance from MCMC chains.",
    )
    ap.add_argument(
        "--planck-chain-root",
        default="./planck_chains/COM_CosmoParams_base-plikHM_R3.01/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing",
        help=(
            "File root for Planck chains (WITHOUT extension). Example:\n"
            "  .../base_plikHM_TTTEEE_lowl_lowE_lensing\n"
            "Must have .paramnames and _1.txt, _2.txt, ... next to it."
        ),
    )

    ap.add_argument("--fix-g-complex", type=float, default=None,
                    help="Fix LTA gC (powerlaw B) parameter to this value.")
    ap.add_argument("--fix-g-life", type=float, default=None,
                    help="Fix LTA gL (powerlaw p) parameter to this value.")

    ap.add_argument(
        "--cv-predictive",
        default="laplace",
        choices=["plugin", "laplace"],
        help=(
            "How to score held-out folds.\n"
            "  - plugin:  current behavior using point estimate theta_hat\n"
            "  - laplace: approximate posterior predictive by Laplace (Gaussian) parameter posterior + linearized residuals\n"
        ),
    )
    ap.add_argument("--cv-laplace-eps-scale", type=float, default=1.0,
                    help="Scale factor for finite-diff steps used in Laplace Hessian/Jacobian.")
    ap.add_argument("--cv-laplace-ridge", type=float, default=1e-6,
                    help="Relative eigenvalue floor used to regularize Hessian before inversion.")
    ap.add_argument("--cv-laplace-jitter-frac", type=float, default=1e-12,
                    help="Fractional jitter used if predictive covariance is not PD.")

    # ----------------------------
    # Leave-out (jackknife) refit analysis
    # ----------------------------
    ap.add_argument("--do-loo", action="store_true",
                    help="Run leave-out refits: remove survey/telescope groups and refit baseline+LTA on remaining data.")
    ap.add_argument("--loo-form", default="powerlaw",
                    help="Which LTA form to use for leave-out refits. Default: --cv-form if set, else first --lta-forms.")
    ap.add_argument("--loo-group", default="telescope", choices=["idsurvey", "telescope"],
                    help="Grouping for leave-out refits: 'idsurvey' removes IDSURVEY groups; 'telescope' removes groups defined by --loo-map.")
    ap.add_argument("--loo-map", default=None,
                    help="CSV/JSON mapping IDSURVEY -> telescope/instrument label (required if --loo-group telescope).")
    ap.add_argument("--loo-max-combo-size", type=int, default=1,
                    help="Remove up to this many groups at once. 1=leave-one-out; 2=all pairs; etc.")
    ap.add_argument("--loo-max-combos", type=int, default=0,
                    help="Cap number of group-removal combinations evaluated (0=no cap).")
    ap.add_argument("--loo-seed", type=int, default=0,
                    help="Seed used to shuffle group order (matters when --loo-max-combos > 0).")


    args = ap.parse_args()

    fixed_h0 = None if args.fix_h0 is None else float(args.fix_h0)
    if fixed_h0 is not None:
        if (not np.isfinite(fixed_h0)) or (fixed_h0 <= 0.0):
            raise ValueError(f"--fix-h0 must be a positive finite number, got {args.fix_h0}")
        
    fixed_alpha = None if args.fix_alpha_rd is None else float(args.fix_alpha_rd)
    if fixed_alpha is not None:
        if (not np.isfinite(fixed_alpha)) or (fixed_alpha <= 0.0):
            raise ValueError(f"--fix-alpha-rd must be a positive finite number, got {args.fix_alpha_rd}")
    
    fixed_om = None if args.fix_om is None else float(args.fix_om)
    if fixed_om is not None:
        if (not np.isfinite(fixed_om)) or (fixed_om <= 0.0) or (fixed_om >= 1.0):
            raise ValueError(f"--fix-om must satisfy 0 < Om < 1, got {args.fix_om}")

    global EARLY_PRIORS
    EARLY_PRIORS = None

    if args.use_planck_priors:
        rd_fid = float(args.bao_rd_fid)

        # Which Planck components to include
        include_h0 = (not bool(getattr(args, "planck_prior_no_h0", False)))
        include_om = (not bool(getattr(args, "planck_prior_no_om", False)))
        include_a  = (not bool(getattr(args, "planck_prior_no_alpha", False)))

        if args.planck_prior_mode == "chain":
            if not args.planck_chain_root:
                raise ValueError("--planck-prior-mode chain requires --planck-chain-root")

            mv_full = build_planck_prior_from_chain(
                chain_root=args.planck_chain_root,
                rd_fid=rd_fid,
            )

            keep_idx: list[int] = []
            keep_labels: list[str] = []
            if include_h0:
                keep_idx.append(0); keep_labels.append("H0")
            if include_om:
                keep_idx.append(1); keep_labels.append("Omega_m")
            if include_a:
                keep_idx.append(2); keep_labels.append("alpha_rd")

            if len(keep_idx) == 0:
                EARLY_PRIORS = None
                print("[prior] Planck priors requested, but all components were disabled (--planck-prior-no-h0/--no-om/--no-alpha).")
            else:
                mean_sel = mv_full.mean[np.array(keep_idx, dtype=int)]
                cov_sel  = mv_full.cov[np.ix_(keep_idx, keep_idx)]

                mv = MVGaussianPrior(
                    mean=mean_sel,
                    cov=cov_sel,
                    labels=tuple(keep_labels),
                    label=f"{mv_full.label} (subset)",
                )

                EARLY_PRIORS = EarlyPriors(mv=mv, mv_idx=tuple(keep_idx))

                # Optional: print correlation matrix for sanity
                C = mv.cov
                sig = np.sqrt(np.diag(C))
                sig = np.maximum(sig, np.finfo(float).tiny)
                corr = C / np.outer(sig, sig)
                print("[prior] Planck CHAIN correlated prior enabled (subset):")
                print("  labels =", mv.labels if mv.labels else keep_labels)
                print("  mean   =", mv.mean)
                print("  sig    =", sig)
                print("  corr   =\n", corr)

        else:
            # Diagonal (uncorrelated) priors
            h0_prior = None
            om_prior = None
            alpha_prior = None

            if include_h0:
                h0_prior = GaussianPrior(float(args.planck_h0_mean), float(args.planck_h0_sigma), "H0 (Planck)")
            if include_om:
                om_prior = GaussianPrior(float(args.planck_om_mean), float(args.planck_om_sigma), "Omega_m (Planck)")
            if include_a:
                alpha_mean = float(args.planck_rdrag_mean) / rd_fid
                alpha_sigma = float(args.planck_rdrag_sigma) / rd_fid
                alpha_prior = GaussianPrior(alpha_mean, alpha_sigma, "alpha_rd (from Planck r_drag)")

            if (h0_prior is None) and (om_prior is None) and (alpha_prior is None):
                EARLY_PRIORS = None
                print("[prior] Planck priors requested, but all components were disabled (--planck-prior-no-h0/--no-om/--no-alpha).")
            else:
                EARLY_PRIORS = EarlyPriors(
                    h0=h0_prior,
                    om=om_prior,
                    alpha_rd=alpha_prior,
                )

    forms_to_run = [f.strip() for f in args.lta_forms.split(",") if f.strip()]
    print("Will run LTA forms:", forms_to_run)

    outdir = None
    plot_formats = [f.strip().lower() for f in args.plot_formats.split(",") if f.strip()]

    if args.outdir is not None:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

    run_tag = str(args.run_tag).strip()
    if not run_tag:
        run_tag = time.strftime("%Y%m%d-%H%M%S")
    print(f"[run-tag] {run_tag}")

    args.run_tag = run_tag

    def save_fig(fig: plt.Figure, stem: str) -> None:
        if outdir is None:
            # If not saving and running headless, close to avoid figure buildup
            if args.no_show:
                plt.close(fig)
            return

        for fmt in plot_formats:
            figpath = outdir / f"{stem}_{run_tag}.{fmt}"
            fig.savefig(figpath, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {stem}_{run_tag} -> {outdir.resolve()}")

        plt.close(fig)


    epochs = LTAEpochs(
        t_digital_gyr=args.t_digital_gyr,
        t_complex_gyr=args.t_complex_gyr,
        t_life_gyr=args.t_life_gyr,
    )

    sn = build_sn_data(
        args.sn_dat,
        args.sn_cov,
        ycol=args.sn_ycol,
        zmin=args.sn_zmin,
        zmax=args.sn_zmax,
        cov_mode=args.sn_cov_mode,
        errcol=args.sn_errcol,
        sample=args.sn_sample,
        filter_query=args.sn_filter,
        anchor_m_to_calibrators=bool(getattr(args, "sn_anchor_m_to_calibrators", False)),
    )
    print(f"[SN] ycol={args.sn_ycol}  cov_mode={args.sn_cov_mode}  sample={args.sn_sample}  N={len(sn.y)}")

    gap_selection_sanity(sn, args=args, save_fig=save_fig)

    n_cal = int(np.sum(sn.is_calibrator)) if sn.is_calibrator is not None else 0
    if n_cal > 0:
        mu_c = sn.mu_ceph[sn.is_calibrator]
        print(f"[SN] calibrators in selection: N={n_cal}")
        print(f"[SN] CEPH_DIST (calibrators) range: {float(np.nanmin(mu_c)):.3f} .. {float(np.nanmax(mu_c)):.3f}")

        diag = np.diag(sn.cov)
        print(f"[SN] cov diag median (calib)    : {float(np.median(diag[sn.is_calibrator])):.6g}")
        print(f"[SN] cov diag median (non-calib): {float(np.median(diag[~sn.is_calibrator])):.6g}")

    bao = build_bao_data(args.bao_res, args.bao_cov)

    # Make sure zmax_table covers both datasets
    zmax_needed = float(max(np.max(sn.zHD), np.max(bao.z))) + 0.05
    zmax_table = max(args.zmax_table, zmax_needed)

    # ---- Fit baseline ΛCDM ----
    x0_base = np.array([70.0, 0.30, 1.0], dtype=float)
    bounds_base = [(40.0, 100.0), (0.05, 0.60), (0.6, 1.4)]

    # Optional: fix H0 globally
    if fixed_h0 is not None:
        x0_base[0] = fixed_h0
        bounds_base[0] = (fixed_h0, fixed_h0)
        print(f"[fix] H0 fixed to {fixed_h0:.6f} km/s/Mpc")

    # Optional: fix alpha_rd globally
    if fixed_alpha is not None:
        x0_base[2] = fixed_alpha
        bounds_base[2] = (fixed_alpha, fixed_alpha)
        print(f"[fix] alpha_rd fixed to {fixed_alpha:.6f}")
    
    if fixed_om is not None:
        x0_base[1] = fixed_om
        bounds_base[1] = (fixed_om, fixed_om)
        print(f"[fix] Om fixed to {fixed_om:.6f}")

    print("\nFitting baseline flat ΛCDM (no LTA)...")
    res_base = minimize(
        lambda x: total_chi2(x, sn, bao, epochs, use_lta=False, zmax_table=zmax_table),
        x0_base,
        method="Powell",
        bounds=bounds_base,
    )
    if not res_base.success:
        print("Baseline fit did not converge:", res_base.message)
    H0_b, Om_b, alpha_b = res_base.x
    chi2_b = res_base.fun
    print(f"Baseline best-fit: H0={H0_b:.4f}, Om={Om_b:.5f}, chi2={chi2_b:.3f}, alpha_rd={alpha_b:.5f}")

    # ---- Sanity test: LTA can reproduce baseline when s_anchor=0 ----
    # If this is not true (to ~1e-6), something is inconsistent in the implementation.
    x_nested = np.array([H0_b, Om_b, alpha_b, epochs.t_life_gyr, 0.0, 0.0, 0.0], dtype=float)
    chi2_nested = total_chi2(x_nested, sn, bao, epochs, use_lta=True, zmax_table=zmax_table)
    print(f"Sanity check: chi2(LTA with s_anchor=0,g=0) = {chi2_nested:.6f}  vs  chi2(baseline) = {chi2_b:.6f}")

    # after zmax_table is chosen
    tmp = build_cosmology_tables(H0=float(H0_b), Om=float(Om_b), zmax=zmax_table)
    chi_data_max = float(tmp.chi_of_z(zmax_needed))
    chi_table_max = float(tmp.chi_mpc[-1])

    if chi_table_max < 2.0 * chi_data_max:
        print("[warn] zmax_table may be too small for Earth-retarded mapping (needs chi(zmax) >= 2*chi(data)). "
            "Increase --zmax-table to avoid clamping in z_of_chi.")
            
    # ------------------------------------------------------------------
    # Choose the amplitude anchor time t_anchor from the SN sample
    # (baseline approx: zcos≈zobs, so chi≈chi(zHD))
    # ------------------------------------------------------------------
    global LTA_T_ANCHOR_GYR
    tables_diag = build_cosmology_tables(H0=float(H0_b), Om=float(Om_b), zmax=zmax_table)
    chi_end = tables_diag.chi_of_z(sn.zHD)
    tret_end = earth_retarded_lookback_gyr(chi_end, tables_diag)

    print(f"[diag] SN zHD range: {float(np.min(sn.zHD)):.5f} .. {float(np.max(sn.zHD)):.5f}")
    print(f"[diag] SN t_ret range (baseline approx): {float(np.min(tret_end)):.3f} .. {float(np.max(tret_end)):.3f} Gyr")

    # Choose anchor from the HF/non-calibrator subset if calibrators are present.
    # Calibrators don't use μ(z) in the likelihood (they use CEPH_DIST), so they
    # should NOT control the LTA amplitude anchor.
    tret_for_anchor = tret_end
    if (sn.is_calibrator is not None) and np.any(sn.is_calibrator) and (not sn.y_is_mu):
        noncal = ~sn.is_calibrator
        if np.any(noncal):
            tret_for_anchor = tret_end[noncal]
            print(f"[diag] t_anchor computed from non-calibrators only (N={int(np.sum(noncal))}).")

    if args.t_anchor_gyr is None:
        # robust "near-min" anchor
        t_anchor = float(np.quantile(tret_for_anchor, 0.05))
    else:
        t_anchor = float(args.t_anchor_gyr)

    # keep it inside (0, t_life)
    t_anchor = float(np.clip(t_anchor, 1e-6, float(epochs.t_life_gyr) - 1e-6))

    LTA_T_ANCHOR_GYR = t_anchor
    print(f"[diag] Using LTA amplitude anchor: t_anchor={LTA_T_ANCHOR_GYR:.3f} Gyr (Earth retarded lookback)")

    # Baseline predictions (independent of LTA form)
    def compute_predictions(H0: float, Om: float, alpha_rd: float,
                        t_life_gyr: float,
                        s_anchor: float = 0.0, gC: float = 0.0, gL: float = 0.0,
                        use_lta: bool = False,
                        lta_override=None):

        """
        Compute SN μ(z) and BAO (D_M, H_z) predictions for given parameters.
        t_life_gyr is used only if use_lta=True.
        """
        tables = build_cosmology_tables(H0=H0, Om=Om, zmax=zmax_table)

        if use_lta:
            epochs_local = LTAEpochs(
                t_digital_gyr=epochs.t_digital_gyr,
                t_complex_gyr=epochs.t_complex_gyr,
                t_life_gyr=t_life_gyr,
            )
            if lta_override is not None:
                lta = lta_override
            else:
                lta = LTAParams(s_anchor_km_s_per_mpc=s_anchor, g_complex=gC, g_life=gL)
        else:
            epochs_local = epochs
            lta = LTAParams(s_anchor_km_s_per_mpc=0.0, g_complex=0.0, g_life=0.0)


        invmap = build_inverse_zmap(
            tables, lta, epochs_local,
            zmax=zmax_table, nz=4000,
        )

        # SN
        zcos_sn = invmap(sn.zHD)
        chi_sn = tables.chi_of_z(zcos_sn)
        dL_sn = (1.0 + sn.zHEL) * chi_sn
        mu_sn = 5.0 * np.log10(dL_sn) + 25.0

        # BAO
        zcos_bao = invmap(bao.z)
        chi_bao = tables.chi_of_z(zcos_bao)

        I_bao = lta_integral_I(chi_bao, tables, lta, epochs_local)

        DM_pred = chi_bao / alpha_rd

        Hcos = tables.H_of_z(zcos_bao)
        jac = dzobs_dzcos(zcos_bao, tables, lta, epochs_local)
        Hz_pred = (Hcos * jac) * alpha_rd

        return mu_sn, DM_pred, Hz_pred

    # Baseline predictions (no LTA)
    mu_b, DM_b, Hz_b = compute_predictions(
        H0_b, Om_b, alpha_b,
        epochs.t_life_gyr,
        0.0, 0.0, 0.0,
        use_lta=False,
    )

    def sn_residuals(mu_pred: np.ndarray) -> np.ndarray:
        mu_pred = np.asarray(mu_pred, dtype=float)
        mu_ref = sn_mu_reference(sn, mu_pred)  # HF: mu_pred, Calibrators: CEPH_DIST

        if sn.y_is_mu:
            return sn.y - mu_ref

        r0 = sn.y - mu_ref
        Cinv_r0 = cho_solve(sn.cho, r0, check_finite=False)
        M_best = float((sn.ones @ Cinv_r0) / sn.ones_Cinv_ones)
        return sn.y - (mu_ref + M_best)

    # ---- Loop over requested LTA forms ----
    fit_real: dict[str, FitBundle] = {}

    for form in forms_to_run:
        print("\n" + "=" * 80)
        print(f"Fitting ΛCDM + LTA with form: {form}")
        print("=" * 80)

        try:
            fit = fit_form(
                form=form,
                sn=sn,
                bao=bao,
                epochs=epochs,
                zmax_table=zmax_table,
                H0_b=float(H0_b),
                Om_b=float(Om_b),
                alpha_b=float(alpha_b),
                args=args,
            )
        except ValueError as e:
            print(f"Skipping form '{form}': {e}")
            continue

        res_lta = fit.res
        x0_lta = fit.x0_used
        bounds_lta = fit.bounds
        H0_l, Om_l, alpha_l, tL_l = fit.H0, fit.Om, fit.alpha_rd, fit.tL
        sA_l, gC_l, gL_l = fit.s_anchor, fit.gC, fit.gL
        lta_best_obj = fit.lta_best_obj
        epochs_best = fit.epochs_best

        print("LTA optimizer diagnostics:")
        print("  success:", res_lta.success)
        print("  message:", res_lta.message)
        print("  nit:", getattr(res_lta, "nit", None))
        print("  x0_lta:", x0_lta)
        print("  xbest :", res_lta.x)
        print("  delta :", res_lta.x - x0_lta)

        chi2_l = float(res_lta.fun)
        dchi2 = float(chi2_b - chi2_l)

        # Reporting mirrors your existing behavior
        print(
            f"LTA({form}) best-fit: H0={H0_l:.4f}, Om={Om_l:.5f}, alpha_rd={alpha_l:.5f}, "
            f"t_life={tL_l:.3f} Gyr, s_anchor={sA_l:.4f} km/s/Mpc, gC={gC_l:.3f}, gL={gL_l:.3f}, "
            f"chi2={chi2_l:.3f}"
        )
        print(f"Δchi2 (baseline - LTA[{form}]) = {dchi2:.3f}  (positive favors LTA)")

        # Store for null/CV use
        fit_real[form] = fit

        mu_l, DM_l, Hz_l = compute_predictions(
            H0_l, Om_l, alpha_l,
            tL_l,
            use_lta=True,
            lta_override=lta_best_obj,
        )

        mu_l_fixed, DM_l_fixed, Hz_l_fixed = compute_predictions(
            H0_b, Om_b, alpha_b,
            tL_l,
            use_lta=True,
            lta_override=lta_best_obj,
        )


        # Chi2 breakdown
        tables_b = build_cosmology_tables(H0=H0_b, Om=Om_b, zmax=zmax_table)
        params_b = {
            "H0": H0_b,
            "Om": Om_b,
            "alpha_rd": alpha_b,
            "s_anchor": 0.0,
            "g_complex": 0.0,
            "g_life": 0.0,
        }
        chi2_sn_b, _ = chi2_sn(params_b, sn, tables_b, epochs, use_lta=False)
        chi2_bao_b = chi2_bao(params_b, bao, tables_b, epochs, use_lta=False)

        tables_l = build_cosmology_tables(H0=H0_l, Om=Om_l, zmax=zmax_table)

        gA = float(earth_history_g(np.array([t_anchor]), lta_best_obj, epochs_best)[0])
        s_now = float(lta_local_s(np.array([0.0]), tables_l, lta_best_obj, epochs_best)[0])
        print(f"[diag] g(t_anchor)={gA:.6f}  =>  s_now=s_anchor/gA ≈ {float(fit.s_anchor)/gA:.6f}  (direct s_now={s_now:.6f})")

        # Use the best-fit life epoch for the LTA breakdown
        epochs_best = LTAEpochs(
            t_digital_gyr=epochs.t_digital_gyr,
            t_complex_gyr=epochs.t_complex_gyr,
            t_life_gyr=tL_l,
        )
        params_l = {
            "H0": H0_l,
            "Om": Om_l,
            "alpha_rd": alpha_l,
            "s_anchor": sA_l,
            "g_complex": gC_l,
            "g_life": gL_l,
        }
        chi2_sn_l, _ = chi2_sn(
            params_l, sn, tables_l, epochs_best,
            use_lta=True, lta_override=lta_best_obj, invmap=None
        )
        chi2_bao_l = chi2_bao(
            params_l, bao, tables_l, epochs_best,
            use_lta=True, lta_override=lta_best_obj, invmap=None
        )

        chi2_prior_b = 0.0
        chi2_prior_l = 0.0
        if EARLY_PRIORS is not None:
            chi2_prior_b = EARLY_PRIORS.chi2(H0_b, Om_b, alpha_b)
            chi2_prior_l = EARLY_PRIORS.chi2(H0_l, Om_l, alpha_l)

        print("Chi2 breakdown:")
        print(f"  baseline: chi2_SN={chi2_sn_b:.3f}, chi2_BAO={chi2_bao_b:.3f}, chi2_prior={chi2_prior_b:.3f}, chi2_tot={chi2_sn_b+chi2_bao_b+chi2_prior_b:.3f}")
        print(f"  {form:7}: chi2_SN={chi2_sn_l:.3f}, chi2_BAO={chi2_bao_l:.3f}, chi2_prior={chi2_prior_l:.3f}, chi2_tot={chi2_sn_l+chi2_bao_l+chi2_prior_l:.3f}")

        chi2_data_l = chi2_sn_l + chi2_bao_l
        chi2_prior_l = 0.0
        if EARLY_PRIORS is not None:
            chi2_prior_l = EARLY_PRIORS.chi2(H0_l, Om_l, alpha_l)

        print(f"  chi2_data   ≈ {chi2_data_l:.6f}")
        print(f"  chi2_prior  ≈ {chi2_prior_l:.6f}")
        print(f"  chi2_total  ≈ {chi2_l:.6f}  (what optimizer minimized)")
 

        # Model selection
        Nsn = len(sn.y)
        Nbao = len(bao.data_vector)
        Ntot = Nsn + Nbao
        # If using m_b_corr-like observable, we profile out an intercept M analytically.
        # Count it as 1 parameter for AIC/BIC fairness.
        k_M = 0
        if (not sn.y_is_mu) and (sn.anchor is None):
            # only count M if we actually profile/fit it from the SN likelihood
            k_M = 1

        k_base = count_free_params(bounds_base) + k_M
        k_lta  = count_free_params(bounds_lta)  + k_M

        chi2tot_b = chi2_sn_b + chi2_bao_b + chi2_prior_b
        chi2tot_l = chi2_sn_l + chi2_bao_l + chi2_prior_l

        Nprior = EARLY_PRIORS.n_constraints() if EARLY_PRIORS is not None else 0
        Ntot = Nsn + Nbao + Nprior

        # Covariance effective dimension: tells you how many independent modes you really have.
        # N_eff = (sum λ)^2 / sum(λ^2).  Equals N only if all modes contribute equally and independently.
        evals = np.linalg.eigvalsh(sn.cov)
        evals = evals[evals > 0]
        neff = (evals.sum() ** 2) / np.sum(evals ** 2)
        print(f"[SN] covariance effective dimension N_eff≈{neff:.1f} (out of N={len(sn.y)})")


        print("Model selection (lower better):")
        print(f"  AIC baseline={aic(chi2tot_b, k_base):.3f}, AIC {form}={aic(chi2tot_l, k_lta):.3f}")
        print(f"  BIC baseline={bic(chi2tot_b, k_base, Ntot):.3f}, BIC {form}={bic(chi2tot_l, k_lta, Ntot):.3f}")


        # Local diagnostic: s evaluated essentially at the observer (chi=0 => z_emit=0, t_ret=0)
        s_local = float(lta_local_s(np.array([0.0]), tables_l, lta_best_obj, epochs_best)[0])
        print(f"  H0_eff_local({form}) ≈ H0 + s(chi≈0) = {H0_l + s_local:.3f} km/s/Mpc")

        # Saturated integrated effect: maximum I reached once t_ret crosses t_life
        z_L = float(tables_l.z_of_tlb(epochs_best.t_life_gyr))
        chi_front = float(tables_l.chi_of_z(z_L))
        chi_cut = 0.5 * chi_front  # where t_ret=t_life along the photon path
        I_sat = float(lta_integral_I(np.array([chi_cut]), tables_l, lta_best_obj, epochs_best)[0])
        print(f"  I_sat(t_ret=t_life) ≈ {I_sat:.6g}  (this is max Δln(1+z) in the model)")

        # --- Exact χ² contribution decomposition by subset: contrib_i = r_i * (C^{-1} r)_i ---
        r_b = sn_residuals(mu_b)
        r_l = sn_residuals(mu_l)

        Cinvr_b = cho_solve(sn.cho, r_b, check_finite=False)
        Cinvr_l = cho_solve(sn.cho, r_l, check_finite=False)

        contrib_b = r_b * Cinvr_b
        contrib_l = r_l * Cinvr_l
        if sn.anchor is None:
            if (sn.is_calibrator is not None) and np.any(sn.is_calibrator) and (not sn.y_is_mu):
                cal = sn.is_calibrator
                hf  = ~cal
                dcal = float(np.sum(contrib_b[cal]) - np.sum(contrib_l[cal]))
                dhf  = float(np.sum(contrib_b[hf])  - np.sum(contrib_l[hf]))
                print(f"[diag] SN Δχ² split: calibrators={dcal:+.3f}   HF/noncal={dhf:+.3f}   total={dcal+dhf:+.3f}")
        else:
            # True anchored breakdown:
            _, info_b = chi2_sn(params_b, sn, tables_b, epochs, use_lta=False)
            _, info_l = chi2_sn(params_l, sn, tables_l, epochs_best, use_lta=True,
                                lta_override=lta_best_obj, invmap=None)
            print("[anchor] Anchored SN breakdown (this matches the likelihood used in the fit):")
            print(f"[anchor]   chi2_cal_const = {info_b['chi2_cal_const']:.3f}  (constant, same for all models)")
            print(f"[anchor]   chi2_hf baseline = {info_b['chi2_hf']:.3f}")
            print(f"[anchor]   chi2_hf {form}   = {info_l['chi2_hf']:.3f}")
            print(f"[anchor]   Δchi2_hf (baseline − {form}) = {info_b['chi2_hf'] - info_l['chi2_hf']:+.3f}")

        fig_dmu = plt.figure()
        plt.scatter(sn.zHD, mu_l - mu_b, s=6, alpha=0.5)
        plt.xscale("log")
        plt.xlabel("zHD (observed)")
        plt.ylabel("Δμ = μ_LTA - μ_baseline [mag]")
        plt.title(f"Pure LTA imprint on SN distance modulus ({form})")
        plt.tight_layout()
        save_fig(fig_dmu, f"delta_mu_{form}")

        fig_dmu_pure = plt.figure()
        plt.scatter(sn.zHD, mu_l_fixed - mu_b, s=6, alpha=0.5)
        plt.xscale("log")
        plt.xlabel("zHD (observed)")
        plt.ylabel("Δμ_pure = μ(fixed cosmo + LTA) - μ(baseline) [mag]")
        plt.title(f"PURE LTA remapping imprint (cosmology fixed) ({form})")
        plt.tight_layout()
        save_fig(fig_dmu_pure, f"delta_mu_PURE_fixedcosmo_{form}")

        fig_dmu_full = plt.figure()
        plt.scatter(sn.zHD, mu_l - mu_b, s=6, alpha=0.5)
        plt.xscale("log")
        plt.xlabel("zHD (observed)")
        plt.ylabel("Δμ_full = μ(best-fit LTA cosmo) - μ(best-fit baseline) [mag]")
        plt.title(f"FULL model difference (includes cosmology shift) ({form})")
        plt.tight_layout()
        save_fig(fig_dmu_full, f"delta_mu_FULL_{form}")

        # SN residuals
        fig1 = plt.figure()
        plt.scatter(sn.zHD, sn_residuals(mu_b), s=6, alpha=0.5, label="Residuals vs baseline")
        plt.scatter(sn.zHD, sn_residuals(mu_l), s=6, alpha=0.5, label=f"Residuals vs LTA ({form})")
        plt.axhline(0.0)
        plt.xscale("log")
        plt.xlabel("zHD (observed)")
        plt.ylabel("SN residual (data - model)")
        plt.title(f"Pantheon+SH0ES residuals: baseline vs ΛCDM+LTA ({form})")
        plt.legend()
        plt.tight_layout()
        save_fig(fig1, f"sn_residuals_{form}")

        # BAO D_M
        fig2 = plt.figure()
        plt.scatter(bao.z, bao.DM, label="D_M obs")
        plt.scatter(bao.z, DM_b, label="D_M pred baseline")
        plt.scatter(bao.z, DM_l, label=f"D_M pred LTA ({form})")
        plt.xlabel("z (observed)")
        plt.ylabel("D_M [Mpc]")
        plt.title(f"BAO D_M comparison ({form})")
        plt.legend()
        plt.tight_layout()
        save_fig(fig2, f"bao_DM_{form}")

        # BAO H_z
        fig3 = plt.figure()
        plt.scatter(bao.z, bao.Hz, label="H_z obs")
        plt.scatter(bao.z, Hz_b, label="H_z pred baseline")
        plt.scatter(bao.z, Hz_l, label=f"H_z pred LTA ({form})")
        plt.xlabel("z (observed)")
        plt.ylabel("H_z [km/s/Mpc]")
        plt.title(f"BAO H_z comparison ({form})")
        plt.legend()
        plt.tight_layout()
        save_fig(fig3, f"bao_Hz_{form}")

        # z mapping
        fig4 = plt.figure()
        zgrid = np.linspace(0.0, max(np.max(sn.zHD), np.max(bao.z)) * 1.05, 400)
        tables_lta = build_cosmology_tables(H0=H0_l, Om=Om_l, zmax=zmax_table)
        epochs_best = LTAEpochs(
            t_digital_gyr=epochs.t_digital_gyr,
            t_complex_gyr=epochs.t_complex_gyr,
            t_life_gyr=tL_l,
        )
        lta_best = lta_best_obj

        zobs_grid = zobs_from_zcos(
            zgrid, tables_lta, lta_best, epochs_best
        )

        plt.plot(zgrid, zobs_grid, label=f"z_obs(z_cos) with LTA ({form})")
        plt.plot(zgrid, zgrid, linestyle="--", label="no LTA (z_obs=z_cos)")
        plt.xlabel("z_cos")
        plt.ylabel("z_obs")
        plt.title(f"Redshift remapping caused by LTA ({form})")
        plt.legend()
        plt.tight_layout()
        save_fig(fig4, f"z_mapping_{form}")

        figI_z = plt.figure()
        I_from_map = np.log((1.0 + zobs_grid) / (1.0 + zgrid))
        plt.plot(zgrid, I_from_map)
        plt.xlabel("z_cos")
        plt.ylabel("Δ ln(1+z) = ln[(1+z_obs)/(1+z_cos)]")
        plt.title(f"Implied integrated LTA factor vs z_cos ({form})")
        plt.tight_layout()
        save_fig(figI_z, f"delta_ln1pz_{form}")

        # Zoomed z-mapping at low z
        fig4z = plt.figure()
        zgrid_low = np.linspace(0.0, 0.2, 400)
    
        zobs_low = zobs_from_zcos(
            zgrid_low, tables_lta, lta_best, epochs_best
        )

        plt.plot(zgrid_low, zobs_low, label=f"z_obs(z_cos) with LTA ({form})")
        plt.plot(zgrid_low, zgrid_low, linestyle="--", label="no LTA (z_obs=z_cos)")
        plt.xlabel("z_cos")
        plt.ylabel("z_obs")
        plt.title(f"Redshift remapping at low z ({form})")
        plt.legend()
        plt.tight_layout()
        save_fig(fig4z, f"z_mapping_low_{form}")

        # LTA shape diagnostics: s(chi) and I(chi)
        chi_max = float(tables_lta.chi_of_z(max(np.max(sn.zHD), np.max(bao.z)) * 1.05))
        chi_plot = np.linspace(0.0, chi_max, 600)

        s_plot = lta_local_s(chi_plot, tables_lta, lta_best, epochs_best)
        I_plot = lta_integral_I(chi_plot, tables_lta, lta_best, epochs_best)

        # ------------------------------------------------------------
        # New: express the reconstructed shape in z_cos and z_obs coordinates
        # ------------------------------------------------------------
        zmax_diag = float(min(zmax_table, max(np.max(sn.zHD), np.max(bao.z)) * 1.05))
        zcos_diag = np.linspace(0.0, zmax_diag, 1200)

        chi_diag = tables_lta.chi_of_z(zcos_diag)
        I_diag = lta_integral_I(chi_diag, tables_lta, lta_best, epochs_best)
        s_diag = lta_local_s(chi_diag, tables_lta, lta_best, epochs_best)
        zobs_diag = zobs_from_zcos(zcos_diag, tables_lta, lta_best, epochs_best)

        order = np.argsort(zobs_diag)

        fig_szcos = plt.figure()
        plt.plot(zcos_diag, s_diag)
        plt.xlabel("z_cos")
        plt.ylabel("s(z_cos) [km/s/Mpc]")
        plt.title(f"LTA local strength vs cosmological redshift ({form})")
        plt.tight_layout()
        save_fig(fig_szcos, f"lta_s_vs_zcos_{form}")

        fig_szobs = plt.figure()
        plt.plot(zobs_diag[order], s_diag[order])
        plt.xlabel("z_obs")
        plt.ylabel("s(z_obs) [km/s/Mpc]")
        plt.title(f"LTA local strength vs observed redshift ({form})")
        plt.tight_layout()
        save_fig(fig_szobs, f"lta_s_vs_zobs_{form}")

        fig_Izobs = plt.figure()
        plt.plot(zobs_diag[order], I_diag[order])
        plt.xlabel("z_obs")
        plt.ylabel("I(z_obs) (dimensionless)")
        plt.title(f"Integrated LTA factor vs observed redshift ({form})")
        plt.tight_layout()
        save_fig(fig_Izobs, f"lta_I_vs_zobs_{form}")

        fig_dz = plt.figure()
        plt.plot(zcos_diag, zobs_diag - zcos_diag)
        plt.xlabel("z_cos")
        plt.ylabel("Δz = z_obs - z_cos")
        plt.title(f"Redshift shift vs z_cos ({form})")
        plt.tight_layout()
        save_fig(fig_dz, f"delta_z_vs_zcos_{form}")

        # Extra diagnostic: I as a function of Earth retarded emission time along the line of sight.
        tret_plot = earth_retarded_lookback_gyr(chi_plot, tables_lta)
        order = np.argsort(tret_plot)

        fig_I_tret = plt.figure()
        plt.plot(tret_plot[order], I_plot[order])
        plt.xlabel("Earth retarded lookback time t_ret [Gyr] (0=now)")
        plt.ylabel("I(t_ret) (dimensionless)")
        plt.title(f"Integrated LTA factor along LOS vs Earth time ({form})")
        plt.tight_layout()
        save_fig(fig_I_tret, f"lta_I_vs_tret_LOS_{form}")

        fig_shape = plt.figure()
        plt.plot(chi_plot, s_plot)
        plt.xlabel("χ [Mpc]")
        plt.ylabel("s(χ) [km/s/Mpc]")
        plt.title(f"LTA local strength profile s(χ) ({form})")
        plt.tight_layout()
        save_fig(fig_shape, f"lta_s_of_chi_{form}")
        
        # Earth-time view of the reconstructed/parametric local strength
        fig_t = plt.figure()
        t_edges = np.linspace(0.0, epochs_best.t_life_gyr, 600)
        # Convert t_ret -> s(t_ret) by evaluating at chi that maps to that t_ret is messy;
        
        t_ret_demo = t_edges
        g_demo = earth_history_g(t_ret_demo, lta_best_obj, epochs_best)
        gA_demo = _anchor_g(lta_best_obj, epochs_best)
        s_demo = float(lta_best_obj.s_anchor_km_s_per_mpc) * (g_demo / gA_demo)

        plt.plot(t_ret_demo, s_demo)
        plt.xlabel("Earth retarded lookback time t_ret [Gyr] (0=now)")
        plt.ylabel("s(t_ret) [km/s/Mpc]")
        plt.title(f"LTA implied strength vs Earth time ({form})")
        plt.tight_layout()
        save_fig(fig_t, f"lta_s_of_tret_{form}")

        fig_I = plt.figure()
        plt.plot(chi_plot, I_plot)
        plt.xlabel("χ [Mpc]")
        plt.ylabel("I(χ) = (1/c)∫ s dχ (dimensionless)")
        plt.title(f"LTA integrated effect I(χ) ({form})")
        plt.tight_layout()
        save_fig(fig_I, f"lta_I_of_chi_{form}")

        invmap_best = build_inverse_zmap(tables_lta, lta_best, epochs_best, zmax=zmax_table, nz=4000)

        # Jacobian diagnostic: how much LTA rescales radial BAO via dzobs/dzcos
        zcos_bao_diag = invmap_best(bao.z)
        jac_bao = dzobs_dzcos(zcos_bao_diag, tables_lta, lta_best, epochs_best)

        fig_jac = plt.figure()
        plt.scatter(bao.z, jac_bao)
        plt.xlabel("z (observed)")
        plt.ylabel("dz_obs/dz_cos")
        plt.title(f"LTA Jacobian (drives BAO H(z)) ({form})")
        plt.tight_layout()
        save_fig(fig_jac, f"bao_jacobian_{form}")

        fig_inv = plt.figure()
        zobs_test = np.linspace(0, 2.5, 400)
        plt.plot(zobs_test, invmap_best(zobs_test))
        plt.plot(zobs_test, zobs_test, "--")
        plt.xlabel("z_obs")
        plt.ylabel("z_cos")
        plt.title(f"Inverse mapping z_cos(z_obs) ({form})")
        plt.tight_layout()
        save_fig(fig_inv, f"z_inverse_{form}")

        # ------------------------------------------------------------
        # Diagnostic only: compare transverse BAO D_M under conventions A vs B
        # (Likelihood should use A only.)
        # ------------------------------------------------------------
        chi_bao_diag = tables_lta.chi_of_z(zcos_bao_diag)
        I_bao_diag = lta_integral_I(chi_bao_diag, tables_lta, lta_best, epochs_best)

        DM_A = chi_bao_diag / alpha_l
        DM_B_diag = (chi_bao_diag * np.exp(np.clip(I_bao_diag, -EXP_CLIP, EXP_CLIP))) / alpha_l

        fig_bao_dmAB = plt.figure()
        plt.scatter(bao.z, bao.DM, label="D_M obs")
        plt.scatter(bao.z, DM_A, label="D_M pred (A)")
        plt.scatter(bao.z, DM_B_diag, label="D_M pred (B diagnostic)")
        plt.xlabel("z (observed)")
        plt.ylabel("D_M [Mpc]")
        plt.title(f"BAO transverse diagnostic: A vs B ({form})")
        plt.legend()
        plt.tight_layout()
        save_fig(fig_bao_dmAB, f"bao_DM_A_vs_B_diagnostic_{form}")

        # ------------------------------------------------------------
        # Diagnostic: whitened residual improvement (handles full covariance)
        # ------------------------------------------------------------
        r_b = sn_residuals(mu_b)
        r_l = sn_residuals(mu_l)

        Lsn = sn.cho[0]  # lower-triangular Cholesky factor (because cho_factor(..., lower=True))
        w_b = solve_triangular(Lsn, r_b, lower=True, check_finite=False)
        w_l = solve_triangular(Lsn, r_l, lower=True, check_finite=False)

        dchi2_point = (w_b * w_b) - (w_l * w_l)

        fig_dchi2 = plt.figure()
        plt.scatter(sn.zHD, dchi2_point, s=6, alpha=0.35)
        plt.axhline(0.0)
        plt.xscale("log")
        plt.xlabel("zHD (observed)")
        plt.ylabel("Δχ² contribution (baseline − LTA) in whitened space")
        plt.title(f"Where the SN χ² improvement comes from ({form})")
        plt.tight_layout()
        save_fig(fig_dchi2, f"sn_delta_chi2_whitened_{form}")

    def plot_binned_residuals(sn, mu_base, mu_lta, z_bins=20, outpath="binned_residuals_zoomed.png"):
        # 1. Calculate best-fit M for baseline to center the plot
        # (Simple weighted average of raw residuals)
        r_raw_b = sn.y - sn_mu_reference(sn, mu_base)
        w_diag = 1.0 / np.diag(sn.cov)
        M_b = np.sum(r_raw_b * w_diag) / np.sum(w_diag)

        # 2. Subtract M so we see the variation around zero
        # Note: We subtract the SAME M_b from both. This is crucial. 
        # We want to see how LTA differs from the Baseline's best fit.
        r_base = r_raw_b - M_b
        r_lta = (sn.y - sn_mu_reference(sn, mu_lta)) - M_b 

        # Create bins
        z_min = float(np.min(sn.zHD))
        z_max = float(np.max(sn.zHD))
        z_edges = np.logspace(np.log10(z_min), np.log10(z_max), z_bins + 1)
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
        
        bin_base = []
        bin_lta = []
        
        for i in range(z_bins):
            mask = (sn.zHD >= z_edges[i]) & (sn.zHD < z_edges[i+1])
            if np.sum(mask) > 0:
                w = w_diag[mask]
                w_sum = np.sum(w)
                # Weighted mean of the centered residuals
                mean_b = np.sum(r_base[mask] * w) / w_sum
                mean_l = np.sum(r_lta[mask] * w) / w_sum
                bin_base.append(mean_b)
                bin_lta.append(mean_l)
            else:
                bin_base.append(np.nan)
                bin_lta.append(np.nan)
                
        fig = plt.figure(figsize=(10, 6))
        plt.axhline(0, color='k', linestyle='-', alpha=0.3)
        
        # Plot baseline in RED
        plt.plot(z_centers, bin_base, 'o-', color='red', label='Baseline Residuals', alpha=0.7)
        # Plot LTA in BLUE
        plt.plot(z_centers, bin_lta, 'o-', color='blue', label='LTA Residuals', alpha=0.7)
        
        plt.xscale('log')
        plt.xlabel('Redshift (zHD)')
        plt.ylabel('Hubble Residual [mag] (M_base subtracted)')
        plt.title('Binned Hubble Residuals: The "Local Void" Correction')
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        
        # --- DYNAMIC Y-LIMITS ---
        # Convert to arrays to handle NaNs safely
        b_arr = np.array(bin_base, dtype=float)
        l_arr = np.array(bin_lta, dtype=float)
        
        # Find global min/max ignoring NaNs
        valid_vals = np.concatenate([b_arr[np.isfinite(b_arr)], l_arr[np.isfinite(l_arr)]])
        
        if valid_vals.size > 0:
            y_min, y_max = np.min(valid_vals), np.max(valid_vals)
            # Add 10% padding
            margin = 0.1 * (y_max - y_min) if (y_max - y_min) > 0 else 0.05
            plt.ylim(y_min - margin, y_max + margin)
        
        plt.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        print(f"Saved zoomed residuals to {outpath}")
    
    plot_path = outdir / f"binned_residuals_{form}_{args.run_tag}.png" if outdir else f"binned_residuals_{form}.png"
    plot_binned_residuals(
            sn=sn, 
            mu_base=mu_b, 
            mu_lta=mu_l, 
            z_bins=20, 
            outpath=str(plot_path)
        )

    # ----------------------------
    # Optional: Cross-validation & Null injections
    # ----------------------------
    if args.do_cv:
        cv_form = str(args.cv_form).strip()
        run_cross_validation(
            form=cv_form,
            sn=sn,
            bao=bao,
            epochs=epochs,
            zmax_table=zmax_table,
            bounds_base=bounds_base,
            args=args,
            save_fig=save_fig,
        )

        # --- telescope/survey-group held-out scoring ---
        if sn.group is not None:
            labels = np.asarray(sn.group)
            uniq = np.unique(labels.astype(str))
            if uniq.size >= 2:
                grp_ids, sid_map = _ids_from_group_labels(labels)

                # Shallow copy SNData; reuse all heavy arrays and anchor; just swap idsurvey codes
                sn_grp = copy.copy(sn)
                sn_grp.idsurvey = grp_ids

                # Use the existing leave-one-survey-out fold builder by setting cv_block="survey"
                args_grp = copy.copy(args)
                args_grp.cv_block = "survey"
                args_grp.run_tag = str(args.run_tag) + "_group"

                def save_fig_grp(fig: plt.Figure, stem: str) -> None:
                    save_fig(fig, "group_" + stem)

                print("\n" + "=" * 80)
                print(f"[CV][group] Running leave-one-GROUP-out held-out scoring over sn.group (n_groups={int(uniq.size)})")
                print("=" * 80)

                run_cross_validation(
                    form=cv_form,
                    sn=sn_grp,
                    bao=bao,
                    epochs=epochs,
                    zmax_table=zmax_table,
                    bounds_base=bounds_base,
                    args=args_grp,
                    save_fig=save_fig_grp,
                    sid_to_label=sid_map,
                )
            else:
                print("[CV][group] sn.group exists but has <2 unique labels; skipping group-held-out CV.")
        else:
            print("[CV][group] sn.group is None; skipping group-held-out CV (no telescope/survey labels found).")
    
    if getattr(args, "do_cv_null", False):
        cv_null_form = str(getattr(args, "cv_null_form", "") or "").strip()
        if not cv_null_form:
            cv_null_form = str(args.cv_form).strip()

        run_cv_null_injections(
            form=cv_null_form,
            sn=sn,
            bao=bao,
            epochs=epochs,
            zmax_table=zmax_table,
            H0_base=float(H0_b),
            Om_base=float(Om_b),
            alpha_base=float(alpha_b),
            bounds_base=bounds_base,
            args=args,
            save_fig=save_fig,
        )

        # --- CV-null for telescope/survey-group held-out scoring ---
        if sn.group is not None:
            labels = np.asarray(sn.group)
            uniq = np.unique(labels.astype(str))
            if uniq.size >= 2:
                grp_ids, sid_map = _ids_from_group_labels(labels)
                sn_grp = copy.copy(sn)
                sn_grp.idsurvey = grp_ids

                args_grp = copy.copy(args)
                args_grp.cv_block = "survey"
                args_grp.run_tag = str(args.run_tag) + "_group"

                def save_fig_grp(fig: plt.Figure, stem: str) -> None:
                    save_fig(fig, "group_" + stem)

                print("\n" + "=" * 80)
                print(f"[cv-null][group] Running CV-null for GROUP-held-out statistic (n_groups={int(uniq.size)})")
                print("=" * 80)

                run_cv_null_injections(
                    form=cv_null_form,
                    sn=sn_grp,
                    bao=bao,
                    epochs=epochs,
                    zmax_table=zmax_table,
                    H0_base=float(H0_b),
                    Om_base=float(Om_b),
                    alpha_base=float(alpha_b),
                    bounds_base=bounds_base,
                    args=args_grp,
                    save_fig=save_fig_grp,
                )
            else:
                print("[cv-null][group] sn.group exists but has <2 unique labels; skipping group CV-null.")
        else:
            print("[cv-null][group] sn.group is None; skipping group CV-null.")

    if args.do_null:
        null_form = str(args.null_form).strip()
        if null_form not in fit_real:
            fit_real[null_form] = fit_form(
                form=null_form,
                sn=sn,
                bao=bao,
                epochs=epochs,
                zmax_table=zmax_table,
                H0_b=float(H0_b),
                Om_b=float(Om_b),
                alpha_b=float(alpha_b),
                args=args,
            )

        chi2_lta_real = float(fit_real[null_form].res.fun)

        run_null_injections(
            form=null_form,
            sn=sn,
            bao=bao,
            epochs=epochs,
            zmax_table=zmax_table,
            H0_base=float(H0_b),
            Om_base=float(Om_b),
            alpha_base=float(alpha_b),
            chi2_base_real=float(chi2_b),
            chi2_lta_real=float(chi2_lta_real),
            bounds_base=bounds_base,
            args=args,
            save_fig=save_fig,
        )
        ztest = np.array([0.01, 0.023, 0.05, 0.10, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0])
        Heff = Heff_of_zobs(ztest, H0=67.4, Om=0.315, zmax_table=zmax_table,
                            lta_obj=lta_best_obj, epochs=epochs_best)

        for z, h in zip(ztest, Heff):
            print(f"z_obs={z:6.3f}  Heff≈{h:8.3f} km/s/Mpc")
        
        # --- Diagnostic: what Earth-retarded times do the SN z-cuts probe? ---
        tables_diag = build_cosmology_tables(H0=67.4, Om=0.315, zmax=zmax_table)
        # Approx: for this diagnostic, treat zcos ~ zHD under baseline
        chi_diag = tables_diag.chi_of_z(sn.zHD)
        tret_diag = earth_retarded_lookback_gyr(chi_diag, tables_diag)

        print(f"[diag] SN zHD range: {sn.zHD.min():.5f} .. {sn.zHD.max():.5f}")
        print(f"[diag] SN t_ret range (baseline approx): {tret_diag.min():.3f} .. {tret_diag.max():.3f} Gyr")

    # ----------------------------
    # Optional: Leave-out (jackknife) refits
    # ----------------------------
    if getattr(args, "do_loo", False):
        loo_form = str(getattr(args, "loo_form", "") or "").strip()
        if not loo_form:
            # default: cv_form if provided, else first requested LTA form
            loo_form = str(getattr(args, "cv_form", "") or "").strip()
        if not loo_form:
            loo_form = forms_to_run[0] if forms_to_run else "powerlaw"

        run_leaveout_refits(
            form=loo_form,
            sn=sn,
            bao=bao,
            epochs=epochs,
            zmax_table=zmax_table,
            bounds_base=bounds_base,
            args=args,
        )

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())