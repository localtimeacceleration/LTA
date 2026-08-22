"""
OTA derivation-program numerics harness.

Imports lta_power.py as a library (NO edits to it) and reproduces the paper's
fiducial configuration (README reproduction command, run-tag 20260112-013808):

    ycol=m_b_corr, cov_mode=file, sample=shoes_global, anchor_m_to_calibrators,
    BAO consensus dM/Hz with rd_fid=147.78, Planck chain prior in
    (omega_m, alpha_rd) space, LTA form=powerlaw with t_life=3.8 fixed,
    B (g_complex)=1.0 fixed, p (g_life)=1.0 fixed  ->  free: H0, Om, alpha_rd, s_anchor.

It also provides a custom line-of-sight I(chi)-profile injection layer used by the
D4/D5/D8 forward models (compact source with physical dilution, extended
chronometric well, linear-theory void outflow). Injection is done by rebinding
lta_power.lta_integral_I / lta_power.lta_local_s at runtime, so every downstream
consumer (SN likelihood, BAO Jacobian, inverse z-mapping) sees a consistent
mapping. With no active profile the original functions are called unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import lta_power as lp  # noqa: E402

C_KM_S = lp.C_KM_S
RD_FID = 147.78
T_LIFE_FID = 3.8
PLANCK_CHAIN_ROOT = str(
    REPO
    / "planck_chains/COM_CosmoParams_base-plikHM_R3.01/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing"
)


# ----------------------------------------------------------------------
# Fiducial setup (mirrors lta_power.main() wiring for the README command)
# ----------------------------------------------------------------------

def setup_fiducial(use_planck_priors: bool = True, prior_space: str = "h0_om_alpha"):
    sn = lp.build_sn_data(
        str(REPO / "Pantheon+SH0ES.dat"),
        str(REPO / "Pantheon+SH0ES_STAT+SYS.cov"),
        ycol="m_b_corr",
        cov_mode="file",
        sample="shoes_global",
        anchor_m_to_calibrators=True,
    )
    bao = lp.build_bao_data(
        str(REPO / "BAO_consensus_results_dM_Hz.txt"),
        str(REPO / "BAO_consensus_covtot_dM_Hz.txt"),
        rd_fid=RD_FID,
    )

    lp.EARLY_PRIORS = None
    if use_planck_priors:
        mv_full = lp.build_planck_prior_from_chain(
            chain_root=PLANCK_CHAIN_ROOT, rd_fid=RD_FID, space=prior_space
        )
        if prior_space == "h0_om_alpha":
            keep_idx, labels = [0, 1, 2], ("H0", "Omega_m", "alpha_rd")
        else:
            keep_idx, labels = [0, 1], ("omega_m", "alpha_rd")
        mv = lp.MVGaussianPrior(
            mean=mv_full.mean[np.array(keep_idx)],
            cov=mv_full.cov[np.ix_(keep_idx, keep_idx)],
            labels=labels,
            label=f"{mv_full.label} (subset)",
        )
        lp.EARLY_PRIORS = lp.EarlyPriors(mv=mv, mv_idx=tuple(keep_idx), mv_space=prior_space)

    epochs = lp.LTAEpochs()  # t_life=3.8 default
    zmax_table = max(2.5, float(np.max(sn.zHD)) + 0.05, float(np.max(bao.z)) + 0.05)
    lp.LTA_FORM = "powerlaw"
    return sn, bao, epochs, zmax_table


def set_anchor_from_baseline(sn, H0_b: float, Om_b: float, zmax_table: float, epochs) -> float:
    """Same rule as main(): 5th percentile of t_ret over non-calibrator SNe."""
    tables = lp.build_cosmology_tables(H0=float(H0_b), Om=float(Om_b), zmax=zmax_table)
    chi_end = tables.chi_of_z(sn.zHD)
    tret_end = lp.earth_retarded_lookback_gyr(chi_end, tables)
    sel = np.ones(len(sn.zHD), dtype=bool)
    if (sn.is_calibrator is not None) and np.any(sn.is_calibrator) and (not sn.y_is_mu):
        sel = ~sn.is_calibrator
    t_anchor = float(np.quantile(tret_end[sel], 0.05))
    t_anchor = float(np.clip(t_anchor, 1e-6, float(epochs.t_life_gyr) - 1e-6))
    lp.LTA_T_ANCHOR_GYR = t_anchor
    return t_anchor


def fit_baseline(sn, bao, epochs, zmax_table):
    res = lp.fit_baseline(sn, bao, epochs, zmax_table)
    return res


def fit_stepA(sn, bao, epochs, zmax_table, H0_b, Om_b, alpha_b):
    """Fiducial step-A fit: powerlaw, t_life fixed 3.8, B=1, p=1 fixed."""
    deactivate_profile()
    lp.LTA_FORM = "powerlaw"
    x0 = np.array([H0_b, Om_b, alpha_b, T_LIFE_FID, 6.0, 1.0, 1.0])
    bounds = [
        (40.0, 100.0),
        (0.05, 0.60),
        (0.6, 1.4),
        (T_LIFE_FID, T_LIFE_FID),
        (0.0, 50.0),
        (1.0, 1.0),
        (1.0, 1.0),
    ]
    obj = lambda x: lp.total_chi2(x, sn, bao, epochs, use_lta=True, zmax_table=zmax_table)
    res = minimize(obj, x0, method="Powell", bounds=bounds, options={"maxiter": 400})
    # nested-safe guard, as in fit_form()
    x_nested = np.array([H0_b, Om_b, alpha_b, T_LIFE_FID, 0.0, 1.0, 1.0])
    chi2_nested = float(obj(x_nested))
    if np.isfinite(chi2_nested) and chi2_nested < float(res.fun):
        res.x, res.fun = x_nested, chi2_nested
    return res


# ----------------------------------------------------------------------
# Custom I(chi)-profile injection
# ----------------------------------------------------------------------
#
# A profile supplies a normalized dimensionless shape I_sh(chi) with
# I_sh(0)=0 and max|I_sh|=1.  The parameter in the s_anchor slot is then the
# dimensionless amplitude A, so I(chi) = A * I_sh(chi) and
# s(chi) = c * dI/dchi (km/s/Mpc).  Everything downstream is consistent.

_ORIG_INTEGRAL_I = lp.lta_integral_I
_ORIG_LOCAL_S = lp.lta_local_s

_ACTIVE = {"profile": None}
_GRID_CACHE = {"key": None, "chi": None, "I_sh": None, "s_sh": None}


class IShapeProfile:
    """Base: subclasses implement _raw_I(chi, tables, epochs) -> unnormalized I."""

    tag = "base"

    def __init__(self, **params):
        self.params = dict(params)

    def key(self):
        return (self.tag,) + tuple(sorted(self.params.items()))

    def _raw_I(self, chi, tables, epochs):
        raise NotImplementedError

    def grids(self, tables, epochs):
        chi_max = float(tables.chi_mpc[-1])
        u = np.linspace(0.0, 1.0, 4000)
        chi = chi_max * u * u
        I_raw = np.asarray(self._raw_I(chi, tables, epochs), dtype=float)
        I_raw = I_raw - I_raw[0]  # enforce I(0)=0
        norm = float(np.max(np.abs(I_raw)))
        if norm <= 0.0 or not np.isfinite(norm):
            norm = 1.0
        I_sh = I_raw / norm
        s_sh = C_KM_S * np.gradient(I_sh, chi)
        return chi, I_sh, s_sh


def activate_profile(profile: IShapeProfile) -> None:
    _ACTIVE["profile"] = profile
    _GRID_CACHE["key"] = None


def deactivate_profile() -> None:
    _ACTIVE["profile"] = None
    _GRID_CACHE["key"] = None


def _get_grids(tables, epochs):
    prof = _ACTIVE["profile"]
    key = (int(id(tables.chi_mpc)), prof.key(), float(epochs.t_life_gyr))
    if _GRID_CACHE["key"] != key:
        chi, I_sh, s_sh = prof.grids(tables, epochs)
        _GRID_CACHE.update(key=key, chi=chi, I_sh=I_sh, s_sh=s_sh)
    return _GRID_CACHE["chi"], _GRID_CACHE["I_sh"], _GRID_CACHE["s_sh"]


def _patched_integral_I(chi_mpc, tables, lta, epochs):
    if _ACTIVE["profile"] is None:
        return _ORIG_INTEGRAL_I(chi_mpc, tables, lta, epochs)
    chi = np.maximum(np.asarray(chi_mpc, dtype=float), 0.0)
    A = float(getattr(lta, "s_anchor_km_s_per_mpc", 0.0))
    if A == 0.0:
        return np.zeros_like(chi)
    g_chi, g_I, _ = _get_grids(tables, epochs)
    return A * np.interp(chi, g_chi, g_I, left=0.0, right=float(g_I[-1]))


def _patched_local_s(chi_mpc, tables, lta, epochs):
    if _ACTIVE["profile"] is None:
        return _ORIG_LOCAL_S(chi_mpc, tables, lta, epochs)
    chi = np.maximum(np.asarray(chi_mpc, dtype=float), 0.0)
    A = float(getattr(lta, "s_anchor_km_s_per_mpc", 0.0))
    if A == 0.0:
        return np.zeros_like(chi)
    g_chi, _, g_s = _get_grids(tables, epochs)
    return A * np.interp(chi, g_chi, g_s, left=float(g_s[0]), right=0.0)


lp.lta_integral_I = _patched_integral_I
lp.lta_local_s = _patched_local_s


def fit_custom(sn, bao, epochs, zmax_table, H0_b, Om_b, alpha_b, profile,
               A_max=0.05, A0=3e-3):
    """Fit (H0, Om, alpha_rd, A) for an injected I-shape profile."""
    activate_profile(profile)
    lp.LTA_FORM = "powerlaw"  # guard path only; B=p=1 kept valid
    x0 = np.array([H0_b, Om_b, alpha_b, T_LIFE_FID, A0, 1.0, 1.0])
    bounds = [
        (40.0, 100.0),
        (0.05, 0.60),
        (0.6, 1.4),
        (T_LIFE_FID, T_LIFE_FID),
        (0.0, float(A_max)),
        (1.0, 1.0),
        (1.0, 1.0),
    ]
    obj = lambda x: lp.total_chi2(x, sn, bao, epochs, use_lta=True, zmax_table=zmax_table)
    res = minimize(obj, x0, method="Powell", bounds=bounds, options={"maxiter": 400})
    x_nested = np.array([H0_b, Om_b, alpha_b, T_LIFE_FID, 0.0, 1.0, 1.0])
    chi2_nested = float(obj(x_nested))
    if np.isfinite(chi2_nested) and chi2_nested < float(res.fun):
        res.x, res.fun = x_nested, chi2_nested
    deactivate_profile()
    return res


# ----------------------------------------------------------------------
# Profiles for D4 / D5 / D8
# ----------------------------------------------------------------------

def _powerlaw_g(t_gyr, B=1.0, p=1.0, tL=T_LIFE_FID):
    """Paper's bounded powerlaw activation, B=p=1 closed form."""
    t = np.clip(np.asarray(t_gyr, dtype=float), 0.0, tL)
    return (B * (tL - t)) / (tL * (B + t))


class CompactSourceProfile(IShapeProfile):
    """
    D4 branch A: observer-compact source with PHYSICAL dilution restored.

    psi_emit(chi) = g(t_ret(2chi)) * exp(-chi/lam) * R_reg / max(chi, R_reg)
    I(chi) proportional to psi_emit(0) - psi_emit(chi)  (endpoint form).
    Two-leg timing retained (correct for a compact source at the observer).
    params: lam_mpc (Yukawa range), R_reg_mpc (source coherence scale).
    """

    tag = "compact"

    def _raw_I(self, chi, tables, epochs):
        lam = float(self.params["lam_mpc"])
        R = float(self.params["R_reg_mpc"])
        tret = lp.earth_retarded_lookback_gyr(chi, tables)
        g = _powerlaw_g(tret, tL=float(epochs.t_life_gyr))
        P = g * np.exp(-chi / lam) * (R / np.maximum(chi, R))
        return P[0] - P  # P[0] at chi=0 -> dilution factor 1 (chi<R)


class ExtendedWellProfile(IShapeProfile):
    """
    D4 branch B / D5: correlated-tier chronometric well of comoving scale R_w,
    one-leg timing, growth-modulated:
    psi(chi) = D(t_lb(chi)) * W(chi),  W = 1 / (1 + (chi/R_w)^n)
    I proportional to psi(0) - psi(chi).
    """

    tag = "well"

    def _raw_I(self, chi, tables, epochs):
        Rw = float(self.params["R_w_mpc"])
        n = float(self.params["n"])
        tlb = tables.tlb_of_z(tables.z_of_chi(chi))
        H0 = float(tables.H_km_s_mpc[0])
        f_growth = 0.53  # Om^0.55 at Om~0.3
        H0_per_gyr = H0 / 978.5  # km/s/Mpc -> 1/Gyr  (1/H0[Gyr] = 978.5/H0)
        D = np.clip(1.0 - f_growth * H0_per_gyr * np.asarray(tlb), 0.0, None)
        W = 1.0 / (1.0 + (chi / Rw) ** n)
        P = D * W
        return P[0] - P


class LinearVoidProfile(IShapeProfile):
    """
    D8 mundane alternative: linear-theory outflow from a centered underdensity.
    Cumulative contrast  bar-delta(<r) = delta_c / (1 + (r/R_v)^gamma)   (delta_c < 0),
    v(r) = -(1/3) f H0 bar-delta(<r) r,     I(chi) = v(chi)/c.
    The fitted amplitude A maps to |delta_c| via the stored norm (see fit output).
    """

    tag = "void"

    def _raw_I(self, chi, tables, epochs):
        Rv = float(self.params["R_v_mpc"])
        gam = float(self.params["gamma"])
        H0 = float(tables.H_km_s_mpc[0])
        f_growth = 0.53
        shape = chi / (1.0 + (chi / Rv) ** gam)  # bar-delta shape * chi
        v = (f_growth * H0 / 3.0) * shape  # km/s per unit |delta_c|
        self.last_norm_v = float(np.max(np.abs(v / C_KM_S)))  # I per |delta_c|
        return v / C_KM_S


class StepAShapeControl(IShapeProfile):
    """Control: reproduces the step-A drift shape through the injection layer.
    s(chi) proportional to g(t_ret(2chi)); validates the machinery end to end."""

    tag = "stepA-control"

    def _raw_I(self, chi, tables, epochs):
        tret = lp.earth_retarded_lookback_gyr(chi, tables)
        g = _powerlaw_g(tret, tL=float(epochs.t_life_gyr))
        from scipy.integrate import cumulative_trapezoid
        return cumulative_trapezoid(g, chi, initial=0.0)


# ----------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------

def describe_fit(tag, res, chi2_baseline):
    x = np.asarray(res.x, dtype=float)
    out = {
        "tag": tag,
        "H0": float(x[0]),
        "Om": float(x[1]),
        "alpha_rd": float(x[2]),
        "amplitude": float(x[4]),
        "chi2": float(res.fun),
        "dchi2_vs_baseline": float(chi2_baseline - res.fun),
    }
    return out
