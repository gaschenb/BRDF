"""Kulla–Conty multiple-scattering compensation: tabulated E(μ, α) and E_avg(α) for F = 1."""
from functools import lru_cache

import numpy as np
from scipy.stats import qmc

from brdftoolbox.core import microfacet as mf
from brdftoolbox.core.frames import EPS

N_MU, N_ALPHA, N_SAMPLES = 32, 32, 4096


def _alpha_grid():
    return np.linspace(0.03, 1.0, N_ALPHA) ** 2


def _mu_grid():
    x, w = np.polynomial.legendre.leggauss(N_MU)
    return 0.5 * (x + 1.0), 0.5 * w


def single_scatter_albedo(mu, alpha, ndf="ggx", g2="height-correlated", n=N_SAMPLES, seed=0):
    """E(μ, α) for an isotropic microfacet BRDF with F = 1. mu, alpha broadcast; returns their shape."""
    mu, alpha = np.broadcast_arrays(np.asarray(mu, float), np.asarray(alpha, float))
    u = qmc.Sobol(2, scramble=True, seed=seed).random(n)
    st = np.sqrt(np.maximum(1.0 - mu * mu, 0.0))
    wo = np.stack([st, np.zeros_like(st), mu], -1)[..., None, :]
    a = alpha[..., None]
    if ndf == "ggx":
        h = mf.sample_vndf_ggx(wo, u[:, 0], u[:, 1], a, a)
    else:
        h = mf.sample_ndf(u[:, 0], u[:, 1], a, a, ndf)
    wi = mf.reflect_about(wo, h)
    g = mf.G2(wi, wo, a, a, ndf, g2)
    if ndf == "ggx":
        w = g / mf.G1(wo, a, a, ndf)
    else:
        w = g * np.sum(wo * h, -1) / (np.maximum(wo[..., 2], EPS) * np.maximum(h[..., 2], EPS))
    return np.mean(np.where(wi[..., 2] > 0, w, 0.0), -1)


@lru_cache(maxsize=None)
def table(ndf="ggx", g2="height-correlated"):
    mu, wmu = _mu_grid()
    al = _alpha_grid()
    E = np.clip(single_scatter_albedo(mu[:, None], al[None, :], ndf, g2), 0.0, 1.0)
    Eavg = 2.0 * np.sum(E * (mu * wmu)[:, None], 0)
    return mu, al, E, Eavg


def _interp_alpha(al, alpha):
    a = np.clip(alpha, al[0], al[-1])
    j = np.clip(np.searchsorted(al, a) - 1, 0, len(al) - 2)
    t = (a - al[j]) / (al[j + 1] - al[j])
    return j, t


def lookup_E(mu, alpha, ndf="ggx", g2="height-correlated"):
    mus, al, E, _ = table(ndf, g2)
    mu, alpha = np.broadcast_arrays(np.asarray(mu, float), np.asarray(alpha, float))
    m = np.clip(mu, mus[0], mus[-1])
    i = np.clip(np.searchsorted(mus, m) - 1, 0, len(mus) - 2)
    s = (m - mus[i]) / (mus[i + 1] - mus[i])
    j, t = _interp_alpha(al, alpha)
    return ((1 - s) * (1 - t) * E[i, j] + s * (1 - t) * E[i + 1, j]
            + (1 - s) * t * E[i, j + 1] + s * t * E[i + 1, j + 1])


def lookup_Eavg(alpha, ndf="ggx", g2="height-correlated"):
    _, al, _, Eavg = table(ndf, g2)
    j, t = _interp_alpha(al, np.asarray(alpha, float))
    return (1 - t) * Eavg[j] + t * Eavg[j + 1]


def f_ms(ci, co, alpha, favg, ndf="ggx", g2="height-correlated"):
    Ei, Eo = lookup_E(ci, alpha, ndf, g2), lookup_E(co, alpha, ndf, g2)
    Eavg = lookup_Eavg(alpha, ndf, g2)
    fms = (1.0 - Ei) * (1.0 - Eo) / (np.pi * np.maximum(1.0 - Eavg, EPS))
    return fms * favg * favg * Eavg / (1.0 - favg * (1.0 - Eavg))
