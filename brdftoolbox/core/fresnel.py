"""Fresnel terms. eta = n_t / n_i; conductors take complex IOR eta + i k (per channel if arrays)."""
import numpy as np

from brdftoolbox.core.frames import EPS

# RGB (≈650/550/450 nm) complex IOR
METALS = {
    "gold": ((0.143, 0.374, 1.442), (3.983, 2.385, 1.603)),
    "silver": ((0.155, 0.117, 0.138), (4.828, 3.122, 2.147)),
    "copper": ((0.200, 0.924, 1.102), (3.912, 2.452, 2.142)),
    "aluminium": ((1.657, 0.880, 0.521), (9.224, 6.270, 4.837)),
}


def schlick_weight(c):
    m = 1.0 - np.clip(c, 0.0, 1.0)
    return (m * m) * (m * m) * m


def schlick(c, f0, f90=1.0):
    return f0 + (f90 - f0) * schlick_weight(c)


def dielectric_sp(c, eta):
    ci = np.clip(c, 0.0, 1.0)
    s2t = (1.0 - ci * ci) / (eta * eta)
    ct = np.sqrt(np.clip(1.0 - s2t, 0.0, 1.0))
    rs = (ci - eta * ct) / (ci + eta * ct + EPS)
    rp = (eta * ci - ct) / (eta * ci + ct + EPS)
    tir = s2t > 1.0
    return np.where(tir, 1.0, rs * rs), np.where(tir, 1.0, rp * rp)


def dielectric(c, eta):
    rs, rp = dielectric_sp(c, eta)
    return 0.5 * (rs + rp)


def conductor(c, eta, k):
    rs, rp = conductor_sp(c, eta, k)
    return 0.5 * (rs + rp)


def conductor_sp(c, eta, k):
    ci = np.clip(c, EPS, 1.0)
    c2 = ci * ci
    s2 = 1.0 - c2
    e2, k2 = eta * eta, k * k
    t0 = e2 - k2 - s2
    a2b2 = np.sqrt(np.maximum(t0 * t0 + 4.0 * e2 * k2, 0.0))
    t1 = a2b2 + c2
    a = np.sqrt(np.maximum(0.5 * (a2b2 + t0), 0.0))
    t2 = 2.0 * a * ci
    rs = (t1 - t2) / (t1 + t2 + EPS)
    t3 = c2 * a2b2 + s2 * s2
    t4 = t2 * s2
    rp = rs * (t3 - t4) / (t3 + t4 + EPS)
    return rs, rp


def f0_from_ior(eta):
    return ((eta - 1.0) / (eta + 1.0)) ** 2


def ior_from_f0(f0):
    s = np.sqrt(np.clip(f0, 0.0, 0.999))
    return (1.0 + s) / (1.0 - s)


def hemispherical_average(F, n=64, ndim=0):
    """2 ∫ F(μ) μ dμ; μ is passed with shape (n,) + (1,)*ndim so F can broadcast parameter arrays."""
    x, w = np.polynomial.legendre.leggauss(n)
    mu = 0.5 * (x + 1.0)
    shape = (n,) + (1,) * ndim
    vals = np.asarray(F(mu.reshape(shape)))
    return np.sum(vals * (mu * w).reshape((n,) + (1,) * (vals.ndim - 1)), axis=0)


def schlick_average(f0):
    return f0 + (1.0 - f0) / 21.0


def brewster_angle(eta):
    return np.arctan(eta)
