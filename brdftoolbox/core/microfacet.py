"""Anisotropic microfacet distributions, Smith masking-shadowing, and half-vector samplers."""
import numpy as np
from scipy.special import erf

from brdftoolbox.core.frames import EPS, normalize

NDFS = ("ggx", "beckmann", "blinn")
G2_MODES = ("height-correlated", "separable")


def _slope2(h, ax, ay):
    z2 = np.maximum(h[..., 2] ** 2, EPS)
    return (h[..., 0] ** 2 / (ax * ax) + h[..., 1] ** 2 / (ay * ay)) / z2, z2


def blinn_exponent(alpha):
    return np.maximum(2.0 / (alpha * alpha) - 2.0, 0.0)


def D(h, ax, ay, ndf="ggx"):
    hz = h[..., 2]
    if ndf == "ggx":
        q = h[..., 0] ** 2 / (ax * ax) + h[..., 1] ** 2 / (ay * ay) + hz * hz
        q = np.maximum(q, EPS)
        val = 1.0 / (np.pi * ax * ay * q * q)
    elif ndf == "beckmann":
        t2, z2 = _slope2(h, ax, ay)
        val = np.exp(-t2) / (np.pi * ax * ay * z2 * z2)
    elif ndf == "blinn":
        n = blinn_exponent(np.sqrt(ax * ay))
        val = (n + 2.0) / (2.0 * np.pi) * np.clip(hz, 0.0, 1.0) ** n
    else:
        raise ValueError(ndf)
    return np.where(hz > 0, val, 0.0)


def Lambda(w, ax, ay, ndf="ggx"):
    z2 = np.maximum(w[..., 2] ** 2, EPS)
    a2t2 = (ax * ax * w[..., 0] ** 2 + ay * ay * w[..., 1] ** 2) / z2
    if ndf == "ggx":
        return 0.5 * (np.sqrt(1.0 + a2t2) - 1.0)
    a = 1.0 / np.sqrt(np.maximum(a2t2, EPS))
    return np.where(a > 8.0, 0.0, 0.5 * (erf(a) - 1.0) + np.exp(-a * a) / (2.0 * a * np.sqrt(np.pi)))


def G1(w, ax, ay, ndf="ggx"):
    return 1.0 / (1.0 + Lambda(w, ax, ay, ndf))


def G2(wi, wo, ax, ay, ndf="ggx", mode="height-correlated"):
    li, lo = Lambda(wi, ax, ay, ndf), Lambda(wo, ax, ay, ndf)
    if mode == "height-correlated":
        return 1.0 / (1.0 + li + lo)
    return 1.0 / ((1.0 + li) * (1.0 + lo))


def sample_vndf_ggx(wo, u1, u2, ax, ay):
    """Visible-normal sampling via spherical caps (Dupuy & Benyoub 2023). Returns h ~ D_wo(h)."""
    s = np.stack(np.broadcast_arrays(ax, ay, 1.0), -1)
    v = normalize(wo * s)
    phi = 2.0 * np.pi * u1
    z = (1.0 - u2) * (1.0 + v[..., 2]) - v[..., 2]
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    c = np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)
    return normalize((c + v) * s)


def pdf_vndf(h, wo, ax, ay, ndf="ggx"):
    """Density of h under VNDF sampling, w.r.t. solid angle of h."""
    return G1(wo, ax, ay, ndf) * np.maximum(np.sum(wo * h, -1), 0.0) * D(h, ax, ay, ndf) / np.maximum(wo[..., 2], EPS)


def sample_ndf(u1, u2, ax, ay, ndf="ggx"):
    """Sample h ∝ D(h) cosθh. Exact for anisotropic GGX/Beckmann (elliptical azimuth warp)."""
    if ndf == "blinn":
        n = blinn_exponent(np.sqrt(ax * ay))
        ct = u1 ** (1.0 / (n + 2.0))
        phi = 2.0 * np.pi * u2
    else:
        a = 2.0 * np.pi * u2
        phi = np.arctan2(ay * np.sin(a), ax * np.cos(a))
        inv_a2 = np.cos(phi) ** 2 / (ax * ax) + np.sin(phi) ** 2 / (ay * ay)
        u = np.clip(u1, 0.0, 1.0 - 1e-12)
        t2 = u / (1.0 - u) / inv_a2 if ndf == "ggx" else -np.log1p(-u) / inv_a2
        ct = 1.0 / np.sqrt(1.0 + t2)
    st = np.sqrt(np.maximum(1.0 - ct * ct, 0.0))
    return np.stack([st * np.cos(phi), st * np.sin(phi), ct], -1)


def pdf_ndf(h, ax, ay, ndf="ggx"):
    return D(h, ax, ay, ndf) * np.maximum(h[..., 2], 0.0)


def gtr1(c, a):
    a2 = a * a
    return (a2 - 1.0) / (np.pi * np.log(a2) * (1.0 + (a2 - 1.0) * c * c))


def sample_gtr1(u1, u2, a):
    a2 = a * a
    c2 = (1.0 - a2 ** (1.0 - u1)) / (1.0 - a2)
    ct = np.sqrt(np.clip(c2, 0.0, 1.0))
    st = np.sqrt(np.maximum(1.0 - c2, 0.0))
    phi = 2.0 * np.pi * u2
    return np.stack([st * np.cos(phi), st * np.sin(phi), ct], -1)


def reflect_about(wo, h):
    return 2.0 * np.sum(wo * h, -1, keepdims=True) * h - wo


def jacobian_reflect(wo, h):
    """dω_h/dω_i for reflection."""
    return 1.0 / np.maximum(4.0 * np.abs(np.sum(wo * h, -1)), EPS)


def roughness_to_alpha(roughness, anisotropy=0.0, remap=True):
    a = roughness * roughness if remap else roughness
    s = np.sqrt(1.0 - 0.9 * np.abs(anisotropy))
    ax, ay = a / s, a * s
    ax, ay = np.where(anisotropy >= 0, ax, ay), np.where(anisotropy >= 0, ay, ax)
    return np.maximum(ax, 1e-3), np.maximum(ay, 1e-3)
