"""Fit analytic BRDFs to measured / tabulated data with bounded multi-start least squares."""
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from brdftoolbox.core import models as M
from brdftoolbox.core.measured import Measured, valid_samples

FIT_PRESETS = {
    "plastic": (lambda: M.plastic(), ["diffuse.albedo", "specular.f0"], ["specular.roughness"]),
    "plastic-aniso": (lambda: M.plastic(), ["diffuse.albedo", "specular.f0"],
                      ["specular.roughness", "specular.anisotropy"]),
    "plastic-beckmann": (lambda: M.plastic(ndf="beckmann"), ["diffuse.albedo", "specular.f0"],
                         ["specular.roughness"]),
    "ggx": (lambda: M.CookTorrance(), ["f0"], ["roughness"]),
    "ward-plastic": (lambda: M.ward_plastic(), ["diffuse.albedo", "specular.ks"], ["specular.alpha_x"],
                     {"specular.alpha_y": "specular.alpha_x"}),
    "ward-plastic-aniso": (lambda: M.ward_plastic(), ["diffuse.albedo", "specular.ks"],
                           ["specular.alpha_x", "specular.alpha_y"]),
    "phong-plastic": (lambda: M.phong_plastic(), ["diffuse.albedo", "specular.ks"], ["specular.n"]),
    "blinn-plastic": (lambda: M.Sum(diffuse=M.Lambert(), specular=M.BlinnPhong()),
                      ["diffuse.albedo", "specular.ks"], ["specular.n"]),
    "oren-nayar-ggx": (lambda: M.Sum(diffuse=M.OrenNayar(), specular=M.CookTorrance()),
                       ["diffuse.albedo", "specular.f0"], ["diffuse.sigma", "specular.roughness"]),
    "disney": (lambda: M.Disney(), ["base_color"],
               ["metallic", "specular", "roughness", "sheen", "clearcoat", "clearcoat_gloss"]),
}

LOG_PARAMS = {"n", "specular.n"}


@dataclass
class FitResult:
    model: M.BRDF
    params: dict
    cost: float
    rmse: float
    rel_rmse: float
    n_samples: int
    seconds: float
    starts: list = field(default_factory=list)

    def summary(self):
        ps = "\n".join(f"  {k:<22} {np.round(v, 4).tolist() if np.ndim(v) else round(float(v), 4)}"
                       for k, v in self.params.items())
        return (f"{self.model.describe()}\n{ps}\n  loss rmse {self.rmse:.4f}   relative rmse "
                f"{self.rel_rmse:.3f}   samples {self.n_samples}   {self.seconds:.1f}s   "
                f"starts {[round(c, 4) for c in self.starts]}")


def _data(data, n, seed):
    if isinstance(data, Measured):
        data = data.table
    if isinstance(data, np.ndarray):
        wi, wo, f = valid_samples(data)
    else:
        wi, wo, f = data
    f = f if f.ndim == 2 else f[:, None]
    if n and len(f) > n:
        idx = np.random.default_rng(seed).choice(len(f), n, replace=False)
        wi, wo, f = wi[idx], wo[idx], f[idx]
    return wi, wo, f


def residual_fn(loss, wi, wo):
    ci, co = wi[:, 2:3], wo[:, 2:3]
    if loss == "log":
        return lambda fm, f: np.log1p(fm * ci * co) - np.log1p(f * ci * co)
    if loss == "cosine":
        return lambda fm, f: (fm - f) * ci * co
    if loss == "rel":
        return lambda fm, f: (fm - f) / (f + 0.01)
    raise ValueError(loss)


def fit(model, data, shared=None, channel=(), tie=None, loss="log", n_samples=20000, n_starts=4,
        seed=0, max_nfev=300):
    """Fit `model` to data (Measured, MERL table, or (wi, wo, f)).
    shared: scalar parameters; channel: parameters fitted per colour channel; tie: {param: source param}.
    MERL tables are isotropic (φh = 0), so anisotropic parameters are only partly identifiable from them."""
    tie = dict(tie or {})
    t0 = time.time()
    wi, wo, f = _data(data, n_samples, seed)
    C = f.shape[1]
    specs = model.param_specs()
    shared = list(shared if shared is not None else [k for k in specs if k not in channel and k not in tie])
    channel = list(channel)
    names = shared + [f"{k}[{c}]" for k in channel for c in range(C)]
    base = [specs[k] for k in shared] + [specs[k] for k in channel for _ in range(C)]
    logs = np.array([k.split("[")[0] in LOG_PARAMS for k in names])
    lo = np.array([b[1] for b in base], float)
    hi = np.array([b[2] for b in base], float)
    lo_t = np.where(logs, np.log(np.where(logs, lo, 1.0)), lo)
    hi_t = np.where(logs, np.log(np.where(logs, hi, 1.0)), hi)
    cur = model.params()
    x0 = []
    for k in shared:
        x0.append(float(np.mean(cur[k])))
    for k in channel:
        x0.extend(np.broadcast_to(np.asarray(cur[k], float), (C,)).tolist() if np.ndim(cur[k]) else [float(cur[k])] * C)
    x0 = np.clip(np.where(logs, np.log(np.maximum(x0, 1e-12)), x0), lo_t, hi_t)
    res = residual_fn(loss, wi, wo)
    wi1, wo1 = wi[:, None, :], wo[:, None, :]

    def build(x):
        x = np.where(logs, np.exp(x), x)
        kw = {k: float(x[i]) for i, k in enumerate(shared)}
        j = len(shared)
        for k in channel:
            kw[k] = np.array(x[j:j + C])
            j += C
        kw.update({k: kw[v] for k, v in tie.items()})
        return model.replace(**kw), kw

    def r(x):
        m, _ = build(x)
        fm = np.broadcast_to(m.eval(wi1, wo1), f.shape)
        return res(fm, f).ravel()

    starts = [x0]
    if n_starts > 1:
        u = np.random.default_rng(seed).random((n_starts - 1, len(x0)))
        starts += list(lo_t + u * (hi_t - lo_t))
    best, costs = None, []
    pad = 1e-9 * (hi_t - lo_t)
    for s in starts:
        s = np.clip(s, lo_t + pad, hi_t - pad)
        sol = least_squares(r, s, bounds=(lo_t, hi_t), method="trf", x_scale="jac", max_nfev=max_nfev)
        costs.append(float(sol.cost))
        if best is None or sol.cost < best.cost:
            best = sol
    m, kw = build(best.x)
    fm = np.broadcast_to(m.eval(wi1, wo1), f.shape)
    rel = np.sqrt(np.mean(((fm - f) / (f + 0.01)) ** 2))
    return FitResult(m, kw, float(best.cost), float(np.sqrt(np.mean(best.fun ** 2))), float(rel),
                     len(f), time.time() - t0, costs)


def fit_preset(name, data, **kw):
    factory, channel, shared, *tie = FIT_PRESETS[name]
    return fit(factory(), data, shared=shared, channel=channel, tie=tie[0] if tie else None, **kw)


def albedo_comparison(measured, fitted, thetas=np.deg2rad([0, 20, 40, 60, 75])):
    from brdftoolbox.core.analytics import directional_albedo
    Em, _ = directional_albedo(measured, thetas, n=8192)
    Ef, _ = directional_albedo(fitted, thetas, n=8192)
    return {"theta_deg": np.rad2deg(thetas), "measured": Em, "fitted": np.broadcast_to(Ef, np.shape(Em))}
