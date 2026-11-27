"""Analytic BRDF models. Directions are (..., 3) unit vectors in the local frame (n = +z, t = +x).

Parameters broadcast elementwise against the cosines, so per-channel (RGB) or spatially varying
values work by giving directions a trailing singleton axis, e.g. wi[..., None, :].
"""
import copy

import numpy as np

from brdftoolbox.core import fresnel as fr
from brdftoolbox.core import microfacet as mf
from brdftoolbox.core import multiscatter as ms
from brdftoolbox.core.frames import (EPS, cosine_hemisphere, lobe_sample, normalize, reflect)


def _lum(c):
    c = np.asarray(c, float)
    if c.ndim and c.shape[-1] == 3:
        return np.sum(c * np.array([0.3, 0.6, 0.1]), -1, keepdims=True)
    return c


def _mix(a, b, t):
    return a + (b - a) * t


def _dot(a, b):
    return np.sum(a * b, -1)


class BRDF:
    key = ""
    label = ""
    PARAMS = {}
    OPTIONS = {}
    native_rgb = False

    def __init__(self, **kw):
        self.p = {k: v[0] for k, v in self.PARAMS.items()}
        self.o = {k: v[0] for k, v in self.OPTIONS.items()}
        self._set(kw)

    def _set(self, kw):
        for k, v in kw.items():
            if k in self.PARAMS:
                self.p[k] = v
            elif k in self.OPTIONS:
                if v not in self.OPTIONS[k][1]:
                    raise ValueError(f"{k}={v!r} not in {self.OPTIONS[k][1]}")
                self.o[k] = v
            else:
                raise TypeError(f"{type(self).__name__} has no parameter {k!r}")
        self._cache = {}

    def replace(self, **kw):
        new = copy.copy(self)
        new.p, new.o = dict(self.p), dict(self.o)
        new._set(kw)
        return new

    def active_params(self):
        return list(self.PARAMS)

    def param_specs(self):
        return {k: self.PARAMS[k] for k in self.active_params()}

    def params(self):
        return {k: self.p[k] for k in self.active_params()}

    def eval(self, wi, wo):
        ok = (wi[..., 2] > 0) & (wo[..., 2] > 0)
        return np.where(ok, self._eval(wi, wo), 0.0)

    __call__ = eval

    def sample(self, wo, u):
        """u: (..., 3) uniforms. Returns wi, pdf (solid angle)."""
        wi = cosine_hemisphere(u[..., 0], u[..., 1])
        return wi, self.pdf(wi, wo)

    def pdf(self, wi, wo):
        return np.where(wi[..., 2] > 0, np.maximum(wi[..., 2], 0.0) / np.pi, 0.0)

    def channels(self):
        return max([np.shape(v)[-1] for v in self.p.values() if np.ndim(v)] or [0])

    def describe(self):
        ps = ", ".join(f"{k}={np.round(v, 4).tolist() if np.ndim(v) else round(float(v), 4)}"
                       for k, v in self.params().items())
        os_ = ", ".join(f"{k}={v}" for k, v in self.o.items())
        return f"{self.label}({ps}{'; ' + os_ if os_ else ''})"

    __repr__ = describe


class Lambert(BRDF):
    key, label = "lambert", "Lambert"
    PARAMS = {"albedo": (0.8, 0.0, 1.0)}

    def _eval(self, wi, wo):
        return self.p["albedo"] / np.pi + 0.0 * wi[..., 2]


class OrenNayar(BRDF):
    key, label = "oren-nayar", "Oren-Nayar"
    PARAMS = {"albedo": (0.8, 0.0, 1.0), "sigma": (0.35, 0.0, 1.2)}

    def _eval(self, wi, wo):
        s2 = self.p["sigma"] ** 2
        A = 1.0 - 0.5 * s2 / (s2 + 0.33)
        B = 0.45 * s2 / (s2 + 0.09)
        cphi = wi[..., 0] * wo[..., 0] + wi[..., 1] * wo[..., 1]
        term = np.maximum(cphi, 0.0) / np.maximum(np.maximum(wi[..., 2], wo[..., 2]), EPS)
        return self.p["albedo"] / np.pi * (A + B * term)


class Phong(BRDF):
    """Energy-normalized modified Phong (Lafortune & Willems)."""
    key, label = "phong", "Phong"
    PARAMS = {"ks": (0.8, 0.0, 1.0), "n": (40.0, 1.0, 2000.0)}

    def _eval(self, wi, wo):
        n = self.p["n"]
        c = np.maximum(_dot(wi, reflect(wo)), 0.0)
        return self.p["ks"] * (n + 2.0) / (2.0 * np.pi) * c ** n

    def sample(self, wo, u):
        r = np.broadcast_to(reflect(wo), np.broadcast_shapes(wo.shape, u.shape))
        wi, pdf = lobe_sample(r, u[..., 0], u[..., 1], self.p["n"])
        return wi, pdf

    def pdf(self, wi, wo):
        n = self.p["n"]
        return (n + 1.0) / (2.0 * np.pi) * np.maximum(_dot(wi, reflect(wo)), 0.0) ** n


class BlinnPhong(BRDF):
    key, label = "blinn-phong", "Blinn-Phong"
    PARAMS = {"ks": (0.8, 0.0, 1.0), "n": (80.0, 1.0, 4000.0)}

    def _eval(self, wi, wo):
        n = self.p["n"]
        h = normalize(wi + wo)
        return self.p["ks"] * (n + 8.0) / (8.0 * np.pi) * np.maximum(h[..., 2], 0.0) ** n

    def sample(self, wo, u):
        z = np.zeros(np.broadcast_shapes(wo.shape, u.shape))
        z[..., 2] = 1.0
        h, _ = lobe_sample(z, u[..., 0], u[..., 1], self.p["n"])
        wi = mf.reflect_about(wo, h)
        return wi, self.pdf(wi, wo)

    def pdf(self, wi, wo):
        n = self.p["n"]
        h = normalize(wi + wo)
        return (n + 1.0) / (2.0 * np.pi) * np.maximum(h[..., 2], 0.0) ** n * mf.jacobian_reflect(wo, h)


class Ward(BRDF):
    """Anisotropic Ward with the Geisler-Moroder & Dür normalization (reciprocal, energy-bounded)."""
    key, label = "ward", "Ward"
    PARAMS = {"ks": (0.6, 0.0, 1.0), "alpha_x": (0.15, 0.01, 1.0), "alpha_y": (0.15, 0.01, 1.0)}

    def _eval(self, wi, wo):
        ax, ay = self.p["alpha_x"], self.p["alpha_y"]
        h = wi + wo
        hz2 = np.maximum(h[..., 2] ** 2, EPS)
        e = np.exp(-(h[..., 0] ** 2 / ax ** 2 + h[..., 1] ** 2 / ay ** 2) / hz2)
        return self.p["ks"] / (np.pi * ax * ay) * _dot(h, h) / (hz2 * hz2) * e

    def sample(self, wo, u):
        h = mf.sample_ndf(u[..., 0], u[..., 1], self.p["alpha_x"], self.p["alpha_y"], "beckmann")
        wi = mf.reflect_about(wo, h)
        return wi, self.pdf(wi, wo)

    def pdf(self, wi, wo):
        h = normalize(wi + wo)
        return mf.pdf_ndf(h, self.p["alpha_x"], self.p["alpha_y"], "beckmann") * mf.jacobian_reflect(wo, h)


class CookTorrance(BRDF):
    """Microfacet specular f = D F G2 / (4 |n·wi| |n·wo|), anisotropic, optional Kulla–Conty term."""
    key, label = "cook-torrance", "Cook-Torrance"
    PARAMS = {"roughness": (0.3, 0.01, 1.0), "anisotropy": (0.0, -0.95, 0.95),
              "f0": (0.04, 0.0, 1.0), "eta": (1.5, 1.0, 3.0), "k": (0.0, 0.0, 10.0),
              "pol_s": (0.5, 0.0, 1.0)}
    OPTIONS = {"ndf": ("ggx", mf.NDFS), "g2": ("height-correlated", mf.G2_MODES),
               "fresnel": ("schlick", ("schlick", "dielectric", "conductor", "none")),
               "remap": (True, (True, False)), "multiscatter": (False, (False, True)),
               "sampling": ("vndf", ("vndf", "ndf"))}

    def active_params(self):
        extra = {"schlick": ["f0"], "dielectric": ["eta"], "conductor": ["eta", "k"], "none": []}
        return ["roughness", "anisotropy"] + extra[self.o["fresnel"]]

    def alphas(self):
        return mf.roughness_to_alpha(self.p["roughness"], self.p["anisotropy"], self.o["remap"])

    def F(self, c):
        f = self.o["fresnel"]
        if f == "schlick":
            return fr.schlick(c, self.p["f0"])
        if f in ("dielectric", "conductor"):
            rs, rp = (fr.dielectric_sp(c, self.p["eta"]) if f == "dielectric"
                      else fr.conductor_sp(c, self.p["eta"], self.p["k"]))
            s = self.p["pol_s"]
            return s * rs + (1.0 - s) * rp
        return np.ones_like(c)

    def favg(self):
        f = self.o["fresnel"]
        if f == "schlick":
            return fr.schlick_average(self.p["f0"])
        if f == "none":
            return 1.0
        nd = max(np.ndim(self.p.get("eta", 0)), np.ndim(self.p.get("k", 0)), np.ndim(self.p["pol_s"]))
        return fr.hemispherical_average(self.F, ndim=nd)

    def _eval(self, wi, wo):
        ax, ay = self.alphas()
        ci, co = np.maximum(wi[..., 2], EPS), np.maximum(wo[..., 2], EPS)
        h = normalize(wi + wo)
        f = (mf.D(h, ax, ay, self.o["ndf"]) * mf.G2(wi, wo, ax, ay, self.o["ndf"], self.o["g2"])
             * self.F(_dot(wi, h)) / (4.0 * ci * co))
        if self.o["multiscatter"]:
            f = f + ms.f_ms(ci, co, np.sqrt(ax * ay), self.favg(), self.o["ndf"], self.o["g2"])
        return f

    def _use_vndf(self):
        return self.o["sampling"] == "vndf" and self.o["ndf"] == "ggx"

    def _p_ms(self, wo):
        if not self.o["multiscatter"]:
            return 0.0
        ax, ay = self.alphas()
        return 1.0 - ms.lookup_E(np.maximum(wo[..., 2], EPS), np.sqrt(ax * ay), self.o["ndf"], self.o["g2"])

    def sample(self, wo, u):
        ax, ay = self.alphas()
        if self._use_vndf():
            h = mf.sample_vndf_ggx(wo, u[..., 0], u[..., 1], ax, ay)
        else:
            h = mf.sample_ndf(u[..., 0], u[..., 1], ax, ay, self.o["ndf"])
        wi = mf.reflect_about(wo, h)
        if self.o["multiscatter"]:
            wi = np.where((u[..., 2] < self._p_ms(wo))[..., None], cosine_hemisphere(u[..., 0], u[..., 1]), wi)
        return wi, self.pdf(wi, wo)

    def _pdf_h(self, h, wo, ax, ay):
        if self._use_vndf():
            return mf.pdf_vndf(h, wo, ax, ay, "ggx")
        return mf.pdf_ndf(h, ax, ay, self.o["ndf"])

    def pdf(self, wi, wo):
        ax, ay = self.alphas()
        h = normalize(wi + wo)
        p = self._pdf_h(h, wo, ax, ay) * mf.jacobian_reflect(wo, h)
        if self.o["multiscatter"]:
            q = self._p_ms(wo)
            p = (1.0 - q) * p + q * np.maximum(wi[..., 2], 0.0) / np.pi
        return p


class Disney(BRDF):
    """Burley 2012 principled BRDF (reflection only), with height-correlated Smith G for the base layer."""
    key, label = "disney", "Disney principled"
    PARAMS = {"base_color": (0.8, 0.0, 1.0), "metallic": (0.0, 0.0, 1.0), "subsurface": (0.0, 0.0, 1.0),
              "specular": (0.5, 0.0, 1.0), "specular_tint": (0.0, 0.0, 1.0), "roughness": (0.5, 0.01, 1.0),
              "anisotropic": (0.0, 0.0, 0.95), "sheen": (0.0, 0.0, 1.0), "sheen_tint": (0.5, 0.0, 1.0),
              "clearcoat": (0.0, 0.0, 1.0), "clearcoat_gloss": (1.0, 0.0, 1.0)}

    def _alphas(self):
        return mf.roughness_to_alpha(self.p["roughness"], self.p["anisotropic"])

    def _eval(self, wi, wo):
        p = self.p
        ci, co = np.maximum(wi[..., 2], EPS), np.maximum(wo[..., 2], EPS)
        h = normalize(wi + wo)
        cd = _dot(wi, h)
        base = np.asarray(p["base_color"], float)
        lum = _lum(base)
        tint = np.where(lum > 0, base / np.maximum(lum, EPS), 1.0)
        fl, fv, fh = fr.schlick_weight(ci), fr.schlick_weight(co), fr.schlick_weight(cd)
        r = p["roughness"]
        fd90 = 0.5 + 2.0 * cd * cd * r
        fd = (1.0 + (fd90 - 1.0) * fl) * (1.0 + (fd90 - 1.0) * fv)
        fss90 = cd * cd * r
        fss = (1.0 + (fss90 - 1.0) * fl) * (1.0 + (fss90 - 1.0) * fv)
        ss = 1.25 * (fss * (1.0 / (ci + co) - 0.5) + 0.5)
        sheen = fh * p["sheen"] * _mix(1.0, tint, p["sheen_tint"])
        diffuse = (base / np.pi * _mix(fd, ss, p["subsurface"]) + sheen) * (1.0 - p["metallic"])
        c0 = _mix(p["specular"] * 0.08 * _mix(1.0, tint, p["specular_tint"]), base, p["metallic"])
        ax, ay = self._alphas()
        spec = (mf.D(h, ax, ay, "ggx") * mf.G2(wi, wo, ax, ay, "ggx") * _mix(c0, 1.0, fh)
                / (4.0 * ci * co))
        a_cc = _mix(0.1, 0.001, p["clearcoat_gloss"])
        g_cc = mf.G2(wi, wo, 0.25, 0.25, "ggx", "separable")
        cc = (0.25 * p["clearcoat"] * mf.gtr1(h[..., 2], a_cc) * _mix(0.04, 1.0, fh) * g_cc
              / (4.0 * ci * co))
        return diffuse + spec + cc

    def _weights(self):
        p = self.p
        base = np.asarray(p["base_color"], float)
        lum = _lum(base)
        tint = np.where(lum > 0, base / np.maximum(lum, EPS), 1.0)
        m = float(np.mean(p["metallic"]))
        c0 = _mix(p["specular"] * 0.08 * _mix(1.0, tint, p["specular_tint"]), base, p["metallic"])
        wd = max((1.0 - m) * float(np.mean(lum)), 1e-3)
        ws = max(float(np.mean(_lum(fr.schlick_average(c0)))), 0.02)
        wc = 0.25 * float(np.mean(p["clearcoat"])) * fr.schlick_average(0.04)
        s = wd + ws + wc
        return wd / s, ws / s, wc / s

    def sample(self, wo, u):
        wd, ws, wc = self._weights()
        ax, ay = self._alphas()
        u3 = u[..., 2]
        wi_d = cosine_hemisphere(u[..., 0], u[..., 1])
        wi_s = mf.reflect_about(wo, mf.sample_vndf_ggx(wo, u[..., 0], u[..., 1], ax, ay))
        a_cc = _mix(0.1, 0.001, self.p["clearcoat_gloss"])
        wi_c = mf.reflect_about(wo, mf.sample_gtr1(u[..., 0], u[..., 1], a_cc))
        sel = u3[..., None]
        wi = np.where(sel < wd, wi_d, np.where(sel < wd + ws, wi_s, wi_c))
        return wi, self.pdf(wi, wo)

    def pdf(self, wi, wo):
        wd, ws, wc = self._weights()
        ax, ay = self._alphas()
        h = normalize(wi + wo)
        j = mf.jacobian_reflect(wo, h)
        a_cc = _mix(0.1, 0.001, self.p["clearcoat_gloss"])
        p = (wd * np.maximum(wi[..., 2], 0.0) / np.pi
             + ws * mf.pdf_vndf(h, wo, ax, ay, "ggx") * j
             + wc * mf.gtr1(h[..., 2], a_cc) * np.maximum(h[..., 2], 0.0) * j)
        return np.where(wi[..., 2] > 0, p, 0.0)


class Sum(BRDF):
    """Sum of named component BRDFs; parameters are addressed as 'name.param'."""
    key, label = "sum", "Sum"

    def __init__(self, **components):
        self.c = dict(components)
        self.p, self.o, self._cache = {}, {}, {}

    @property
    def native_rgb(self):
        return any(c.native_rgb for c in self.c.values())

    def replace(self, **kw):
        groups = {n: {} for n in self.c}
        for k, v in kw.items():
            n, _, q = k.partition(".")
            if n not in groups:
                raise TypeError(f"no component {n!r}")
            groups[n][q] = v
        return Sum(**{n: (c.replace(**groups[n]) if groups[n] else c) for n, c in self.c.items()})

    def active_params(self):
        return [f"{n}.{k}" for n, c in self.c.items() for k in c.active_params()]

    def param_specs(self):
        return {f"{n}.{k}": v for n, c in self.c.items() for k, v in c.param_specs().items()}

    def params(self):
        return {f"{n}.{k}": v for n, c in self.c.items() for k, v in c.params().items()}

    def eval(self, wi, wo):
        return sum(c.eval(wi, wo) for c in self.c.values())

    def _weights(self):
        if "w" not in self._cache:
            from brdftoolbox.core.analytics import directional_albedo
            w = np.array([max(float(np.mean(directional_albedo(_scalar(c), 0.0, method="mc", n=512)[0])), 1e-3)
                          for c in self.c.values()])
            self._cache["w"] = w / w.sum()
        return self._cache["w"]

    def sample(self, wo, u):
        w = self._weights()
        cw = np.concatenate([[0.0], np.cumsum(w)])
        out = None
        for i, c in enumerate(self.c.values()):
            v = np.clip((u[..., 2] - cw[i]) / max(w[i], EPS), 0.0, 1.0 - 1e-12)
            wi_k, _ = c.sample(wo, np.stack([u[..., 0], u[..., 1], v], -1))
            sel = ((u[..., 2] >= cw[i]) & (u[..., 2] < cw[i + 1]))[..., None]
            out = wi_k if out is None else np.where(sel, wi_k, out)
        return out, self.pdf(out, wo)

    def pdf(self, wi, wo):
        return sum(wk * c.pdf(wi, wo) for wk, c in zip(self._weights(), self.c.values()))

    def channels(self):
        return max(c.channels() for c in self.c.values())

    def describe(self):
        return " + ".join(f"{n}:{c.describe()}" for n, c in self.c.items())

    __repr__ = describe


def polarization_keys(model):
    if isinstance(model, CookTorrance) and model.o["fresnel"] in ("dielectric", "conductor"):
        return ["pol_s"]
    if isinstance(model, Sum):
        return [f"{n}.{k}" for n, c in model.c.items() for k in polarization_keys(c)]
    return []


def _scalar(model):
    """Copy with every array parameter replaced by its mean (for sampling heuristics)."""
    if model.native_rgb:
        return model
    return model.replace(**{k: float(np.mean(v)) for k, v in model.params().items() if np.ndim(v)})


def eval_c(model, wi, wo):
    """Evaluate with a trailing channel axis when the model has per-channel parameters."""
    if model.native_rgb or not model.channels():
        return model.eval(wi, wo)
    return model.eval(wi[..., None, :], wo[..., None, :])


def eval_rgb(model, wi, wo):
    f = eval_c(model, wi, wo)
    if f.shape[-1:] != (3,) or f.shape == wi.shape[:-1]:
        f = f[..., None] if f.shape == wi.shape[:-1] else f
    return np.broadcast_to(f, wi.shape[:-1] + (3,))


MODELS = {m.key: m for m in (Lambert, OrenNayar, Phong, BlinnPhong, Ward, CookTorrance, Disney)}


def plastic(albedo=0.5, roughness=0.3, f0=0.04, **opts):
    return Sum(diffuse=Lambert(albedo=albedo), specular=CookTorrance(roughness=roughness, f0=f0, **opts))


def phong_plastic(albedo=0.5, ks=0.3, n=60.0):
    return Sum(diffuse=Lambert(albedo=albedo), specular=Phong(ks=ks, n=n))


def ward_plastic(albedo=0.5, ks=0.2, alpha_x=0.15, alpha_y=0.15):
    return Sum(diffuse=Lambert(albedo=albedo), specular=Ward(ks=ks, alpha_x=alpha_x, alpha_y=alpha_y))


def metal(name="gold", roughness=0.3, **opts):
    eta, k = fr.METALS[name]
    return CookTorrance(roughness=roughness, fresnel="conductor", eta=np.array(eta), k=np.array(k), **opts)


PRESETS = {
    "lambert": Lambert, "oren-nayar": OrenNayar, "phong": Phong, "blinn-phong": BlinnPhong,
    "ward": Ward, "cook-torrance": CookTorrance, "disney": Disney,
    "plastic": plastic, "phong-plastic": phong_plastic, "ward-plastic": ward_plastic, "metal": metal,
}


def make(model, /, **kw):
    """Build a preset by name. `model` is positional-only, so presets that take their own
    `name` keyword (metal(name="copper")) can be passed it through **kw."""
    if model not in PRESETS:
        raise KeyError(f"unknown model {model!r}; choose from {sorted(PRESETS)}")
    return PRESETS[model](**kw)
