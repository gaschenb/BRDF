"""Physical-plausibility checks and estimator diagnostics: albedo, furnace, reciprocity, sampler χ², efficiency."""
import numpy as np
from scipy import stats
from scipy.stats import qmc

from brdftoolbox.core import microfacet as mf
from brdftoolbox.core.models import eval_c
from brdftoolbox.core.frames import EPS, sph_dir, uniform_hemisphere, cosine_hemisphere


def _f(model, wi, wo):
    return eval_c(model, wi, wo)


def _qmc(n, seed=0, d=3):
    m = int(np.ceil(np.log2(max(n, 2))))
    return qmc.Sobol(d, scramble=True, seed=seed).random_base2(m)[:n]


def hemisphere_quadrature(n_mu=96, n_phi=128):
    x, w = np.polynomial.legendre.leggauss(n_mu)
    mu, wmu = 0.5 * (x + 1.0), 0.5 * w
    phi = (np.arange(n_phi) + 0.5) * 2.0 * np.pi / n_phi
    MU, PH = np.meshgrid(mu, phi, indexing="ij")
    wi = sph_dir(np.arccos(MU), PH)
    wts = (wmu[:, None] * np.full(n_phi, 2.0 * np.pi / n_phi)[None, :])
    return wi, wts


def _albedo_quad(model, wo, n_mu, n_phi):
    wi, wts = hemisphere_quadrature(n_mu, n_phi)
    f = _f(model, wi, np.broadcast_to(wo, wi.shape))
    g = f * wi[..., 2].reshape(wi.shape[:2] + (1,) * (np.ndim(f) - 2)) * wts.reshape(wts.shape + (1,) * (np.ndim(f) - 2))
    return np.sum(g, axis=(0, 1)), np.zeros(np.shape(g)[2:])


def _albedo_mc(model, wo, n, seed):
    u = _qmc(n, seed)
    wob = np.broadcast_to(wo, u.shape)
    wi, pdf = model.sample(wob, u)
    f = _f(model, wi, wob)
    ok = (pdf > EPS) & (wi[..., 2] > 0)
    shp = ok.shape + (1,) * (np.ndim(f) - ok.ndim)
    w = np.where(ok.reshape(shp), f * (np.maximum(wi[..., 2], 0.0) / np.maximum(pdf, EPS)).reshape(shp), 0.0)
    return np.mean(w, 0), np.std(w, 0) / np.sqrt(n)


def directional_albedo(model, theta_o, phi_o=0.0, method="auto", n=4096, n_mu=96, n_phi=128, seed=0):
    """E(ωo) = ∫ f(ωi, ωo) cosθi dωi. Returns (value, stderr); vectorized over theta_o.
    'auto' uses BRDF importance sampling (robust for sharp lobes)."""
    th = np.atleast_1d(np.asarray(theta_o, float))
    vals, errs = [], []
    for t in th:
        wo = sph_dir(t, phi_o)
        if method == "quad":
            v, e = _albedo_quad(model, wo, n_mu, n_phi)
        else:
            v, e = _albedo_mc(model, wo, n, seed)
        vals.append(v)
        errs.append(e)
    vals, errs = np.array(vals), np.array(errs)
    if np.ndim(theta_o) == 0:
        return vals[0], errs[0]
    return vals, errs


def hemispherical_albedo(model, n_mu=24, **kw):
    x, w = np.polynomial.legendre.leggauss(n_mu)
    mu = 0.5 * (x + 1.0)
    E, _ = directional_albedo(model, np.arccos(mu), **kw)
    shp = (n_mu,) + (1,) * (E.ndim - 1)
    return np.sum(E * (mu * w).reshape(shp), 0)


def energy_check(model, n_theta=16, tol=0.005, **kw):
    th = np.linspace(0.0, np.deg2rad(89.0), n_theta)
    E, err = directional_albedo(model, th, **kw)
    m = float(np.max(E))
    return {"theta_deg": np.rad2deg(th), "albedo": E, "stderr": err, "max_albedo": m,
            "conserving": bool(m <= 1.0 + tol)}


def reciprocity_check(model, n=4096, seed=1):
    rng = np.random.default_rng(seed)
    wi = cosine_hemisphere(rng.random(n), rng.random(n))
    wo = cosine_hemisphere(rng.random(n), rng.random(n))
    a, b = _f(model, wi, wo), _f(model, wo, wi)
    rel = np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), EPS)
    return {"max_rel_err": float(np.max(rel)), "rms_rel_err": float(np.sqrt(np.mean(rel ** 2))),
            "min_value": float(np.min(a)), "reciprocal": bool(np.max(rel) < 1e-6),
            "non_negative": bool(np.min(a) >= 0.0)}


def ndf_checks(ax, ay, ndf="ggx", theta_o=np.deg2rad(60.0), n_mu=256, n_phi=256):
    """∫ D cosθh dω = 1 and the weak white furnace ∫ G1(ωo) <ωo,h>⁺ D dω = cosθo."""
    x, w = np.polynomial.legendre.leggauss(n_mu)
    t = 0.5 * (x + 1.0) * (np.pi / 2.0)
    wt = 0.5 * w * (np.pi / 2.0)
    phi = (np.arange(n_phi) + 0.5) * 2.0 * np.pi / n_phi
    T, P = np.meshgrid(t, phi, indexing="ij")
    h = sph_dir(T, P)
    dw = (wt[:, None] * np.sin(T)) * (2.0 * np.pi / n_phi)
    Dh = mf.D(h, ax, ay, ndf)
    norm = float(np.sum(Dh * h[..., 2] * dw))
    wo = sph_dir(theta_o, 0.3)
    proj = float(np.sum(mf.G1(wo, ax, ay, ndf) * np.maximum(np.sum(h * wo, -1), 0.0) * Dh * dw))
    return {"ndf_norm": norm, "projected_area": proj, "cos_theta_o": float(np.cos(theta_o))}


def pdf_mass(model, theta_o, n_mu=128, n_phi=256):
    """∫ pdf over the upper hemisphere (= 1 minus the sample mass that falls below the horizon)."""
    wi, wts = hemisphere_quadrature(n_mu, n_phi)
    p = model.pdf(wi, np.broadcast_to(sph_dir(theta_o), wi.shape))
    return float(np.sum(p * wts))


def sampler_chi2(model, theta_o, n=200_000, n_theta=16, n_phi=32, sub=6, seed=3):
    """Pearson χ² between sampled directions and ∫pdf over (cosθ, φ) bins (pbrt-style)."""
    wo = sph_dir(theta_o, 0.0)
    rng = np.random.default_rng(seed)
    u = rng.random((n, 3))
    wi, _ = model.sample(np.broadcast_to(wo, u.shape), u)
    up = wi[:, 2] > 0
    ci = np.clip(wi[up, 2], 0.0, 1.0)
    ph = np.mod(np.arctan2(wi[up, 1], wi[up, 0]), 2.0 * np.pi)
    obs, _, _ = np.histogram2d(ci, ph, bins=[n_theta, n_phi], range=[[0, 1], [0, 2 * np.pi]])
    gx, gw = np.polynomial.legendre.leggauss(sub)
    gx, gw = 0.5 * (gx + 1.0), 0.5 * gw
    cb = (np.arange(n_theta)[:, None] + gx[None, :]) / n_theta
    pb = (np.arange(n_phi)[:, None] + gx[None, :]) * 2.0 * np.pi / n_phi
    C, P = np.meshgrid(cb.ravel(), pb.ravel(), indexing="ij")
    d = sph_dir(np.arccos(C), P)
    pdf = model.pdf(d, np.broadcast_to(wo, d.shape)).reshape(n_theta, sub, n_phi, sub)
    exp = n * np.einsum("isjt,s,t->ij", pdf, gw, gw) / n_theta * (2.0 * np.pi / n_phi)
    e, o = exp.ravel(), obs.ravel()
    order = np.argsort(e)
    e, o = e[order], o[order]
    pe, po, chi, dof = 0.0, 0.0, 0.0, 0
    for ei, oi in zip(e, o):
        pe, po = pe + ei, po + oi
        if pe >= 5.0:
            chi += (po - pe) ** 2 / pe
            dof += 1
            pe = po = 0.0
    if pe > 0:
        chi += (po - pe) ** 2 / max(pe, EPS)
        dof += 1
    dof = max(dof - 1, 1)
    p = float(stats.chi2.sf(chi, dof))
    return {"chi2": float(chi), "dof": dof, "p_value": p, "pass": p > 0.01,
            "expected": exp, "observed": obs}


def sampling_efficiency(model, theta_o, n=4096, trials=32, seed=5):
    """Variance of the albedo estimator under uniform, cosine and BRDF importance sampling."""
    wo = sph_dir(theta_o, 0.0)
    out = {}
    rng = np.random.default_rng(seed)
    for name in ("uniform", "cosine", "brdf"):
        ests = []
        for _ in range(trials):
            u = rng.random((n, 3))
            wob = np.broadcast_to(wo, u.shape)
            if name == "uniform":
                wi, pdf = uniform_hemisphere(u[:, 0], u[:, 1]), np.full(n, 1.0 / (2.0 * np.pi))
            elif name == "cosine":
                wi = cosine_hemisphere(u[:, 0], u[:, 1])
                pdf = wi[:, 2] / np.pi
            else:
                wi, pdf = model.sample(wob, u)
            f = eval_c(model, wi, wob)
            f = np.mean(f, -1) if f.ndim > 1 else f
            w = np.where((pdf > EPS) & (wi[:, 2] > 0), f * np.maximum(wi[:, 2], 0) / np.maximum(pdf, EPS), 0.0)
            ests.append(w.mean())
        out[name] = {"mean": float(np.mean(ests)), "var": float(np.var(ests) * n)}
    ref = out["uniform"]["var"]
    for v in out.values():
        v["speedup_vs_uniform"] = ref / v["var"] if v["var"] > 1e-12 * max(ref, 1e-30) else float("inf")
    return out


def report(model, theta_o=np.deg2rad(45.0), quick=False):
    n = 2048 if quick else 8192
    en = energy_check(model, n=n)
    rc = reciprocity_check(model)
    r = {"model": model.describe(), "energy": en, "reciprocity": rc,
         "hemispherical_albedo": hemispherical_albedo(model, n=n // 2)}
    if not model.native_rgb:
        r["pdf_mass"] = pdf_mass(model, theta_o)
        r["chi2"] = sampler_chi2(model, theta_o, n=50_000 if quick else 200_000)
        r["efficiency"] = sampling_efficiency(model, theta_o, n=1024 if quick else 4096,
                                              trials=16 if quick else 32)
    return r


def _fmt(v):
    v = np.asarray(v)
    return np.array2string(v, precision=4) if v.ndim else f"{float(v):.4f}"


def format_report(r):
    en, rc = r["energy"], r["reciprocity"]
    lines = [r["model"], "",
             f"max directional albedo   {_fmt(en['max_albedo'])}   energy-conserving: {en['conserving']}",
             f"hemispherical albedo     {_fmt(r['hemispherical_albedo'])}",
             f"reciprocity max rel err  {rc['max_rel_err']:.2e}   reciprocal: {rc['reciprocal']}",
             f"non-negative             {rc['non_negative']}"]
    lines.append("albedo E(θo): " + "  ".join(
        f"{t:.0f}°:{_fmt(np.mean(e))}" for t, e in zip(en["theta_deg"][::3], en["albedo"][::3])))
    if "chi2" in r:
        c = r["chi2"]
        lines += [f"pdf mass (θo=45°)        {r['pdf_mass']:.4f}",
                  f"sampler χ²               {c['chi2']:.1f} / dof {c['dof']}  p = {c['p_value']:.3f}  "
                  f"pass: {c['pass']}",
                  "estimator variance × N   " + "  ".join(
                      f"{k}: {v['var']:.3g} ({v['speedup_vs_uniform']:.1f}×)" if np.isfinite(v['speedup_vs_uniform'])
                      else f"{k}: 0 (exact)" for k, v in r["efficiency"].items())]
    return "\n".join(lines)
