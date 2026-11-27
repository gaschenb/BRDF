"""Orthographic sphere renderer: point lights plus an optional environment integrated with BRDF importance sampling."""
import numpy as np
from scipy.stats import qmc

from brdftoolbox.core.frames import EPS, frame_from_tangent, normalize, sph_dir, to_local, to_world
from brdftoolbox.core import lights as LT
from brdftoolbox.core.models import eval_rgb, polarization_keys


def sphere_geometry(res):
    t = (np.arange(res) + 0.5) / res * 2.0 - 1.0
    X, Y = np.meshgrid(t, -t)
    R2 = X * X + Y * Y
    mask = R2 <= 1.0
    Z = np.sqrt(np.clip(1.0 - R2, 0.0, 1.0))
    N = np.stack([X, Y, Z], -1)[mask]
    hint = np.stack([N[:, 2], np.zeros(len(N)), -N[:, 0]], -1)
    T, B = frame_from_tangent(N, hint)
    u = (0.5 + np.arctan2(X, Z) / (2.0 * np.pi))[mask]
    v = (0.5 - np.arcsin(np.clip(Y, -1, 1)) / np.pi)[mask]
    return {"mask": mask, "N": N, "T": T, "B": B, "u": u, "v": v}


def checker(u, v, n=10):
    return (np.floor(u * n) + np.floor(v * n)) % 2


TEXTURES = {
    "checker": lambda u, v: np.where(checker(u, v)[:, None] > 0, [0.85, 0.85, 0.85], [0.18, 0.05, 0.05]),
    "uv": lambda u, v: np.stack([u, v, np.full_like(u, 0.35)], -1),
    "stripes": lambda u, v: np.where((np.floor(u * 24) % 2)[:, None] > 0, [0.9, 0.55, 0.1], [0.1, 0.25, 0.6]),
}


def studio_env(d):
    """Procedural studio: graded sky, ground, two rectangular soft boxes."""
    y = d[..., 1]
    sky = np.where(y[..., None] > 0, (0.1 + 0.3 * y[..., None]) * np.array([0.85, 0.9, 1.0]),
                   0.04 * np.array([0.9, 0.85, 0.8]))
    box1 = (np.abs(d[..., 0] + 0.55) < 0.25) & (np.abs(y - 0.55) < 0.18) & (d[..., 2] > 0)
    box2 = (np.abs(d[..., 0] - 0.7) < 0.12) & (np.abs(y - 0.1) < 0.4) & (d[..., 2] > -0.2)
    return sky + 4.0 * box1[..., None] + 2.0 * box2[..., None] * np.array([1.0, 0.9, 0.8])


ENVIRONMENTS = {"studio": studio_env, "uniform": lambda d: np.ones(d.shape[:-1] + (3,))}


def tonemap(rgb, exposure=1.0, mode="aces"):
    x = np.maximum(rgb * exposure, 0.0)
    if mode == "aces":
        x = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    elif mode == "reinhard":
        x = x / (1.0 + x)
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def _maps(maps, geo):
    out = {}
    for k, m in (maps or {}).items():
        v = m(geo["u"], geo["v"]) if callable(m) else np.asarray(m)[geo["mask"]]
        out[k] = v if np.ndim(v) == 2 else v[:, None]
    return out


def _sub(maps, s):
    return {k: v[s] for k, v in maps.items()}


def _as_light(l):
    return l if isinstance(l, LT.Light) else LT.Light(theta=l[0], phi=l[1], intensity=l[2])


def render_sphere(model, res=256, lights=(LT.Light(),), env=None, spp=32, exposure=0.8,
                  ambient=0.03, maps=None, tonemapper="aces", seed=0, chunk=4096, hdr=False, speckle=True):
    """lights: Light objects (or legacy (θ°, φ°, E) tuples). env: name or callable(d)->rgb. Returns (res, res, 3)."""
    geo = sphere_geometry(res)
    N, T, B = geo["N"], geo["T"], geo["B"]
    P = len(N)
    mp = _maps(maps, geo)
    view = np.array([0.0, 0.0, 1.0])
    wo = to_local(np.broadcast_to(view, N.shape), T, B, N)
    rgb = np.zeros((P, 3))
    env_fn = ENVIRONMENTS[env] if isinstance(env, str) else env
    rng = np.random.default_rng(seed)
    base = qmc.Sobol(3, scramble=True, seed=seed).random(max(spp, 1))
    lights = [_as_light(l) for l in lights]
    pkeys = polarization_keys(model)
    lt = []
    for i, L in enumerate(lights):
        sp = None
        if speckle and L.coherence != "incoherent":
            sp = LT.speckle_pattern(res, L.speckle_grain, L.coherence_stats()["M"], seed=L.seed + 17 * i)[geo["mask"]]
        ub = qmc.Sobol(2, scramble=True, seed=L.seed + i).random(max(L.samples, 1))
        lt.append((L, L.rgb(), sp, ub))
    for s in range(0, P, chunk):
        sl = slice(s, s + chunk)
        n = len(N[sl])
        wo_c = wo[sl]
        sub = _sub(mp, sl)
        m = model.replace(**sub) if mp else model
        me = model.replace(**{k: v[:, None] for k, v in sub.items()}) if mp else model
        for L, col, sp, ub in lt:
            u = np.mod(ub[None] + rng.random((n, 1, 2)), 1.0)
            dirs, w = LT.incident(L, N[sl], u)
            wi = to_local(dirs, T[sl][:, None], B[sl][:, None], N[sl][:, None])
            ml = me
            pol = LT.polarization_s_fraction(L, dirs, view) if pkeys else None
            if pol is not None:
                ml = me.replace(**{k: pol[..., None] for k in pkeys})
            f = eval_rgb(ml, wi, np.broadcast_to(wo_c[:, None, :], wi.shape))
            c = np.sum(f * (np.maximum(wi[..., 2], 0.0) * w)[..., None], 1) * col
            rgb[sl] += c * sp[sl][:, None] if sp is not None else c
        if ambient:
            n_up = np.broadcast_to([0.0, 0.0, 1.0], wo_c.shape)
            rgb[sl] += ambient * np.pi * eval_rgb(m, n_up, wo_c)
        if env_fn is not None and spp > 0:
            mono = {k: np.mean(v, -1, keepdims=True) for k, v in sub.items()}
            ms = model.replace(**mono) if mono else model
            u = np.mod(base[None] + rng.random((n, 1, 3)), 1.0)
            wob = np.broadcast_to(wo_c[:, None, :], (n, spp, 3))
            wi, pdf = ms.sample(wob, u)
            f = eval_rgb(me, wi, wob)
            d = to_world(wi, T[sl][:, None], B[sl][:, None], N[sl][:, None])
            ok = ((pdf > EPS) & (wi[..., 2] > 0))[..., None]
            w = np.where(ok, f * np.maximum(wi[..., 2:3], 0.0) / np.maximum(pdf, EPS)[..., None], 0.0)
            rgb[sl] += np.mean(w * env_fn(d), 1)
    img = np.zeros((res, res, 3))
    if env_fn is not None:
        t = (np.arange(res) + 0.5) / res * 2.0 - 1.0
        X, Y = np.meshgrid(t, -t)
        img[:] = 0.35 * env_fn(normalize(np.stack([X * 0.6, Y * 0.6, -np.ones_like(X)], -1)))
    img[geo["mask"]] = rgb
    return img if hdr else tonemap(img, exposure, tonemapper)


def irradiance_map(light, res=128):
    """Normal-weighted irradiance E(p) on the sphere (no speckle), NaN outside the disk."""
    from brdftoolbox.core.models import Lambert
    white = light.replace(spectrum="white", coherence="incoherent")
    img = render_sphere(Lambert(albedo=1.0), res=res, lights=[white], ambient=0.0, hdr=True, speckle=False)
    E = np.pi * img[..., 1]
    E[~sphere_geometry(res)["mask"]] = np.nan
    return E
