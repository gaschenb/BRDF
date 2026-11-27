"""Light sources for the sphere renderer.

kinds
  directional  distant source; angular_radius > 0 gives a finite disk (e.g. the sun, 0.27°)
  point        isotropic emitter at `distance` (sphere radii, sphere at origin): per-point incidence and
               inverse-square falloff; radius > 0 makes it a spherical area emitter (near-field penumbra)
  beam         Gaussian beam from `distance`, 1/e² waist w0 and half-angle divergence β:
               w(z)² = w0² + (z tan β)², rays diverge from a virtual apex w0/tan β behind the source
`intensity` is the normal irradiance delivered at the sphere centre (on-axis for a beam), so distance and
spread reshape the illumination without changing its overall level.

Spectrum: white, blackbody (temperature) or Gaussian (wavelength, bandwidth = FWHM). Its centroid and
width give the coherence length l_c = λ²/Δλ.

Coherence (speckle in the image of a rough surface, Goodman's statistics):
  incoherent  no speckle
  coherent    ideal monochromatic point source: fully developed speckle, contrast 1 (1/√2 if depolarizing)
  partial     M = M_λ · M_Ω · M_pol independent patterns, contrast C = 1/√M, intensity ~ Gamma(M, 1/M)
              M_λ = √(1 + (4π σ_h / l_c)²)      spectral averaging over surface height spread σ_h
              M_Ω = 1 + (α_s / α_c)²,  α_c = λ / (2π σ_h)   angular averaging over the source extent α_s
              M_pol = 2 for a depolarizing surface
  speckle_grain is the mean speckle size in pixels (≈ λ / NA of the imaging system / pixel pitch).

Polarization: "unpolarized" or a linear E-field angle (deg) about the incident direction, 0 = in the plane
spanned by the light and view directions (p for the mirror highlight), 90 = perpendicular (s); the s-fraction
(ê·ŝ)², with ŝ normal to the plane of incidence, weights Rs and Rp in exact (dielectric/conductor) Fresnel.
"""
from dataclasses import dataclass, field, replace

import numpy as np

from brdftoolbox.core.frames import EPS, basis, normalize, sph_dir

KINDS = ("directional", "point", "beam")
SPECTRA = ("white", "blackbody", "gaussian")
COHERENCE = ("incoherent", "coherent", "partial")

_XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
_LAM = np.arange(380.0, 781.0, 1.0)


def _g(l, mu, s1, s2):
    return np.exp(-0.5 * ((l - mu) / np.where(l < mu, s1, s2)) ** 2)


def cie_xyz(lam):
    """CIE 1931 2° matching functions, multi-lobe fit of Wyman, Sloan & Shirley (2013)."""
    x = 1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7) - 0.065 * _g(lam, 501.1, 20.4, 26.2)
    y = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)


def planck(lam_nm, T):
    l = lam_nm * 1e-9
    return 1.0 / (l ** 5 * np.expm1(1.438777e-2 / (l * T)))


def spectrum_to_rgb(S):
    """Linear sRGB of spectrum S(_LAM), normalized to channel sum 3 (white = 1,1,1); out-of-gamut colours
    are desaturated toward white."""
    xyz = S @ cie_xyz(_LAM)
    rgb = _XYZ2RGB @ (xyz / max(xyz[1], EPS))
    rgb = rgb - min(rgb.min(), 0.0)
    return 3.0 * rgb / max(rgb.sum(), EPS)


@dataclass
class Light:
    kind: str = "directional"
    theta: float = 40.0
    phi: float = -35.0
    intensity: float = 2.5
    distance: float = 4.0
    radius: float = 0.0
    angular_radius: float = 0.0
    waist: float = 0.3
    divergence: float = 5.0
    aim_x: float = 0.0
    aim_y: float = 0.0
    spectrum: str = "white"
    temperature: float = 6500.0
    wavelength: float = 550.0
    bandwidth: float = 1.0
    coherence: str = "incoherent"
    roughness_um: float = 1.0
    speckle_grain: float = 3.0
    depolarizing: bool = False
    polarization: object = "unpolarized"
    samples: int = 16
    seed: int = 0

    def replace(self, **kw):
        return replace(self, **kw)

    @property
    def direction(self):
        return sph_dir(np.deg2rad(self.theta), np.deg2rad(self.phi))

    @property
    def position(self):
        return self.direction * self.distance

    def spectrum_samples(self):
        if self.spectrum == "white":
            return np.ones_like(_LAM)
        if self.spectrum == "blackbody":
            return planck(_LAM, self.temperature)
        s = max(self.bandwidth, 0.5) / 2.3548
        return np.exp(-0.5 * ((_LAM - self.wavelength) / s) ** 2)

    def rgb(self):
        if self.spectrum == "white":
            return np.ones(3)
        return spectrum_to_rgb(self.spectrum_samples())

    def centroid_bandwidth(self):
        if self.spectrum == "gaussian":
            return self.wavelength, max(self.bandwidth, 1e-6)
        S = self.spectrum_samples()
        m = np.sum(S * _LAM) / np.sum(S)
        return m, 2.3548 * np.sqrt(np.sum(S * (_LAM - m) ** 2) / np.sum(S))

    def source_angle(self):
        """Angular radius (rad) of the source seen from the sphere centre."""
        if self.kind == "directional":
            return np.deg2rad(self.angular_radius)
        if self.kind == "point":
            return float(np.arcsin(min(self.radius / max(self.distance, EPS), 1.0)))
        return 0.0

    def coherence_stats(self):
        lam, dlam = self.centroid_bandwidth()
        lc_um = lam ** 2 / dlam * 1e-3
        sh = max(self.roughness_um, 1e-6)
        m_pol = 2.0 if self.depolarizing else 1.0
        if self.coherence == "incoherent":
            m_l = m_a = np.inf
        elif self.coherence == "coherent":
            m_l = m_a = 1.0
        else:
            m_l = np.sqrt(1.0 + (4.0 * np.pi * sh / lc_um) ** 2)
            a_c = lam * 1e-3 / (2.0 * np.pi * sh)
            m_a = 1.0 + (self.source_angle() / a_c) ** 2
        M = m_l * m_a * m_pol
        return {"wavelength_nm": lam, "bandwidth_nm": dlam, "coherence_length_um": lc_um,
                "M_spectral": m_l, "M_angular": m_a, "M_polarization": m_pol, "M": M,
                "contrast": 0.0 if not np.isfinite(M) else 1.0 / np.sqrt(M)}

    def describe(self):
        geo = {"directional": f"α={self.angular_radius:.3g}°", "point": f"d={self.distance:.3g} R={self.radius:.3g}",
               "beam": f"d={self.distance:.3g} w0={self.waist:.3g} β={self.divergence:.3g}°"}[self.kind]
        return (f"{self.kind} θ={self.theta:.3g}° φ={self.phi:.3g}° {geo} E0={self.intensity:.3g} "
                f"{self.spectrum} {self.coherence}")


def _cone_dirs(c, alpha, u):
    """Uniform directions in the cone of half-angle alpha about c. c (..., 3); u (S, 2) -> (..., S, 3)."""
    ca = np.cos(alpha)[..., None]
    ct = 1.0 - u[..., 0] * (1.0 - ca)
    st = np.sqrt(np.maximum(1.0 - ct * ct, 0.0))
    ph = 2.0 * np.pi * u[..., 1]
    t, b = basis(c)
    return (st * np.cos(ph))[..., None] * t[..., None, :] + (st * np.sin(ph))[..., None] * b[..., None, :] \
        + ct[..., None] * c[..., None, :]


def _beam_frame(light):
    d = light.direction
    t, b = basis(d)
    L = light.position
    aim = light.aim_x * t + light.aim_y * b
    a = normalize(aim - L)
    return L, a


def _beam_w(light, z):
    return np.sqrt(light.waist ** 2 + (z * np.tan(np.deg2rad(light.divergence))) ** 2)


def incident(light, p, u):
    """Incident directions and per-direction irradiance weights at surface points p (n, 3).
    Returns wi (n, S, 3) world directions and w (n, S): E_perp contribution of each direction
    (sum over S of w gives the normal irradiance; the cosine with the surface is applied by the caller)."""
    n = len(p)
    if light.kind == "directional":
        c = np.broadcast_to(light.direction, (n, 3))
        E = np.full(n, light.intensity)
        alpha = np.full(n, np.deg2rad(light.angular_radius))
    elif light.kind == "point":
        v = light.position - p
        r = np.linalg.norm(v, axis=-1)
        c = v / r[:, None]
        E = light.intensity * (light.distance / np.maximum(r, EPS)) ** 2
        alpha = np.arcsin(np.clip(light.radius / np.maximum(r, EPS), 0.0, 1.0))
    else:
        L, a = _beam_frame(light)
        rel = p - L
        z = rel @ a
        rho2 = np.sum(rel * rel, -1) - z * z
        w = _beam_w(light, z)
        wc = _beam_w(light, -L @ a)
        E = np.where(z > 0, light.intensity * (wc / w) ** 2 * np.exp(-2.0 * rho2 / (w * w)), 0.0)
        tb = np.tan(np.deg2rad(light.divergence))
        if tb > 1e-9:
            apex = L - a * (light.waist / tb)
            c = normalize(apex - p)
        else:
            c = np.broadcast_to(-a, (n, 3))
        alpha = np.zeros(n)
    if np.all(alpha <= 1e-9) or u is None:
        return c[:, None, :], E[:, None]
    S = u.shape[-2]
    ca = np.cos(alpha)
    omega = 2.0 * np.pi * (1.0 - ca)
    # uniform disk radiance Le with E_perp = Le·π sin²α, so each sample carries Le·Ω/S
    frac = np.where(alpha > 1e-9, omega / np.maximum(np.pi * np.sin(alpha) ** 2, EPS), 1.0)
    dirs = _cone_dirs(c, alpha, u)
    return dirs, np.repeat((E * frac / S)[:, None], S, 1)


def speckle_weights(M):
    """K equal-variance patterns with weights giving contrast 1/√M (exact for any real M ≥ 1)."""
    if not np.isfinite(M) or M <= 1.0 + 1e-9:
        return np.ones(1) if np.isfinite(M) else np.zeros(0)
    K = int(np.ceil(M))
    if K == int(M):
        return np.ones(K)
    k1 = K - 1
    # (k1 + w²)/(k1 + w)² = 1/M  ->  (M-1)w² - 2k1 w + k1(M - k1) = 0
    A, B, C = M - 1.0, -2.0 * k1, k1 * (M - k1)
    disc = max(B * B - 4 * A * C, 0.0)
    w = (-B - np.sqrt(disc)) / (2 * A)
    return np.concatenate([np.ones(k1), [np.clip(w, 0.0, 1.0)]])


def speckle_pattern(res, grain, M=1.0, seed=0):
    """Unit-mean speckle intensity image (res, res) with mean grain size `grain` px and contrast 1/√M."""
    wts = speckle_weights(M)
    if wts.size == 0:
        return np.ones((res, res))
    rng = np.random.default_rng(seed)
    fx = np.fft.fftfreq(res)
    FX, FY = np.meshgrid(fx, fx)
    pupil = (FX ** 2 + FY ** 2) <= (0.5 / max(grain, 1.0)) ** 2
    out = np.zeros((res, res))
    for w in wts:
        g = rng.standard_normal((res, res)) + 1j * rng.standard_normal((res, res))
        I = np.abs(np.fft.ifft2(np.fft.fft2(g) * pupil)) ** 2
        out += w * I / I.mean()
    return out / wts.sum()


def polarization_s_fraction(light, wi, view):
    """(ê·ŝ)² for linear polarization; 0.5 for unpolarized. wi (..., 3) world, view (3,)."""
    if isinstance(light.polarization, str):
        return None
    d = light.direction
    u = np.asarray(view, float) - np.dot(view, d) * d
    if np.linalg.norm(u) < 1e-6:
        u, _ = basis(d)
    u = u / np.linalg.norm(u)
    v = np.cross(d, u)
    psi = np.deg2rad(float(light.polarization))
    e = np.cos(psi) * u + np.sin(psi) * v
    e = normalize(e - np.sum(e * wi, -1, keepdims=True) * wi)
    s = np.cross(wi, np.broadcast_to(view, wi.shape))
    ns = np.linalg.norm(s, axis=-1)
    s = s / np.maximum(ns, EPS)[..., None]
    return np.where(ns > 1e-6, np.sum(e * s, -1) ** 2, 0.5)


def report(light):
    st = light.coherence_stats()
    lam, dlam = st["wavelength_nm"], st["bandwidth_nm"]
    rows = [light.describe(),
            f"colour (lin. sRGB, Σ=3)   {np.round(light.rgb(), 3).tolist()}",
            f"centroid / FWHM            {lam:.0f} nm / {dlam:.1f} nm",
            f"coherence length           {st['coherence_length_um']:.3g} µm",
            f"source angular radius      {np.rad2deg(light.source_angle()):.3g}°"]
    if light.coherence != "incoherent":
        rows += [f"speckle M = M_λ·M_Ω·M_pol  {st['M_spectral']:.3g} · {st['M_angular']:.3g} · "
                 f"{st['M_polarization']:.0f} = {st['M']:.3g}",
                 f"predicted contrast         {st['contrast']:.3f}"]
    if light.kind == "beam":
        L, a = _beam_frame(light)
        rows.append(f"beam radius at sphere      {_beam_w(light, -L @ a):.3g}  (waist {light.waist:.3g})")
    return "\n".join(rows)


def parse(spec):
    """'θ,φ,I' or 'k=v,k=v' -> Light."""
    if "=" not in spec:
        t, p, i = (float(x) for x in spec.split(","))
        return Light(theta=t, phi=p, intensity=i)
    kw = {}
    for item in spec.split(","):
        k, _, v = item.partition("=")
        k = k.strip()
        typ = type(getattr(Light, k)) if k != "polarization" else object
        if typ is bool:
            kw[k] = v.lower() in ("1", "true", "yes")
        elif typ in (int, float):
            kw[k] = typ(float(v))
        elif k == "polarization":
            kw[k] = v if v == "unpolarized" else float(v)
        else:
            kw[k] = v
    return Light(**kw)
