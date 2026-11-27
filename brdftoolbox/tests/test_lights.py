import numpy as np
import pytest

from brdftoolbox.core import fresnel as fr
from brdftoolbox.core import lights as LT
from brdftoolbox.core import models as M
from brdftoolbox.core import render as R
from brdftoolbox.core.frames import sph_dir


def _E(light, p):
    dirs, w = LT.incident(light, np.atleast_2d(p), np.random.default_rng(0).random((4096, 2)))
    n = np.atleast_2d(p) / np.linalg.norm(p)
    return float(np.sum(np.maximum(np.sum(dirs * n[:, None], -1), 0) * w))


def test_point_inverse_square():
    L = LT.Light(kind="point", theta=0, phi=0, distance=5.0, intensity=1.0)
    p = np.array([0.0, 0.0, 1.0])
    assert _E(L, p) == pytest.approx((5.0 / 4.0) ** 2, rel=1e-9)
    L2 = L.replace(distance=1e6)
    assert _E(L2, p) == pytest.approx(1.0, rel=1e-5)


def test_area_light_irradiance_matches_point_when_facing():
    p = np.array([0.0, 0.0, 1.0])
    for a in (2.0, 10.0, 30.0):
        L = LT.Light(theta=0, phi=0, intensity=1.0, angular_radius=a)
        assert _E(L, p) == pytest.approx(1.0, rel=2e-3)
    small = LT.Light(kind="point", theta=0, phi=0, distance=6.0, radius=1e-3, intensity=1.0)
    big = small.replace(radius=0.8)
    assert _E(big, p) == pytest.approx(_E(small, p), rel=2e-3)


def test_area_light_softens_terminator():
    p = sph_dir(np.deg2rad(95.0), 0.0)
    assert _E(LT.Light(theta=0, phi=0, intensity=1.0), p) == 0.0
    assert _E(LT.Light(theta=0, phi=0, intensity=1.0, angular_radius=15), p) > 0.0


def test_beam_power_conservation():
    L = LT.Light(kind="beam", theta=0, phi=0, distance=3.0, waist=0.2, divergence=5.0, intensity=1.0)
    w = LT._beam_w(L, 3.0)
    r = np.linspace(0, 5 * w, 4001)
    E = L.intensity * np.exp(-2 * r ** 2 / w ** 2)
    P = np.trapezoid(E * 2 * np.pi * r, r)
    assert P == pytest.approx(np.pi * w ** 2 / 2, rel=1e-4)
    assert _E(L, np.array([0.0, 0.0, 1.0])) == pytest.approx((LT._beam_w(L, 3.0) / LT._beam_w(L, 2.0)) ** 2, rel=1e-6)


@pytest.mark.parametrize("M_", [1.0, 2.0, 3.7, 12.0])
def test_speckle_contrast(M_):
    S = LT.speckle_pattern(512, 3.0, M_, seed=1)
    assert S.mean() == pytest.approx(1.0, rel=1e-9)
    assert S.std() == pytest.approx(1 / np.sqrt(M_), rel=0.03)


def test_partial_coherence_limits():
    laser = LT.Light(spectrum="gaussian", wavelength=633, bandwidth=1e-3, coherence="partial", roughness_um=1)
    assert laser.coherence_stats()["contrast"] > 0.99
    led = laser.replace(bandwidth=30.0, roughness_um=10.0)
    assert led.coherence_stats()["contrast"] < 0.4
    assert laser.replace(depolarizing=True).coherence_stats()["contrast"] == pytest.approx(2 ** -0.5, rel=1e-3)
    assert LT.Light().coherence_stats()["contrast"] == 0.0


def test_spectrum_colours():
    assert np.allclose(LT.Light().rgb(), 1.0)
    red = LT.Light(spectrum="gaussian", wavelength=650, bandwidth=10).rgb()
    assert red[0] > 5 * red[1] and red[0] > 5 * red[2]
    warm = LT.Light(spectrum="blackbody", temperature=2700).rgb()
    cool = LT.Light(spectrum="blackbody", temperature=10000).rgb()
    assert warm[0] > warm[2] and cool[2] > cool[0]


def test_brewster_polarization():
    eta = 1.5
    tb = fr.brewster_angle(eta)
    rs, rp = fr.dielectric_sp(np.cos(tb), eta)
    assert rp < 1e-12 and rs > 0.1
    ct = M.CookTorrance(roughness=0.05, fresnel="dielectric", eta=eta)
    wi, wo = sph_dir(tb, np.pi), sph_dir(tb, 0.0)
    assert ct.replace(pol_s=0.0).eval(wi, wo) < 1e-3 * ct.replace(pol_s=1.0).eval(wi, wo)


def test_render_with_rich_lights():
    m = M.Sum(diffuse=M.Lambert(albedo=0.4), specular=M.CookTorrance(fresnel="dielectric"))
    lights = [LT.Light(kind="point", distance=2.0, radius=0.5, spectrum="blackbody", temperature=3000),
              LT.Light(kind="beam", distance=4.0, waist=0.1, divergence=3, spectrum="gaussian", wavelength=532,
                       coherence="partial", roughness_um=2, polarization=0.0)]
    img = R.render_sphere(m, res=64, lights=lights)
    assert img.shape == (64, 64, 3) and np.isfinite(img).all()
    E = R.irradiance_map(lights[1], 64)
    assert np.nanmax(E) > 0


def test_parse():
    L = LT.parse("kind=beam,distance=5,divergence=3,depolarizing=true,polarization=45,samples=8")
    assert L.kind == "beam" and L.distance == 5 and L.depolarizing and L.polarization == 45.0 and L.samples == 8
    assert LT.parse("30,10,2").theta == 30


def test_polarized_highlight_near_brewster():
    m = M.CookTorrance(roughness=0.15, fresnel="dielectric", eta=1.5)
    s = R.render_sphere(m, res=96, lights=[LT.Light(theta=88, phi=180, polarization=90.0)], ambient=0, hdr=True)
    p = R.render_sphere(m, res=96, lights=[LT.Light(theta=88, phi=180, polarization=0.0)], ambient=0, hdr=True)
    assert s.max() > 5 * p.max()
