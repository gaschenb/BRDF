import numpy as np
import pytest

from brdftoolbox.core import analytics as A
from brdftoolbox.core import fitting as F
from brdftoolbox.core import frames as fr
from brdftoolbox.core import microfacet as mf
from brdftoolbox.core import models as M
from brdftoolbox.core import render as R
from brdftoolbox.core.measured import Measured, load_merl, save_merl, tabulate

ALL = {k: M.make(k) for k in M.PRESETS}
ALL["ward-aniso"] = M.Ward(alpha_x=0.08, alpha_y=0.4)
ALL["ct-aniso-beckmann"] = M.CookTorrance(roughness=0.5, anisotropy=0.7, ndf="beckmann")
ALL["ct-blinn"] = M.CookTorrance(ndf="blinn", g2="separable")
ALL["ct-dielectric"] = M.CookTorrance(fresnel="dielectric", eta=1.7)
ALL["disney-full"] = M.Disney(base_color=np.array([0.7, 0.3, 0.1]), metallic=0.3, sheen=1, clearcoat=1,
                              anisotropic=0.5, subsurface=0.5)


@pytest.mark.parametrize("name", ALL)
def test_reciprocity_positivity(name):
    r = A.reciprocity_check(ALL[name])
    assert r["reciprocal"] and r["non_negative"]


@pytest.mark.parametrize("name", [k for k in ALL if not k.startswith("disney")])
def test_energy_conservation(name):
    assert A.energy_check(ALL[name], n=8192)["conserving"]


def test_disney_not_energy_conserving_at_grazing():
    assert A.energy_check(M.Disney(), n=8192)["max_albedo"] > 1.01


@pytest.mark.parametrize("name", [k for k in ALL if k not in ("lambert",)])
def test_sampler_matches_pdf(name):
    assert A.sampler_chi2(ALL[name], np.deg2rad(50.0), n=100_000)["p_value"] > 1e-3


@pytest.mark.parametrize("ndf", mf.NDFS)
def test_ndf_normalization_and_projected_area(ndf):
    ax, ay = (0.2, 0.5) if ndf != "blinn" else (0.3, 0.3)
    c = A.ndf_checks(ax, ay, ndf)
    assert abs(c["ndf_norm"] - 1) < 2e-3
    assert abs(c["projected_area"] - c["cos_theta_o"]) < 2e-3


def test_white_furnace_kulla_conty():
    for r in (0.3, 0.7, 1.0):
        m = M.CookTorrance(roughness=r, fresnel="none", multiscatter=True)
        E, _ = A.directional_albedo(m, np.deg2rad([0, 30, 60, 80]), n=1 << 14)
        assert np.all(np.abs(E - 1) < 0.01)
        m1 = m.replace(multiscatter=False)
        assert A.directional_albedo(m1, 0.0, n=1 << 14)[0] < 0.995


def test_lambert_albedo_exact():
    E, _ = A.directional_albedo(M.Lambert(albedo=0.6), np.deg2rad([0, 45, 80]), method="quad")
    assert np.allclose(E, 0.6, atol=1e-6)


def test_quad_matches_mc():
    m = M.CookTorrance(roughness=0.7)
    q, _ = A.directional_albedo(m, 0.6, method="quad", n_mu=128, n_phi=256)
    mc, _ = A.directional_albedo(m, 0.6, n=1 << 15)
    assert abs(q - mc) < 2e-3


def test_importance_sampling_beats_uniform():
    e = A.sampling_efficiency(M.CookTorrance(roughness=0.2), np.deg2rad(40.0), n=1024, trials=16)
    assert e["brdf"]["var"] < 0.05 * e["uniform"]["var"]


def test_half_diff_roundtrip():
    rng = np.random.default_rng(0)
    wi = fr.cosine_hemisphere(rng.random(500), rng.random(500))
    wo = fr.cosine_hemisphere(rng.random(500), rng.random(500))
    a, b = fr.from_half_diff(*fr.to_half_diff(wi, wo))
    assert np.allclose(a, wi, atol=1e-10) and np.allclose(b, wo, atol=1e-10)


def test_merl_roundtrip_and_lookup(tmp_path):
    truth = M.plastic(albedo=np.array([0.6, 0.3, 0.2]), roughness=0.4)
    T = tabulate(truth)
    p = tmp_path / "x.binary"
    save_merl(p, T)
    T2 = load_merl(p)
    assert np.allclose(T, T2, rtol=1e-12, atol=1e-12)
    m = Measured(T2)
    rng = np.random.default_rng(1)
    wi = fr.cosine_hemisphere(rng.random(2000), rng.random(2000))
    wo = fr.cosine_hemisphere(rng.random(2000), rng.random(2000))
    rel = np.abs(m.eval(wi, wo) - M.eval_rgb(truth, wi, wo)) / (M.eval_rgb(truth, wi, wo) + 1e-2)
    assert np.median(rel) < 1e-3


def test_fit_recovers_parameters():
    truth = M.plastic(albedo=np.array([0.6, 0.3, 0.2]), roughness=0.35, f0=0.05)
    r = F.fit_preset("plastic", Measured.from_model(truth), n_samples=6000, n_starts=2)
    assert abs(r.params["specular.roughness"] - 0.35) < 1e-3
    assert np.allclose(r.params["diffuse.albedo"], [0.6, 0.3, 0.2], atol=2e-3)
    assert np.allclose(r.params["specular.f0"], 0.05, atol=2e-3)


def test_fit_ward_recovers_anisotropy():
    truth = M.ward_plastic(albedo=0.4, ks=0.3, alpha_x=0.1, alpha_y=0.3)
    rng = np.random.default_rng(2)
    wo = fr.cosine_hemisphere(rng.random(8000), rng.random(8000))
    wi, _ = truth.sample(wo, rng.random((8000, 3)))
    ok = wi[:, 2] > 0
    data = (wi[ok], wo[ok], truth.eval(wi[ok], wo[ok]))
    r = F.fit_preset("ward-plastic-aniso", data, n_samples=0, n_starts=2)
    assert abs(r.params["specular.alpha_x"] - 0.1) < 5e-3 and abs(r.params["specular.alpha_y"] - 0.3) < 5e-3


def test_render_shapes_and_maps():
    m = M.plastic()
    img = R.render_sphere(m, res=48, env="studio", spp=4,
                          maps={"diffuse.albedo": R.TEXTURES["checker"],
                                "specular.roughness": lambda u, v: 0.2 + 0.5 * u})
    assert img.shape == (48, 48, 3) and np.isfinite(img).all()
    img = R.render_sphere(M.metal("copper"), res=32, env=None)
    assert img[16, 16].std() > 0


def test_cli_smoke(tmp_path):
    from brdftoolbox.cli import main
    main(["check", "--model", "ward", "--set", "alpha_y=0.3", "--quick"])
    main(["render", "--model", "disney", "--set", "clearcoat=1", "--res", "32", "--spp", "4",
          "--out", str(tmp_path / "s.png")])
    main(["fit", "--synthetic", "plastic", "--fit", "ggx", "--samples", "2000", "--starts", "1",
          "--out", str(tmp_path / "fit")])
    assert (tmp_path / "fit" / "fit_comparison.png").exists()
