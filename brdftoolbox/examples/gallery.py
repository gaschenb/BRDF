"""Generate the example figures in brdf_output/."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brdftoolbox.core import analytics as A
from brdftoolbox.core import fitting as F
from brdftoolbox.core import lights as LT
from brdftoolbox.core import models as M
from brdftoolbox.core import plots as P
from brdftoolbox.core import render as R
from brdftoolbox.core.measured import Measured


def main(out="brdf_output"):
    os.makedirs(out, exist_ok=True)
    P.overview_figure(M.CookTorrance(roughness=0.45, anisotropy=0.7), np.deg2rad(50)).savefig(
        os.path.join(out, "overview_ggx_aniso.png"), dpi=120)

    spheres = {
        "Lambert": M.Lambert(albedo=0.7),
        "Oren-Nayar σ=0.8": M.OrenNayar(albedo=0.7, sigma=0.8),
        "Phong plastic": M.phong_plastic(albedo=np.array([0.1, 0.3, 0.6]), ks=0.3, n=80),
        "Ward aniso": M.ward_plastic(albedo=np.array([0.3, 0.05, 0.05]), ks=0.3, alpha_x=0.05, alpha_y=0.35),
        "GGX plastic": M.plastic(albedo=np.array([0.6, 0.1, 0.1]), roughness=0.3),
        "Gold (conductor)": M.metal("gold", 0.3),
        "Copper aniso": M.metal("copper", 0.4, anisotropy=0.8),
        "Disney clearcoat": M.Disney(base_color=np.array([0.1, 0.25, 0.6]), roughness=0.6, clearcoat=1.0),
    }
    fig, ax = plt.subplots(2, 4, figsize=(14, 7.4))
    for a, (n, m) in zip(ax.ravel(), spheres.items()):
        a.imshow(R.render_sphere(m, res=256, env="studio", spp=128))
        a.set_title(n, fontsize=10)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "spheres.png"), dpi=110)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    P.plot_albedo(ax[0], {f"r={r}": M.CookTorrance(roughness=r, fresnel="none") for r in (0.2, 0.5, 0.8, 1.0)}, n=8192)
    ax[0].set_title("white furnace, single scattering (F=1)", fontsize=9)
    P.plot_albedo(ax[1], {f"r={r}": M.CookTorrance(roughness=r, fresnel="none", multiscatter=True)
                          for r in (0.2, 0.5, 0.8, 1.0)}, n=8192)
    ax[1].set_title("white furnace, + Kulla–Conty (F=1)", fontsize=9)
    for a in ax:
        a.set_ylim(0.4, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "furnace.png"), dpi=120)

    mat = M.Sum(diffuse=M.Lambert(albedo=0.45), specular=M.CookTorrance(roughness=0.3, fresnel="dielectric"))
    setups = {
        "directional": LT.Light(),
        "sun-like disk α=10°": LT.Light(angular_radius=10.0),
        "near point d=1.6": LT.Light(kind="point", distance=1.6),
        "area R=0.8, d=2.5": LT.Light(kind="point", distance=2.5, radius=0.8, samples=64),
        "2700 K blackbody": LT.Light(spectrum="blackbody", temperature=2700),
        "flashlight beam β=12°": LT.Light(kind="beam", distance=3.0, waist=0.15, divergence=12.0),
        "laser 532 nm, coherent": LT.Light(kind="beam", distance=4.0, waist=0.25, divergence=1.0, spectrum="gaussian",
                                           wavelength=532, bandwidth=0.01, coherence="coherent", speckle_grain=2.5),
        "LED 532±30 nm, partial": LT.Light(kind="beam", distance=4.0, waist=0.25, divergence=1.0, spectrum="gaussian",
                                           wavelength=532, bandwidth=30, coherence="partial", roughness_um=3,
                                           speckle_grain=2.5),
        "s-polarized, θ=88°": LT.Light(theta=88.0, phi=180.0, polarization=90.0),
        "p-polarized, θ=88° (near Brewster)": LT.Light(theta=88.0, phi=180.0, polarization=0.0),
    }
    fig, ax = plt.subplots(2, 5, figsize=(17, 7.4))
    for a, (n, L) in zip(ax.ravel(), setups.items()):
        a.imshow(R.render_sphere(mat, res=240, lights=[L], ambient=0.01))
        c = L.coherence_stats()["contrast"]
        a.set_title(n + (f"\nspeckle contrast {c:.2f}" if L.coherence != "incoherent" else ""), fontsize=9)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "lighting.png"), dpi=110)
    P.lighting_figure(setups["LED 532±30 nm, partial"].replace(depolarizing=True), mat).savefig(
        os.path.join(out, "lighting_analytics.png"), dpi=120)

    checks = {}
    for n, m in spheres.items():
        r = A.report(m, quick=True)
        checks[n] = {"max_albedo": float(np.max(r["energy"]["max_albedo"])),
                     "reciprocity_max_rel_err": r["reciprocity"]["max_rel_err"],
                     "chi2_p": r["chi2"]["p_value"],
                     "is_speedup_vs_uniform": r["efficiency"]["brdf"]["speedup_vs_uniform"],
                     "cosine_speedup_vs_uniform": r["efficiency"]["cosine"]["speedup_vs_uniform"]}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    P.plot_sampling(ax[0], ax[1], M.Ward(alpha_x=0.08, alpha_y=0.4), np.deg2rad(50))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sampling_ward.png"), dpi=120)

    truth = M.Disney(base_color=np.array([0.7, 0.35, 0.1]), roughness=0.4, specular=0.6, sheen=0.5)
    data = Measured.from_model(truth, "Disney (synthetic, σ=0.05)")
    rng = np.random.default_rng(0)
    t = data.table
    data = Measured(np.where(t >= 0, t * np.exp(0.05 * rng.standard_normal(t.shape)), t), data.name)
    fits = {n: F.fit_preset(n, data, n_samples=20000, n_starts=3)
            for n in ("plastic", "oren-nayar-ggx", "ward-plastic", "phong-plastic", "disney")}
    k = len(fits)
    fig, ax = plt.subplots(2, k + 1, figsize=(3.4 * (k + 1), 6.8))
    ax[0, 0].imshow(R.render_sphere(data, res=200, env="studio", spp=64))
    ax[0, 0].set_title(data.name, fontsize=9)
    P.plot_half_diff(ax[1, 0], data, title="data log10 f")
    for i, (n, r) in enumerate(fits.items(), 1):
        ax[0, i].imshow(R.render_sphere(r.model, res=200, env="studio", spp=64))
        ax[0, i].set_title(f"{n}  loss rmse {r.rmse:.3f}", fontsize=9)
        P.plot_fit_slices(ax[1, i], data, r.model)
    for a in ax[0]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fit_comparison.png"), dpi=110)

    res = {"checks": checks,
           "fits": {n: {"rmse_log": r.rmse, "rel_rmse": r.rel_rmse,
                        "params": {p: np.round(v, 4).tolist() for p, v in r.params.items()}}
                    for n, r in fits.items()}}
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
