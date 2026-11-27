"""Matplotlib views: lobes, microfacet terms, energy, sampling, half/diff slices, fits, renders."""
import numpy as np

from brdftoolbox.core import analytics as A
from brdftoolbox.core import fresnel as fr
from brdftoolbox.core import microfacet as mf
from brdftoolbox.core.frames import EPS, from_half_diff, sph_dir
from brdftoolbox.core.models import CookTorrance, Sum, eval_c

COLORS = ("#3b6fb6", "#d1603d", "#3f9b5a", "#8e5ea2", "#c49a2c", "#5b5b5b")


def _mono(f, nd=1):
    return np.mean(f, -1) if f.ndim > nd else f


def incidence_plane(model, theta_i, phi_plane=0.0, n=721, weighted=True):
    t = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, n)
    wi = np.broadcast_to(sph_dir(theta_i, phi_plane + np.pi), (n, 3))
    wo = sph_dir(np.abs(t), np.where(t >= 0, phi_plane, phi_plane + np.pi))
    f = _mono(eval_c(model, wi, wo))
    if weighted:
        f = f * np.cos(t)
    return t, f


def plot_polar_lobe(ax, model, theta_i, phi_plane=0.0, weighted=True, label=None, color=None, log=False):
    t, f = incidence_plane(model, theta_i, phi_plane, weighted=weighted)
    if log:
        f = np.log10(1.0 + f / max(f.max(), EPS) * 1e3) / 3.0
    ax.plot(np.pi / 2 - t, f, lw=1.6, label=label, color=color)
    ax.plot([np.pi / 2 + theta_i] * 2, [0, max(f.max(), EPS) * 1.05], "--", color="#c33", lw=1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_yticklabels([])
    return f.max()


def lobe_surface(model, theta_i, phi_i=np.pi, n_t=60, n_p=120):
    t = np.linspace(0, np.pi / 2, n_t)
    p = np.linspace(0, 2 * np.pi, n_p)
    T, P = np.meshgrid(t, p, indexing="ij")
    wo = sph_dir(T, P)
    wi = np.broadcast_to(sph_dir(theta_i, phi_i), wo.shape)
    f = _mono(eval_c(model, wi, wo), 2) * np.cos(T)
    return wo * f[..., None], f


def plot_lobe_3d(ax, model, theta_i, phi_i=np.pi):
    from matplotlib import cm
    xyz, f = lobe_surface(model, theta_i, phi_i)
    c = f / max(f.max(), EPS)
    ax.plot_surface(xyz[..., 0], xyz[..., 1], xyz[..., 2], facecolors=cm.viridis(c), rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    s = max(np.abs(xyz).max(), EPS)
    wi = sph_dir(theta_i, phi_i) * s
    ax.plot([0, wi[0]], [0, wi[1]], [0, wi[2]], color="#c33", lw=2)
    ax.set_xlim(-s, s); ax.set_ylim(-s, s); ax.set_zlim(0, s)
    ax.set_box_aspect((2, 2, 1), zoom=1.5)
    ax.view_init(elev=22, azim=-70)
    ax.set_axis_off()


def _ct(model):
    if isinstance(model, CookTorrance):
        return model
    if isinstance(model, Sum):
        for c in model.c.values():
            if isinstance(c, CookTorrance):
                return c
    return None


def plot_ndf(ax, model, log=False):
    ct = _ct(model) or CookTorrance()
    ax_, ay_ = ct.alphas()
    th = np.linspace(0, np.pi / 2 - 1e-3, 500)
    for i, n in enumerate(mf.NDFS):
        cur = n == ct.o["ndf"]
        for phi, ls in ((0.0, "-"), (np.pi / 2, ":")):
            if phi and np.isclose(ax_, ay_):
                continue
            ax.plot(np.rad2deg(th), mf.D(sph_dir(th, phi), ax_, ay_, n), ls, color=COLORS[i],
                    lw=2 if cur else 1, alpha=1 if cur else 0.45, label=n if phi == 0 else None)
    if log:
        ax.set_yscale("log")
    ax.set_title(f"D(θh)  αx={float(np.mean(ax_)):.3f} αy={float(np.mean(ay_)):.3f}", fontsize=9)
    ax.set_xlabel("θh (deg)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def plot_fresnel(ax, f0=0.04, eta=1.5, metals=True):
    th = np.linspace(0, np.pi / 2, 400)
    c = np.cos(th)
    d = np.rad2deg(th)
    ax.plot(d, fr.schlick(c, f0), color=COLORS[0], label=f"Schlick F0={f0:.2f}")
    ax.plot(d, fr.dielectric(c, eta), "--", color=COLORS[1], label=f"dielectric η={eta:.2f}")
    ax.plot(d, fr.schlick(c, fr.f0_from_ior(eta)), ":", color=COLORS[1], label="Schlick(F0(η))")
    if metals:
        for i, (k, (e, kk)) in enumerate(fr.METALS.items()):
            ax.plot(d, np.mean(fr.conductor(c[:, None], np.array(e), np.array(kk)), -1), lw=0.9,
                    color=COLORS[2 + i], alpha=0.8, label=k)
    ax.set_ylim(0, 1.02)
    ax.set_title("Fresnel", fontsize=9)
    ax.set_xlabel("θ (deg)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)


def plot_masking(ax, model, theta_i):
    ct = _ct(model) or CookTorrance()
    a, b = ct.alphas()
    th = np.linspace(0, np.pi / 2 - 1e-3, 400)
    wo = sph_dir(th, 0.0)
    wi = np.broadcast_to(sph_dir(theta_i, np.pi), wo.shape)
    n = ct.o["ndf"]
    d = np.rad2deg(th)
    ax.plot(d, mf.G1(wo, a, b, n), color=COLORS[0], label="G1(θ)")
    ax.plot(d, mf.G2(wi, wo, a, b, n, "height-correlated"), color=COLORS[1], label="G2 height-corr.")
    ax.plot(d, mf.G2(wi, wo, a, b, n, "separable"), "--", color=COLORS[2], label="G2 separable")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Smith masking ({n})", fontsize=9)
    ax.set_xlabel("θo (deg)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def plot_albedo(ax, models, n_theta=24, n=4096):
    th = np.linspace(0, np.deg2rad(89), n_theta)
    for i, (name, m) in enumerate(models.items()):
        E, err = A.directional_albedo(m, th, n=n)
        E = _mono(E)
        ax.plot(np.rad2deg(th), E, color=COLORS[i % len(COLORS)], label=name)
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("θo (deg)")
    ax.set_title("directional albedo E(θo)", fontsize=9)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def plot_sampling(ax_h, ax_p, model, theta_o, n=100_000):
    c = A.sampler_chi2(model, theta_o, n=n)
    kw = dict(aspect="auto", origin="lower", extent=[0, 360, 0, 1], cmap="magma")
    vmax = max(c["observed"].max(), c["expected"].max())
    ax_h.imshow(c["observed"], vmin=0, vmax=vmax, **kw)
    ax_h.set_title("sample histogram", fontsize=9)
    ax_p.imshow(c["expected"], vmin=0, vmax=vmax, **kw)
    ax_p.set_title(f"∫pdf   χ²={c['chi2']:.0f}/{c['dof']}  p={c['p_value']:.2f}", fontsize=9)
    for a in (ax_h, ax_p):
        a.set_xlabel("φ (deg)")
        a.set_ylabel("cos θi")
    return c


def half_diff_slice(model, phi_d=np.pi / 2, n=128, channel_mean=False):
    th = (np.linspace(0, 1, n) ** 2) * (np.pi / 2)
    td = np.linspace(0, np.pi / 2, n)
    TH, TD = np.meshgrid(th, td, indexing="ij")
    wi, wo = from_half_diff(TH, 0.0, TD, np.full_like(TH, phi_d))
    f = _mono(eval_c(model, wi, wo), 2) if channel_mean else eval_c(model, wi, wo)
    return th, td, f


def plot_half_diff(ax, model, phi_d=np.pi / 2, channel=None, title=None, vmin=-3, vmax=3):
    th, td, f = half_diff_slice(model, phi_d)
    if f.ndim == 3:
        f = f[..., channel] if channel is not None else np.mean(f, -1)
    im = ax.imshow(np.log10(np.maximum(f, 1e-4)).T, origin="lower", aspect="auto", cmap="inferno",
                   extent=[0, 1, 0, 90], vmin=vmin, vmax=vmax)
    ax.set_xlabel("√(θh / 90°)")
    ax.set_ylabel("θd (deg)")
    ax.set_title(title or f"log10 f, φd={np.rad2deg(phi_d):.0f}°", fontsize=9)
    return im


def plot_fit_slices(ax, measured, fitted, theta_ds=(5, 30, 60, 80), phi_d=np.pi / 2):
    th = (np.linspace(0, 1, 200) ** 2) * (np.pi / 2)
    for i, d in enumerate(theta_ds):
        wi, wo = from_half_diff(th, 0.0, np.deg2rad(d), np.full_like(th, phi_d))
        ok = np.minimum(wi[:, 2], wo[:, 2]) > 0.02
        fm = np.where(ok, _mono(eval_c(measured, wi, wo)), np.nan)
        ff = np.where(ok, _mono(eval_c(fitted, wi, wo)), np.nan)
        ax.semilogy(np.rad2deg(th), np.maximum(fm, 1e-4), color=COLORS[i], lw=1.8, label=f"θd={d}° data")
        ax.semilogy(np.rad2deg(th), np.maximum(ff, 1e-4), "--", color=COLORS[i], lw=1.2)
    ax.set_xlabel("θh (deg)")
    ax.set_title("measured (solid) vs fit (dashed)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")


def overview_figure(model, theta_i=np.deg2rad(45.0), fig=None):
    import matplotlib.pyplot as plt
    fig = fig or plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    a = fig.add_subplot(gs[0, 0], projection="polar")
    plot_polar_lobe(a, model, theta_i, label="φ=0")
    plot_polar_lobe(a, model, theta_i, phi_plane=np.pi / 2, color=COLORS[1], label="φ=90°")
    a.set_title("f·cosθo, incidence planes", fontsize=9)
    a.legend(fontsize=7, loc="upper right")
    plot_lobe_3d(fig.add_subplot(gs[0, 1], projection="3d"), model, theta_i)
    plot_half_diff(fig.add_subplot(gs[0, 2]), model)
    plot_ndf(fig.add_subplot(gs[1, 0]), model)
    plot_masking(fig.add_subplot(gs[1, 1]), model, theta_i)
    plot_albedo(fig.add_subplot(gs[1, 2]), {"model": model})
    fig.suptitle(model.describe(), fontsize=9)
    return fig


def lighting_figure(light, model=None, fig=None, res=160):
    """Irradiance map and profile, spectrum, speckle statistics, s/p Fresnel for the material."""
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.size": 7}):
        return _lighting_figure(light, model, fig, res)


def _lighting_figure(light, model, fig, res):
    import matplotlib.pyplot as plt
    from scipy import stats
    from brdftoolbox.core import lights as LT
    from brdftoolbox.core.render import irradiance_map
    own = fig is None
    fig = fig or plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.35, left=0.05, right=0.97, top=0.9 if own else 0.93,
                          bottom=0.08)
    E = irradiance_map(light, res)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(E, cmap="magma", extent=[-1, 1, -1, 1])
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("irradiance E(p) = ∫ L cosθi dω", fontsize=7.5)
    ax.axis("off")
    ax = fig.add_subplot(gs[0, 1])
    x = np.linspace(-1, 1, res)
    i, j = np.unravel_index(np.nanargmax(E), E.shape)
    ax.plot(x, E[i], color=COLORS[0], label="horizontal")
    ax.plot(x, E[:, j][::-1], color=COLORS[1], label="vertical")
    ax.set_title("profiles through the maximum", fontsize=7.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[0, 2])
    S = light.spectrum_samples()
    ax.fill_between(LT._LAM, S / S.max(), color=np.clip(light.rgb() / max(light.rgb().max(), 1e-9), 0, 1), alpha=0.8)
    lam, dl = light.centroid_bandwidth()
    st = light.coherence_stats()
    ax.set_title(f"spectrum  λ̄={lam:.0f} nm  Δλ={dl:.3g} nm\nl_c={st['coherence_length_um']:.3g} µm", fontsize=7.5)
    ax.set_xlabel("λ (nm)")
    ax.set_xlim(380, 780)
    ax = fig.add_subplot(gs[1, 0])
    if light.coherence != "incoherent":
        sp = LT.speckle_pattern(res, light.speckle_grain, st["M"], seed=light.seed)
        ax.imshow(sp, cmap="gray", vmin=0, vmax=np.percentile(sp, 99))
        ax.set_title(f"speckle pattern (grain {light.speckle_grain:.2g} px)", fontsize=7.5)
        ax.axis("off")
        ax = fig.add_subplot(gs[1, 1])
        ax.hist(sp.ravel(), bins=80, density=True, color=COLORS[0], alpha=0.6, label="pattern")
        I = np.linspace(1e-3, np.percentile(sp, 99.5), 300)
        M = st["M"]
        ax.plot(I, stats.gamma.pdf(I, a=M, scale=1.0 / M), color=COLORS[1], label=f"Gamma(M={M:.2f})")
        ax.set_title(f"intensity pdf: contrast {sp.std() / sp.mean():.3f}\n(theory 1/√M = {st['contrast']:.3f})", fontsize=7.5)
        ax.set_xlabel("I / ⟨I⟩")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "incoherent: no speckle", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.add_subplot(gs[1, 1]).axis("off")
    ax = fig.add_subplot(gs[1, 2])
    ct = _ct(model) if model is not None else None
    th = np.linspace(0, np.pi / 2, 400)
    if ct is not None and ct.o["fresnel"] == "conductor":
        rs, rp = fr.conductor_sp(np.cos(th)[:, None], np.asarray(ct.p["eta"]), np.asarray(ct.p["k"]))
        rs, rp, name = np.mean(np.atleast_2d(rs.T).T, -1), np.mean(np.atleast_2d(rp.T).T, -1), "conductor"
    else:
        eta = float(np.mean(ct.p["eta"])) if ct is not None and ct.o["fresnel"] == "dielectric" else 1.5
        rs, rp = fr.dielectric_sp(np.cos(th), eta)
        name = f"dielectric η={eta:.2f}"
        ax.axvline(np.rad2deg(fr.brewster_angle(eta)), color="k", lw=0.7, ls=":")
    d = np.rad2deg(th)
    ax.plot(d, rs, color=COLORS[0], label="Rs")
    ax.plot(d, rp, color=COLORS[1], label="Rp")
    ax.plot(d, 0.5 * (rs + rp), "--", color="k", lw=0.8, label="unpolarized")
    ax.set_title(f"polarized Fresnel ({name})", fontsize=7.5)
    ax.set_xlabel("θ (deg)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    if own:
        fig.suptitle(light.describe(), fontsize=9)
    return fig
