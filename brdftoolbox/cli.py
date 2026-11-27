"""BRDF toolbox CLI.

Models: lambert, oren-nayar, phong, blinn-phong, ward, cook-torrance, disney,
        plastic, phong-plastic, ward-plastic, metal
Parameters/options via --set k=v (comma list = per-channel RGB; composites use name.k).

Examples:
  brdf-toolbox list
  brdf-toolbox plot   --model cook-torrance --set roughness=0.4 --set anisotropy=0.6 --theta 50
  brdf-toolbox check  --model disney --set clearcoat=1
  brdf-toolbox check  --model cook-torrance --set fresnel=none --set multiscatter=true --set roughness=1
  brdf-toolbox sample --model ward --set alpha_x=0.1 --set alpha_y=0.4 --theta 60
  brdf-toolbox render --model metal --set name=copper --set roughness=0.25 --env studio --spp 64
  brdf-toolbox render --model plastic --texture checker --roughness-map
  brdf-toolbox render --model plastic --light kind=point,distance=1.8,radius=0.4,spectrum=blackbody,temperature=3000
  brdf-toolbox render --model plastic --light kind=beam,distance=5,waist=0.1,divergence=3,spectrum=gaussian,wavelength=532,bandwidth=0.01,coherence=partial,roughness_um=5,polarization=90
  brdf-toolbox light  --light kind=point,radius=0.5,coherence=partial --out light.png
  brdf-toolbox fit    --merl gold-metallic-paint.binary --fit plastic,ward-plastic,disney
  brdf-toolbox fit    --synthetic disney --set base_color=0.7,0.3,0.1 --set roughness=0.35 --fit all
  brdf-toolbox tabulate --model ward --out ward.binary
"""
import argparse
import json
import os
import sys

import numpy as np


def _value(v):
    lo = v.lower()
    if lo in ("true", "false"):
        return lo == "true"
    try:
        xs = [float(x) for x in v.split(",")]
        return np.array(xs) if len(xs) > 1 else xs[0]
    except ValueError:
        return v


def parse_sets(sets):
    return {k: _value(v) for k, _, v in (s.partition("=") for s in sets or [])}


def build_model(name, sets):
    from brdftoolbox.core import models as M
    kw = parse_sets(sets)
    if name in M.MODELS or name == "metal":
        return M.make(name, **kw)
    m = M.make(name)
    return m.replace(**kw) if kw else m


def _json(o):
    if isinstance(o, dict):
        return {k: _json(v) for k, v in o.items() if k not in ("expected", "observed")}
    if isinstance(o, (np.ndarray, np.generic)):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    return o


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def cmd_list(a):
    from brdftoolbox.core import models as M
    from brdftoolbox.core.fitting import FIT_PRESETS
    for k in M.PRESETS:
        m = M.make(k)
        print(f"{k:<14} {m.describe()}")
        for p, (d, lo, hi) in m.param_specs().items():
            print(f"    {p:<24} default {d}  [{lo}, {hi}]")
        opts = getattr(m, "o", {}) or {}
        for o, v in opts.items():
            print(f"    {o:<24} option  {v}  {m.OPTIONS[o][1]}")
    print("fit presets:", ", ".join(FIT_PRESETS))


def cmd_plot(a):
    from brdftoolbox.core import plots as P
    plt = _plt()
    m = build_model(a.model, a.set)
    os.makedirs(a.out, exist_ok=True)
    fig = P.overview_figure(m, np.deg2rad(a.theta))
    p = os.path.join(a.out, f"{a.model}_overview.png")
    fig.savefig(p, dpi=130)
    fig2, ax = plt.subplots(figsize=(6, 4))
    P.plot_fresnel(ax, float(np.mean(m.params().get("f0", 0.04))), float(np.mean(m.params().get("eta", 1.5))))
    fig2.savefig(os.path.join(a.out, "fresnel.png"), dpi=130)
    print("wrote", p)


def cmd_check(a):
    from brdftoolbox.core import analytics as A
    from brdftoolbox.core import plots as P
    m = build_model(a.model, a.set)
    r = A.report(m, np.deg2rad(a.theta), quick=a.quick)
    print(A.format_report(r))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"{a.model}_check.json"), "w") as fh:
            json.dump(_json(r), fh, indent=1)
        plt = _plt()
        fig, ax = plt.subplots(figsize=(6, 4))
        P.plot_albedo(ax, {a.model: m})
        fig.savefig(os.path.join(a.out, f"{a.model}_albedo.png"), dpi=130)
    return r


def cmd_sample(a):
    from brdftoolbox.core import analytics as A
    from brdftoolbox.core import plots as P
    m = build_model(a.model, a.set)
    th = np.deg2rad(a.theta)
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    c = P.plot_sampling(ax[0], ax[1], m, th, n=a.n)
    e = A.sampling_efficiency(m, th)
    print(f"χ² = {c['chi2']:.1f}, dof {c['dof']}, p = {c['p_value']:.3f}  ({'pass' if c['pass'] else 'FAIL'})")
    print(f"pdf mass over hemisphere: {A.pdf_mass(m, th):.4f}")
    for k, v in e.items():
        print(f"{k:<8} var×N {v['var']:.4g}   speedup {v['speedup_vs_uniform']:.1f}×")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fig.savefig(os.path.join(a.out, f"{a.model}_sampling.png"), dpi=130)


def _maps(a, m):
    from brdftoolbox.core import render as R
    maps = {}
    names = m.params()
    if a.texture:
        key = next((k for k in ("albedo", "diffuse.albedo", "base_color") if k in names), None)
        if key:
            maps[key] = R.TEXTURES[a.texture]
    if a.roughness_map:
        key = next((k for k in ("roughness", "specular.roughness") if k in names), None)
        if key:
            r0 = float(np.mean(names[key]))
            maps[key] = lambda u, v: np.where(R.checker(u, v, 8) > 0, r0, min(r0 * 3 + 0.1, 1.0))
    return maps


def cmd_render(a):
    from brdftoolbox.core import render as R
    plt = _plt()
    m = build_model(a.model, a.set)
    from brdftoolbox.core import lights as LT
    lights = [LT.parse(l) for l in a.light] if a.light else [LT.Light()]
    img = R.render_sphere(m, res=a.res, lights=lights, env=None if a.env == "none" else a.env, spp=a.spp,
                          exposure=a.exposure, maps=_maps(a, m))
    out = a.out or f"{a.model}_sphere.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.imsave(out, img)
    print("wrote", out)


def cmd_light(a):
    from brdftoolbox.core import lights as LT
    from brdftoolbox.core import plots as P
    L = LT.parse(a.light)
    print(LT.report(L))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        _plt()
        P.lighting_figure(L, build_model(a.model, a.set)).savefig(a.out, dpi=120)
        print("wrote", a.out)


def cmd_tabulate(a):
    from brdftoolbox.core.measured import save_merl, tabulate
    m = build_model(a.model, a.set)
    save_merl(a.out, tabulate(m))
    print("wrote", a.out)


def cmd_fit(a):
    from brdftoolbox.core import fitting as F
    from brdftoolbox.core import plots as P
    from brdftoolbox.core import render as R
    from brdftoolbox.core.measured import Measured
    if a.merl:
        data = Measured.from_file(a.merl)
    elif a.synthetic:
        truth = build_model(a.synthetic, a.set)
        data = Measured.from_model(truth, f"synthetic {a.synthetic}")
        if a.noise:
            rng = np.random.default_rng(a.seed)
            t = data.table
            data = Measured(np.where(t >= 0, t * np.exp(a.noise * rng.standard_normal(t.shape)), t), data.name)
        print("truth:", truth.describe())
    else:
        raise SystemExit("need --merl or --synthetic")
    names = list(F.FIT_PRESETS) if a.fit == "all" else a.fit.split(",")
    results = {}
    for n in names:
        r = F.fit_preset(n, data, loss=a.loss, n_samples=a.samples, n_starts=a.starts, seed=a.seed)
        results[n] = r
        print(f"\n[{n}]\n{r.summary()}")
    best = min(results, key=lambda k: results[k].rmse)
    print(f"\nbest: {best}  (loss rmse {results[best].rmse:.4f})")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        plt = _plt()
        k = len(results)
        fig, ax = plt.subplots(2, k + 1, figsize=(3.6 * (k + 1), 7), squeeze=False)
        ax[0, 0].imshow(R.render_sphere(data, res=160, env="studio", spp=32))
        ax[0, 0].set_title(data.name, fontsize=9)
        P.plot_half_diff(ax[1, 0], data, title="data log10 f")
        for i, (n, r) in enumerate(results.items(), 1):
            ax[0, i].imshow(R.render_sphere(r.model, res=160, env="studio", spp=32))
            ax[0, i].set_title(f"{n}  rmse {r.rmse:.3f}", fontsize=9)
            P.plot_fit_slices(ax[1, i], data, r.model)
        for x in ax[0]:
            x.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "fit_comparison.png"), dpi=120)
        with open(os.path.join(a.out, "fit_results.json"), "w") as fh:
            json.dump({n: {"model": r.model.describe(), "params": _json(r.params), "rmse": r.rmse,
                           "rel_rmse": r.rel_rmse, "seconds": r.seconds} for n, r in results.items()}, fh, indent=1)
        print("wrote", a.out)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, model=True, theta=False):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if model:
            p.add_argument("--model", default="cook-torrance")
            p.add_argument("--set", action="append", metavar="K=V")
        if theta:
            p.add_argument("--theta", type=float, default=45.0, help="incident/outgoing polar angle (deg)")
        return p

    add("list", cmd_list, model=False)
    add("plot", cmd_plot, theta=True).add_argument("--out", default="brdf_output")
    p = add("check", cmd_check, theta=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out")
    p = add("sample", cmd_sample, theta=True)
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--out")
    p = add("render", cmd_render)
    p.add_argument("--res", type=int, default=384)
    p.add_argument("--env", default="studio", choices=["studio", "uniform", "none"])
    p.add_argument("--spp", type=int, default=64)
    p.add_argument("--exposure", type=float, default=0.8)
    p.add_argument("--light", action="append", metavar="SPEC",
                   help="'θ,φ,E' or 'k=v,...' with keys of Light, e.g. kind=beam,distance=5,divergence=3")
    p.add_argument("--texture", choices=["checker", "uv", "stripes"])
    p.add_argument("--roughness-map", action="store_true")
    p.add_argument("--out")
    p = add("light", cmd_light)
    p.add_argument("--light", default="kind=directional")
    p.add_argument("--out")
    p = add("tabulate", cmd_tabulate)
    p.add_argument("--out", required=True)
    p = add("fit", cmd_fit)
    p.add_argument("--merl")
    p.add_argument("--synthetic", metavar="MODEL")
    p.add_argument("--noise", type=float, default=0.0, help="multiplicative log-normal noise σ for --synthetic")
    p.add_argument("--fit", default="plastic,ward-plastic,disney")
    p.add_argument("--loss", default="log", choices=["log", "cosine", "rel"])
    p.add_argument("--samples", type=int, default=20000)
    p.add_argument("--starts", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="fit_output")
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main() and 0)
