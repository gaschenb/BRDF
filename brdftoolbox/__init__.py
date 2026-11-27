"""BRDF toolbox: analytic and measured BRDFs, physical checks, importance sampling, fitting, rendering."""

from brdftoolbox.core.models import (BRDF, Lambert, OrenNayar, Phong, BlinnPhong, Ward, CookTorrance, Disney,
                                     Sum, plastic, metal, make, eval_rgb, MODELS, PRESETS)
from brdftoolbox.core.measured import Measured, load_merl, save_merl, tabulate
from brdftoolbox.core.analytics import (directional_albedo, hemispherical_albedo, energy_check,
                                        reciprocity_check, sampler_chi2, sampling_efficiency, report)
from brdftoolbox.core.fitting import fit, fit_preset, FIT_PRESETS
from brdftoolbox.core.render import render_sphere

__version__ = "2.0.0"
__all__ = [
    "BRDF", "Lambert", "OrenNayar", "Phong", "BlinnPhong", "Ward", "CookTorrance", "Disney", "Sum",
    "plastic", "metal", "make", "eval_rgb", "MODELS", "PRESETS",
    "Measured", "load_merl", "save_merl", "tabulate",
    "directional_albedo", "hemispherical_albedo", "energy_check", "reciprocity_check",
    "sampler_chi2", "sampling_efficiency", "report",
    "fit", "fit_preset", "FIT_PRESETS", "render_sphere",
]
