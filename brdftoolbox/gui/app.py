"""PyQt5 GUI: model explorer, microfacet terms, energy/reciprocity, sampling, sphere render, MERL fitting."""
import argparse
import sys

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from brdftoolbox.core import analytics as A
from brdftoolbox.core import fitting as F
from brdftoolbox.core import lights as LT
from brdftoolbox.core import models as M
from brdftoolbox.core import plots as P
from brdftoolbox.core import render as R
from brdftoolbox.core.measured import Measured

GUI_MODELS = {
    "Cook-Torrance (microfacet)": lambda: M.CookTorrance(),
    "Plastic (Lambert + GGX)": lambda: M.plastic(),
    "Disney principled": lambda: M.Disney(),
    "Ward (anisotropic)": lambda: M.Ward(alpha_y=0.4),
    "Phong (normalized)": lambda: M.Phong(),
    "Blinn-Phong": lambda: M.BlinnPhong(),
    "Oren-Nayar": lambda: M.OrenNayar(),
    "Lambert": lambda: M.Lambert(),
    "Phong plastic": lambda: M.phong_plastic(),
    "Ward plastic": lambda: M.ward_plastic(),
    **{f"Metal: {k}": (lambda k=k: M.metal(k)) for k in ("gold", "silver", "copper", "aluminium")},
}


def options_of(model):
    if isinstance(model, M.Sum):
        return {f"{n}.{k}": v for n, c in model.c.items() for k, v in options_of(c).items()}
    return {k: (model.o[k], model.OPTIONS[k][1]) for k in model.o}


class FloatSlider(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self, label, lo, hi, init, steps=1000, fmt="{:.3f}", log=False):
        super().__init__()
        self.lo, self.hi, self.steps, self.fmt, self.log = lo, hi, steps, fmt, log
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.name = QtWidgets.QLabel(label)
        self.name.setMinimumWidth(120)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, steps)
        self.readout = QtWidgets.QLabel()
        self.readout.setMinimumWidth(56)
        lay.addWidget(self.name)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.readout)
        self.slider.valueChanged.connect(self._emit)
        self.setValue(init)

    def _t(self, v):
        return (np.log(v) - np.log(self.lo)) / (np.log(self.hi) - np.log(self.lo)) if self.log \
            else (v - self.lo) / (self.hi - self.lo)

    def value(self):
        t = self.slider.value() / self.steps
        return float(np.exp(np.log(self.lo) + t * (np.log(self.hi) - np.log(self.lo)))) if self.log \
            else self.lo + (self.hi - self.lo) * t

    def setValue(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(np.clip(self._t(v), 0, 1) * self.steps)))
        self.slider.blockSignals(False)
        self.readout.setText(self.fmt.format(self.value()))

    def _emit(self, _):
        self.readout.setText(self.fmt.format(self.value()))
        self.valueChanged.emit(self.value())


def _clamp(lo, hi, home):
    inv = lo > hi
    a, b = sorted((lo, hi))
    A, B = sorted(home)
    if b - a >= B - A:
        a, b = A, B
    elif a < A:
        a, b = A, A + (b - a)
    elif b > B:
        a, b = B - (b - a), B
    return (b, a) if inv else (a, b)


def limit_zoom(fig):
    """Keep every 2-D axes inside the extent it had when drawn: zooming out or panning stops at the data."""
    for ax in fig.axes:
        if ax.name not in ("rectilinear", "polar"):
            continue
        ax._home = {"x": ax.get_xlim(), "y": ax.get_ylim()}
        ax._busy = False

        def cb(a, axis):
            if a._busy:
                return
            get, set_ = (a.get_xlim, a.set_xlim) if axis == "x" else (a.get_ylim, a.set_ylim)
            cur = get()
            new = _clamp(*cur, a._home[axis])
            if not np.allclose(new, cur):
                a._busy = True
                set_(new)
                a._busy = False

        ax.callbacks.connect("xlim_changed", lambda a: cb(a, "x"))
        ax.callbacks.connect("ylim_changed", lambda a: cb(a, "y"))


class Canvas(QtWidgets.QWidget):
    def __init__(self, parent, figsize=(9, 7)):
        super().__init__()
        self.fig = Figure(figsize=figsize)
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(NavigationToolbar2QT(self.canvas, parent))
        lay.addWidget(self.canvas, 1)

    def finish(self):
        limit_zoom(self.fig)
        self.canvas.draw_idle()


def to_pixmap(img):
    a = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    h, w, _ = a.shape
    return QtGui.QPixmap.fromImage(QtGui.QImage(a.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy())


class FitWorker(QtCore.QThread):
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, data, presets, kw):
        super().__init__()
        self.data, self.presets, self.kw = data, presets, kw

    def run(self):
        try:
            self.done.emit({n: F.fit_preset(n, self.data, **self.kw) for n in self.presets})
        except Exception as e:
            self.failed.emit(repr(e))


class MainWindow(QtWidgets.QMainWindow):
    TABS = ("Lobes", "Microfacet terms", "Energy", "Sampling", "Render", "Fit", "Lighting")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BRDF Toolbox")
        self.resize(1480, 900)
        self.model = None
        self.data = None
        self.fits = {}
        self.worker = None
        self.dirty = set(range(len(self.TABS)))

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)
        self.c_lobe = Canvas(self)
        self.c_terms = Canvas(self)
        self.c_energy = Canvas(self)
        self.c_sampling = Canvas(self)
        for c, name in zip((self.c_lobe, self.c_terms), self.TABS[:2]):
            self.tabs.addTab(c, name)
        self.tabs.addTab(self._energy_tab(), "Energy")
        self.tabs.addTab(self.c_sampling, "Sampling")
        self.tabs.addTab(self._render_tab(), "Render")
        self.tabs.addTab(self._fit_tab(), "Fit")
        self.c_light = Canvas(self)
        self.tabs.addTab(self.c_light, "Lighting")
        self.tabs.currentChanged.connect(lambda _: self.refresh())

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(360)
        pl = QtWidgets.QVBoxLayout(panel)
        self.cb_model = QtWidgets.QComboBox()
        self.cb_model.addItems(GUI_MODELS)
        pl.addWidget(QtWidgets.QLabel("Model"))
        pl.addWidget(self.cb_model)
        self.s_theta = FloatSlider("θi (deg)", 0.0, 89.0, 45.0, fmt="{:.1f}")
        pl.addWidget(self.s_theta)
        self.param_box = QtWidgets.QGroupBox("Parameters")
        self.param_lay = QtWidgets.QVBoxLayout(self.param_box)
        pl.addWidget(self.param_box)
        self.opt_box = QtWidgets.QGroupBox("Options")
        self.opt_lay = QtWidgets.QFormLayout(self.opt_box)
        pl.addWidget(self.opt_box)
        self.chk_log = QtWidgets.QCheckBox("log NDF / log lobe")
        pl.addWidget(self.chk_log)
        self.readout = QtWidgets.QLabel()
        self.readout.setStyleSheet("font-family: monospace;")
        self.readout.setWordWrap(True)
        pl.addWidget(self.readout)
        btn = QtWidgets.QPushButton("Save current figure…")
        btn.clicked.connect(self.save_figure)
        pl.addWidget(btn)
        pl.addStretch(1)
        root.addWidget(panel)

        self.s_theta.valueChanged.connect(self.changed)
        self.chk_log.toggled.connect(self.changed)
        self.cb_model.currentTextChanged.connect(self.select_model)
        self.select_model(self.cb_model.currentText())

    def _energy_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.addWidget(self.c_energy, 3)
        self.report_text = QtWidgets.QPlainTextEdit(readOnly=True)
        self.report_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        lay.addWidget(self.report_text, 2)
        return w

    def _render_tab(self):
        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)
        left = QtWidgets.QVBoxLayout()
        self.img = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.img.setMinimumSize(420, 420)
        left.addWidget(self.img, 1)
        self.s_exp = FloatSlider("exposure", 0.1, 4.0, 0.8, fmt="{:.2f}")
        self.s_res = FloatSlider("resolution", 128, 512, 320, fmt="{:.0f}")
        row = QtWidgets.QHBoxLayout()
        self.cb_env = QtWidgets.QComboBox()
        self.cb_env.addItems(["none", "studio", "uniform"])
        self.sp_spp = QtWidgets.QSpinBox(minimum=1, maximum=1024, value=64)
        self.cb_tex = QtWidgets.QComboBox()
        self.cb_tex.addItems(["none"] + list(R.TEXTURES))
        self.chk_rmap = QtWidgets.QCheckBox("roughness map")
        for lab, wd in (("env", self.cb_env), ("env spp", self.sp_spp), ("texture", self.cb_tex)):
            row.addWidget(QtWidgets.QLabel(lab))
            row.addWidget(wd)
        row.addWidget(self.chk_rmap)
        row.addStretch(1)
        for s_ in (self.s_exp, self.s_res):
            left.addWidget(s_)
            s_.valueChanged.connect(self.changed_render)
        left.addLayout(row)
        for wd in (self.cb_env, self.cb_tex):
            wd.currentTextChanged.connect(self.changed_render)
        self.sp_spp.valueChanged.connect(self.changed_render)
        self.chk_rmap.toggled.connect(self.changed_render)
        outer.addLayout(left, 1)

        panel = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(panel)
        d = LT.Light()

        def combo(items, cur, tip):
            cb = QtWidgets.QComboBox()
            cb.addItems(items)
            cb.setCurrentText(cur)
            cb.setToolTip(tip)
            cb.currentTextChanged.connect(self.changed_light)
            return cb

        def slider(label, lo, hi, v, fmt="{:.3g}", log=False, tip=""):
            s_ = FloatSlider(label, lo, hi, v, fmt=fmt, log=log)
            s_.setToolTip(tip)
            s_.valueChanged.connect(self.changed_light)
            return s_

        g1 = QtWidgets.QGroupBox("Source geometry")
        l1 = QtWidgets.QVBoxLayout(g1)
        self.cb_kind = combo(list(LT.KINDS), d.kind,
                             "directional: distant source; point: isotropic emitter at a finite distance; "
                             "beam: Gaussian beam with waist and divergence")
        self.l_sl = {
            "theta": slider("θ (deg)", 0.0, 89.0, d.theta, "{:.0f}", tip="polar angle of the source from the view axis"),
            "phi": slider("φ (deg)", -180.0, 180.0, d.phi, "{:.0f}"),
            "intensity": slider("E0", 0.05, 20.0, d.intensity, log=True,
                                tip="normal irradiance at the sphere centre (on-axis for a beam)"),
            "distance": slider("distance", 1.05, 50.0, d.distance, log=True,
                               tip="source distance in sphere radii: per-point incidence and 1/r² falloff"),
            "radius": slider("emitter radius", 0.0, 2.0, d.radius,
                             tip="spherical emitter radius (sphere radii): soft highlights and penumbra"),
            "angular_radius": slider("angular radius°", 0.0, 30.0, d.angular_radius,
                                     tip="half-angle of a distant disk source (sun ≈ 0.27°)"),
            "waist": slider("beam waist w0", 0.01, 2.0, d.waist, log=True, tip="1/e² beam radius at the source"),
            "divergence": slider("divergence°", 0.0, 45.0, d.divergence,
                                 tip="beam half-angle divergence: w(z)² = w0² + (z tan β)²"),
            "aim_x": slider("aim x", -1.5, 1.5, 0.0, tip="beam target offset in the plane through the centre"),
            "aim_y": slider("aim y", -1.5, 1.5, 0.0),
        }
        l1.addWidget(self.cb_kind)
        for s_ in self.l_sl.values():
            l1.addWidget(s_)
        self.sp_lsamples = QtWidgets.QSpinBox(minimum=1, maximum=512, value=d.samples)
        self.sp_lsamples.setToolTip("directions sampled over the source's solid angle per pixel")
        self.sp_lsamples.valueChanged.connect(self.changed_light)
        r2 = QtWidgets.QHBoxLayout()
        r2.addWidget(QtWidgets.QLabel("area samples"))
        r2.addWidget(self.sp_lsamples)
        l1.addLayout(r2)
        pl.addWidget(g1)

        g2 = QtWidgets.QGroupBox("Spectrum")
        l2 = QtWidgets.QVBoxLayout(g2)
        self.cb_spec = combo(list(LT.SPECTRA), d.spectrum, "light colour and spectral width")
        self.l_sp = {
            "temperature": slider("temperature K", 1500.0, 12000.0, d.temperature, "{:.0f}"),
            "wavelength": slider("λ0 (nm)", 380.0, 780.0, d.wavelength, "{:.0f}"),
            "bandwidth": slider("Δλ FWHM (nm)", 0.001, 300.0, d.bandwidth, log=True,
                                tip="sets coherence length l_c = λ²/Δλ (laser ≲ 0.01 nm, LED ≈ 20 nm)"),
        }
        l2.addWidget(self.cb_spec)
        for s_ in self.l_sp.values():
            l2.addWidget(s_)
        pl.addWidget(g2)

        g3 = QtWidgets.QGroupBox("Coherence and polarization")
        l3 = QtWidgets.QVBoxLayout(g3)
        self.cb_coh = combo(list(LT.COHERENCE), d.coherence,
                            "incoherent: no interference. coherent: ideal laser, fully developed speckle "
                            "(contrast 1). partial: speckle averaged over bandwidth, source size and polarization")
        self.l_co = {
            "roughness_um": slider("surface σh (µm)", 0.01, 100.0, d.roughness_um, log=True,
                                   tip="rms surface height: path-length spread that decorrelates speckle"),
            "speckle_grain": slider("speckle grain px", 1.0, 16.0, d.speckle_grain, "{:.1f}",
                                    tip="mean speckle size ≈ λ/NA in pixels"),
        }
        self.chk_depol = QtWidgets.QCheckBox("depolarizing surface (M ×2)")
        self.chk_depol.toggled.connect(self.changed_light)
        self.cb_pol = combo(["unpolarized", "linear"], "unpolarized",
                            "linear: E-field angle about the incident direction; weights Rs/Rp in exact Fresnel "
                            "(dielectric/conductor materials)")
        self.s_pol = slider("E angle (deg)", 0.0, 180.0, 90.0, "{:.0f}", tip="0 = E in the light/view plane (p at the highlight), 90 = perpendicular (s)")
        l3.addWidget(self.cb_coh)
        for s_ in self.l_co.values():
            l3.addWidget(s_)
        l3.addWidget(self.chk_depol)
        l3.addWidget(self.cb_pol)
        l3.addWidget(self.s_pol)
        pl.addWidget(g3)
        self.light_info = QtWidgets.QLabel()
        self.light_info.setStyleSheet("font-family: monospace; font-size: 10px;")
        self.light_info.setWordWrap(True)
        pl.addWidget(self.light_info)
        pl.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(400)
        outer.addWidget(scroll)
        self.update_light_widgets()
        return w

    def current_light(self):
        kw = {k: s_.value() for k, s_ in {**self.l_sl, **self.l_sp, **self.l_co}.items()}
        return LT.Light(kind=self.cb_kind.currentText(), spectrum=self.cb_spec.currentText(),
                        coherence=self.cb_coh.currentText(), depolarizing=self.chk_depol.isChecked(),
                        polarization="unpolarized" if self.cb_pol.currentText() == "unpolarized" else self.s_pol.value(),
                        samples=self.sp_lsamples.value(), **kw)

    def update_light_widgets(self):
        k, sp, co = self.cb_kind.currentText(), self.cb_spec.currentText(), self.cb_coh.currentText()
        vis = {"distance": k != "directional", "radius": k == "point", "angular_radius": k == "directional",
               "waist": k == "beam", "divergence": k == "beam", "aim_x": k == "beam", "aim_y": k == "beam",
               "temperature": sp == "blackbody", "wavelength": sp == "gaussian", "bandwidth": sp == "gaussian",
               "roughness_um": co == "partial", "speckle_grain": co != "incoherent"}
        for name, s_ in {**self.l_sl, **self.l_sp, **self.l_co}.items():
            s_.setVisible(vis.get(name, True))
        self.chk_depol.setVisible(co != "incoherent")
        self.s_pol.setVisible(self.cb_pol.currentText() == "linear")
        self.light_info.setText(LT.report(self.current_light()))

    def changed_light(self, *_):
        self.update_light_widgets()
        self.dirty.update({4, 6})
        self.refresh()

    def _fit_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        self.c_fit = Canvas(self)
        lay.addWidget(self.c_fit, 3)
        side = QtWidgets.QVBoxLayout()
        b1 = QtWidgets.QPushButton("Load MERL .binary…")
        b1.clicked.connect(self.load_merl)
        b2 = QtWidgets.QPushButton("Use current model as data")
        b2.clicked.connect(self.synth_data)
        self.data_label = QtWidgets.QLabel("no data")
        self.data_label.setWordWrap(True)
        self.preset_list = QtWidgets.QListWidget()
        self.preset_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.preset_list.addItems(F.FIT_PRESETS)
        for i in range(3):
            self.preset_list.item(i).setSelected(True)
        form = QtWidgets.QFormLayout()
        self.sp_samples = QtWidgets.QSpinBox(minimum=1000, maximum=200000, value=10000, singleStep=1000)
        self.sp_starts = QtWidgets.QSpinBox(minimum=1, maximum=32, value=3)
        self.cb_loss = QtWidgets.QComboBox()
        self.cb_loss.addItems(["log", "cosine", "rel"])
        self.sp_noise = QtWidgets.QDoubleSpinBox(minimum=0.0, maximum=1.0, value=0.0, singleStep=0.05)
        form.addRow("samples", self.sp_samples)
        form.addRow("starts", self.sp_starts)
        form.addRow("loss", self.cb_loss)
        form.addRow("synthetic noise σ", self.sp_noise)
        self.btn_fit = QtWidgets.QPushButton("▶ Fit")
        self.btn_fit.clicked.connect(self.run_fit)
        self.fit_text = QtWidgets.QPlainTextEdit(readOnly=True)
        self.fit_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        for wd in (b1, b2, self.data_label, QtWidgets.QLabel("Fit models"), self.preset_list):
            side.addWidget(wd)
        side.addLayout(form)
        side.addWidget(self.btn_fit)
        side.addWidget(self.fit_text, 1)
        sw = QtWidgets.QWidget()
        sw.setLayout(side)
        sw.setFixedWidth(330)
        lay.addWidget(sw)
        return w

    def select_model(self, name):
        self.set_model(GUI_MODELS[name]())

    def set_model(self, model):
        self.model = model
        for lay in (self.param_lay, self.opt_lay):
            while lay.count():
                it = lay.takeAt(0)
                w = it.widget()
                if w:
                    w.setParent(None)
                    w.deleteLater()
        self.sliders = {}
        cur = model.params()
        for k, (d, lo, hi) in model.param_specs().items():
            v = cur[k]
            if np.ndim(v):
                self.param_lay.addWidget(QtWidgets.QLabel(f"{k} = {np.round(v, 3).tolist()}"))
                continue
            s = FloatSlider(k, lo, hi, float(v), log=lo > 0 and hi / lo >= 100)
            s.valueChanged.connect(self.param_changed)
            self.param_lay.addWidget(s)
            self.sliders[k] = s
        self.opts = {}
        for k, (v, choices) in options_of(model).items():
            cb = QtWidgets.QComboBox()
            cb.addItems([str(c) for c in choices])
            cb.setCurrentText(str(v))
            cb.currentTextChanged.connect(lambda t, k=k, ch=choices: self.option_changed(k, t, ch))
            self.opt_lay.addRow(k, cb)
            self.opts[k] = cb
        self.opt_box.setVisible(bool(self.opts))
        self.changed()

    def param_changed(self, *_):
        self.model = self.model.replace(**{k: s.value() for k, s in self.sliders.items()})
        self.changed()

    def option_changed(self, key, text, choices):
        val = next(c for c in choices if str(c) == text)
        self.set_model(self.model.replace(**{key: val}))

    def theta(self):
        return np.deg2rad(self.s_theta.value())

    def changed(self, *_):
        self.dirty = set(range(len(self.TABS)))
        self.update_readout()
        self.refresh()

    def changed_render(self, *_):
        self.dirty.add(4)
        self.refresh()

    def draw_lighting(self):
        self.c_light.fig.clear()
        P.lighting_figure(self.current_light(), self.model, fig=self.c_light.fig, res=128)
        self.c_light.finish()

    def update_readout(self):
        E, _ = A.directional_albedo(self.model, self.theta(), n=2048)
        rc = A.reciprocity_check(self.model, n=512)
        self.readout.setText(f"E(θi) = {np.round(E, 4)}\n"
                             f"reciprocity max rel err {rc['max_rel_err']:.1e}\n"
                             f"min f = {rc['min_value']:.2e}")

    def refresh(self):
        i = self.tabs.currentIndex()
        if i in self.dirty:
            self.dirty.discard(i)
            [self.draw_lobes, self.draw_terms, self.draw_energy, self.draw_sampling, self.draw_render,
             self.draw_fit, self.draw_lighting][i]()

    def draw_lobes(self):
        fig = self.c_lobe.fig
        fig.clear()
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25, left=0.06, right=0.97, top=0.93, bottom=0.07)
        ax = fig.add_subplot(gs[0, 0], projection="polar")
        log = self.chk_log.isChecked()
        P.plot_polar_lobe(ax, self.model, self.theta(), label="φ=0", log=log)
        P.plot_polar_lobe(ax, self.model, self.theta(), np.pi / 2, color=P.COLORS[1], label="φ=90°", log=log)
        ax.set_title("f·cosθo in two planes" + (" (log)" if log else ""), fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        P.plot_lobe_3d(fig.add_subplot(gs[0, 1], projection="3d"), self.model, self.theta())
        im = P.plot_half_diff(fig.add_subplot(gs[1, 0]), self.model)
        fig.colorbar(im, ax=fig.axes[-1], fraction=0.05)
        P.plot_half_diff(fig.add_subplot(gs[1, 1]), self.model, phi_d=0.0)
        self.c_lobe.finish()

    def draw_terms(self):
        fig = self.c_terms.fig
        fig.clear()
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25, left=0.07, right=0.97, top=0.93, bottom=0.08)
        p = self.model.params()
        f0 = float(np.mean(next((v for k, v in p.items() if k.endswith("f0")), 0.04)))
        eta = float(np.mean(next((v for k, v in p.items() if k.endswith("eta")), 1.5)))
        P.plot_ndf(fig.add_subplot(gs[0, 0]), self.model, log=self.chk_log.isChecked())
        P.plot_fresnel(fig.add_subplot(gs[0, 1]), f0, eta)
        P.plot_masking(fig.add_subplot(gs[1, 0]), self.model, self.theta())
        ax = fig.add_subplot(gs[1, 1])
        ct = P._ct(self.model) or M.CookTorrance()
        a = float(np.sqrt(np.prod([np.mean(x) for x in ct.alphas()])))
        from brdftoolbox.core import multiscatter as MS
        mu = np.linspace(0.02, 1, 100)
        for i, al in enumerate(sorted({0.05, 0.2, 0.5, 1.0, round(a, 3)})):
            ax.plot(np.rad2deg(np.arccos(mu)), MS.lookup_E(mu, al, ct.o["ndf"], ct.o["g2"]),
                    color=P.COLORS[i % 6], lw=2 if np.isclose(al, a) else 1, label=f"α={al}")
        ax.set_title("single-scatter E(θ), F=1 (Kulla–Conty table)", fontsize=9)
        ax.set_xlabel("θ (deg)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        self.c_terms.finish()

    def draw_energy(self):
        fig = self.c_energy.fig
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        models = {"model": self.model}
        ct = P._ct(self.model)
        if ct is not None and not ct.o["multiscatter"]:
            models["+ Kulla–Conty"] = self.model.replace(
                **{("specular." if isinstance(self.model, M.Sum) else "") + "multiscatter": True})
            if ct is self.model:
                models["F=1 (furnace)"] = ct.replace(fresnel="none")
                models["F=1 + Kulla–Conty"] = ct.replace(fresnel="none", multiscatter=True)
        P.plot_albedo(ax, models, n=2048)
        self.c_energy.finish()
        self.report_text.setPlainText(A.format_report(A.report(self.model, self.theta(), quick=True)))

    def draw_sampling(self):
        fig = self.c_sampling.fig
        fig.clear()
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        if self.model.native_rgb:
            self.c_sampling.finish()
            return
        P.plot_sampling(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), self.model, self.theta(), n=60_000)
        e = A.sampling_efficiency(self.model, self.theta(), n=1024, trials=16)
        ax = fig.add_subplot(gs[1, :])
        names = list(e)
        ax.bar(names, [max(e[k]["var"], 1e-12) for k in names], color=P.COLORS[:3])
        ax.set_yscale("log")
        ax.set_title("albedo-estimator variance × N (lower is better)", fontsize=9)
        self.c_sampling.finish()

    def render_maps(self):
        maps = {}
        names = self.model.params()
        t = self.cb_tex.currentText()
        if t != "none":
            key = next((k for k in ("albedo", "diffuse.albedo", "base_color") if k in names), None)
            if key:
                maps[key] = R.TEXTURES[t]
        if self.chk_rmap.isChecked():
            key = next((k for k in ("roughness", "specular.roughness") if k in names), None)
            if key:
                r0 = float(np.mean(names[key]))
                maps[key] = lambda u, v: np.where(R.checker(u, v, 8) > 0, r0, min(r0 * 3 + 0.1, 1.0))
        return maps

    def draw_render(self):
        env = self.cb_env.currentText()
        img = R.render_sphere(self.model, res=int(self.s_res.value()),
                              lights=[self.current_light()],
                              env=None if env == "none" else env, spp=self.sp_spp.value(),
                              exposure=self.s_exp.value(), maps=self.render_maps())
        self.last_render = img
        self.img.setPixmap(to_pixmap(img))

    def load_merl(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "MERL BRDF", "", "MERL (*.binary);;All (*)")
        if path:
            self.set_data(Measured.from_file(path))

    def synth_data(self):
        d = Measured.from_model(self.model, f"synthetic {self.cb_model.currentText()}")
        s = self.sp_noise.value()
        if s > 0:
            t = d.table
            n = np.random.default_rng(0).standard_normal(t.shape)
            d = Measured(np.where(t >= 0, t * np.exp(s * n), t), d.name + f" σ={s}")
        self.set_data(d)

    def set_data(self, d):
        self.data = d
        self.fits = {}
        self.data_label.setText(f"data: {d.name}")
        self.dirty.add(5)
        self.refresh()

    def run_fit(self):
        if self.data is None or (self.worker and self.worker.isRunning()):
            return
        presets = [i.text() for i in self.preset_list.selectedItems()]
        kw = dict(n_samples=self.sp_samples.value(), n_starts=self.sp_starts.value(), loss=self.cb_loss.currentText())
        self.fit_text.setPlainText(f"fitting {', '.join(presets)} …")
        self.btn_fit.setEnabled(False)
        self.worker = FitWorker(self.data, presets, kw)
        self.worker.done.connect(self.fit_done)
        self.worker.failed.connect(lambda m: (self.fit_text.setPlainText(m), self.btn_fit.setEnabled(True)))
        self.worker.start()

    def fit_done(self, results):
        self.fits = results
        self.btn_fit.setEnabled(True)
        txt = "\n\n".join(f"[{n}]\n{r.summary()}" for n, r in results.items())
        best = min(results, key=lambda k: results[k].rmse)
        self.fit_text.setPlainText(f"best: {best}\n\n{txt}")
        self.dirty.add(5)
        self.refresh()

    def draw_fit(self):
        fig = self.c_fit.fig
        fig.clear()
        if self.data is None:
            self.c_fit.finish()
            return
        k = len(self.fits)
        gs = fig.add_gridspec(2, k + 1, hspace=0.3, wspace=0.3)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(R.render_sphere(self.data, res=160, env="studio", spp=16))
        ax.set_title(self.data.name, fontsize=8)
        ax.axis("off")
        P.plot_half_diff(fig.add_subplot(gs[1, 0]), self.data, title="data log10 f")
        for i, (n, r) in enumerate(self.fits.items(), 1):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(R.render_sphere(r.model, res=160, env="studio", spp=16))
            ax.set_title(f"{n}\nloss rmse {r.rmse:.3f}", fontsize=8)
            ax.axis("off")
            P.plot_fit_slices(fig.add_subplot(gs[1, i]), self.data, r.model)
        self.c_fit.finish()

    def save_figure(self):
        i = self.tabs.currentIndex()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save", f"{self.TABS[i].lower()}.png", "PNG (*.png)")
        if not path:
            return
        if i == 4:
            import matplotlib.pyplot as plt
            plt.imsave(path, self.last_render)
        else:
            {0: self.c_lobe, 1: self.c_terms, 2: self.c_energy, 3: self.c_sampling, 5: self.c_fit,
             6: self.c_light}[i].fig.savefig(
                path, dpi=150)


def main(argv=None):
    ap = argparse.ArgumentParser(description="BRDF toolbox GUI")
    ap.add_argument("--merl", help="MERL .binary to load into the Fit tab")
    a = ap.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    win = MainWindow()
    if a.merl:
        win.load_merl(a.merl)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    main()
