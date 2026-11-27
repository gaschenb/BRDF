import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")


def test_gui_tabs_and_fit(tmp_path):
    from PyQt5 import QtCore
    from brdftoolbox.gui import app as gui
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = gui.MainWindow()
    win.show()
    for name in list(gui.GUI_MODELS)[:4]:
        win.cb_model.setCurrentText(name)
        for i in range(len(win.TABS) - 1):
            win.tabs.setCurrentIndex(i)
            qapp.processEvents()
    win.tabs.setCurrentIndex(4)
    win.cb_kind.setCurrentText("beam")
    win.cb_spec.setCurrentText("gaussian")
    win.cb_coh.setCurrentText("partial")
    win.cb_pol.setCurrentText("linear")
    qapp.processEvents()
    L = win.current_light()
    assert L.kind == "beam" and L.coherence == "partial" and isinstance(L.polarization, float)
    win.tabs.setCurrentIndex(6)
    qapp.processEvents()
    win.tabs.setCurrentIndex(1)
    qapp.processEvents()
    ax = win.c_terms.fig.axes[0]
    x0, y0 = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(x0[0] - 50, x0[1] + 50)
    ax.set_ylim(y0[0] - 10, y0[1] + 10)
    assert np.allclose(ax.get_xlim(), x0) and np.allclose(ax.get_ylim(), y0)
    w = x0[1] - x0[0]
    ax.set_xlim(x0[1] - 0.2 * w, x0[1] + 0.2 * w)
    assert np.isclose(ax.get_xlim()[1], x0[1]) and np.isclose(ax.get_xlim()[1] - ax.get_xlim()[0], 0.4 * w)
    win.cb_model.setCurrentText("Cook-Torrance (microfacet)")
    win.sliders["roughness"].setValue(0.5)
    win.sliders["roughness"].valueChanged.emit(0.5)
    win.opts["ndf"].setCurrentText("beckmann")
    assert win.model.o["ndf"] == "beckmann"
    win.cb_model.setCurrentText("Plastic (Lambert + GGX)")
    win.synth_data()
    for i in range(win.preset_list.count()):
        win.preset_list.item(i).setSelected(win.preset_list.item(i).text() == "plastic")
    win.sp_samples.setValue(3000)
    win.sp_starts.setValue(1)
    win.tabs.setCurrentIndex(5)
    win.run_fit()
    t = QtCore.QElapsedTimer()
    t.start()
    while win.worker.isRunning() and t.elapsed() < 120_000:
        qapp.processEvents()
        QtCore.QThread.msleep(20)
    qapp.processEvents()
    assert "plastic" in win.fits and win.fits["plastic"].rmse < 1e-3
    win.grab().save(str(tmp_path / "fit.png"))
    win.close()
