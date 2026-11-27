"""MERL-format tabulated BRDFs: binary I/O, half/diff indexing, trilinear lookup, and tabulation of analytic models."""
import struct

import numpy as np

from brdftoolbox.core.frames import from_half_diff, to_half_diff
from brdftoolbox.core.models import BRDF, eval_rgb

RES = (90, 90, 180)
SCALE = np.array([1.0 / 1500.0, 1.15 / 1500.0, 1.66 / 1500.0])


def load_merl(path):
    with open(path, "rb") as fh:
        dims = struct.unpack("3i", fh.read(12))
        n = int(np.prod(dims))
        data = np.frombuffer(fh.read(), dtype="<f8", count=3 * n)
    if tuple(dims) != RES:
        raise ValueError(f"unexpected MERL dims {dims}")
    t = data.reshape(3, *RES).copy()
    return np.where(t < 0, -1.0, t * SCALE[:, None, None, None])


def save_merl(path, table):
    t = np.where(table < 0, -1.0, table / SCALE[:, None, None, None])
    with open(path, "wb") as fh:
        fh.write(struct.pack("3i", *RES))
        fh.write(np.ascontiguousarray(t, dtype="<f8").tobytes())


def grid_angles():
    i, j, k = (np.arange(r, dtype=float) for r in RES)
    return (i / RES[0]) ** 2 * (np.pi / 2), j / RES[1] * (np.pi / 2), k / RES[2] * np.pi


def continuous_index(th, td, pd):
    x = np.sqrt(np.clip(th, 0.0, np.pi / 2) / (np.pi / 2)) * RES[0]
    y = np.clip(td, 0.0, np.pi / 2) / (np.pi / 2) * RES[1]
    z = np.mod(pd, np.pi) / np.pi * RES[2]
    return x, y, z


def tabulate(model, chunk=90 * 180 * 10):
    """Evaluate a BRDF on the MERL grid. Returns (3, 90, 90, 180); entries below the horizon are -1."""
    th, td, pd = grid_angles()
    TH, TD, PD = np.meshgrid(th, td, pd, indexing="ij")
    wi, wo = from_half_diff(TH.ravel(), 0.0, TD.ravel(), PD.ravel())
    out = np.empty((wi.shape[0], 3))
    for s in range(0, wi.shape[0], chunk):
        out[s:s + chunk] = eval_rgb(model, wi[s:s + chunk], wo[s:s + chunk])
    valid = (wi[:, 2] > 1e-6) & (wo[:, 2] > 1e-6)
    out[~valid] = -1.0
    return out.T.reshape(3, *RES)


def valid_samples(table):
    """All valid grid entries as (wi, wo, f[n, 3])."""
    th, td, pd = grid_angles()
    idx = np.nonzero(np.all(table >= 0, axis=0))
    wi, wo = from_half_diff(th[idx[0]], 0.0, td[idx[1]], pd[idx[2]])
    ok = (wi[:, 2] > 1e-6) & (wo[:, 2] > 1e-6)
    return wi[ok], wo[ok], table[(slice(None),) + idx].T[ok]


class Measured(BRDF):
    key, label = "measured", "Measured"
    native_rgb = True

    def __init__(self, table, name="table"):
        super().__init__()
        self.table = np.asarray(table, float)
        self.name = name
        self._pad = np.concatenate([np.maximum(self.table, 0.0), np.maximum(self.table[..., :1], 0.0)], -1)

    @classmethod
    def from_file(cls, path):
        import os
        return cls(load_merl(path), os.path.splitext(os.path.basename(path))[0])

    @classmethod
    def from_model(cls, model, name=None):
        return cls(tabulate(model), name or model.label)

    def channels(self):
        return 3

    def eval(self, wi, wo):
        wi, wo = np.broadcast_arrays(wi, wo)
        th, _, td, pd = to_half_diff(wi, wo)
        x, y, z = continuous_index(th, td, pd)
        x = np.clip(x, 0.0, RES[0] - 1.0)
        y = np.clip(y, 0.0, RES[1] - 1.0)
        i0 = np.minimum(x.astype(int), RES[0] - 2)
        j0 = np.minimum(y.astype(int), RES[1] - 2)
        k0 = np.minimum(z.astype(int), RES[2] - 1)
        fx, fy, fz = (x - i0)[..., None], (y - j0)[..., None], (z - k0)[..., None]
        T = self._pad
        out = 0.0
        for di, wx in ((0, 1 - fx), (1, fx)):
            for dj, wy in ((0, 1 - fy), (1, fy)):
                for dk, wz in ((0, 1 - fz), (1, fz)):
                    out = out + wx * wy * wz * np.moveaxis(T[:, i0 + di, j0 + dj, k0 + dk], 0, -1)
        ok = ((wi[..., 2] > 0) & (wo[..., 2] > 0))[..., None]
        return np.where(ok, out, 0.0)

    __call__ = eval

    def describe(self):
        return f"Measured({self.name})"

    __repr__ = describe
