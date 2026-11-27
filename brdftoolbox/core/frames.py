"""Directions, local frames and Rusinkiewicz half/difference coordinates. Local frame: n = +z, t = +x."""
import numpy as np

EPS = 1e-9


def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), EPS)


def sph_dir(theta, phi=0.0):
    theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=-1)


def to_spherical(w):
    return np.arccos(np.clip(w[..., 2], -1.0, 1.0)), np.arctan2(w[..., 1], w[..., 0])


def reflect(w):
    return w * np.array([-1.0, -1.0, 1.0])


def half_vector(wi, wo):
    return normalize(wi + wo)


def rot_z(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


def rot_y(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * v[..., 0] + s * v[..., 2], v[..., 1], -s * v[..., 0] + c * v[..., 2]], -1)


def to_half_diff(wi, wo):
    h = half_vector(wi, wo)
    th, ph = to_spherical(h)
    d = rot_y(rot_z(wi, -ph), -th)
    td, pd = to_spherical(d)
    return th, ph, td, pd


def from_half_diff(th, ph, td, pd):
    h = sph_dir(th, ph)
    wi = rot_z(rot_y(sph_dir(td, pd), th), ph)
    wo = 2.0 * np.sum(wi * h, -1, keepdims=True) * h - wi
    return wi, wo


def basis(n):
    """Branchless orthonormal basis (Duff et al. 2017). Returns t, b with (t, b, n) right-handed."""
    s = np.where(n[..., 2] >= 0, 1.0, -1.0)
    a = -1.0 / (s + n[..., 2])
    b = n[..., 0] * n[..., 1] * a
    t = np.stack([1.0 + s * n[..., 0] ** 2 * a, s * b, -s * n[..., 0]], -1)
    bb = np.stack([b, s + n[..., 1] ** 2 * a, -n[..., 1]], -1)
    return t, bb


def frame_from_tangent(n, t_hint):
    t = normalize(t_hint - np.sum(t_hint * n, -1, keepdims=True) * n)
    bad = np.linalg.norm(t_hint - np.sum(t_hint * n, -1, keepdims=True) * n, axis=-1) < 1e-6
    if np.any(bad):
        t0, _ = basis(n)
        t = np.where(bad[..., None], t0, t)
    return t, np.cross(n, t)


def to_local(v, t, b, n):
    return np.stack([np.sum(v * t, -1), np.sum(v * b, -1), np.sum(v * n, -1)], -1)


def to_world(v, t, b, n):
    return v[..., 0:1] * t + v[..., 1:2] * b + v[..., 2:3] * n


def cosine_hemisphere(u1, u2):
    r, phi = np.sqrt(u1), 2.0 * np.pi * u2
    return np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(np.maximum(1.0 - u1, 0.0))], -1)


def uniform_hemisphere(u1, u2):
    z, phi = u1, 2.0 * np.pi * u2
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)


def lobe_sample(axis, u1, u2, n):
    """Sample w ∝ max(0, w·axis)^n. Returns w, pdf."""
    c = u1 ** (1.0 / (np.asarray(n, float) + 1.0))
    s = np.sqrt(np.maximum(1.0 - c * c, 0.0))
    phi = 2.0 * np.pi * u2
    loc = np.stack([s * np.cos(phi), s * np.sin(phi), c], -1)
    t, b = basis(axis)
    return to_world(loc, t, b, axis), (n + 1.0) / (2.0 * np.pi) * c ** n
