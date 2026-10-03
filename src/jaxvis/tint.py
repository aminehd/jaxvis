"""Per-point colour, so a transform stays visible.

    from jaxvis import tint

    hue = tint.by_angle(x)

A cloud with no colour only shows its silhouette, so anything that moves
points around inside an unchanged outline -- a rotation, a shuffle, a
whitening -- looks like nothing happened. Colour fixed to each point at the
start is what makes the motion readable.

Each function takes the input and returns an (n, 3) array to hand to
`colors=` on any of the drawing decorators.
"""
import numpy as np

from .render import ramp

__all__ = ["by_angle", "by_radius", "by_value", "by_flag"]


def _asnp(a):
    return np.asarray(a, dtype=float)


def by_angle(x, centre=None):
    """A colour wheel around the centroid.

    The classic companion to a linear map: the wheel is carried along, so you
    can see which part of the cloud went where.
    """
    p = _asnp(x)
    c = p.mean(0) if centre is None else np.asarray(centre, dtype=float)
    a = (np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]) + np.pi) / (2 * np.pi)
    w = np.stack([np.sin(np.pi * a), np.sin(np.pi * (a + 1 / 3)),
                  np.sin(np.pi * (a + 2 / 3))], 1) ** 2
    return w / w.max(1, keepdims=True)


def by_radius(x, palette="ultra", centre=None):
    "Colour by distance from the centroid -- rings rather than wedges."
    p = _asnp(x)
    c = p.mean(0) if centre is None else np.asarray(centre, dtype=float)
    return by_value(np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]), palette)


def by_value(v, palette="ultra"):
    "Map any per-point number through a palette."
    u = _asnp(v).ravel()
    u = (u - u.min()) / (np.ptp(u) + 1e-9)
    c = ramp(palette)[(u * 255).astype(int)] / 255.0
    return c / np.maximum(c.max(1, keepdims=True), 1e-9)


def by_flag(mask, yes=(1.0, 0.35, 0.5), no=(0.3, 0.8, 1.0)):
    "Two colours from a boolean -- outliers against everything else."
    m = np.asarray(mask).reshape(-1, 1)
    return np.where(m, np.asarray(yes, dtype=float), np.asarray(no, dtype=float))
