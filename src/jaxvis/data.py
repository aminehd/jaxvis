"""Ready-made inputs, so an example can start at the formula.

    from jaxvis import data

    x = data.tilted(40_000)

Every generator takes `n` and a `seed` and returns an (n, 2) jnp array.
Nothing here is clever -- it is the three lines of `rng.something` that
otherwise open every example and tell the reader nothing.
"""
import numpy as np

import jax.numpy as jnp

__all__ = ["tilted", "blob", "ring", "squares", "uniform", "bar", "grid_points"]


def _rng(seed):
    return np.random.default_rng(seed)


def tilted(n=40_000, mean=(0.9, -0.5), cov=((2.2, 1.6), (1.6, 1.4)), seed=0):
    """A correlated Gaussian -- off-centre and stretched along a diagonal.

    The usual "raw data" of a preprocessing example: whitening, PCA,
    standardising all have something to undo here.
    """
    return jnp.asarray(_rng(seed).multivariate_normal(mean, cov, n))


def blob(n=40_000, seed=0):
    "A plain isotropic Gaussian, centred at the origin."
    return jnp.asarray(_rng(seed).normal(size=(n, 2)))


def ring(n=40_000, r=1.6, width=0.12, seed=0):
    "An annulus. Has a hole, so anything that fills it is obvious."
    g = _rng(seed)
    th = g.uniform(0, 2 * np.pi, n)
    rad = r + g.normal(scale=width, size=n)
    return jnp.asarray(np.stack([rad * np.cos(th), rad * np.sin(th)], 1))


def squares(n=40_000, k=3, spread=2.6, size=0.85, seed=0):
    """`k` square clouds on a circle.

    Broad flat-sided shapes survive being summed with noise, which a finely
    detailed start does not -- useful whenever something should dissolve
    slowly enough to watch.
    """
    g = _rng(seed)
    th = np.arange(k) * 2 * np.pi / k + np.pi / 2
    centres = np.stack([np.cos(th), np.sin(th)], 1) * spread
    return jnp.asarray(centres[g.integers(0, k, n)]
                       + g.uniform(-size, size, size=(n, 2)))


def uniform(n=40_000, extent=1.0, seed=0):
    "A filled square. Flat density, so a transform's stretching shows."
    return jnp.asarray(_rng(seed).uniform(-extent, extent, size=(n, 2)))


def bar(n=40_000, length=2.2, height=0.45, seed=0):
    "A long thin rectangle. Anisotropic, so rotations are visible."
    g = _rng(seed)
    return jnp.asarray(g.uniform(-1, 1, size=(n, 2)) * np.array([length, height]))


def grid_points(n=40_000, extent=1.0, jitter=0.0, seed=0):
    "A square lattice of roughly n points, optionally jittered."
    side = int(np.sqrt(n))
    g = np.linspace(-extent, extent, side)
    xs, ys = np.meshgrid(g, g)
    p = np.stack([xs.ravel(), ys.ravel()], 1)
    if jitter:
        p = p + _rng(seed).normal(scale=jitter, size=p.shape)
    return jnp.asarray(p)
