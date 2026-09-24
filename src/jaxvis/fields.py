"""A field is a function of coordinates, not an array.

    @jaxvis.draw_field(then=jnp.sin, palette="ultra")
    def blocks(x, y):
        return tiles.tumbling(x, y)

Writing f(x, y) instead of a precomputed array is what makes the knobs
possible: the framework can evaluate it wherever it likes -- on warped
coordinates, at a different t, composed with something else -- because it has
the function, not just its output.

    warp=g     f(g(x, y))        change WHERE it is sampled
    then=h     h(f(x, y))        change WHAT comes out
    t=n        f(x, y, t)        sweep t over n frames, looping
    scale=     "fixed" | "auto"  how colour maps onto value

`tiles` holds tessellations that survive being pushed through sin -- crisp
edges and repeated cells, so the deformation has something to bite on.
"""
import numpy as np

TAU = 2 * np.pi


# ------------------------------------------------------------------- tiles

class tiles:
    """Tessellations. Each is f(x, y) -> value on [-1, 1]^2."""

    @staticmethod
    def tumbling(x, y, k=5.0):
        "Isometric cubes: three faces meeting at 120 degrees."
        a = [np.sin(k * (x * np.cos(t) + y * np.sin(t)))
             for t in (0.0, TAU / 3, 2 * TAU / 3)]
        # whichever of the three directions is highest picks the face, which
        # is what gives the flat-shaded cube look rather than a ripple
        return np.argmax(np.stack(a), 0).astype(float) + 0.12 * sum(a)

    @staticmethod
    def pinwheel(x, y, k=4.0, turn=0.5):
        "Squares on a lattice, each rotated a little."
        cx, cy = np.floor(x * k), np.floor(y * k)
        a = turn * (cx + cy)                       # per-cell rotation
        u = (x * k % 1) - 0.5
        v = (y * k % 1) - 0.5
        ru = u * np.cos(a) - v * np.sin(a)
        rv = u * np.sin(a) + v * np.cos(a)
        return np.maximum(np.abs(ru), np.abs(rv))  # square distance -> squares

    @staticmethod
    def diagonal(x, y, k=6.0):
        "A checker turned 45 degrees -- diamonds."
        u, v = (x + y) * k, (x - y) * k
        return ((np.floor(u) + np.floor(v)) % 2) + 0.25 * (u % 1)

    @staticmethod
    def checker(x, y, k=6.0):
        "Plain squares."
        return (np.floor(x * k) + np.floor(y * k)) % 2

    @staticmethod
    def herringbone(x, y, k=5.0):
        "Interlocking bricks."
        cy = np.floor(y * k)
        u = x * k + 0.5 * cy                       # every row offset
        return ((np.floor(u) % 2) + (cy % 2)) % 2 + 0.3 * (u % 1)

    @staticmethod
    def rings(x, y, k=7.0):
        "Concentric, for comparison -- the one thing that is NOT a tessellation."
        return np.sin(k * np.hypot(x, y)) * (1 - np.hypot(x, y))


# ----------------------------------------------------------------- warps

class warps:
    """Coordinate transforms, for `warp=`. Each is g(x, y) -> (x', y')."""

    @staticmethod
    def swirl(x, y, k=1.6):
        a = k * np.hypot(x, y)
        return x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)

    @staticmethod
    def pinch(x, y, k=0.55):
        s = 1 + k * np.exp(-3 * (x * x + y * y))
        return x * s, y * s

    @staticmethod
    def ripple(x, y, k=0.16, f=9.0):
        r = np.hypot(x, y) + 1e-9
        d = k * np.sin(f * r)
        return x + d * x / r, y + d * y / r

    @staticmethod
    def shear(x, y, k=0.5):
        return x + k * np.sin(2.3 * y), y + k * np.sin(2.3 * x)


def grid(n=180, extent=1.0):
    "The coordinate pair every field is evaluated on."
    ax = np.linspace(-extent, extent, n)
    Y, X = np.meshgrid(ax, ax, indexing="ij")
    return X, Y
