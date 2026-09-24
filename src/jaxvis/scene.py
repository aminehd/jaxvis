"""The composition layer — manim-flavoured, for training runs.

The auto-visualiser (`visualize`) is zero-effort but takes what it's given.
This is the other half: you say what you want, in order, with durations.

    from jaxvis import Scene, points, heatmap, flow

    with Scene("film.gif", size=560) as s:
        s.play(points(a).to(b), dur=40)
        s.play(heatmap(scores).reveal(), dur=24)
        s.play(flow(velocity, n=60_000), dur=90, trail=0.92)

A Clip is anything that yields (S, S) float canvases in [0, 1]. Because every
clip shares that one type, clips COMPOSE -- you can cross-fade a particle cloud
into a heatmap without either knowing the other exists.
"""
import numpy as np

from .reps import as_heatmap, as_points, as_strip, pick, _norm, _splat, _blank
from .render import colour, save_gif


def _ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


class Clip:
    """A sequence of canvases. Subclasses implement frame(t, S) for t in [0,1]."""

    label = ""

    def frame(self, t, S):
        raise NotImplementedError

    def render(self, dur, S):
        return [self.frame(i / max(1, dur - 1), S) for i in range(dur)]


# ------------------------------------------------------------------- clips

class _Morph(Clip):
    def __init__(self, a, b, rep=None, label=""):
        self.a, self.b = np.asarray(a), np.asarray(b)
        self.rep = rep or pick(self.a)[0]
        self.label = label

    def frame(self, t, S):
        e = _ease(t)
        return self.rep(self.a * (1 - e) + self.b * e, S)


class _Static(Clip):
    def __init__(self, a, rep=None, label="", reveal=False):
        self.a = np.asarray(a)
        self.rep = rep or pick(self.a)[0]
        self.label, self._reveal = label, reveal

    def frame(self, t, S):
        c = self.rep(self.a, S)
        return c * _ease(t) if self._reveal else c


class _Data:
    """Handle returned by points()/heatmap()/strip() — lets you chain .to()."""

    def __init__(self, a, rep, label):
        self.a, self.rep, self.label = np.asarray(a), rep, label

    def to(self, b, label=None):
        return _Morph(self.a, np.asarray(b), self.rep, label or self.label)

    def reveal(self):
        return _Static(self.a, self.rep, self.label, reveal=True)

    def hold(self):
        return _Static(self.a, self.rep, self.label)


def points(a, label="points"):
    return _Data(a, as_points, label)


def heatmap(a, label="heatmap"):
    return _Data(a, as_heatmap, label)


def strip(a, label="strip"):
    return _Data(a, as_strip, label)


class flow(Clip):
    """A particle field. `velocity(pts, t) -> dpts` — any JAX function.

    This is the one that looks alive: trails accumulate in a buffer that fades,
    so motion leaves a wake.
    """

    def __init__(self, velocity, n=40_000, seed=0, step=0.003, trail=0.9,
                 gain=2.0, extent=1.5, label="flow"):
        self.v, self.n, self.step = velocity, n, step
        self.trail, self.gain, self.extent = trail, gain, extent
        self.label = label
        rng = np.random.default_rng(seed)
        self.p0 = rng.uniform(-extent, extent, (n, 2))

    def render(self, dur, S, warm=40):
        import jax.numpy as jnp
        p = jnp.asarray(self.p0)
        buf = _blank(S)
        out = []
        for i in range(-warm, dur):                  # warm the trails first
            t = 2 * np.pi * i / dur
            p = p + self.v(p, t) * self.step
            p = jnp.where(jnp.abs(p) > self.extent * 1.05, -p * 0.97, p)
            buf *= self.trail
            q = np.asarray(p)
            _splat(q[:, 0] / self.extent * S * 0.46 + S / 2,
                   q[:, 1] / self.extent * S * 0.46 + S / 2, buf)
            if i >= 0:
                out.append(_norm(np.log1p(buf * self.gain)) ** 0.85)
        return out


# ------------------------------------------------------------------ scene

class Scene:
    def __init__(self, out="scene.gif", size=520, palette="ember",
                 fade=10, duration=55, caption=False):
        self.out, self.S, self.palette = out, size, palette
        self.fade, self.duration, self.caption = fade, duration, caption
        self.canvases, self.labels = [], []

    def play(self, clip, dur=36, **kw):
        new = clip.render(dur, self.S, **kw)
        if self.canvases and self.fade:              # cross-fade between clips
            a, b = self.canvases[-1], new[0]
            for i in range(self.fade):
                e = _ease((i + 1) / (self.fade + 1))
                self.canvases.append(a * (1 - e) + b * e)
                self.labels.append("")
        self.canvases += new
        self.labels += [clip.label] * len(new)
        return self

    def save(self):
        from PIL import ImageDraw
        frames = []
        for c, lbl in zip(self.canvases, self.labels):
            im = colour(c, self.palette)
            if self.caption and lbl:
                ImageDraw.Draw(im).text((12, self.S - 20), lbl, fill=(255, 190, 90))
            frames.append(im)
        return save_gif(frames, self.out, duration=self.duration)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            print(f"  {self.out}  <-  {len(self.canvases)} frames")
            self.save()
