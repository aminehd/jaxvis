"""Let the framework decide how to animate a formula.

    from jaxvis import explore
    explore(lambda t: jnp.tanh((J - I - t) * 0.05), out="f.gif")

Given f(t) -> array, it:
  1. probes t over a wide range and measures how much the output moves
  2. picks the window where something actually happens
  3. picks a representation from the output's shape
  4. warns if the output is boolean-ish, because that can only snap
  5. renders with the formula and the live value beside it

What it CANNOT decide is how much to soften a hard comparison -- it can only
tell you that you need to.
"""
import inspect

import numpy as np

from .reps import pick
from .render import colour, save_gif, linear_mean, smoothstep, _font


def probe(f, lo=-3.0, hi=3.0, n=48):
    """How much does the output move as t sweeps? Returns (ts, movement)."""
    ts = np.linspace(lo, hi, n)
    outs = [np.asarray(f(float(t)), dtype=float) for t in ts]
    mv = [float(np.abs(outs[i + 1] - outs[i]).mean()) for i in range(len(outs) - 1)]
    return ts, np.array(mv), outs


def discreteness(a):
    """0 = continuous, 1 = only takes a couple of distinct values."""
    a = np.asarray(a, dtype=float).ravel()
    u = np.unique(np.round(a, 6))
    return 1.0 if len(u) <= 3 else max(0.0, 1.0 - len(u) / min(64, a.size))


def choose_window(ts, mv, keep=0.80):
    """The sub-range containing most of the movement -- skip the dead ends."""
    c = np.cumsum(mv)
    if c[-1] <= 0:
        return ts[0], ts[-1]
    lo_i = int(np.searchsorted(c, c[-1] * (1 - keep) / 2))
    hi_i = int(np.searchsorted(c, c[-1] * (1 + keep) / 2))
    return float(ts[lo_i]), float(ts[min(hi_i + 1, len(ts) - 1)])


def _panel(lines, live, W, H, fs=15):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (W, H), (16, 16, 22))
    d = ImageDraw.Draw(im)
    y = 18
    for ln in lines:
        d.text((16, y), ln, font=_font(fs), fill=(198, 202, 220))
        y += fs + 7
    y += 10
    for k, v in live.items():
        d.text((16, y), f"{k} = {v}", font=_font(fs + 1), fill=(255, 176, 60))
        y += fs + 9
    return im


def explore(f, out="explore.gif", lo=-3.0, hi=3.0, n=56, size=460,
            samples=5, palette="ultra", rep=None, formula=None, verbose=True):
    ts, mv, outs = probe(f, lo, hi)
    a, b = choose_window(ts, mv)
    render_fn, rname = pick(outs[len(outs) // 2], rep)
    disc = discreteness(outs[len(outs) // 2])

    if verbose:
        print(f"  probed t in [{lo}, {hi}]  -> active window [{a:.2f}, {b:.2f}]")
        print(f"  representation: {rname}   discreteness {disc:.2f}")
        if disc > 0.6:
            print("  ! output is near-boolean -- it will SNAP between states.")
            print("    soften the comparison, e.g. tanh((x - t) * k), to get motion.")

    src = formula
    if src is None:
        try:
            src = inspect.getsource(f).strip().splitlines()
        except Exception:
            src = ["<formula unavailable>"]
    if isinstance(src, str):
        src = src.splitlines()

    frames = []
    for i in range(n):
        u = i / n
        t = a + (b - a) * (0.5 - 0.5 * np.cos(2 * np.pi * u)) if True else 0
        subs = [render_fn(np.asarray(f(float(
            a + (b - a) * (0.5 - 0.5 * np.cos(2 * np.pi * (u + s / samples / n)))
        )), dtype=float), size) for s in range(samples)]
        viz = colour(linear_mean(subs), palette)
        pan = _panel(src, {"t": f"{t:8.3f}"}, int(size * 1.05), size)
        from PIL import Image
        both = Image.new("RGB", (pan.size[0] + 8 + size, size), (16, 16, 22))
        both.paste(pan, (0, 0)); both.paste(viz, (pan.size[0] + 8, 0))
        frames.append(both)

    p = save_gif(frames, out, duration=60, colors=64)
    if verbose:
        print(f"  -> {p}  ({len(frames)} frames)")
    return p
