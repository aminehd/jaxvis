"""jaxvis — animate any JAX function by interpreting its jaxpr.

    from jaxvis import visualize
    visualize(my_fn, x, y, out="film.gif")

or as a decorator that also runs the function normally:

    from jaxvis import animate

    @animate("attention.gif")
    def attention(Q, K, V):
        ...
"""
import functools
import inspect
import numpy as np

from .interp import trace_values
from .reps import pick
from .render import (morph, save_gif, colour, code_panel, side_by_side,
                     linear_mean, smoothstep, blurred, ease, wave_ease, phase_field)
from .scene import Scene, points, heatmap, strip, flow
from .explore import explore, probe, discreteness

__all__ = ["visualize", "animate", "draw", "draw_field", "draw_sim",
           "output_to",
           "trace_values", "pick",
           "morph", "save_gif", "colour", "code_panel", "side_by_side",
           "Scene", "points", "heatmap", "strip", "flow",
           "explore", "probe", "discreteness"]
__version__ = "0.1.0"


def visualize(f, *args, out="jaxvis.gif", size=460, hold=6, tween=14,
              palette="ember", caption=True, verbose=True, value_morph=True,
              structural=False, only=None, code=False, rep=None,
              samples=8, shutter=1.5, ops=None, link=8,
              stagger="auto", spread=0.65, g=2.0, rep_kw=None,
              colors=None, blend="mean", frame="auto", gain=None, **kw):
    """Run `f`, draw every intermediate, write a GIF. Returns the path.

    value_morph=True animates each OPERATION: when an input shares the output's
    shape, the data is interpolated (`lerp(src, out, t)`) and re-rendered, so the
    structure travels. Falls back to a canvas cross-fade when shapes differ
    (a matmul, say), and for the very first frame.
    """
    steps = trace_values(f, *args, structural=structural)
    rk = dict(rep_kw or {})
    if gain is not None:        # a renderer option, not a gif one --
        rk["gain"] = gain       # without this it lands in save_gif()

    def _opts(fn):
        o = _opts_raw(fn)
        if box is not None and getattr(fn, "__name__", "") in (
                "as_points", "fn"):
            o = {**o, "frame": box}
        return o

    def _opts_raw(fn):
        """rep_kw filtered to what THIS renderer accepts.

        A function's steps rarely all render the same way -- a matrix draws as
        a heatmap, its row-means as a strip -- and an option meant for one is a
        TypeError on the other. Pass each renderer only what it understands.
        """
        if not rk:
            return {}
        try:
            names = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return rk
        if any(p.kind == p.VAR_KEYWORD for p in names.values()):
            return rk
        return {k: v for k, v in rk.items() if k in names}
    steps = [s for s in steps if pick(s.out, rep, s.name)[1] != "skip"]
    if only:
        want = {only} if isinstance(only, str) else set(only)
        steps = [s for s in steps if pick(s.out, rep, s.name)[1] in want]
    if ops:
        # keep only these primitives. a loop body is usually ONE op (the update)
        # surrounded by dozens of intermediates -- this is how you get just it.
        want_ops = {ops} if isinstance(ops, str) else set(ops)
        steps = [s for s in steps if s.name in want_ops or s.name == "input"]
    if not steps:
        raise ValueError(
            f"nothing left to draw: only={only!r} ops={ops!r} filtered every "
            f"step away. `only` matches REP names (points, heatmap, liquid, "
            f"strip, batch, warp); `ops` matches PRIMITIVE names (mul, add, "
            f"sin...). Run with verbose=True to see what this function has.")
    # One window for every point cloud in the clip, unless frame="auto".
    # Re-framing each step would cancel out translations and uniform scalings
    # -- a subtract-the-mean or divide-by-a-constant step would draw as
    # nothing happening at all.
    # frame="auto"  (default) each cloud re-centred and re-scaled to fill the
    #                canvas -- tidy, but a shift or uniform scale is invisible
    # frame="fixed"  one window for the whole clip, so x - mu slides and x / 2
    #                shrinks, and you see true relative size
    # frame=(cx, cy, r)  a window you choose
    box = None
    if frame in ("fixed", "clip"):
        pts = [np.asarray(q)[:, :2] for s_ in steps
               for q in (s_.out, s_.src) if q is not None
               and np.ndim(q) == 2 and np.shape(q)[1] >= 2
               and pick(np.asarray(q), rep, s_.name)[1] == "points"]
        if pts:
            allp = np.concatenate(pts)
            lo, hi = np.percentile(allp, 0.5, 0), np.percentile(allp, 99.5, 0)
            mid = (lo + hi) / 2
            box = (float(mid[0]), float(mid[1]),
                   float(np.max(hi - lo) / 2 * 1.08) + 1e-9)
    elif isinstance(frame, tuple):
        box = frame

    canvases, labels, all_sites = [], [], []

    for s in steps:
        render_fn, rname = pick(s.out, rep, s.name)
        # colors= gives every point its own RGB. Only point steps can use it;
        # a heatmap has no per-point anything, so those stay on the palette.
        if colors is not None and rname == "points":
            from .reps import coloured_points
            render_fn = coloured_points(colors, blend=blend)
        # Scale is INTERPOLATED alongside the values. A fixed union range makes
        # one end of a morph render nearly black when the two ends differ in
        # magnitude; per-frame auto-scaling makes brightness flicker. Sliding
        # the range is the middle path.
        def _r(x):
            x = np.ravel(x)
            return float(np.nanmin(x)), float(np.nanmax(x))
        vr_out = _r(s.out)
        vr_src = _r(s.src) if s.src is not None else vr_out

        def vr_at(e):
            return (vr_src[0] * (1 - e) + vr_out[0] * e,
                    vr_src[1] * (1 - e) + vr_out[1] * e)
        vr = vr_out
        label = f"{s.name}  {tuple(s.out.shape)}  [{rname}]"
        sites = []
        morphing = (value_morph and s.src is not None
                    and s.src.shape == s.out.shape)
        # Hold the colour scale FIXED across the morph. Without a vrange every
        # frame renormalises to its own min/max -- auto-exposure -- which both
        # lies about the values and re-synchronises the wave, flattening the
        # phase offset back into a crossfade.
        # The scale INTERPOLATES with the values (vr_at). A fixed union range
        # renders one end nearly black whenever the two ends differ wildly --
        # div spans 0..1 while its source spans 1..34. Interpolating keeps
        # every frame exposed, and because vr_at(0) == vr_src and vr_at(1) ==
        # vr_out, the entry fade and the exit frame line up exactly: no cut.
        # Between consecutive ops the arrays are unrelated, so the picture cuts.
        # Fade the last canvas into this step's first one -- rendered at
        # vr_morph, NOT vr_out, or the fade lands on one normalisation and the
        # morph starts on another. Same data, different brightness, hard cut:
        # it reads as the animation jumping back to the start of the step.
        if link and canvases:
            first = render_fn(s.src if morphing else s.out, size,
                              vr_at(0.0) if morphing else vr,
                              **_opts(render_fn))
            prev = canvases[-1]
            for i in range(link):
                e = smoothstep((i + 1) / (link + 1))
                canvases.append(prev * (1 - e) + first * e)
                labels.append(label)
                all_sites.append(s.site)
        if morphing:
            # the operation IS the animation
            dt = 1.0 / tween
            # beesandbombs phase offset: a grid eased in lockstep is a
            # crossfade; the identical motion offset per element is a wave.
            # Point clouds are already sparse and read fine in lockstep --
            # it is the solid grids that look like a slide transition.
            kind = ({"heatmap": "radial", "liquid": "noise",
                     "strip": "diagonal"}.get(rname) if stagger == "auto"
                    else (None if stagger in (None, False) else stagger))
            ph = phase_field(s.out, kind) if kind else None
            for t_ in np.linspace(0, 1, tween, endpoint=False):
                # motion blur: several sub-frames across the shutter, averaged
                # in linear light (beesandbombs)
                canvases.append(blurred(
                    lambda u: render_fn(
                        s.src * (1 - wave_ease(u, ph, spread, g))
                        + s.out * wave_ease(u, ph, spread, g),
                        size, vr_at(ease(u, g)), **_opts(render_fn)),
                    t_, dt, samples, shutter))
                labels.append(label)
                sites.append(s.site)
        # ...and leave on the same normalisation the morph ended on, or the
        # last morph frame and this one show identical data at different
        # brightness -- the same hard cut, at the other end of the step.
        canvases.append(render_fn(s.out, size,
                                  vr_at(1.0) if morphing else vr,
                                  **_opts(render_fn)))
        labels.append(label)
        sites.append(s.site)
        for _ in range(hold - 1):
            canvases.append(canvases[-1])
            labels.append(label)
            sites.append(s.site)
        all_sites.extend(sites)

    if verbose:
        seen = []
        for s in steps:
            seen.append(f"{s.name}  {tuple(s.out.shape)}  "
                        f"[{pick(s.out, rep, s.name)[1]}]"
                        f"{'  value-morph' if (value_morph and s.src is not None) else ''}")
        for l in seen:
            print("  " + l)
        print(f"  {len(steps)} steps")

    tw = 1 if value_morph else tween
    frames = morph(canvases, labels, hold=1, tween=tw,
                   palette=palette, caption=caption)

    if code:
        # pair every frame with the source line that produced it
        sites = all_sites + [all_sites[-1]] * (len(frames) - len(all_sites))
        paired = []
        for fr, site in zip(frames, sites):
            path, _, ln = (site or "").rpartition(":")
            panel = code_panel(path, int(ln) if ln.isdigit() else None,
                               int(size * 1.15), size)
            paired.append(side_by_side(panel, fr))
        frames = paired

    p = save_gif(frames, out, **kw)
    if verbose:
        print(f"  -> {p}  ({len(frames)} frames)")
    return p


def animate(out="jaxvis.gif", **vopts):
    """Decorator. The function still runs and returns normally."""
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if kwargs:
                raise TypeError("jaxvis.animate: positional args only")
            visualize(f, *args, out=out, **vopts)
            return f(*args)
        return wrapper
    return deco


# ----------------------------------------------------------- page-level output

_OUT = [None]


def output_to(directory):
    """Send every later `@draw` into this directory. Set it once per file.

        jaxvis.output_to("renders/")

        @jaxvis.draw(X, palette="ultra")
        def sigmoid(x):
            return 1 / (1 + jnp.exp(-x))     # -> renders/sigmoid.gif

    Saves repeating a path in every decorator, and keeps the decorators
    showing only what the picture actually depends on.
    """
    from pathlib import Path as _P
    d = _P(directory)
    d.mkdir(parents=True, exist_ok=True)
    _OUT[0] = d
    return d


def draw(*args, name=None, cache=True, **vopts):
    """Render this function NOW, to <output_to()>/<function name>.gif.

    Unlike `animate`, nothing is deferred -- decorating is the render. The
    function comes back untouched, so you can still call it normally.

    cache=True skips the render when the function's source, the options and a
    fingerprint of the inputs are all unchanged. Set JAXVIS_FORCE=1 to ignore
    the cache.
    """
    import hashlib
    import inspect
    import json
    import os
    from pathlib import Path as _P

    d = _OUT[0] or _P(".")

    def deco(fn):
        tag = name or fn.__name__
        gif = d / f"{tag}.gif"

        key = None
        if cache:
            h = hashlib.md5()
            try:
                h.update(inspect.getsource(fn).encode())
            except OSError:
                h.update(tag.encode())
            h.update(repr(sorted(vopts.items())).encode())
            for a in args:                    # cheap fingerprint, not the data
                x = np.asarray(a)
                h.update(f"{x.shape}{x.dtype}{float(x.sum()):.6g}".encode())
            key = h.hexdigest()[:16]
            cf = d / ".jaxvis-cache.json"
            try:
                old = json.loads(cf.read_text())
            except Exception:
                old = {}
            done = gif.exists() or (d / f"{tag}.mp4").exists()
            if old.get(tag) == key and done and not os.environ.get("JAXVIS_FORCE"):
                print(f"  = {tag} (unchanged)")
                return fn

        vopts.setdefault("verbose", False)
        try:
            visualize(fn, *args, out=str(gif), **vopts)
            if cache:
                old[tag] = key
                cf.write_text(json.dumps(old, indent=1))
            print(f"  {tag}.gif")
        except Exception as e:
            print(f"  x {tag}: {type(e).__name__}: {e}")
        return fn
    return deco


def draw_field(t=None, then=None, warp=None, n=180, extent=1.0,
               scale="fixed", name=None, size=680, palette="ultra",
               tween=22, hold=6, duration=60, grain=0.22, cycles=1.0,
               **kw):
    """Animate a FUNCTION of coordinates rather than a fixed array.

        @jaxvis.draw_field(then=jnp.sin, warp=warps.swirl)
        def blocks(x, y):
            return tiles.tumbling(x, y)

    The decorated function is f(x, y), or f(x, y, t) when `t` is set. Because
    the framework has the function and not just its output, it can evaluate it
    somewhere else:

      warp=g    f(g(x, y))     sampled on transformed coordinates
      then=h    h(f(x, y))     transformed after the fact
      t=n       f(x, y, t)     swept over n frames, t going 0 -> cycles

    scale decides how colour maps onto value:

      "fixed"   ONE range across every frame. A value keeps its colour for the
                whole clip, so you can compare frames -- and nothing flickers.
      "auto"    each frame renormalised to its own min/max. Always uses the
                full palette, so faint structure stays visible, but the same
                colour means different numbers at different times.
    """
    from .fields import grid
    from .render import morph, save_gif
    from .reps import as_heatmap

    d = _OUT[0] or __import__("pathlib").Path(".")

    def deco(f):
        tag = name or f.__name__
        X, Y = grid(n, extent)
        gx, gy = warp(X, Y) if warp is not None else (X, Y)

        states, labels = [], []
        if t is not None:
            for k in range(t):
                v = np.asarray(f(gx, gy, cycles * k / t), dtype=float)
                states.append(then(v) if then is not None else v)
                labels.append(f"{tag}  t={cycles * k / t:.2f}")
        else:
            v = np.asarray(f(X, Y), dtype=float)
            states.append(v); labels.append(tag)
            if warp is not None:
                states.append(np.asarray(f(gx, gy), dtype=float))
                labels.append(f"{tag}  warped")
            if then is not None:
                states.append(np.asarray(then(states[-1]), dtype=float))
                labels.append(f"{then.__name__}({tag})")

        states = [np.asarray(s, dtype=float) for s in states]
        vr = None
        if scale == "fixed":
            vr = (min(s.min() for s in states), max(s.max() for s in states))

        canvases = [as_heatmap(s, size, vr, grain=grain) for s in states]
        # a t-sweep is already continuous, so it needs no tween between frames
        frames = (morph(canvases, labels, hold=1, tween=1, palette=palette,
                        caption=kw.pop("caption", True))
                  if t is not None else
                  morph(canvases, labels, hold=hold, tween=tween,
                        palette=palette, caption=kw.pop("caption", True)))
        save_gif(frames, str(d / f"{tag}.gif"), duration=duration)
        print(f"  {tag}.gif  ({len(states)} states, {len(frames)} frames)")
        return f
    return deco


def draw_sim(init, frames=90, every=40, show=None, name=None, size=680,
             palette="ultra", duration=50, grain=0.18, scale="fixed",
             caption=True, rep="heatmap", colors=None, blend="mean",
             gain=2.0, frame="fixed", tween=1):
    """Animate a SIMULATION: a step function applied over and over.

        @jaxvis.draw_sim(init=(u0, v0), show=lambda s: s[1])
        def gray_scott(state):
            u, v = state
            ...
            return u_next, v_next

    This is where a heatmap earns its place. A transform morphs one array into
    another; a PDE is a field that evolves under its own rules, and the only
    honest way to show it is frame after frame of the field itself.

    `every` steps are fused into one jitted `fori_loop` per frame, so a clip
    with thousands of solver steps costs one compile and runs on the device.

      init     the starting state -- an array, or a tuple of them
      show     state -> the 2-D array to draw (default: the state itself)
      frames   snapshots in the clip
      every    solver steps between snapshots
    """
    import jax
    from .render import morph, save_gif
    from .reps import as_heatmap

    d = _OUT[0] or __import__("pathlib").Path(".")
    view = show or (lambda s: s)

    def deco(step):
        tag = name or step.__name__

        @jax.jit
        def advance(s):
            return jax.lax.fori_loop(0, every, lambda i, x: step(x), s)

        s, snaps = init, []
        for _ in range(frames):
            snaps.append(np.asarray(view(s), dtype=float))
            s = advance(s)

        # A solver step is a jump. Blending between consecutive states gives
        # the eye something continuous to follow -- points glide instead of
        # teleporting -- without changing what was computed.
        if tween > 1:
            from .render import ease
            smooth = []
            for a, b in zip(snaps, snaps[1:]):
                for j in range(tween):
                    e = float(ease(j / tween))
                    smooth.append(a * (1 - e) + b * e)
            snaps = smooth + [snaps[-1]]

        if rep == "points":
            from .reps import as_points, coloured_points
            # one window across every snapshot, so a cloud that spreads or
            # slides actually spreads or slides instead of being re-framed
            box = None
            if frame == "fixed":
                allp = np.concatenate([a[:, :2] for a in snaps])
                lo = np.percentile(allp, 0.5, 0)
                hi = np.percentile(allp, 99.5, 0)
                mid = (lo + hi) / 2
                box = (float(mid[0]), float(mid[1]),
                       float(np.max(hi - lo) / 2 * 1.08) + 1e-9)
            elif isinstance(frame, tuple):
                box = frame
            fn = (coloured_points(colors, blend=blend, gain=gain)
                  if colors is not None else as_points)
            canvases = [fn(a, size, frame=box) for a in snaps]
        else:
            vr = None
            if scale == "fixed":
                vr = (min(a.min() for a in snaps), max(a.max() for a in snaps))
            canvases = [as_heatmap(a, size, vr, grain=grain) for a in snaps]
        # one label per RENDERED frame -- tweening multiplies them
        per = max(1, tween)
        labels = [f"{tag}  step {(i // per) * every}"
                  for i in range(len(canvases))]
        out = morph(canvases, labels, hold=1, tween=1, palette=palette,
                    caption=caption)
        save_gif(out, str(d / f"{tag}.gif"), duration=duration)
        print(f"  {tag}.gif  ({frames} frames, {frames * every:,} solver steps)")
        return step
    return deco
