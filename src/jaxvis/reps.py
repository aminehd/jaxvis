"""Representation selection.

Every renderer returns an (S, S) float array in [0, 1]. Because they all share
that output type, morphing BETWEEN different representations is just a blend —
which is what lets a film mix a point cloud, a heatmap and a strip and still
read as one continuous thing.
"""
import numpy as np


def _norm(a, vrange=None):
    """Scale to [0,1].

    `vrange=(lo,hi)` fixes the scale. Without it every frame normalises to its
    own min/max -- so a change in the DATA's range reads as a brightness jump
    even when nothing moved. That flicker is the main enemy of smoothness.
    """
    a = np.asarray(a, dtype=float)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = vrange if vrange else (a.min(), a.max())
    return np.clip((a - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(a)


def _blank(S):
    return np.zeros((S, S))


def _splat(x, y, buf, w=None):
    "Bilinear splat — sub-pixel placement, which is what reads as smooth."
    S = buf.shape[0]
    ok = (x > 1) & (x < S - 2) & (y > 1) & (y < S - 2)
    x, y = x[ok], y[ok]
    w = np.ones_like(x) if w is None else np.asarray(w)[ok]
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    for dx, dy, f in ((0,0,(1-fx)*(1-fy)), (1,0,fx*(1-fy)),
                      (0,1,(1-fx)*fy),     (1,1,fx*fy)):
        np.add.at(buf, (y0+dy, x0+dx), f * w)


# ---------------------------------------------------------------- renderers

def project(a, k=2, onto=None):
    """High-dimensional points -> k dims.

    Two modes, both from Shai et al.'s belief-geometry work:

      onto=None   PCA. unsupervised, no assumptions.
      onto=B      REGRESSION onto coordinates you already know:
                  solve a @ W + c ~= B, then plot a @ W. That is what they do
                  to show the residual stream sitting on the belief simplex --
                  PCA would not find it, because the interesting axes are not
                  the high-variance ones.
    """
    a = np.asarray(a, dtype=float)
    if onto is not None:
        B = np.asarray(onto, dtype=float)
        A1 = np.hstack([a, np.ones((len(a), 1))])
        W, *_ = np.linalg.lstsq(A1, B, rcond=None)
        return (A1 @ W)[:, :k]
    c = a - a.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    return c @ Vt[:k].T


def _place(p, S, frame):
    """Cloud coordinates -> pixel coordinates.

    frame=None re-centres and re-scales THIS cloud to fill the canvas, which
    looks tidy but silently undoes any translation or uniform scaling: x - mu
    and x / 2 render identically to x. frame=(cx, cy, r) is a fixed window, so
    those moves are actually visible.
    """
    if frame is None:
        c = p - p.mean(0)
        r = np.abs(c).max() + 1e-9
    else:
        cx, cy, r = frame
        c = p - np.array([cx, cy])
    return c[:, 0] / r * S * 0.42 + S / 2, c[:, 1] / r * S * 0.42 + S / 2


def as_points(a, S, vrange=None, gain=3.0, onto=None, frame=None):
    """(n, d) -> a particle cloud. Projects to 2-D when d > 3."""
    p = np.asarray(a, dtype=float)
    if p.ndim == 2 and p.shape[1] > 3:
        p = project(p, 2, onto)
    if p.ndim == 1 or (p.ndim == 2 and p.shape[1] == 1):
        # a single column is not a cloud -- plot it against its own index
        v = p.reshape(-1)
        p = np.stack([np.linspace(-1, 1, len(v)), v], 1)
    p = p[:, :2]
    buf = _blank(S)
    _splat(*_place(p, S, frame), buf)
    return _norm(np.log1p(buf * gain))


def as_points_rgb(a, values, S, gain=3.0, onto=None, blend="mass",
                  frame=None):
    """Points coloured by what each one HOLDS, not by how many landed.

    `as_points` colours by density -- bright means many particles hit this
    pixel. This colours by value instead. Shai et al. read a 3-state belief
    distribution straight off as RGB: three numbers that sum to 1 are already
    a colour, no palette needed.

    `values` is (n, 3) -> used as RGB directly. (n,) or (n, k) is padded or
    projected to 3.

    `blend` decides what happens when two points share a pixel:

      "mean"  average them              honest; blends to mud where dense
      "max"   brightest one wins        vivid; hides how much overlap there is
      "mass"  mean, then scale          both signals at once   <- default
              brightness by count

    Returns (S, S, 3) -- NOT the (S, S) the other renderers return, because
    colour cannot be recovered from one channel. Feed it to `rgb_image`.
    """
    p = np.asarray(a, dtype=float)
    if p.ndim == 2 and p.shape[1] > 3:
        p = project(p, 2, onto)
    if p.ndim == 1 or (p.ndim == 2 and p.shape[1] == 1):
        # a single column is not a cloud -- plot it against its own index
        v = p.reshape(-1)
        p = np.stack([np.linspace(-1, 1, len(v)), v], 1)
    p = p[:, :2]

    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    if v.shape[1] > 3:
        v = project(v, 3)
    if v.shape[1] < 3:
        v = np.hstack([v, np.tile(v[:, -1:], (1, 3 - v.shape[1]))])
    v = (v - v.min()) / (v.max() - v.min() + 1e-9)

    x, y = _place(p, S, frame)

    if blend == "max":
        o = np.argsort(v.mean(1))            # dim first, bright last wins
        xi, yi = x[o].astype(int), y[o].astype(int)
        ok = (xi > 1) & (xi < S - 2) & (yi > 1) & (yi < S - 2)
        out = np.zeros((S, S, 3))
        out[yi[ok], xi[ok]] = v[o][ok]
        return np.clip(out, 0, 1)

    cnt = _blank(S)
    _splat(x, y, cnt)
    ch = []
    for k in range(3):
        b = _blank(S)
        _splat(x, y, b, w=v[:, k])
        ch.append(b)
    col = np.stack(ch, 2) / (cnt[..., None] + 1e-9)      # the average

    if blend == "mean":
        return np.clip(col, 0, 1)
    # Brightness against the 99th percentile, not the max. When a transform
    # collapses many points onto a few pixels (ReLU folding a cluster flat onto
    # an axis), those pixels are so dense that normalising by the max leaves
    # everything else dark. A percentile lets a few extremes saturate instead.
    b = np.log1p(cnt * gain)
    hi = np.percentile(b[b > 0], 99) if (b > 0).any() else 1.0
    return np.clip(col * np.clip(b / (hi + 1e-9), 0, 1)[..., None], 0, 1)


def _paper(S, seed=11):
    """Static fine grain. Deterministic, so it is IDENTICAL every frame --
    that is what makes it read as paper texture rather than film noise."""
    g = np.random.default_rng(seed).normal(size=(S, S))
    # a touch of smoothing kills the single-pixel buzz and leaves fibre
    g = (g + np.roll(g, 1, 0) + np.roll(g, 1, 1) + np.roll(g, (1, 1), (0, 1))) / 4
    return g / (np.abs(g).max() + 1e-9)


def as_heatmap(a, S, vrange=None, smooth=True, grain=0.0, seed=11):
    """(n, m) -> a grid, aspect preserved, centred.

    Upscales with BICUBIC, not np.repeat. Nearest-neighbour turns a 96x96 array
    into 3-pixel blocks with hard edges -- which is the entire reason heatmaps
    used to look chunky next to the bilinearly-splatted point clouds.
    """
    from PIL import Image
    v = _norm(a, vrange)
    h0, w0 = v.shape
    scale = min(S / h0, S / w0)
    h, w = max(1, int(h0 * scale)), max(1, int(w0 * scale))
    if smooth:
        im = Image.fromarray((v * 65535).astype(np.uint16), mode="I;16")
        big = np.asarray(im.resize((w, h), Image.BICUBIC), dtype=float) / 65535.0
    else:
        r = max(1, int(scale))
        big = np.repeat(np.repeat(v, r, 0), r, 1)[:h, :w]
    out = _blank(S)
    hh, ww = min(S, big.shape[0]), min(S, big.shape[1])
    out[(S-hh)//2:(S-hh)//2+hh, (S-ww)//2:(S-ww)//2+ww] = np.clip(big[:hh, :ww], 0, 1)
    if grain:
        # Ink on fibre: grain is strongest in the midtones and vanishes at
        # pure black and pure white, which is why 4v(1-v) rather than a flat
        # overlay. A flat overlay fogs the blacks and reads as dirt.
        out = np.clip(out + grain * _paper(S, seed) * (4 * out * (1 - out)), 0, 1)
    return out


def _flow(S, t=0.0, seed=0, octaves=4):
    """A smooth scalar field that LOOPS in t. Sum of directional waves.

    Cheap, analytic, and periodic in t -- so a GIF made by sweeping t=k/n
    closes seamlessly. Real value noise would need interpolation and would not
    loop without extra work.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:S, 0:S] / float(S)
    f, amp, tot = np.zeros((S, S)), 1.0, 0.0
    for o in range(octaves):
        k = 2.0 * (2 ** o)
        ph = rng.uniform(0, 2 * np.pi)
        d = rng.normal(size=2); d /= np.hypot(*d) + 1e-9
        f += amp * np.sin(k * np.pi * (d[0] * xx + d[1] * yy) + ph + 2*np.pi*t)
        tot += amp
        amp *= 0.55
    return f / tot


def _sample(v, x, y):
    "Bilinear read of v at float coords -- the inverse of _splat."
    H, W = v.shape
    x = np.clip(x, 0, W - 1.001); y = np.clip(y, 0, H - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    return (v[y0,   x0  ] * (1-fx) * (1-fy) + v[y0,   x0+1] * fx * (1-fy) +
            v[y0+1, x0  ] * (1-fx) * fy     + v[y0+1, x0+1] * fx * fy)


def as_liquid(a, S, vrange=None, t=0.0, warp=0.05, bands=9.0, caustic=0.45,
              seed=0):
    """A watery heatmap. Same input as `as_heatmap`, but it flows.

    `as_heatmap` BICUBICs a small grid up to full size. That is smooth and
    edge-free, which is exactly why it reads as a flat print: every pixel is a
    weighted average of its neighbours, so there is no detail finer than the
    original grid. Point clouds look alive because they carry detail at the
    pixel level. Three borrowed-from-water tricks put that detail back:

      warp     DOMAIN WARPING -- read the field at coordinates pushed around by
               a smooth noise field, instead of on the grid. Biggest single
               change: straight contours become curled ones. (Inigo Quilez.)
      bands    iso-contours. Water shows depth as banding, and banding is
               pattern at a frequency the source grid never had. <- "more
               pattern" lives here.
      caustic  brighten where the field is steep -- light focusing through a
               wavy surface. The glint.

    `t` drifts the warp and loops at t=1, so sweeping t=k/n across n frames
    gives a seamless flowing loop.
    """
    base = as_heatmap(a, S, vrange)
    yy, xx = np.mgrid[0:S, 0:S].astype(float)

    wx = _flow(S, t, seed * 7 + 1) * warp * S
    wy = _flow(S, t, seed * 7 + 2) * warp * S
    v = _sample(base, xx + wx, yy + wy)

    if bands:
        v = v * (0.72 + 0.28 * np.sin(v * bands * 2 * np.pi))
    if caustic:
        gy, gx = np.gradient(v)
        v = v + caustic * _norm(np.hypot(gx, gy)) * v
    return _norm(v)


def as_strip(a, S, vrange=None, bars=True):
    """(n,) -> a full-height band.

    NOT via as_heatmap: that preserves aspect, which turns a (1, 48) array into
    a 6-pixel sliver in a 300-pixel canvas -- between two heatmaps it reads as a
    black frame. A 1-D array has no meaningful aspect, so fill the canvas.
    """
    v = _norm(a, vrange)
    cols = np.repeat(v, max(1, int(np.ceil(S / len(v)))))[:S]
    if len(cols) < S:
        cols = np.pad(cols, (0, S - len(cols)))
    out = np.tile(cols, (S, 1))
    if bars:
        # height-encode as well, so it reads as data rather than a gradient
        h = (cols * (S - 1)).astype(int)
        mask = np.arange(S)[::-1, None] <= h[None, :]
        out = out * (0.30 + 0.70 * mask)
    return out


def as_batch(a, S, vrange=None):
    "(b, h, w) -> tile the batch into a square-ish contact sheet."
    b = a.shape[0]
    cols = int(np.ceil(np.sqrt(b)))
    rows = int(np.ceil(b / cols))
    h, w = a.shape[1], a.shape[2]
    sheet = np.zeros((rows * h, cols * w))
    for i in range(b):
        r, c = divmod(i, cols)
        sheet[r*h:(r+1)*h, c*w:(c+1)*w] = _norm(a[i], vrange)
    return as_heatmap(sheet, S)


BY_NAME = {
    "liquid": as_liquid, "points": as_points, "heatmap": as_heatmap,
           "strip": as_strip, "batch": as_batch}


def coloured_points(values, blend="mean", gain=2.0):
    """Wrap as_points_rgb so it looks like every other renderer.

    The renderers are called as fn(array, S, vrange) -- there is nowhere to
    pass a colour per point. Closing over the colours fixes that, and means
    `visualize(..., colors=rgb)` can route point steps through here instead
    of the density renderer.
    """
    def fn(a, S, vrange=None, **kw):
        # rep_kw={"gain": ...} reaches here; higher gain log-compresses
        # density so dim regions stay visible next to very dense ones
        return as_points_rgb(a, values, S, gain=kw.get("gain", gain),
                             blend=blend, frame=kw.get("frame"))
    return fn


def pick(a, rep=None, op=None):
    """Choose a renderer.

    `rep` overrides the automatic choice:
        rep="heatmap"                  force everything
        rep={"mul": "points"}          per-primitive
        rep={(3000, 2): "heatmap"}     per-shape
    """
    a = np.asarray(a)
    if isinstance(rep, str):
        return BY_NAME[rep], rep
    if isinstance(rep, dict):
        for key in (op, a.shape, a.ndim):
            if key in rep:
                name = rep[key]
                return BY_NAME[name], name
    return _auto(a)


def _auto(a):
    """The default: choose from the array's shape alone."""
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1] in (2, 3) and a.shape[0] >= 16:
        return as_points, "points"          # looks like a cloud of coordinates
    if (a.ndim == 2 and a.shape[1] > 1 and a.shape[0] >= 64
            and a.shape[0] > a.shape[1] * 4):
        return as_points, "points"          # many rows, few dims -> project it
    if a.ndim == 3:
        return as_batch, "batch"
    if a.ndim == 2:
        return as_heatmap, "heatmap"
    if a.ndim == 1:
        return as_strip, "strip"
    return (lambda x, S, vrange=None: _blank(S)), "skip"


# ------------------------------------------------------------ warp / mesh

def _line(p0, p1, buf, w=1.0, steps=None):
    "Splat a straight segment into the buffer."
    n = steps or max(2, int(np.hypot(p1[0]-p0[0], p1[1]-p0[1])))
    ts = np.linspace(0, 1, n)
    xs = p0[0] + (p1[0]-p0[0]) * ts
    ys = p0[1] + (p1[1]-p0[1]) * ts
    _splat(xs, ys, buf, np.full(n, w / max(1, n) * 6))


def as_warp(a, S, vrange=None, grid=34, amount=0.42, mesh=True):
    """Data as DISTORTION of a regular lattice.

    A square grid is displaced by the field's own gradient -- so you read the
    values as bulges and pinches in the mesh rather than as colour. High
    curvature crowds the lines; flat regions leave them square.
    """
    v = _norm(a, vrange)
    g = min(grid, min(v.shape))
    ys = np.linspace(0, v.shape[0] - 1, g).astype(int)
    xs = np.linspace(0, v.shape[1] - 1, g).astype(int)
    sub = v[np.ix_(ys, xs)]

    gy, gx = np.gradient(sub)
    u = np.linspace(-1, 1, g)
    X, Y = np.meshgrid(u, u)
    k = amount * g * 0.5
    px = (X + gx * k * 0.12) * S * 0.44 + S / 2
    py = (Y + gy * k * 0.12) * S * 0.44 + S / 2

    buf = _blank(S)
    if mesh:
        for i in range(g):
            for j in range(g - 1):
                w = 0.25 + sub[i, j]
                _line((px[i, j], py[i, j]), (px[i, j+1], py[i, j+1]), buf, w)
                _line((px[j, i], py[j, i]), (px[j+1, i], py[j+1, i]), buf, w)
    else:
        _splat(px.ravel(), py.ravel(), buf, (0.3 + sub).ravel())
    return _norm(np.log1p(buf * 2.2))


BY_NAME["warp"] = as_warp   # defined below the table, so register it here
