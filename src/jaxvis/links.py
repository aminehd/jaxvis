"""The impact of each input entry on each output entry, drawn as wiring.

`as_cells` shows a vector before and an answer after. It never shows the
thing in between: WHICH input entry was responsible for which part of the
output. For y = M x that information is not in either vector, it is in the
n x n numbers M[i, j] * x[j] -- one per wire.

So draw the wires. Input entries along the bottom, output entries along the
top, one narrow link per non-zero entry of M, and a pulse that travels from
the input that sent it to the output that receives it. Every link starts at
its own random moment, which is what keeps it from reading as one big
curtain coming down.
"""
import numpy as np

from .reps import _blank, _splat
from .render import ramp, morph, save_gif, ease


BG = np.array([0.045, 0.035, 0.075])


def _bezier(p0, p1, m=150, bend=0.46):
    """A vertical S-curve from p0 to p1.

    Straight lines between two rows of n points cross in a hatch you cannot
    read. Pulling the control points vertically makes every link leave its
    input going straight up and arrive at its output going straight up, so
    the ends stay legible even where the middles overlap.
    """
    (x0, y0), (x1, y1) = p0, p1
    dy = (y1 - y0) * bend
    c0, c1 = (x0, y0 + dy), (x1, y1 - dy)
    def at(t):
        t = t[:, None]
        return ((1 - t) ** 3 * np.array([x0, y0])
                + 3 * (1 - t) ** 2 * t * np.array(c0)
                + 3 * (1 - t) * t ** 2 * np.array(c1)
                + t ** 3 * np.array([x1, y1]))
    # Re-sample by ARC LENGTH. Evenly spaced t gives points that bunch up in
    # the curves and spread out in the straights, which splats as a dotted
    # line exactly where the link is most readable.
    q = at(np.linspace(0.0, 1.0, 64))
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
    n = max(8, int(d[-1] / 0.45))
    p = at(np.interp(np.linspace(0.0, d[-1], n), d, np.linspace(0, 1, 64)))
    return p[:, 0], p[:, 1]


def _square(rgb, cx, cy, half, color):
    "A small axis-aligned square. Fixed size -- only the colour carries value."
    S = rgb.shape[0]
    x0, x1 = int(round(cx - half)), int(round(cx + half)) + 1
    y0, y1 = int(round(cy - half)), int(round(cy + half)) + 1
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(S, x1), min(S, y1)
    if x1 > x0 and y1 > y0:
        rgb[y0:y1, x0:x1] = color


def impact(M, x, name="impact", size=720, palette="ultra", frames=150,
           duration=45, spread=0.55, travel=0.34, seed=0, bend=0.46,
           lattice=0.10, head=1.45, label=None, caption=True, out=None,
           hold=10):
    """Animate y = M x as n^2 links firing, each at its own moment.

        jaxvis.impact(L, v, name="cumulative")

      spread   how spread out the start times are, 0 = all at once
      travel   how long one pulse takes to cross, as a fraction of the clip
      lattice  brightness of the links that are NOT carrying anything --
               the wiring that exists but is multiplied by zero
      head     brightness of the travelling pulse

    Colour is the value, through `palette`, on a scale symmetric about zero,
    so the midpoint of the palette means "nothing here" for both the squares
    and the wires.
    """
    from pathlib import Path as _P
    from . import _OUT

    M = np.asarray(M, dtype=float)
    x = np.asarray(x, dtype=float).reshape(-1)
    n_out, n_in = M.shape
    y = M @ x
    contrib = M * x[None, :]                 # impact of input j on output i

    S = int(size)
    lut = ramp(palette) / 255.0
    vmax = float(max(np.abs(x).max(), np.abs(y).max(), 1e-9))

    def col(v):
        u = np.clip(0.5 + 0.5 * float(v) / vmax, 0.0, 1.0)
        return lut[int(u * (len(lut) - 1))]

    pitch = S * 0.92 / max(n_in, n_out)
    half = max(2.0, pitch * 0.26)
    xs_in = (np.arange(n_in) - (n_in - 1) / 2) * pitch + S / 2
    xs_out = (np.arange(n_out) - (n_out - 1) / 2) * pitch + S / 2
    y_in, y_out = S - S * 0.10, S * 0.10

    rng = np.random.default_rng(seed)
    strength = np.abs(contrib) / (np.abs(contrib).max() + 1e-12)
    wires = []
    for i in range(n_out):
        for j in range(n_in):
            if abs(M[i, j]) < 1e-9:
                continue                     # no wire at all, not a dim one
            px, py = _bezier((xs_in[j], y_in - half),
                             (xs_out[i], y_out + half), bend=bend)
            wires.append(dict(i=i, j=j, px=px, py=py,
                              s=float(strength[i, j]),
                              c=col(contrib[i, j]),
                              d=float(rng.uniform(0.0, 1.0))))

    # The unlit wiring never changes, so it is drawn once and added to every
    # frame instead of being re-splatted 150 times.
    lat = [_blank(S), _blank(S), _blank(S)]
    if lattice:
        for w in wires:
            m = len(w["px"])
            for ch in range(3):
                _splat(w["px"], w["py"], lat[ch],
                       np.full(m, lattice * w["c"][ch]))

    canvases = []
    for t in np.linspace(0.0, 1.0, frames):
        R, G, B = [a.copy() for a in lat]
        acc = np.zeros(n_out)
        for w in wires:
            # each link waits for its own moment, then crosses
            p = float(np.clip((t - w["d"] * spread) / travel, 0.0, 1.0))
            e = float(ease(p))
            acc[w["i"]] += contrib[w["i"], w["j"]] * e
            k = int(e * (len(w["px"]) - 1))
            if k < 2:
                continue
            # One splat carries both the trail and the pulse: a weight that
            # ramps steeply toward the leading end. Two passes would mean two
            # normalisations and a visible seam where they meet.
            a = (0.20 + 0.80 * np.linspace(0.0, 1.0, k) ** 5) \
                * head * (0.30 + 0.70 * w["s"])
            if p >= 1.0:
                a = np.full(k, 0.42 * (0.30 + 0.70 * w["s"]))
            cr, cg, cb = w["c"]
            _splat(w["px"][:k], w["py"][:k], R, a * cr)
            _splat(w["px"][:k], w["py"][:k], G, a * cg)
            _splat(w["px"][:k], w["py"][:k], B, a * cb)

        rgb = np.clip(np.stack([R, G, B], 2) + BG, 0.0, 1.0)
        for j in range(n_in):
            _square(rgb, xs_in[j], y_in, half, col(x[j]))
        for i in range(n_out):
            _square(rgb, xs_out[i], y_out, half, col(acc[i]))
        canvases.append(rgb)

    canvases = [canvases[0]] * hold + canvases + [canvases[-1]] * hold
    lbl = label or f"y = M x   ({n_out}, {n_in}) @ ({n_in},)"
    out_dir = _P(out) if out else (_OUT[0] or _P("."))
    gif = str(out_dir / f"{name}.gif")
    save_gif(morph(canvases, [lbl] * len(canvases), hold=1, tween=1,
                   palette=palette, caption=caption),
             gif, duration=duration)
    print(f"  {name}.gif  ({len(wires)} links, {len(canvases)} frames)")
    return gif


def _hues(n, s=0.78, v=1.0):
    """n maximally separated colours, one per input.

    Not a palette: a palette maps VALUE to colour, and here colour has to
    mean identity instead. Spreading hue evenly round the circle is the
    only encoding where two inputs are never confusable.
    """
    h = (np.arange(n) / n + 0.04) % 1.0
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    out = np.zeros((n, 3))
    for k in range(n):
        out[k] = [(v, t[k], p), (q[k], v, p), (p, v, t[k]),
                  (p, q[k], v), (t[k], p, v), (v, p, q[k])][i[k] % 6]
    return out


def _apportion(w, total):
    "Split `total` whole grains in proportion to w (largest remainder)."
    if w.sum() <= 0:
        return np.zeros(len(w), dtype=int)
    exact = w / w.sum() * total
    cnt = np.floor(exact).astype(int)
    left = total - cnt.sum()
    if left > 0:
        order = np.argsort(-(exact - cnt))
        cnt[order[:left]] += 1
    return cnt


def grains(M, x, name="grains", size=760, k=9, frames=170, duration=45,
           spread=0.62, travel=0.30, seed=0, dot=5.0, boxes=0.16,
           label=None, caption=True, out=None, hold=12):
    """Where each output's content CAME FROM, carried by coloured grains.

        jaxvis.grains(M, x)

    `impact` draws one wire per matrix entry: good for the structure of M,
    but every wire looks alike and you cannot tell by looking at an output
    square which inputs built it. Here each input is a small k x k block of
    grains with its own hue, and hue means WHICH INPUT -- not value.

    Input j's grains are divided among the outputs in proportion to
    |M[i,j] x_j|, so the share of j that lands on i is the share of j that
    output i actually used. Each grain crosses on its own path at its own
    moment. What piles up at the top is a mixture, and the colour of that
    mixture is literally the provenance of the output.

    An output that comes from one place is one clean colour. An output
    that averages everything is grey.
    """
    from pathlib import Path as _P
    from . import _OUT

    M = np.asarray(M, dtype=float)
    x = np.asarray(x, dtype=float).reshape(-1)
    n_out, n_in = M.shape
    rng = np.random.default_rng(seed)

    S = int(size)
    pitch = S * 0.92 / max(n_in, n_out)
    side = pitch * 0.58
    xs_in = (np.arange(n_in) - (n_in - 1) / 2) * pitch + S / 2
    xs_out = (np.arange(n_out) - (n_out - 1) / 2) * pitch + S / 2
    y_in, y_out = S - S * 0.11, S * 0.11
    hue = _hues(n_in)

    # the k x k lattice inside one block
    u = (np.arange(k) - (k - 1) / 2) / max(1, k - 1) * side
    gx, gy = np.meshgrid(u, u)
    gx, gy = gx.ravel(), gy.ravel()

    P0, P1, col, delay = [], [], [], []
    for j in range(n_in):
        w = np.abs(M[:, j] * x[j])
        cnt = _apportion(w, k * k)
        order = rng.permutation(k * k)      # which grain goes where, shuffled
        at = 0
        for i in range(n_out):
            m = int(cnt[i])
            if m == 0:
                continue
            idx = order[at:at + m]
            at += m
            P0.append(np.stack([xs_in[j] + gx[idx], y_in + gy[idx]], 1))
            # Land anywhere inside the destination block -- more grains than
            # slots is normal, and the pile-up is the point. The half-side
            # matters: the k x k input lattice spans +-side/2, so scattering
            # over +-side would make every pile overlap its neighbours and a
            # permutation would read as a mixture when it is the opposite.
            P1.append(np.stack([xs_out[i] + rng.uniform(-side / 2, side / 2, m),
                                y_out + rng.uniform(-side / 2, side / 2, m)], 1))
            col.append(np.repeat(hue[j][None, :], m, 0))
            delay.append(rng.uniform(0, 1, m))
        if at < k * k:                      # grains with nowhere to go stay put
            idx = order[at:]
            home = np.stack([xs_in[j] + gx[idx], y_in + gy[idx]], 1)
            P0.append(home)
            P1.append(home)
            col.append(np.repeat(hue[j][None, :] * 0.30, len(idx), 0))
            delay.append(np.zeros(len(idx)))

    P0 = np.concatenate(P0); P1 = np.concatenate(P1)
    col = np.concatenate(col); delay = np.concatenate(delay)
    # control points pulled vertically: every grain leaves straight up and
    # arrives straight up, so the two ends stay readable where paths cross
    bend = (P1[:, 1] - P0[:, 1]) * 0.45
    C0 = np.stack([P0[:, 0], P0[:, 1] + bend], 1)
    C1 = np.stack([P1[:, 0], P1[:, 1] - bend], 1)

    def box(rgb, cx, cy, r, w):
        "A faint square outline: the slot a block of grains belongs to."
        S_ = rgb.shape[0]
        x0, x1 = int(cx - r), int(cx + r)
        y0, y1 = int(cy - r), int(cy + r)
        for (a, b, c, d) in ((y0, y0 + 1, x0, x1), (y1, y1 + 1, x0, x1),
                             (y0, y1, x0, x0 + 1), (y0, y1, x1, x1 + 1)):
            a, b = max(0, a), min(S_, b)
            c, d = max(0, c), min(S_, d)
            if b > a and d > c:
                rgb[a:b, c:d] = np.maximum(rgb[a:b, c:d], w)

    canvases = []
    for t in np.linspace(0.0, 1.0, frames):
        p = np.clip((t - delay * spread) / travel, 0.0, 1.0)
        e = ease(p)[:, None]
        pos = ((1 - e) ** 3 * P0 + 3 * (1 - e) ** 2 * e * C0
               + 3 * (1 - e) * e ** 2 * C1 + e ** 3 * P1)
        chans = []
        for ch in range(3):
            b = _blank(S)
            _splat(pos[:, 0], pos[:, 1], b, col[:, ch] * dot)
            for dx, dy in ((0.8, 0), (-0.8, 0), (0, 0.8), (0, -0.8)):
                _splat(pos[:, 0] + dx, pos[:, 1] + dy, b, col[:, ch] * dot * 0.4)
            chans.append(b)
        rgb = np.clip(np.stack(chans, 2) + BG, 0.0, 1.0)
        if boxes:
            r = side * 0.68
            for j in range(n_in):
                box(rgb, xs_in[j], y_in, r, boxes * hue[j])
            for i in range(n_out):
                box(rgb, xs_out[i], y_out, r, boxes * 0.85)
        canvases.append(rgb)

    canvases = [canvases[0]] * hold + canvases + [canvases[-1]] * hold
    lbl = label or f"y = M x   colour = which input   ({n_out}, {n_in})"
    out_dir = _P(out) if out else (_OUT[0] or _P("."))
    gif = str(out_dir / f"{name}.gif")
    save_gif(morph(canvases, [lbl] * len(canvases), hold=1, tween=1,
                   palette="mono", caption=caption),
             gif, duration=duration)
    print(f"  {name}.gif  ({len(P0)} grains, {len(canvases)} frames)")
    return gif
