"""Turn a list of (S,S) canvases into a smooth, saveable film."""
import numpy as np
from PIL import Image, ImageDraw

# Palettes are colour STOPS, interpolated. Easier to design than curves, and
# you can add your own: STOPS["mine"] = [(0.0, (r,g,b)), ..., (1.0, (r,g,b))]
STOPS = {
    "ember":  [(0.00, (  4,  2,  6)), (0.35, (120, 40, 12)),
               (0.70, (245, 140, 30)), (1.00, (255, 240, 200))],
    "neon":   [(0.00, ( 10,  2, 22)), (0.30, (120, 10, 110)),
               (0.62, (255, 40, 160)), (0.85, (255, 120, 210)),
               (1.00, (255, 235, 250))],
    "acid":   [(0.00, (  2,  8,  6)), (0.32, ( 10, 110, 80)),
               (0.65, ( 70, 230, 120)), (0.88, (190, 255, 90)),
               (1.00, (245, 255, 210))],
    "bloom":  [(0.00, ( 12,  2, 20)), (0.28, (190, 20, 120)),   # pink lows
               (0.55, (255, 80, 170)),
               (0.78, ( 80, 230, 170)),                          # green highs
               (1.00, (225, 255, 240))],
    "vapor":  [(0.00, ( 14,  4, 30)), (0.34, (150, 30, 170)),
               (0.62, (255, 80, 190)), (0.84, ( 90, 220, 255)),
               (1.00, (235, 250, 255))],
    "spectral": [(0.00, (  4,  2, 28)), (0.16, ( 40, 20, 130)),
                 (0.34, (140, 30, 190)), (0.50, (255, 40, 150)),
                 (0.66, (255, 105, 90)), (0.80, (255, 190, 60)),
                 (0.92, (220, 255, 130)), (1.00, (245, 255, 235))],
    "ultra":    [(0.00, (  2, 10, 26)), (0.18, (  0, 90, 160)),
                 (0.36, (  0, 200, 190)), (0.54, (120, 255, 120)),
                 (0.70, (255, 220, 60)), (0.85, (255, 90, 130)),
                 (1.00, (255, 225, 250))],
    "ice":    [(0.00, (  2,  4, 12)), (0.38, ( 20, 70, 150)),
               (0.72, ( 90, 190, 255)), (1.00, (230, 245, 255))],
    "mono":   [(0.00, (  0,  0,  0)), (1.00, (250, 250, 250))],
}


def ramp(name, n=256):
    "Expand a palette's stops into an n x 3 lookup table."
    stops = STOPS[name]
    xs = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops], dtype=float)
    g = np.linspace(0, 1, n)
    return np.stack([np.interp(g, xs, cs[:, i]) for i in range(3)], 1)


def rgb_image(v):
    "(S, S, 3) float in [0,1] -> an Image. The colour path; skips the palette."
    return Image.fromarray((np.clip(v, 0, 1) * 255).astype(np.uint8))


def colour(v, palette="ember"):
    lut = ramp(palette)
    idx = (np.clip(v, 0, 1) * (len(lut) - 1)).astype(int)
    return Image.fromarray(lut[idx].astype(np.uint8))


def morph(canvases, labels=None, hold=6, tween=14, palette="ember", caption=True):
    """Hold each canvas, then ease into the next.

    Because every renderer emits the same (S,S) float type, a cross-fade works
    even when the REPRESENTATION changes between steps.
    """
    frames = []
    for i, c in enumerate(canvases):
        lbl = labels[i] if labels else None
        for _ in range(hold):
            frames.append(_draw(c, lbl, palette, caption))
        if i + 1 < len(canvases):
            nxt = canvases[i + 1]
            for t in np.linspace(0, 1, tween, endpoint=False)[1:]:
                e = 0.5 - 0.5 * np.cos(np.pi * t)          # ease in-out
                frames.append(_draw(c*(1-e) + nxt*e, lbl, palette, caption))
    return frames


def _draw(v, label, palette, caption, font_px=None):
    """Colour the canvas, and put the label in its OWN BAND below it.

    Drawing the label onto the picture meant it sat on top of the data, in
    PIL's bitmap default font, at whatever size that happens to be -- which is
    unreadable once the render is downscaled for a web page. A band costs a
    few rows of pixels and keeps the frame clean.
    """
    # An (S, S, 3) canvas carries its own colour -- it never touches a palette.
    im = rgb_image(v) if np.ndim(v) == 3 else colour(v, palette)
    if not (caption and label):
        return im
    W, H = im.size
    fp = font_px or max(13, W // 30)
    bh = int(fp * 2.1)
    out = Image.new("RGB", (W, H + bh), (11, 11, 16))
    out.paste(im, (0, 0))
    ImageDraw.Draw(out).text((int(fp * 0.8), H + (bh - fp) // 2 - 1),
                             label, font=_font(fp), fill=(233, 230, 240))
    return out


def ease(p, g=2.0):
    """The beesandbombs ease, verbatim from Dave Whyte's Processing template.

        if (p < 0.5) return 0.5 * pow(2*p, g);
        else         return 1 - 0.5 * pow(2*(1-p), g);

    `g` is the knob `smoothstep` does not have: g=1 is linear, g=2 is gentle,
    g=3+ snaps. Zero derivative at both ends either way, which is what stops a
    loop from ticking at the seam.
    """
    p = np.clip(p, 0.0, 1.0)
    return np.where(p < 0.5, 0.5 * (2 * p) ** g, 1 - 0.5 * (2 * (1 - p)) ** g)


def ease_cos(p):
    "The template's simpler alternative: (1 - cos(pi p)) / 2."
    return (1 - np.cos(np.pi * np.clip(p, 0.0, 1.0))) / 2


def _flow_2d(h, w, seed=0, octaves=4):
    "Smooth scalar noise over a grid -- sum of directional waves."
    rng = np.random.default_rng(seed)
    gy, gx = np.mgrid[0:h, 0:w]
    yy, xx = gy / max(h - 1, 1), gx / max(w - 1, 1)
    f, amp, tot = np.zeros((h, w)), 1.0, 0.0
    for o in range(octaves):
        k = 2.0 * (2 ** o)
        ph = rng.uniform(0, 2 * np.pi)
        d = rng.normal(size=2); d /= np.hypot(*d) + 1e-9
        f += amp * np.sin(k * np.pi * (d[0] * xx + d[1] * yy) + ph)
        tot += amp; amp *= 0.55
    return f / tot


def phase_field(a, kind="radial"):
    """Per-element phase in [0,1]: WHERE each element sits in the sweep.

    "radial"    centre first, edges last
    "noise"     smooth random regions lead and lag -- what the template
                actually uses: offset = 9*noise(0.02*x, 0.02*y)
    "diagonal"  a wipe across the array
    "value"     biggest values arrive first -- the sweep carries information
                instead of just decorating
    """
    a = np.asarray(a)
    if kind == "value":
        v = np.abs(a.astype(float))
        return 1.0 - (v - v.min()) / (np.ptp(v) + 1e-9)    # big = early
    if a.ndim < 2:
        n = a.shape[0] if a.ndim else 1
        return np.linspace(0, 1, n)
    h, w = a.shape[:2]
    if kind == "noise":
        f = _flow_2d(h, w, seed=3)
        f = (f - f.min()) / (np.ptp(f) + 1e-9)
        while f.ndim < a.ndim:
            f = f[..., None]
        return f
    gy, gx = np.mgrid[0:h, 0:w]
    yy, xx = gy / max(h - 1, 1), gx / max(w - 1, 1)
    f = np.hypot(yy - 0.5, xx - 0.5) if kind == "radial" else (yy + xx) / 2
    f = (f - f.min()) / (np.ptp(f) + 1e-9)
    while f.ndim < a.ndim:
        f = f[..., None]
    return f


def wave_ease(u, phase=None, spread=0.65, g=2.0):
    """Smoothstep with a PER-ELEMENT phase offset -- the beesandbombs wave.

    A plain smoothstep(u) advances every element in lockstep, which reads as a
    crossfade. Offsetting each element's start makes the identical motion
    travel across the field instead. `spread` is how much of the timeline the
    sweep occupies: 0 gives back lockstep, 1 means the last element starts
    exactly as the first one finishes.
    """
    if phase is None:
        return ease(u, g)
    return ease(np.clip(u * (1 + spread) - phase * spread, 0, 1), g)


def save_gif(frames, path, duration=55, colors=128):
    """Write a GIF, preserving holds.

    PIL merges consecutive IDENTICAL frames on save -- which silently deletes
    every pause, because a hold is just the same canvas repeated. So collapse
    the runs ourselves and give the survivor a proportionally longer duration.
    """
    # Build the palette from a SAMPLE of the whole film, not just frame 0.
    # With a code panel the first frame is mostly the visualisation, so the
    # panel's greys and its orange highlight get quantised into nonsense.
    n = len(frames)
    picks = [frames[i] for i in range(0, n, max(1, n // 12))][:12]
    w, h = frames[0].size
    montage = Image.new("RGB", (w, h * len(picks)))
    for i, f in enumerate(picks):
        montage.paste(f, (0, i * h))
    pal = montage.quantize(colors=colors)
    qs = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]

    keep, durs = [], []
    for q in qs:
        b = q.tobytes()
        if keep and b == keep[-1][1]:
            durs[-1] += duration            # extend the hold instead
        else:
            keep.append((q, b))
            durs.append(duration)

    imgs = [k[0] for k in keep]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=durs, loop=0, optimize=True)
    return path


# ------------------------------------------------------- code panel

import functools
import linecache
import os

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


@functools.lru_cache(maxsize=8)
def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(MONO, size)
    except Exception:
        return ImageFont.load_default()


@functools.lru_cache(maxsize=64)
def source_lines(path):
    "The source of the function being visualised, as a list of lines."
    if not path or not os.path.exists(path):
        return []
    linecache.checkcache(path)
    return [l.rstrip("\n") for l in linecache.getlines(path)]


def code_panel(path, lineno, W, H, fsize=13, window=None):
    """Render the source with `lineno` highlighted. Returns a PIL image."""
    from PIL import Image, ImageDraw
    lines = source_lines(path)
    im = Image.new("RGB", (W, H), (16, 16, 22))
    d = ImageDraw.Draw(im)
    f = _font(fsize)
    if not lines:
        d.text((14, 14), "(source unavailable)", font=f, fill=(110, 110, 130))
        return im

    step = fsize + 6
    rows = max(1, (H - 24) // step)
    window = window or rows
    lo = max(0, (lineno or 1) - window // 2)
    hi = min(len(lines), lo + rows)
    lo = max(0, hi - rows)

    y = 12
    for n in range(lo, hi):
        cur = (n + 1) == lineno
        if cur:
            d.rectangle([6, y - 3, W - 6, y + fsize + 3], fill=(46, 38, 20))
        d.text((12, y), f"{n+1:>3}", font=f,
               fill=(255, 176, 60) if cur else (78, 78, 96))
        d.text((48, y), lines[n][:110], font=f,
               fill=(255, 226, 178) if cur else (146, 150, 170))
        y += step
    return im


def side_by_side(code_im, viz_im, gap=10):
    from PIL import Image
    W = code_im.size[0] + gap + viz_im.size[0]
    H = max(code_im.size[1], viz_im.size[1])
    out = Image.new("RGB", (W, H), (16, 16, 22))
    out.paste(code_im, (0, (H - code_im.size[1]) // 2))
    out.paste(viz_im, (code_im.size[0] + gap, (H - viz_im.size[1]) // 2))
    return out


# ------------------------------------------------- motion blur (beesandbombs)
# Technique from Dave Whyte's Processing sketches (gist.github.com/beesandbombs):
# render several sub-frames across a shutter angle and average them -- but
# average in LINEAR light, not sRGB. Squaring before the mean and square-rooting
# after is the difference between a bright blur and a muddy one.

def linear_mean(canvases):
    """Average canvases in linear light. sq -> mean -> sqrt."""
    a = np.stack([np.asarray(c, dtype=float) for c in canvases])
    return np.sqrt(np.clip((a ** 2).mean(0), 0, 1))


def smoothstep(p):
    "ease(p) = 3p^2 - 2p^3 -- Dave's default easing."
    p = np.clip(p, 0, 1)
    return 3 * p * p - 2 * p * p * p


def ease_g(p, g=2.0):
    "Adjustable-sharpness easing: gentler or snappier than smoothstep."
    p = np.clip(p, 0, 1)
    return np.where(p < 0.5, 0.5 * (2 * p) ** g, 1 - 0.5 * (2 * (1 - p)) ** g)


def blurred(render_at, t, dt, samples=8, shutter=0.65):
    """One motion-blurred frame.

    render_at(u) -> canvas. Samples across `shutter` of the frame's duration
    and averages in linear light.
    """
    if samples <= 1:
        return render_at(t)
    us = t + np.linspace(0, shutter, samples, endpoint=False) * dt
    return linear_mean([render_at(float(u)) for u in us])
