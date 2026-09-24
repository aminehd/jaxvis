"""Generate the gallery: each entry is a snippet and the GIF it produces.

    python3 jaxvis-pkg/gallery.py

Nothing is hand-drawn. Every image below is whatever `visualize` made from the
snippet printed above it.
"""
import textwrap
from pathlib import Path

OUT = Path("sources/jaxvis")
OUT.mkdir(parents=True, exist_ok=True)

HEADER = '''import jax, jax.numpy as jnp
from jax import random
from jaxvis import visualize
k = random.split(random.key(0), 4)
'''

ENTRIES = [
    ("One operation", "sin", """
def f(x):
    return jnp.sin(x)

X = jnp.linspace(-6, 6, 96)[:, None] * jnp.linspace(-1, 1, 96)[None, :]
visualize(f, X, out=OUT, size=300, tween=22, hold=6)
""", "A single op. The **values** interpolate, so the structure travels."),

    ("A point cloud, transformed", "points", """
def f(p):
    return jnp.tanh(p @ jnp.array([[1.5, 0.7], [-0.6, 1.1]]))

p = random.normal(k[0], (9000, 2))
visualize(f, p, out=OUT, size=300, tween=24, hold=6, palette="neon")
""", "`(n, 2)` is detected as coordinates, so it draws as particles that move."),

    ("Same data, forced to a heatmap", "points_as_heat", """
def f(p):
    return jnp.tanh(p @ jnp.array([[1.5, 0.7], [-0.6, 1.1]]))

p = random.normal(k[0], (9000, 2))
visualize(f, p, out=OUT, size=300, tween=24, hold=6, rep="heatmap")
""", 'Identical snippet, `rep="heatmap"`. You choose what becomes what.'),

    ("Motion blur, beesandbombs style", "blur", """
def f(p):
    a = jnp.arctan2(p[:, 1], p[:, 0])[:, None]
    r = jnp.linalg.norm(p, axis=1, keepdims=True)
    return p * jnp.cos(a * 3 + r * 4) * 1.6

p = random.normal(k[0], (14000, 2))
visualize(f, p, out=OUT, size=300, tween=20, hold=4,
          samples=10, shutter=0.8, palette="bloom")
""", "`samples` sub-frames per output frame, averaged in **linear light** "
     "(square, mean, square-root). Dave Whyte's trick — averaging sRGB directly "
     "makes blur muddy; this keeps it bright."),

    ("A lattice, distorted by its own gradient", "warpmesh", """
N = 220
g = jnp.linspace(-3, 3, N)
X, Y = g[:, None] * jnp.ones(N), jnp.ones(N)[:, None] * g

def field(X, Y):
    r = jnp.sqrt(X**2 + Y**2)
    a = jnp.arctan2(Y, X)
    w = jnp.sin(r * 3.4 - a * 2.0)
    return jnp.tanh(w * jnp.exp(-r * 0.3) * 2.4)

visualize(field, X, Y, out=OUT, size=560, tween=14, hold=1, samples=6,
          link=10, rep="warp", palette="ultra", colors=48)
""", "A new representation: a square mesh **displaced by the field's own "
     "gradient**. You read the values as bulges and pinches instead of colour. "
     "Steep regions crowd the lines together; flat regions stay square."),

    ("Points: rotate, then flee to the edges", "story", """
# a story you can narrate: spin, push outward, saturate
p = random.normal(k[0], (18000, 2)) * 0.9

def act(p):
    r = jnp.linalg.norm(p, axis=1, keepdims=True)
    th = r * 2.6                                  # 1. rotate, more at the rim
    c, s = jnp.cos(th), jnp.sin(th)
    p = jnp.concatenate([p[:, :1]*c - p[:, 1:]*s,
                         p[:, :1]*s + p[:, 1:]*c], 1)
    p = p * (1.0 + 1.9 / (1.0 + r * r))           # 2. shove outward from centre
    return jnp.tanh(p * 1.15)                     # 3. tanh pins them to a square

visualize(act, p, out=OUT, size=560, tween=22, hold=2, samples=12,
          shutter=0.9, only="points", structural=True,   # keep the rotation!
          palette="neon", colors=64)
""", "**The cloud twists, the middle blows outward into a ring, then `tanh` "
     "flattens the ring against a square wall.** Three ops, three sentences -- "
     "the shear comes from rotating by radius, and the square edge is tanh "
     "saturating both coordinates at once. `structural=True` matters here -- the "
     "rotation IS a `concatenate`, and the default filter hides those."),

    ("A continuous field", "ripple", """
N = 600                       # sample densely -- smoothness starts here
g = jnp.linspace(-3.2, 3.2, N)
X, Y = g[:, None] * jnp.ones(N), jnp.ones(N)[:, None] * g

def ripple(X, Y):
    r = jnp.sqrt(X**2 + Y**2)
    a = jnp.arctan2(Y, X)
    w = jnp.sin(r * 5.5 - a * 3.0)          # spiral interference
    v = w * jnp.exp(-r * 0.45)              # falls off from the centre
    return jnp.tanh(v * 2.6)

visualize(ripple, X, Y, out=OUT, size=620, tween=14, hold=1,
          samples=8, link=8, palette="ultra", colors=64)
""", "Every intermediate is an analytic function of position, so nothing is "
     "grainy. Two things make a heatmap smooth: **sample densely** (600x600 "
     "here, so there is no upscaling) and interpolate with BICUBIC rather than "
     "`np.repeat` -- the old renderer duplicated 334 of 400 rows, which is what "
     "the blockiness was."),

    ("The backward pass at 1000x1000", "grad_1k", """
N = 1000                      # renders DOWN to 620 -- supersampling, not stretching
g = jnp.linspace(-3, 3, N)
X = ((jnp.sin(g[:, None] * 2.1) * jnp.cos(g[None, :] * 1.7)) * 1.4
     + random.normal(k[3], (N, N)) * 0.55)

def loss(x):
    h = jnp.tanh(x * 1.6)
    return jnp.sum(h * h)

visualize(jax.grad(loss), X, out=OUT, size=560, tween=12, hold=1,
          samples=8, link=8, palette="spectral", colors=48)
""", "48 MB of intermediates, 0.24s to trace -- cheap. And because 1000 -> 620 "
     "is a **downscale**, every output pixel averages ~2.6 input pixels. That is "
     "free anti-aliasing: the noise becomes fine grain instead of chunks."),

    ("Domain warp", "warp", """
N = 620
g = jnp.linspace(-2.6, 2.6, N)
X, Y = g[:, None] * jnp.ones(N), jnp.ones(N)[:, None] * g

def warp(X, Y):
    qx = jnp.sin(X * 1.7 + jnp.cos(Y * 2.3))      # warp the coordinates...
    qy = jnp.cos(Y * 1.9 + jnp.sin(X * 2.1))
    r = jnp.sqrt(qx**2 + qy**2)
    return jnp.tanh(jnp.sin(r * 4.2 + qx * 2.0) * 2.0)

visualize(warp, X, Y, out=OUT, size=560, tween=12, hold=1,
          samples=8, link=8, palette="vapor", colors=48)
""", "Feed a field into its own coordinates. Every step is analytic, so it stays "
     "liquid -- no grain to amplify."),

    ("A complex map", "complex", """
N = 620
g = jnp.linspace(-1.8, 1.8, N)
X, Y = g[:, None] * jnp.ones(N), jnp.ones(N)[:, None] * g

def julia(X, Y):
    zx, zy = X, Y
    for _ in range(6):                 # z -> z^2 + c, unrolled in the jaxpr
        zx, zy = zx*zx - zy*zy - 0.70, 2*zx*zy + 0.27
        zx, zy = jnp.tanh(zx * 0.9), jnp.tanh(zy * 0.9)
    return jnp.sqrt(zx**2 + zy**2)

visualize(julia, X, Y, out=OUT, size=560, tween=14, hold=2, samples=8,
          link=8, ops="tanh", only="heatmap", palette="ultra", colors=48)
""", "`z -> z^2 + c` six times. The python loop unrolls, so each iteration is a "
     "keyframe. `ops=\"tanh\"` keeps just the iteration boundaries -- without it "
     "the unrolled loop emits over a thousand frames."),

    ("Moire", "moire", """
N = 700
g = jnp.linspace(-3, 3, N)
X, Y = g[:, None] * jnp.ones(N), jnp.ones(N)[:, None] * g

def moire(X, Y):
    th = 0.22
    u = X * jnp.cos(th) - Y * jnp.sin(th)         # a rotated copy
    a = jnp.sin(jnp.sqrt(X**2 + Y**2) * 14.0)
    b = jnp.sin(jnp.sqrt(u**2 + (Y*1.06)**2) * 14.0)
    return jnp.tanh((a * b) * 3.0)

visualize(moire, X, Y, out=OUT, size=560, tween=12, hold=1,
          samples=10, link=8, palette="neon", colors=48)
""", "Two ring gratings, one rotated 0.22 rad. Their product is interference -- "
     "and at 700x620 the beat pattern resolves instead of aliasing."),

    ("grad — the chain rule, backwards", "grad", """
def loss(x):
    h = jnp.tanh(x * 1.6)
    s = h * h
    return jnp.sum(s)

X = random.normal(k[0], (48, 48))
visualize(jax.grad(loss), X, out=OUT, size=320, tween=40, hold=3,
          code=True, samples=14, link=16, palette="bloom")
""", "Watch the highlight go **down the function and then back up**. "
     "Four ops forward, twelve for the gradient: `sub` is `1 - tanh^2`, the "
     "`mul` chain is the chain rule, and `add_any` is gradient accumulation "
     "where a value was used twice."),

    ("Noise falling into a shape", "emerge", """
# the potential's minima ARE a curve, so descending lands you on it
S = 240
s = jnp.linspace(0, 2 * jnp.pi, S)
curve = jnp.stack([jnp.cos(3 * s) * jnp.cos(s),
                   jnp.cos(3 * s) * jnp.sin(s)], 1) * 1.25   # a rose

def potential(p):
    d2 = ((p[:, None, :] - curve[None, :, :]) ** 2).sum(-1)   # (n, S)
    return jnp.sum(jnp.min(d2, axis=1))

field = jax.grad(potential)

def emerge(p):
    for _ in range(11):            # stop short -- a halo survives
        p = p - 0.20 * field(p)
    return p

p = random.normal(k[2], (34000, 2)) * 1.35     # starts as pure noise
visualize(emerge, p, out=OUT, size=620, tween=20, hold=2, samples=14,
          shutter=0.9, only="points", ops="sub", palette="spectral")
""", "**Random noise in, a rose curve out.** The potential is the squared "
     "distance to the nearest point on the curve, so `-grad` pulls every "
     "particle onto it. 620px, `spectral` palette."),

    ("The backward pass, big", "grad_big", """
N = 512
g = jnp.linspace(-3, 3, N)
smooth = (jnp.sin(g[:, None] * 2.1) * jnp.cos(g[None, :] * 1.7)
          + jnp.sin((g[:, None] + g[None, :]) * 1.3) * 0.6)
X = smooth * 1.4 + random.normal(k[3], (N, N)) * 0.55    # structure + noise

def loss(x):
    h = jnp.tanh(x * 1.6)
    s = h * h
    return jnp.sum(s)

visualize(jax.grad(loss), X, out=OUT, size=560, tween=16, hold=1,
          samples=10, link=10, palette="spectral", colors=64)
""", "512x512 in, 560px out. The whole backward pass over a field that is "
     "smooth structure plus noise -- so `tanh` visibly saturates the peaks and "
     "`1 - tanh^2` lights up exactly the edges it flattened."),

    ("The backward pass, with the code walked", "grad_big_code", """
N = 384
g = jnp.linspace(-3, 3, N)
smooth = (jnp.sin(g[:, None] * 2.1) * jnp.cos(g[None, :] * 1.7)
          + jnp.sin((g[:, None] + g[None, :]) * 1.3) * 0.6)
X = smooth * 1.4 + random.normal(k[3], (N, N)) * 0.55

def loss(x):
    h = jnp.tanh(x * 1.6)
    s = h * h
    return jnp.sum(s)

visualize(jax.grad(loss), X, out=OUT, size=460, tween=16, hold=2, samples=10,
          link=10, code=True, palette="spectral", colors=64)
""", "Same backward pass with the source beside it. The highlight walks **down "
     "the function and then back up** -- forward to `sum`, then reverse through "
     "`h * h` and back to the `tanh` line as the chain rule unwinds."),

    ("Gradient descent, 40k particles", "descent", """
def bowl(p):                      # a lumpy potential
    return jnp.sum(jnp.sum(p**2, -1) * 0.5
                   - jnp.cos(p[:, 0] * 3.2) * jnp.cos(p[:, 1] * 3.2) * 0.9)

field = jax.grad(bowl)

def descend(p):
    for _ in range(14):           # unrolls into 14 steps in the jaxpr
        p = p - 0.055 * field(p)
    return p

p = random.normal(k[1], (40000, 2)) * 1.5
visualize(descend, p, out=OUT, size=340, tween=14, hold=1, samples=12,
          shutter=0.9, only="points", ops="sub", palette="neon")
""", "Every particle follows `-grad(bowl)` into the nearest minimum. The python "
     "`for` loop unrolls into 14 steps in the jaxpr. `ops=\"sub\"` keeps just the "
     "updates -- without it the gradient's own 127 intermediates come too. "
     "12 motion-blur samples per frame."),

    ("Gradient descent, with the code", "descent_code", """
def bowl(p):
    return jnp.sum(jnp.sum(p**2, -1) * 0.5
                   - jnp.cos(p[:, 0] * 3.2) * jnp.cos(p[:, 1] * 3.2) * 0.9)

field = jax.grad(bowl)

def descend(p):
    for _ in range(14):
        p = p - 0.055 * field(p)
    return p

p = random.normal(k[1], (40000, 2)) * 1.5
visualize(descend, p, out=OUT, size=320, tween=14, hold=1, samples=12,
          shutter=0.9, only="points", ops="sub", code=True, palette="neon")
""", "The same descent, with the source beside it. The highlight sits on the "
     "update line while 40,000 particles fall into the minima."),

    ("Palettes", "palette_neon", """
def f(p):
    a = jnp.arctan2(p[:, 1], p[:, 0])[:, None]
    return p * (1.4 + jnp.sin(a * 5) * 0.5)

p = random.normal(k[2], (11000, 2))
visualize(f, p, out=OUT, size=300, tween=20, hold=4,
          samples=8, palette="vapor")
""", "`palette=` takes **neon · bloom · acid · vapor · ember · ice · mono**. "
     "They're colour stops in `render.STOPS` — add your own with one line."),

    ("Attention", "attention", """
def f(Q, K, V):
    w = jax.nn.softmax(Q @ K.T / jnp.sqrt(16), axis=-1)
    return w @ V

Q, K, V = (random.normal(k[i], (48, 16)) for i in (1, 2, 3))
visualize(f, Q, K, V, out=OUT, size=300, tween=14, hold=5, palette="acid")
""", "`softmax` is not one op — it's max, broadcast, sub, exp, sum, divide."),
]


def main():
    md = ["---", "title: jaxvis gallery", "---", "",
          "Every GIF below was generated from the snippet above it. Nothing hand-drawn.",
          "", "```bash", "pip install -e jaxvis-pkg", "```", ""]
    for title, slug, body, note in ENTRIES:
        gif = OUT / f"{slug}.gif"
        snip = OUT / "snippets" / f"{slug}.py"
        snip.parent.mkdir(exist_ok=True)
        src = HEADER + body.replace("out=OUT", f'out="{gif}"')
        snip.write_text(src)
        ns = {"__file__": str(snip)}
        exec(compile(src, str(snip), "exec"), ns)
        shown = textwrap.dedent(body).strip().replace("out=OUT", 'out="out.gif"')
        md += [f"## {title}", "", note, "", "```python", shown, "```", "",
               f"![{slug}](./{slug}.gif)", "", "---", ""]
    (OUT / "index.md").write_text("\n".join(md))
    print(f"\n  gallery -> {OUT/'index.md'}  ({len(ENTRIES)} entries)")


if __name__ == "__main__":
    main()
