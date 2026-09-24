"""The playground: decorate a function, it renders, the web page updates.

    from jaxvis.play import show

    X = jnp.linspace(-6, 6, 120)[:, None] * jnp.linspace(-1, 1, 120)[None, :]

    @show(X, palette="neon")
    def ripple(x):
        return jnp.sin(x) * jnp.tanh(x)

Rendering happens at decoration time, so running the cell in molten is all it
takes. The function itself is returned unchanged -- you can still call it.
"""
import inspect
import json
import os
import time
from pathlib import Path

OUT = Path(os.environ.get(
    "JAXVIS_PLAY",
    Path(__file__).resolve().parents[3] / "playground" / "out"))


def _manifest(d):
    f = d / "manifest.json"
    try:
        return json.loads(f.read_text())
    except Exception:
        return []


def show(*args, out=None, style=None, **kw):
    """Render this function and publish it to the playground page.

    style="bars"    force EVERY step to points, including the single-column
                    ones -- they plot value-against-index, which is the
                    bar-of-dots look. Shows the whole computation.

    style="clean"   only 2-column steps are drawn; single-column intermediates
                    are dropped instead of smeared. Shows just the cloud.
                    Needs the function to END on an elementwise op -- if the
                    last op is a concatenate, its source is (n,1), value_morph
                    can't interpolate, and you get a cross-fade with no motion.

    Anything you pass explicitly wins over the style.
    """
    d = Path(out) if out else OUT
    if style:
        import numpy as _np
        n = _np.asarray(args[0]).shape[0] if args else None
        if style == "clean":
            kw.setdefault("rep", {(n, 2): "points"})
            kw.setdefault("only", "points")
            kw.setdefault("structural", True)
        elif style == "bars":
            kw.setdefault("rep", "points")
        else:
            raise ValueError(f"style must be 'bars' or 'clean', got {style!r}")
    d.mkdir(parents=True, exist_ok=True)

    def deco(fn):
        from jaxvis import visualize
        name = fn.__name__
        gif = f"{name}.gif"
        try:
            src = inspect.getsource(fn)
            src = "\n".join(l for l in src.splitlines()
                            if not l.lstrip().startswith("@show"))
        except Exception:
            src = f"def {name}(...): ..."
        t0 = time.time()
        kw.setdefault("verbose", False)
        try:
            visualize(fn, *args, out=str(d / gif), **kw)
            err = None
        except Exception as e:                    # keep the page alive
            err = f"{type(e).__name__}: {e}"
        rows = [r for r in _manifest(d) if r.get("name") != name]
        rows.insert(0, {"name": name, "code": src.rstrip(), "gif": gif,
                        "ts": time.time(), "secs": round(time.time() - t0, 1),
                        "err": err, "opts": {k: str(v) for k, v in kw.items()
                                             if k != "verbose"}})
        (d / "manifest.json").write_text(json.dumps(rows, indent=1))
        print(("  x " + err) if err else
              f"  {name} -> {gif}  ({time.time() - t0:.1f}s)")
        return fn
    return deco
