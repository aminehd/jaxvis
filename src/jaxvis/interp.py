"""Walk a jaxpr, evaluate every equation, keep every intermediate."""
from typing import NamedTuple, Optional

import jax
import numpy as np
from jax.extend import core


# Primitives that move data around without changing it. A jaxpr is full of
# these and they dominate the film if you draw them -- a 3-line function can
# emit 30 steps, 25 of which are reshapes.
STRUCTURAL = frozenset({
    "slice", "squeeze", "reshape", "broadcast_in_dim", "transpose",
    "convert_element_type", "concatenate", "copy", "device_put", "pjit",
    "stop_gradient", "expand_dims", "rev", "select_n",
})


class Step(NamedTuple):
    name: str
    src: Optional[np.ndarray]   # same-shaped input, if any -> value morph
    out: np.ndarray
    site: Optional[str] = None  # "path/to/file.py:12" -- which line made this


def _site(eqn):
    "Which line of Python produced this equation."
    try:
        from jax._src import source_info_util as siu
        s = siu.summarize(eqn.source_info)          # "file.py:12 (fname)"
        return s.split(" ")[0] if s else None
    except Exception:
        return None


def trace_values(f, *args, keep=None, structural=False):
    """Run `f` one jaxpr equation at a time.

    Returns [(label, ndarray), ...] in program order — the inputs first, then
    one entry per primitive that produced something worth drawing.

    `keep(name, array) -> bool` filters which steps are recorded.
    """
    closed = jax.make_jaxpr(f)(*args)
    jaxpr, consts = closed.jaxpr, closed.literals
    env = {}

    def read(v):
        return v.val if isinstance(v, core.Literal) else env[v]

    for v, val in zip(jaxpr.constvars, consts):
        env[v] = val
    for v, val in zip(jaxpr.invars, args):
        env[v] = val

    if keep is None:
        def keep(n, a):
            if not structural and n in STRUCTURAL:
                return False
            return a.ndim <= 3 and a.size > 1
    steps = [Step("input", None, np.asarray(a), None) for a in args
             if np.asarray(a).size > 1]

    for eqn in jaxpr.eqns:
        invals = [read(v) for v in eqn.invars]
        outvals = eqn.primitive.bind(*invals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        for v, val in zip(eqn.outvars, outvals):
            env[v] = val
        out = np.asarray(outvals[0])
        if keep(eqn.primitive.name, out):
            # The "from" state for this op: the first input with a matching
            # shape. When one exists we can interpolate in VALUE space, which
            # makes the operation itself the animation rather than a dissolve.
            src = None
            for iv in invals:
                arr = np.asarray(iv)
                if arr.shape == out.shape:
                    src = arr
                    break
            steps.append(Step(eqn.primitive.name, src, out, _site(eqn)))
    return steps
