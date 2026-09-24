"""jaxvis <module.py> — run a file that defines `fn` and `args`, write the gif."""
import runpy
import sys

from . import visualize


def main():
    if len(sys.argv) < 2:
        print("usage: jaxvis <script.py> [out.gif]\n"
              "  the script must define `fn` and `args`")
        return 1
    ns = runpy.run_path(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "jaxvis.gif"
    visualize(ns["fn"], *ns["args"], out=out)
    return 0
