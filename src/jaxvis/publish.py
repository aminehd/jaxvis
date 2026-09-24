"""GIF -> MP4, for anything going on a public page.

A GIF is a palette of 256 colours with no interframe compression. H.264 is
~12x smaller for identical-looking output, and <video autoplay loop muted
playsinline> behaves exactly like a GIF in a browser.
"""
import shutil
import subprocess
from pathlib import Path


def to_mp4(gif, out=None, crf=32, fps=25):
    """Convert one GIF. Returns (path, gif_kb, mp4_kb).

    ffmpeg 4.x spells constant frame rate `-vsync cfr`; `-fps_mode` is 5.x+.

    crf 32 rather than 20 because this content is OVERSAMPLED and GRAINY, and
    both hide compression. Rendered at ~680px, displayed at ~420: downscaling
    averages artifacts away. And particle splatter / paper grain is noise, so
    it masks the blocking that would show in a smooth gradient. Grain costs a
    lot to encode (a grainy GIF is 3x the bytes) and buys most of it back by
    letting crf go up. Render big, add texture, compress hard, display small.

    `fps` is forced, and should be an even multiple of the GIF's own rate
    (80 ms frames = 12.5 fps, so 25 is exactly 2 video frames each). Let
    ffmpeg choose and it lands GIF frames on an uneven number of video
    frames, which reads as judder even though the total duration is right.
    """
    gif = Path(gif)
    out = Path(out) if out else gif.with_suffix(".mp4")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(gif),
         "-movflags", "+faststart", "-pix_fmt", "yuv420p",
         # H.264 needs even dimensions; odd ones fail with a cryptic error
         "-vf", f"scale=trunc(iw/2)*2:trunc(ih/2)*2,fps={fps}",
         "-vsync", "cfr", "-crf", str(crf), str(out)],
        check=True, capture_output=True)
    return out, gif.stat().st_size // 1024, out.stat().st_size // 1024


VIDEO_TAG = ('<video src="{src}" autoplay loop muted playsinline '
             'style="width:100%;max-width:{w}px;border-radius:8px"></video>')


def video_tag(src, w=520):
    "Markdown renders raw HTML on GitHub and Pages, so this works in both."
    return VIDEO_TAG.format(src=src, w=w)
