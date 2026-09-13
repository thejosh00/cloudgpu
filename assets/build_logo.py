#!/usr/bin/env python3
"""Regenerate the CloudGPU logo lockups.

The wordmark is set in Source Code Pro (SIL OFL 1.1) and converted to outlines, so
the SVGs render identically everywhere without shipping or depending on a font.

    uv run --with fonttools python assets/build_logo.py

Only the lockups (logo*.svg) are generated. The mark itself (logo-mark.svg,
logo-mark-mono.svg, favicon.svg) is hand-authored geometry -- edit those directly.
"""

from pathlib import Path

from fontTools.misc.transform import Transform
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

ASSETS = Path(__file__).parent

FONTS = {
    "regular": "/Users/joshuahill/Library/Fonts/Source Code Pro for Powerline.otf",
    "bold": "/Users/joshuahill/Library/Fonts/Source Code Pro Bold for Powerline.otf",
}
SEGMENTS = [("cloud", "regular"), ("gpu", "bold")]

EM = 34.0          # wordmark em size, in viewBox units
WORDMARK_X = 74.0  # left edge of the wordmark's advance box
BASELINE = 40.5    # chosen so the wordmark's ink centres on the mark's centre (y=32)
PAD_RIGHT = 7.0    # matches the mark's own left padding

INDIGO, CYAN = "#4F46E5", "#06B6D4"
INK, PAPER = "#0F172A", "#F8FAFC"


def outline():
    """Return {weight: path_d}, plus the ink's right edge, in viewBox units."""
    paths, x, right = {}, 0.0, 0.0
    for text, weight in SEGMENTS:
        font = TTFont(FONTS[weight])
        upem = font["head"].unitsPerEm
        glyphs, cmap, hmtx = font.getGlyphSet(), font.getBestCmap(), font["hmtx"]
        scale = EM / upem
        drawn = []
        for ch in text:
            name = cmap[ord(ch)]
            pen = SVGPathPen(glyphs, ntos=lambda v: f"{v:.2f}")
            # font y is up-positive, SVG y is down-positive -> flip while scaling
            glyphs[name].draw(TransformPen(pen, Transform(scale, 0, 0, -scale, x, 0)))
            if cmds := pen.getCommands():
                drawn.append(cmds)
            x += hmtx[name][0] * scale
        paths[weight] = " ".join(drawn)
        font.close()
        right = x
    return paths, right


MARK = """\
  <g fill="{mark}">
    <rect x="7" y="22" width="7" height="4" rx="2"/>
    <rect x="7" y="30" width="7" height="4" rx="2"/>
    <rect x="7" y="38" width="7" height="4" rx="2"/>
    <rect x="50" y="22" width="7" height="4" rx="2"/>
    <rect x="50" y="30" width="7" height="4" rx="2"/>
    <rect x="50" y="38" width="7" height="4" rx="2"/>
    <rect x="22" y="7" width="4" height="7" rx="2"/>
    <rect x="30" y="7" width="4" height="7" rx="2"/>
    <rect x="38" y="7" width="4" height="7" rx="2"/>
    <rect x="22" y="50" width="4" height="7" rx="2"/>
    <rect x="30" y="50" width="4" height="7" rx="2"/>
    <rect x="38" y="50" width="4" height="7" rx="2"/>
  </g>
  <rect x="14" y="14" width="36" height="36" rx="8" fill="{mark}" mask="url(#cg-cloud)"/>"""

GRADIENT = f"""\
    <linearGradient id="cg-grad" x1="14" y1="14" x2="50" y2="50" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{INDIGO}"/>
      <stop offset="1" stop-color="{CYAN}"/>
    </linearGradient>"""

CLOUD_MASK = """\
    <mask id="cg-cloud">
      <rect x="0" y="0" width="{w}" height="64" fill="#fff"/>
      <g fill="#000">
        <circle cx="26.5" cy="35" r="5.5"/>
        <circle cx="37.5" cy="35" r="5.5"/>
        <circle cx="32" cy="30.5" r="7"/>
        <rect x="26.5" y="35" width="11" height="5.5"/>
      </g>
    </mask>"""


def lockup(paths, width, *, mark, word, gradient):
    defs = (GRADIENT + "\n" if gradient else "") + CLOUD_MASK.format(w=width)
    return f"""\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 64" width="{width}" height="64" role="img" aria-label="CloudGPU">
  <title>CloudGPU</title>
  <defs>
{defs}
  </defs>
{MARK.format(mark=mark)}
  <g transform="translate({WORDMARK_X} {BASELINE})" fill="{word}">
    <path d="{paths['regular']}"/>
    <path d="{paths['bold']}"/>
  </g>
</svg>
"""


def main():
    paths, right = outline()
    width = round(WORDMARK_X + right + PAD_RIGHT)
    grad = "url(#cg-grad)"
    for name, mark, word, gradient in [
        ("logo.svg", grad, INK, True),
        ("logo-dark.svg", grad, PAPER, True),
        ("logo-mono.svg", "currentColor", "currentColor", False),
    ]:
        (ASSETS / name).write_text(lockup(paths, width, mark=mark, word=word, gradient=gradient))
        print(f"wrote {name}  ({width}x64)")


if __name__ == "__main__":
    main()
