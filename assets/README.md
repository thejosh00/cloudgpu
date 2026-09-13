# CloudGPU brand assets

The mark is a **chip die with a cloud knocked out of it** — silicon you don't own,
which is the whole idea of the tool. The cloud is a true knockout (transparent), so
the mark sits on any background without a plate behind it.

The wordmark is set in monospace, lowercase, with `gpu` bold: it reads as the command
you actually type.

## Files

| File | Use |
| --- | --- |
| `logo.svg` | Full lockup, for light backgrounds |
| `logo-dark.svg` | Full lockup, for dark backgrounds |
| `logo-mono.svg` | Full lockup in one colour — inherits CSS `color` via `currentColor` |
| `logo-mark.svg` | Mark only, with pins. Use at **32px and up** |
| `logo-mark-mono.svg` | Mark only, one colour via `currentColor` |
| `favicon.svg` | Mark without pins, die enlarged. Use at **32px and down** |

PNG renditions (`logo-488.png`, `logo-dark-488.png`, `logo-mark-256.png`,
`icon-512.png`, `apple-touch-icon-180.png`, `favicon-32.png`) are exported from those
SVGs — regenerate rather than edit them.

### Which mark at which size

`logo-mark.svg` loses its pins below about 32px and turns to mush at favicon sizes.
`favicon.svg` drops the pins and scales the die up to fill the canvas so the cloud
stays readable at 16px. Use the right one; don't scale `logo-mark.svg` down to 16.

## Colour

| | Hex | |
| --- | --- | --- |
| Indigo | `#4F46E5` | gradient start (top-left) |
| Cyan | `#06B6D4` | gradient end (bottom-right) |
| Ink | `#0F172A` | wordmark on light |
| Paper | `#F8FAFC` | wordmark on dark |

The gradient runs across the die only; the pins sample the same gradient so the whole
mark shares one light direction. On a one-colour background, or anywhere the gradient
can't be trusted (embroidery, single-colour print, terminal output), use the `-mono`
files instead of approximating it.

## Clearspace and minimum size

Keep clear on all sides at least the height of one pin group (`7` units in the
viewBox, ~11% of the mark's height). The lockup's built-in padding already satisfies
this — don't crop it.

Minimum sizes: lockup **120px** wide, `logo-mark.svg` **32px**, `favicon.svg` **16px**.

## Regenerating the lockups

The wordmark is **outlined**, not live text, so the SVGs render identically everywhere
with no font to ship or install. To change the wording, size, or spacing, edit the
constants at the top of `build_logo.py` and re-run it:

```bash
uv run --with fonttools python assets/build_logo.py
```

Then re-export the PNGs with `rsvg-convert` (sizes are listed in the table above).

`build_logo.py` reads Source Code Pro from a hardcoded `~/Library/Fonts` path — it
only runs on a machine where that font is installed. It is needed solely to regenerate
the lockups; the committed SVGs have no such dependency.

The mark itself (`logo-mark.svg`, `logo-mark-mono.svg`, `favicon.svg`) is hand-authored
geometry and is **not** generated — edit those files directly.

## Licensing

The wordmark outlines are derived from **Source Code Pro**, licensed under the
[SIL Open Font License 1.1](https://github.com/adobe-fonts/source-code-pro/blob/main/LICENSE.md),
which permits embedding outlines in a document like this.
