"""Figure 1: parts of one key recovery and how they connect (a diagram, no data).

(a) One run, in the order the run meets its parts (Section 5): shaped exponent registers, the first lookup with the
random start, 14 windowed additions over unit-orbit or mixed-radix windows on the borrowed core, the full or root-sign
last addition (also on the core), and the measurements with the visible flags. (b) One key recovery: the random key
shift, the per-run classical preparation (fresh scalar, start point, tables), the 79 to 110 runs, each one a run of (a),
the flagged runs dropped, and the lattice decoder on random subsets within 2^40 classical operations. Each of our parts
carries the challenge it meets (C1-C3 of Section 1, named in the caption).
Colors: blue = our parts, gray = the borrowed point-addition core; the stack of runs is neutral because each of its runs
is all of (a). Box widths follow their text and the gaps are equal, so both rows span the same width.
Run: python3 paper/figures/fig1_overview.py  (writes paper/figures/fig1_overview.pdf)
"""
import os
from matplotlib.patches import FancyBboxPatch
from style import OURS, OTHER, INK, MUTED, tint, figure, save

HERE = os.path.dirname(os.path.abspath(__file__))
OUR = dict(facecolor=tint(OURS, 0.12), edgecolor=OURS)
BORROWED = dict(facecolor=tint(OTHER, 0.16), edgecolor=OTHER)
STACK = dict(facecolor="white", edgecolor=MUTED)
PAD, GAP, BOX_H, CARD = 0.08, 0.26, 0.44, 0.035       # inches: text padding, gap between boxes, box height, stack offset
MARGIN = 0.03

fig = figure(1.0, 1.96)
W, H = fig.get_size_inches()
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
renderer = fig.canvas.get_renderer()


def width(s, size=8, **kw):
    t = ax.text(0, 0, s, fontsize=size, **kw)
    w = t.get_window_extent(renderer).width / fig.dpi
    t.remove()
    return w


def rounded(x, y, w, h, style, r=0.035, lw=0.8, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % r, lw=lw, zorder=z, **style))


def tag(x, y, text, edge):
    """Challenge tag: a small pill centered on (x, y), the top edge of its box."""
    w, h = width(text, 7) + 0.09, 0.14
    rounded(x - w / 2, y - h / 2, w, h, dict(facecolor="white", edgecolor=edge), r=h / 2, lw=0.6, z=4)
    ax.text(x, y, text, ha="center", va="center", fontsize=7, zorder=5)


def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, mutation_scale=7,
                                                        shrinkA=0, shrinkB=0), zorder=1)


def row(items, y, x0, x1):
    """Boxes of one row from x0 to x1, joined by arrows; items are (text, style, tag). Returns their extents."""
    extra_w = [2 * CARD if style is STACK else 0 for _, style, _ in items]
    natural = [max(width(line) for line in text.split("\n")) + 2 * PAD for text, _, _ in items]
    fill = (x1 - x0 - sum(natural) - sum(extra_w) - GAP * (len(items) - 1)) / len(items)
    out, x = [], x0
    for (text, style, t), n, e in zip(items, natural, extra_w):
        w = n + fill
        if style is STACK:
            for k in (2, 1):
                rounded(x + k * CARD, y + k * CARD, w, BOX_H, style, lw=0.7, z=2)
        rounded(x, y, w, BOX_H, style, lw=0.7 if style is STACK else 0.8, z=3 if style is STACK else 2)
        ax.text(x + w / 2, y + BOX_H / 2 - 0.015, text, ha="center", va="center", fontsize=8, linespacing=1.3,
                zorder=3)
        if t:
            tag(x + w - 0.07 - width(t, 7) / 2, y + BOX_H, t, style["edgecolor"])
        out.append((x, w + e))
        x += w + e + GAP
    for (xa, wa), (xb, _) in zip(out, out[1:]):
        arrow(xa + wa + 0.03, y + BOX_H / 2, xb - 0.03, y + BOX_H / 2)
    return out


def header(y, letter, title):
    ax.text(MARGIN, y, letter, ha="left", va="top", fontsize=9, family="cmb10")
    ax.text(MARGIN + width(letter, 9, family="cmb10") + 0.06, y, title, ha="left", va="top", fontsize=9)


# (a) One run.
yb = MARGIN                                  # bottom of row (b); the rows stack upward from here
hb = yb + BOX_H + 0.26                       # top of header (b)
core_y, core_h = hb + 0.16, 0.22             # the borrowed core, under row (a)
ya = core_y + core_h + 0.14                  # bottom of row (a)
header(H - 0.025, "(a)", "One run")
a = row([("Shaped exponent\nregisters", OUR, "C1"),
         ("First lookup with\nrandom start", OUR, "C3"),
         ("14 windowed additions,\nunit-orbit or mixed-radix", OUR, "C2"),
         ("Full or root-sign\nlast addition", OUR, "C2"),
         ("Measurement, flags,\non-curve test", OUR, "C3")], ya, MARGIN, W - MARGIN)
c3, c4 = a[2][0] + a[2][1] / 2, a[3][0] + a[3][1] / 2
core_w = width("Borrowed point-addition core") + 2 * PAD
rounded((c3 + c4) / 2 - core_w / 2, core_y, core_w, core_h, BORROWED)
ax.text((c3 + c4) / 2, core_y + core_h / 2, "Borrowed point-addition core", ha="center", va="center", fontsize=8,
        zorder=3)
for c in (c3, c4):
    ax.plot([c, c], [core_y + core_h, ya], color=OTHER, lw=0.8, zorder=1, solid_capstyle="butt")

# Key to the colors, on the line of the first header.
x = W - MARGIN
for name, style in (("Borrowed", BORROWED), ("Our parts", OUR)):
    x -= width(name)
    ax.text(x, H - 0.025 - 0.065, name, ha="left", va="center", fontsize=8)
    x -= 0.06 + 0.2
    rounded(x, H - 0.025 - 0.065 - 0.055, 0.2, 0.11, style, r=0.025, lw=0.7)
    x -= 0.22

# (b) One key recovery.
header(hb, "(b)", "One key recovery")
out_w = width("Key $d$")
b = row([("Key shift\n$Q+[\\rho]G$", OUR, "C3"),
         ("Fresh scalar, start point\nand tables for each run", OUR, "C3"),
         ("79 to 110 runs,\neach as in (a)", STACK, None),
         ("Drop flagged\nruns", OUR, "C3"),
         ("Lattice decoder on random\nsubsets, $\\leq 2^{40}$ operations", OUR, "C1")], yb, MARGIN,
        W - MARGIN - out_w - GAP - 0.04)
arrow(b[-1][0] + b[-1][1] + 0.03, yb + BOX_H / 2, W - MARGIN - out_w - 0.05, yb + BOX_H / 2)
ax.text(W - MARGIN, yb + BOX_H / 2, "Key $d$", ha="right", va="center", fontsize=8)

save(fig, os.path.join(HERE, "fig1_overview.pdf"))
