"""Shared figure style for the paper.

The paper is set in revtex4-2 (aps, prx, reprint), and every figure is a figure* included at a fraction of
\\textwidth = 510 pt = 7.06 in. Each script draws its figure at exactly that printed width and saves it without
cropping, so every size here is the size on the page: 8 pt text and tick labels, 9 pt axis labels, 9 pt bold panel
labels, all in Computer Modern (matplotlib's cmr10, cmb10 and its cm math fonts) to match the paper's Latin Modern.
Avoid <, >, {, |, _ and ^ outside math mode: cmr10 has the OT1 layout, where they are other glyphs.

Colors carry meaning, the same in every figure: blue = our constructions and what they measure or store, vermillion =
the baseline, gray = other runs, borrowed parts and reference quantities, black = bounds and floors. Blue and vermillion
are the Okabe-Ito pair, which stays distinct under the three common color-vision deficiencies, and every colored mark is
also named in the figure, by a direct label or a key.
"""
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXTWIDTH = 510 / 72.27            # inches; \textwidth of revtex4-2 reprint, the \linewidth of a figure*

OURS = "#0072B2"                   # blue: our constructions
BASE = "#D55E00"                   # vermillion: the baseline
OTHER = "#8C8C8C"                  # gray: other runs we priced
INK = "#1A1A1A"                    # text, axes, bounds and floors
MUTED = "#5A5A5A"                  # secondary text and leader lines
GUIDE = "#C8C8C8"                  # guides: zoom frames, sector lines, outlines

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.6,
    "axes.labelpad": 3,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.pad": 3.5,
    "ytick.major.pad": 3,
    "lines.linewidth": 1.0,
    "lines.markersize": 4.5,
    "lines.markeredgewidth": 0.8,
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "legend.borderaxespad": 0.4,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})
logging.getLogger("fontTools").setLevel(logging.ERROR)   # silence warnings about the old dates in cmr10.ttf

EKERA = r"Eker$\mathrm{å}$"   # cmr10 has no "å"; in math mode matplotlib takes it from its STIX fallback font


def tint(color, f):
    """`color` mixed with white, f = 0 white to 1 full color: an opaque light fill (no transparency in the PDF)."""
    rgb = matplotlib.colors.to_rgb(color)
    return tuple(1 - f * (1 - c) for c in rgb)


def figure(width_frac, height):
    """A figure exactly as wide as the paper prints it (width_frac of \\textwidth) and `height` inches tall."""
    return plt.figure(figsize=(width_frac * TEXTWIDTH, height))


def axes(fig, left, bottom, width, height):
    """Axes placed in inches from the figure's lower left corner, so panels and margins align across figures."""
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def points(ax, x, y, color, filled=True, **kw):
    """Data points: one marker style for every figure; open markers keep a white center."""
    kw = dict(marker="o", linestyle="none", color=color, markeredgecolor=color,
              markerfacecolor=color if filled else "white", zorder=4, clip_on=False, **kw)
    return ax.plot(x, y, **kw)


def label(ax, x, y, text, dx=0, dy=0, ha="left", va="center", color=INK, **kw):
    """Direct label at an offset in points from the data point (x, y)."""
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, color=color, **kw)


def panel_label(fig, x, y, text):
    """Panel label, e.g. "(a)", with its top left corner at (x, y) inches."""
    W, H = fig.get_size_inches()
    fig.text(x / W, y / H, text, ha="left", va="top", fontsize=9, family="cmb10")


def save(fig, path):
    fig.savefig(path, metadata={"CreationDate": None})   # no timestamp, so an unchanged figure rebuilds to the same bytes
    print("wrote", path)
