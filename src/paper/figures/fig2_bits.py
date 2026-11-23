"""Figure 2: exponent bits per run against the number of runs at n = 256.

Floors: Corollary 1 (runs fixed in advance) and Theorem 2 (adaptive), success 0.99, list of 2^80 candidates
(paper/figures/floors.py, same formulas as s3/theory/numbers_n256.py), drawn as smooth curves through their values at
integer run counts.
Points (exponent bits per run, runs per key recovery), every one read from a saved file:
  baseline, Haener et al.: 466 bits (18-bit initial lookup + 28 x 16), 3 attempts       harness/README.md l.104-116
  Ekera's tradeoff on the borrowed core, plain signed windows and plain decoding (the third control), s = 2..5:
    runs per key recovery = batches for 0.99 at the 95 % rule x runs per batch, bits per run from Ekera's table
    (s = 2: 9 x 3 runs of 392 bits; s = 3: 11 x 4 of 350; s = 4: 13 x 5 of 328, the control; s = 5: 17 x 6 of 317)
  same core, unsplit (U1, U2-h: 464 bits; U2-n, U2-nt: 464.5 bits as log2 norms, as in Table 3), 4 runs
Panel (b) enlarges the region of the last three and marks their distance above the floor of Corollary 1 at their own
run counts (floors.py: 12.16, 10.71 and 10.90 bits, Section 7.3).
Run: python3 paper/figures/fig2_bits.py  (writes paper/figures/fig2_bits.pdf)
"""
import csv
import os
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
from style import OURS, BASE, OTHER, INK, MUTED, GUIDE, EKERA, figure, axes, points, label, panel_label, save
from floors import bits_nonadaptive, bits_adaptive

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EKERA_CSV = os.path.join(REPO, "base", "ekera_tradeoff.csv")
EKERA_BITS = {2: 392, 3: 350, 4: 328, 5: 317}
OURS_RUNS = [(79, 270, "Unit-orbit"), (102, 267.88, "Root-sign"), (110, 267.88, "Mixed-radix")]
ZOOM_X, ZOOM_Y = (70, 120), (246, 275)
DASH = (0, (4, 2.5))

with open(EKERA_CSV) as f:
    ek = [r for r in csv.DictReader(f) if int(r["s"]) in EKERA_BITS]
ex = [int(r["batches_95"]) * int(r["runs_per_batch"]) for r in ek]
ey = [EKERA_BITS[int(r["s"])] for r in ek]


def floors(ax, lo, hi):
    R = np.geomspace(lo, hi, 300)
    ax.plot(R, [bits_nonadaptive(x) for x in R], color=INK, zorder=2, label="Floor, runs fixed in advance (Corollary 1)")
    ax.plot(R, [bits_adaptive(x) for x in R], color=INK, ls=DASH, zorder=2, label="Floor, adaptive runs (Theorem 2)")


fig = figure(1.0, 2.48)
axa = axes(fig, 0.48, 0.40, 3.90, 1.95)
axb = axes(fig, 5.00, 0.40, 1.90, 1.95)

# (a) All runs that we priced.
floors(axa, 2, 300)
points(axa, [3], [466], BASE)
label(axa, 3, 466, "Baseline", dy=5, ha="center", va="bottom")
points(axa, [4, 4], [464, 464.5], OTHER)
label(axa, 4, 464.25, "Same core, unsplit", dy=-6, ha="center", va="top")
axa.plot(ex, ey, color=OTHER, lw=0.8, zorder=3)
points(axa, ex, ey, OTHER)
label(axa, ex[0], ey[0], EKERA + "'s tradeoff, same core", dx=6)
points(axa, [25], [288], OURS, filled=False)
label(axa, 25, 288, "Own point addition", dx=6)
points(axa, [r for r, _, _ in OURS_RUNS], [d for _, d, _ in OURS_RUNS], OURS)
axa.add_patch(Rectangle((ZOOM_X[0], ZOOM_Y[0]), ZOOM_X[1] - ZOOM_X[0], ZOOM_Y[1] - ZOOM_Y[0], fill=False,
                        edgecolor=MUTED, lw=0.6, zorder=1))
label(axa, ZOOM_X[1], ZOOM_Y[1], "(b)", dx=1, dy=1, va="bottom")
axa.legend(loc="upper right")
axa.set_xscale("log")
axa.set_xlim(2, 300)
axa.set_ylim(230, 495)
axa.set_xticks([2, 4, 8, 16, 32, 64, 128, 256], ["2", "4", "8", "16", "32", "64", "128", "256"])
axa.xaxis.set_minor_locator(NullLocator())
axa.set_yticks([250, 300, 350, 400, 450])
axa.set_yticks(np.arange(240, 495, 10), minor=True)
axa.set_xlabel("Runs per key recovery")
axa.set_ylabel("Exponent bits per run")

# (b) Our configurations and their distance above the floor of Corollary 1.
floors(axb, ZOOM_X[0], ZOOM_X[1] + 2)
for R, D, name in OURS_RUNS:
    f = bits_nonadaptive(R)
    axb.annotate("", (R, f), (R, D), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.6, mutation_scale=5,
                                                      shrinkA=2.6, shrinkB=0))
    right = R != 102
    label(axb, R, (f + D) / 2, "+%.2f" % (D - f), dx=3.5 if right else -3.5, ha="left" if right else "right")
points(axb, [r for r, _, _ in OURS_RUNS], [d for _, d, _ in OURS_RUNS], OURS)
label(axb, 79, 270, "Unit-orbit", dy=5, ha="center", va="bottom")
label(axb, 102, 267.88, "Root-sign", dx=-5, ha="right")
label(axb, 110, 267.88, "Mixed-radix", dy=6, ha="center", va="bottom")
axb.set_xlim(ZOOM_X[0], ZOOM_X[1] + 2)
axb.set_ylim(*ZOOM_Y)
axb.set_xticks(range(70, 121, 10))
axb.set_xticks(range(70, 121, 5), minor=True)
axb.set_yticks([250, 260, 270])
axb.set_yticks(range(246, 276, 2), minor=True)
axb.set_xlabel("Runs per key recovery")
axb.set_ylabel("Exponent bits per run")

panel_label(fig, 0.02, 2.45, "(a)")
panel_label(fig, 4.52, 2.45, "(b)")
save(fig, os.path.join(HERE, "fig2_bits.pdf"))
