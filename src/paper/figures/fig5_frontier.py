"""Figure 5: volume of the largest run against total volume, both relative to the baseline.

Points (T / T0, B / B0), every one read from a saved file:
  baseline, Haener et al.                                  (1, 1)                         harness/README.md l.104-116
  unsplit on the borrowed core (U1; U2-h, U2-n, U2-nt)     (0.664, 0.497995), (0.643, 0.482583), (0.634, 0.475439),
  Ekera's tradeoff on the borrowed core, plain signed windows and plain decoding (the third control), s = 2..5,
  batches for 0.99 at the 95 % rule (T_T0_95, B_B0); s = 4 is the control (7.7115, 0.355916)
                                                                                           base/ekera_tradeoff.csv
Curve: splitting alone at the baseline's arithmetic, R equal runs at the floor of Corollary 1 (base/frontier.csv
columns T_eq_T0, B_eq_B0), R <= 55. By Appendix A.5 the smallest largest run with total volume T is the curve's value at
T, so the curve is the frontier of every budget up to 10 T0. The dots mark R = 21 (the frontier at 4 T0: 264.24 bits,
0.5504 B0) and R = 55 (at 10 T0: 258.98 bits, 0.5387 B0).
Both axes are logarithmic: the text compares these runs by ratios (2.81 times, 1.29 to 1.38 times further, 11 to 15
times the total volume), and on these axes equal ratios are equal distances.
Run: python3 paper/figures/fig5_frontier.py  (writes paper/figures/fig5_frontier.pdf)
"""
import csv
import os
from matplotlib.ticker import NullFormatter, NullLocator
from style import OURS, BASE, OTHER, INK, MUTED, EKERA, figure, axes, points, label, save

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
THIRD_CONTROL_CSV = os.path.join(REPO, "base", "ekera_tradeoff.csv")
FRONTIER_CSV = os.path.join(REPO, "base", "frontier.csv")
BUDGET = 10


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


fig = figure(1.0, 2.55)
ax = axes(fig, 0.48, 0.40, 6.45, 2.10)

# Splitting alone at the baseline's arithmetic: R equal runs at the floor, inside the budget (R <= 55).
front = [r for r in read_csv(FRONTIER_CSV) if int(r["R"]) <= 55]
ax.plot([float(r["T_eq_T0"]) for r in front], [float(r["B_eq_B0"]) for r in front], color=INK, ls=(0, (4, 2.5)),
        zorder=2)
mark = {int(r["R"]): (float(r["T_eq_T0"]), float(r["B_eq_B0"])) for r in front if int(r["R"]) in (21, 55)}
points(ax, [mark[21][0], mark[55][0]], [mark[21][1], mark[55][1]], INK, markersize=3.2)
label(ax, *mark[21], "21 runs", dy=5, ha="center", va="bottom", color=MUTED)
label(ax, *mark[55], "55 runs", dx=-2, dy=5, ha="right", va="bottom", color=MUTED)
label(ax, 1.3, 0.596, "Splitting alone, baseline arithmetic", dy=6, va="bottom")

ax.axvline(BUDGET, color=INK, lw=0.6, ls=(0, (1.2, 1.8)), zorder=1)
label(ax, BUDGET, 1.1, "Budget", dx=-3, dy=-4, ha="right", va="top")

points(ax, [1], [1], BASE)
label(ax, 1, 1, "Baseline", dx=6)

ux = [0.664, 0.643, 0.634, 0.622]
uy = [0.497995, 0.482583, 0.475439, 0.466767]
points(ax, ux, uy, OTHER)
label(ax, 0.664, 0.497995, "Same core, unsplit", dx=6)

# Third control: Ekera's tradeoff on the same core, s = 2..5 (s = 4 is the control; s = 1, two runs of 519 bits at
# 0.569 B0, lies above the unsplit run and is left out).
ek = [r for r in read_csv(THIRD_CONTROL_CSV) if 2 <= int(r["s"]) <= 5]
tx = [float(r["T_T0_95"]) for r in ek]
ty = [float(r["B_B0"]) for r in ek]
ax.plot(tx, ty, color=OTHER, lw=0.8, zorder=3)
points(ax, tx, ty, OTHER)
label(ax, tx[0], ty[0], EKERA + "'s tradeoff, same core", dx=-6, ha="right")

points(ax, [3.896], [0.4675], OURS, filled=False)
label(ax, 3.896, 0.4675, "Own point addition", dx=6)
points(ax, [7.243276, 9.796623, 8.789276], [0.275061, 0.267181, 0.258508], OURS)
label(ax, 7.243276, 0.275061, "Unit-orbit", dx=-6, ha="right")
label(ax, 8.789276, 0.258508, "Root-sign", dy=-6, ha="center", va="top")
ax.annotate("Mixed-radix", (9.796623, 0.267181), xytext=(8.33, 0.318), textcoords="data", ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=3.5))

ax.set_xscale("log")
ax.set_xlim(0.5, 14)
ax.set_yscale("log")
ax.set_ylim(0.18, 1.1)
ax.set_xticks([0.5, 1, 2, 4, 10], ["0.5", "1", "2", "4", "10"])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0], ["0.2", "0.3", "0.4", "0.5", "0.6", "0.8", "1.0"])
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xlabel("Total volume $T/T_0$")
ax.set_ylabel("Largest run $B/B_0$")
save(fig, os.path.join(HERE, "fig5_frontier.pdf"))
