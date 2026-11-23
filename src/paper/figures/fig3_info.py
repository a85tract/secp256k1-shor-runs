"""Figure 3: one-run information about the key on the tested distributions, against c = |S| / r.

All values are exact computations on small groups (no sampling), read from the primary files:
  QFT information, uniform 2D boxes:
    r = 211:  c = 3.986 -> 0.8795, c = 63.773 -> 3.7695 bits              s3/theory/oversample_info.txt (pad 0)
  Holevo information of the raw run, uniform 2D boxes (hash_check_test1.txt, "raw chi"):
    r = 101 8x8 (c = 0.634) 0.427; r = 103 10x10 (0.971) 0.661; r = 211 12x12 (0.682) 0.469; r = 211 20x20 (1.896) 1.268
  Bound of Theorem 1: log2(1 + c) (with |S| - 1 in place of |S| the difference is below the line width).
Both axes are logarithmic, so the QFT points at small c, which sit within 0.5 bit of zero, stay apart from each other
and from the bound, and a constant fraction of the bound is a constant distance below it.
Run: python3 paper/figures/fig3_info.py  (writes paper/figures/fig3_info.pdf)
"""
import os
import numpy as np
from matplotlib.ticker import NullLocator, NullFormatter
from style import OURS, OTHER, INK, figure, axes, save

HERE = os.path.dirname(os.path.abspath(__file__))

qft_c = [0.25, 1, 2, 4, 16, 64, 256, 3.986, 63.773, 15.943]
qft_i = [0.0598, 0.2455, 0.4853, 0.894, 2.153, 3.79, 5.64, 0.8795, 3.7695, 2.128]
hol_c = [0.634, 0.971, 0.682, 1.896]
hol_i = [0.427, 0.661, 0.469, 1.268]

fig = figure(0.8, 2.25)
ax = axes(fig, 0.50, 0.40, 5.03, 1.80)
c = np.geomspace(0.2, 320, 300)
ax.plot(c, np.log2(1 + c), color=INK, zorder=2, label="Bound of Theorem 1")
ax.plot(hol_c, hol_i, marker="D", markersize=3.8, linestyle="none", color=OTHER, zorder=3,
        label="Holevo information")
ax.plot(qft_c, qft_i, marker="o", linestyle="none", color=OURS, zorder=4, label="QFT measurement")
ax.legend(loc="upper left")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.2, 320)
ax.set_ylim(0.04, 12)
ax.set_xticks([0.25, 1, 4, 16, 64, 256], ["0.25", "1", "4", "16", "64", "256"])
ax.set_xticks([0.5, 2, 8, 32, 128], minor=True)
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10], ["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10"])
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xlabel("$c = |S|/r$")
ax.set_ylabel("Information per run (bits)")
save(fig, os.path.join(HERE, "fig3_info.pdf"))
