"""Figure 4: the 64 residues of Z[omega] modulo 2^3 and their orbits under the six units.

Z / 2^W; omega x = -b + (a - b) omega; the units are +-1, +-omega, +-omega^2. In the complex plane omega = exp(2 pi i / 3),
so the units are the rotations by multiples of 60 degrees. Each residue is drawn at a lift of smallest norm (inside the
hexagonal Voronoi cell of 2^W Z[omega]). One member of each orbit, the stored point, lies in the sector [0, 60) degrees,
at its lift of smallest norm, and the other members are drawn at its rotations, which are lifts of smallest norm too
(asserted below): on the cell's boundary, where a residue has two such lifts, this keeps exactly one member of each
orbit of six in each sector, as the text says. The orbit of three (the nonzero multiples of 2^(W-1)) is drawn at
rotations by 120 degrees. The thin hexagon joins the six members of one orbit.
Count check: 4^W residues, (4^W - 4)/6 orbits of six, one orbit of three, the zero class: (4^W - 4)/6 + 2 = 12 at W = 3.
Run: python3 paper/figures/fig4_orbits.py  (writes paper/figures/fig4_orbits.pdf)
"""
import os
import cmath
import math
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from style import OURS, OTHER, MUTED, GUIDE, tint, figure, axes, label, save

HERE = os.path.dirname(os.path.abspath(__file__))
W = 3
M = 2 ** W
OM = complex(-0.5, math.sqrt(3) / 2)


def mul_omega(x):
    a, b = x
    return ((-b) % M, (a - b) % M)


def neg(x):
    return ((-x[0]) % M, (-x[1]) % M)


def orbit(x):
    out = set()
    y = x
    for _ in range(3):
        out.add(y)
        out.add(neg(y))
        y = mul_omega(y)
    return frozenset(out)


def lift(x):
    best = None
    a0, b0 = x
    for m in range(-2, 3):
        for n in range(-2, 3):
            z = complex(a0 + M * m, 0) + (b0 + M * n) * OM
            key = (round(abs(z), 9), round(math.atan2(z.imag, z.real) % (2 * math.pi), 9))
            if best is None or key < best[0]:
                best = (key, z)
    return best[1]


def residue(z):
    b = round(2 * z.imag / math.sqrt(3))
    return (round(z.real + b / 2) % M, b % M)


def angle(z):
    return math.degrees(math.atan2(z.imag, z.real)) % 360


residues = [(a, b) for a in range(M) for b in range(M)]
orbits = {orbit(x) for x in residues}
assert len(orbits) == (4 ** W - 4) // 6 + 2

reps, others, example = [], [], None
for orb in orbits:
    if orb == frozenset({(0, 0)}):
        reps.append(0j)
        continue
    rep = min((z for z in map(lift, orb) if angle(z) < 60 - 1e-9), key=lambda z: (abs(z), angle(z)))
    turns = 6 if len(orb) == 6 else 3
    members = [rep * cmath.exp(2j * math.pi * k / turns) for k in range(turns)]
    assert {residue(z) for z in members} == set(orb)
    assert all(abs(abs(z) - abs(lift(residue(z)))) < 1e-9 for z in members)
    reps.append(rep)
    others.extend(members[1:])
    if (2, 1) in orb:
        example = members
assert len(reps) == (4 ** W - 4) // 6 + 2 and len(reps) + len(others) == 4 ** W

fig = figure(0.62, 2.45)
ax = axes(fig, 0.02, 0.05, 2.40, 2.35)
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-4.95, 4.95)
ax.set_ylim(-4.85, 4.85)

R = M / math.sqrt(3)                                   # circumradius of the Voronoi cell; its inradius is M / 2
corner = [R * cmath.exp(1j * math.radians(30 + 60 * k)) for k in range(7)]
ax.plot([z.real for z in corner], [z.imag for z in corner], color=GUIDE, lw=0.8, zorder=1)
sector = [0j, M / 2 + 0j, corner[0], M / 2 * cmath.exp(1j * math.pi / 3)]
ax.add_patch(Polygon([(z.real, z.imag) for z in sector], closed=True, facecolor=tint(OURS, 0.13), edgecolor="none",
                     zorder=0))
for k in range(6):
    z = M / 2 * cmath.exp(1j * math.radians(60 * k))
    ax.plot([0, z.real], [0, z.imag], color=GUIDE, lw=0.6, ls=(0, (2, 2)), zorder=1)
ring = example + example[:1]
ax.plot([z.real for z in ring], [z.imag for z in ring], color=MUTED, lw=0.7, zorder=2)

ax.plot([z.real for z in others], [z.imag for z in others], marker="o", markersize=4.2, linestyle="none",
        markerfacecolor="white", markeredgecolor=OTHER, markeredgewidth=0.8, zorder=3)
ax.plot([z.real for z in reps], [z.imag for z in reps], marker="o", linestyle="none", color=OURS, zorder=4)
label(ax, 1, 0, "$1$", dx=0, dy=-4.5, ha="center", va="top")
label(ax, OM.real, OM.imag, r"$\omega$", dx=4.5, dy=0.5)

handles = [Line2D([], [], marker="o", linestyle="none", color=OURS),
           Line2D([], [], marker="o", markersize=4.2, linestyle="none", markerfacecolor="white",
                  markeredgecolor=OTHER, markeredgewidth=0.8),
           Patch(facecolor=tint(OURS, 0.13), edgecolor="none"),
           Line2D([], [], color=MUTED, lw=0.7)]
texts = ["%d stored points, one per orbit" % len(reps), "%d residues reached by a unit" % len(others),
         "Stored sector of $60^\\circ$", "An orbit of six"]
W_IN, H_IN = fig.get_size_inches()
fig.legend(handles, texts, loc="center left", bbox_to_anchor=(2.50 / W_IN, 0.5), handlelength=1.4, handleheight=1.0,
           labelspacing=1.0, borderaxespad=0)
save(fig, os.path.join(HERE, "fig4_orbits.pdf"))
