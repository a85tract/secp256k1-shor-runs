"""Reference conversion for unit-orbit ("hex") windows. The circuit (o1/hexconv.py) and the V-A sampler
(o1/hexva.py) both use the digit defined here, so the table, the controlled addition and the measurement's
correction phases agree on every residue.

Residue x = (a, b) in (Z / M)^2, M = 2^W, meaning a + b w (w^2 + w + 1 = 0; w acts on points as lam).
  v      = 2-adic valuation of x (min of v2(a), v2(b)); x = 0 is special (digit M, i.e. 2^W).
  j      = unit power read from the bit pair (a_v, b_v): (1,0) -> 0, (0,1) -> 1, (1,1) -> 2.
  x'     = w^(-j) x mod M                 (then a'_v = 1, b'_v = 0)
  s      = bit v+1 of a' (0 if v = W-1)   (the sign: -x' flips exactly this bit)
  x''    = (-1)^s x' mod M                (canonical orbit representative: a'' = 2^v mod 2^(v+2), b'' = 0 mod 2^(v+1))
  lift   = hexreduce(centered(x''))       (an element of x'' + M Z[w] in the closed hexagon: minimal norm)
  digit  = (-1)^s w^j lift                (an Eisenstein integer congruent to x mod M)
The digits form a complete residue system mod M Z[w]; the table holds one point per canonical x''
(K = (4^W - 4)/6 + 2 entries incl. the zero class; W = 9: 43,692) and the circuit applies the unit to the
looked-up point ((x, y) -> (beta^j x, (-1)^s y)).
"""
import numpy as np


def mulw(u, v):
    """w (u + v w) = -v + (u - v) w"""
    return (-v, u - v)


def unit(s, j, d):
    for _ in range(j):
        d = mulw(*d)
    return (-d[0], -d[1]) if s else d


def norm(u, v):
    return u * u - u * v + v * v


def centered(a, M):
    return a - M if a >= M // 2 else a


def hexreduce(A, B, M):
    """Box [-M/2, M/2)^2 -> closed Voronoi hexagon of M Z[w] (|2A-B|, |2B-A|, |A+B| <= M), by at most one
    lattice step; the rule is the one the circuit implements (o1/hexconv.py)."""
    p = 2 * A - B > M or 2 * B - A < -M          # corner near (M/2, -M/2)
    q = 2 * A - B < -M or 2 * B - A > M          # corner near (-M/2, M/2)
    if p:
        if A + B >= 0:
            A -= M
        else:
            B += M
    elif q:
        if A + B <= 0:
            A += M
        else:
            B -= M
    return A, B


def convert(a, b, W):
    """x = (a, b) -> (v, s, j, x'') with x'' = (a'', b''); x = 0 -> (W, 0, 0, (0, 0))."""
    M = 1 << W
    if a == 0 and b == 0:
        return W, 0, 0, (0, 0)
    v = 0
    while not ((a >> v) & 1 or (b >> v) & 1):
        v += 1
    av, bv = (a >> v) & 1, (b >> v) & 1
    j = {(1, 0): 0, (0, 1): 1, (1, 1): 2}[(av, bv)]
    x1 = (a, b)
    for _ in range((3 - j) % 3):                    # w^(-j) = w^(3-j)
        x1 = mulw(*x1)
    x1 = (x1[0] % M, x1[1] % M)
    s = (x1[0] >> (v + 1)) & 1 if v + 1 < W else 0
    x2 = ((-x1[0]) % M, (-x1[1]) % M) if s else x1
    return v, s, j, x2


def lift(x2, W):
    M = 1 << W
    if x2 == (0, 0):
        return (M, 0)                               # zero class: digit 2^W (the adder never adds O)
    return hexreduce(centered(x2[0], M), centered(x2[1], M), M)


def digit(a, b, W):
    v, s, j, x2 = convert(a, b, W)
    return unit(s, j, lift(x2, W))


def canonical_list(W):
    """Canonical residues x'' in the order of the lookup tree (spine over v, then the free bits)."""
    M = 1 << W
    out = []
    for v in range(W):
        if v == W - 1:
            out.append((M >> 1, 0))
            continue
        fa = W - v - 2                              # free bits a''[v+2 .. W-1]
        fb = W - v - 1                              # free bits b''[v+1 .. W-1]
        for ia in range(1 << fa):
            for ib in range(1 << fb):
                out.append(((1 << v) | (ia << (v + 2)), ib << (v + 1)))
    out.append((0, 0))
    return out


_CACHE = {}


def digit_arrays(W):
    """d1, d2: int64 arrays of shape (M, M), digit of residue (a, b) = d1[a, b] + d2[a, b] w."""
    if W in _CACHE:
        return _CACHE[W]
    M = 1 << W
    d1 = np.zeros((M, M), dtype=np.int64)
    d2 = np.zeros((M, M), dtype=np.int64)
    for a in range(M):
        for b in range(M):
            d1[a, b], d2[a, b] = digit(a, b, W)
    _CACHE[W] = (d1, d2)
    return d1, d2


def check(W):
    """Exhaustive: complete residue system, minimal norm, conversion consistency, table size."""
    M = 1 << W
    seen = set()
    canon = set(canonical_list(W))
    bad = 0
    for a in range(M):
        for b in range(M):
            v, s, j, x2 = convert(a, b, W)
            d = digit(a, b, W)
            if (d[0] % M, d[1] % M) != (a, b):
                bad += 1
            seen.add(d)
            if x2 not in canon:
                bad += 1
            if (a, b) != (0, 0):
                best = min(norm(a + M * e1, b + M * e2) for e1 in range(-2, 3) for e2 in range(-2, 3))
                if norm(*d) != best:
                    bad += 1
            Ar, Br = lift(x2, W) if x2 != (0, 0) else (0, 0)
            if x2 != (0, 0) and not (abs(2 * Ar - Br) <= M and abs(2 * Br - Ar) <= M and abs(Ar + Br) <= M):
                bad += 1
    K = len(canonical_list(W))
    return dict(W=W, digits=len(seen), complete=len(seen) == M * M, bad=bad, K=K,
                K_formula=(4 ** W - 4) // 6 + 2, max_abs=int(max(max(abs(u), abs(v)) for u, v in seen)))


if __name__ == '__main__':
    import sys
    for W in [int(x) for x in (sys.argv[1:] or ['1', '2', '3', '4', '5', '6'])]:
        print(check(W))
