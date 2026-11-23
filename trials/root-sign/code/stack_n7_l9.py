"""o3/candidates/stack_mix_l9.py (commit 465aefa) with three changes for the harness's mixed-radix
(1) kind names follow the window labels the harness checks: 'n7' for the (3 + w) 2^w window, 'm7' for the ten
    (2 - w) 2^w windows (V-B KIND n7_window_add / m7_window_add), 'w8' for the W = 8 windows, 'term' for the lowest
    window (W = 9 terminal); at toy sizes 'n<w>' / 'm<w>' / 'w<W>' (declared stand-ins, since only n = 256 is priced);
(2) every solve works on Q0 = Q + [rho0] G for one random rho0, so d0 = d + rho0 is uniform for every d and the
    success rate holds for every d; the decoder answers d0 - rho0;
ENUM_BLOCK nodes (it charged them after the loop, so an attempt could execute more nodes than its remaining allowance),
and solve() starts at most floor((2^40 - work so far) / cap) attempts. Every attempt executes at most `cap` units
(try_subset stops it at the first charge that would pass), so a solve executes at most 2^40 units whatever the outcomes.

decoder near the information limit. a = 'h9,n7,m7x10,h8x3,s4' (W9 terminal, 11 norm-7 x 4^7 windows, three W = 8,
the 4-bit sine box top), b = L = 9 sine bits, so the first lookup reads 17 bits (66,588 Toffoli instead of 132,380)
target (-r k_i, 0); entries of m + l + 1 bits, 4 words instead of 8), LLL, Babai, and if Babai's answer fails the
check, a Schnorr-Euchner enumeration of the lattice vectors within ENUM_RAD x the expected error norm (sine kernel
RMS 0.5 per run), at most ENUM_NODES = 2^17 nodes charged dim x words each, every leaf's d checked (at most ENUM_CHECKS).
NEED = 34 runs per subset at n = 256 (40 exact trials: 37 / 39 clean subsets decoded, o3/results/s6_bdd_l9.txt).
Run law for Lemma G: one global scalar per run (run.offset(global_scalar=True)) and subsets of unflagged runs only
"""
import math
from fractions import Fraction

L = 9
LAYOUT = 'h9,n7,m7x10,h8x3,s4'      # lowest first; the last window is the sine box top ('first', in the first lookup)
KINDS = ['term'] + ['n7'] + ['m7'] * 10 + ['w8'] * 3 + ['first']     # h1: names = window labels (n7, m7, w8)
RUNS = 101                           # o3 S7.3: P(fail) <= 1e-3 (109 for 5e-4)
NEED = 34
TRIES = None                         # default: as many as 2^40 / attempt cap allows (minus 2^34 slack)
BRANCH = 0
ENUM_RAD = 1.2
ENUM_NODES = 1 << 17
ENUM_CHECKS = 256
ENUM_BLOCK = 64
GLOBAL = True
CHECKS = 64
LLL_CAP = None                       # default 1.3 x the calibrated LLL charge at dimension NEED + 1 (o1/s4/calib*.py)
METRIC = 'euclid'                    # 'euclid' or 'hex'


def params(m):
    """layout per size: n = 256 uses the module defaults; toy sizes: windows of norm 7 x 4^w0 (alternating 3 + w and
    2 - w) up to the target, one hex window to fill, a 2-4-bit sine top; kinds 'n7w<w>' / 'h<W>' / 'first'"""
    if m >= 256:
        return L, LAYOUT, list(KINDS), NEED, RUNS
    l = 10 if m >= 128 else (8 if m >= 64 else 6)
    wt = 4 if m >= 64 else (3 if m >= 32 else 2)
    w0 = min(7, max(2, (m + 8) // 20))
    b0 = 2 * w0 + math.log2(7)
    target = m + 4 - 2 * wt
    k = int(target // b0)
    rem = target - k * b0
    toks, kinds = [], []
    for i in range(k):
        toks.append(('n' if i % 2 == 0 else 'm') + str(w0))
        kinds.append(('n' if i % 2 == 0 else 'm') + str(w0))       # h1: kind name = window label
    if rem > 1.2:
        W = max(1, int(round(rem / 2)))
        toks.insert(0, 'h%d' % W)
        kinds.insert(0, 'w%d' % W)
    toks.append('s%d' % wt)
    kinds.append('first')
    need = math.ceil(m / (l - 2.2)) + 1
    return l, ','.join(toks), kinds, need, need + 10


# ---- the register's moduli and the u decoder (o1/mixrad.py parse_layout / positions / decode, inlined: the harness
# scan allows only standard imports in a candidate)
def emul(z, t):
    return (z[0] * t[0] - z[1] * t[1], z[0] * t[1] + z[1] * t[0] - z[1] * t[1])


def moduli(spec):
    """window moduli (lowest first) of a layout spec: hW -> 2^W, nW -> (3 + w) 2^W, mW -> (2 - w) 2^W, sW -> 2^W"""
    out = []
    for tok in spec.split(','):
        cnt = 1
        if 'x' in tok:
            tok, c = tok.split('x')
            cnt = int(c)
        k, v = tok[0], int(tok[1:])
        q = {'h': (1, 0), 's': (1, 0), 'n': (3, 1), 'm': (2, -1)}[k]
        out += [emul(q, (1 << v, 0))] * cnt
    return out


def norm(z):
    return z[0] * z[0] - z[0] * z[1] + z[1] * z[1]


def decode_mix(r, lam, Y1, Y2, N, Ptop):
    """u c mod r: the point of {(x1, x2): x2 = lam x1 mod r} closest to r (Y1, Y2) / N in |M_Ptop^T e|^2 (the top
    window's frame; M_z the matrix of multiplication by z in the basis (1, w))"""
    a, b = Ptop
    MT = ((a, b), (-b, a - b))

    def tr(v):
        return (MT[0][0] * v[0] + MT[0][1] * v[1], MT[1][0] * v[0] + MT[1][1] * v[1])

    def q(v):
        t = tr(v)
        return t[0] * t[0] + t[1] * t[1]
    b1, b2 = (1, lam % r), (0, r)
    while True:
        if q(b2) < q(b1):
            b1, b2 = b2, b1
        num = (q((b1[0] + b2[0], b1[1] + b2[1])) - q(b1) - q(b2)) // 2
        mu = (2 * num + q(b1)) // (2 * q(b1))
        if mu == 0:
            break
        b2 = (b2[0] - mu * b1[0], b2[1] - mu * b1[1])
    t = (Fraction(r * Y1, N), Fraction(r * Y2, N))
    det = b1[0] * b2[1] - b1[1] * b2[0]
    c1 = (t[0] * b2[1] - t[1] * b2[0]) / det
    c2 = (b1[0] * t[1] - b1[1] * t[0]) / det
    k1, k2 = round(c1), round(c2)
    best = None
    for i in range(-2, 3):
        for j in range(-2, 3):
            v = (b1[0] * (k1 + i) + b2[0] * (k2 + j), b1[1] * (k1 + i) + b2[1] * (k2 + j))
            e = tr((t[0] - v[0], t[1] - v[1]))
            d = e[0] * e[0] + e[1] * e[1]
            if best is None or d < best[0]:
                best = (d, v[0] % r)
    return best[1]


def lll_cap(dim, words):
    """swap cap per dimension: 1.1 x max(20,000, 13,500 (dim / 23)^1.34) (o3/results/lll_swaps.txt; at dimension
    35 the harness's LLL needed at most 22,731 swaps in 145 trials, o3/results/s6_outlier_stats_l9.txt), charged as
    the harness does, (swaps + dim) dim^2 words"""
    if LLL_CAP is not None:
        return LLL_CAP
    swaps = int(1.1 * max(20000, 13500 * (dim / 23) ** 1.34))
    return (swaps + dim) * dim * dim * words


def budget(m):
    return params(m)[4]


def glv_basis(r, lam):
    """reduced basis of {(x1, x2): x2 = lam x1 mod r} (Lagrange reduction, Euclidean)"""
    b1, b2 = (1, lam % r), (0, r)
    n = lambda v: v[0] * v[0] + v[1] * v[1]
    dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
    while True:
        if n(b1) > n(b2):
            b1, b2 = b2, b1
        mu = (2 * dot(b1, b2) + n(b1)) // (2 * n(b1))
        if mu == 0:
            return b1, b2
        b2 = (b2[0] - mu * b1[0], b2[1] - mu * b1[1])
        if n(b2) >= n(b1):
            return b1, b2


def _rnd(a, b):
    if b < 0:
        a, b = -a, -b
    return (2 * a + b) // (2 * b)


def decode_u(r, basis, Y1, Y2, N):
    """u = first coordinate (mod r) of the lattice vector closest to (r Y1, r Y2) / N, searched over a 5 x 5
    neighbourhood of the rounded coordinates (exact integers)"""
    (a1, a2), (c1, c2) = basis
    t1, t2 = r * Y1, r * Y2
    det = a1 * c2 - a2 * c1
    al0 = _rnd(t1 * c2 - t2 * c1, N * det)
    be0 = _rnd(a1 * t2 - a2 * t1, N * det)
    best = None
    for da in range(-2, 3):
        for db in range(-2, 3):
            x1 = (al0 + da) * a1 + (be0 + db) * c1
            x2 = (al0 + da) * a2 + (be0 + db) * c2
            e1, e2 = N * x1 - t1, N * x2 - t2
            q = e1 * e1 + e2 * e2 if METRIC == 'euclid' else e1 * e1 + e1 * e2 + e2 * e2
            if best is None or q < best[0]:
                best = (q, x1)
    return best[1] % r


def _dot(x, y):
    return sum(p * q for p, q in zip(x, y))


def _gso(b):
    """integral Gram-Schmidt data (Cohen 2.6.7): d[0..n], lam[i][j]"""
    n = len(b)
    d = [1] + [0] * n
    lam = [[0] * n for _ in range(n)]
    for k in range(n):
        for j in range(k + 1):
            u = _dot(b[k], b[j])
            for i in range(j):
                u = (d[i + 1] * u - lam[k][i] * lam[j][i]) // d[i]
            if j < k:
                lam[k][j] = u
            else:
                d[k + 1] = u
    return d, lam


def branch_babai(h, b, target, K, words):
    """nearest plane with two choices at the first K levels processed; returns leaves (residual norm^2, vector)"""
    n = len(b)
    h.charge(n ** 3 * words, 'lattice')
    d, lam = _gso(b)
    w0 = list(map(int, target))
    lw0 = [0] * n
    for j in range(n):
        u = _dot(w0, b[j])
        for i in range(j):
            u = (d[i + 1] * u - lw0[i] * lam[j][i]) // d[i]
        lw0[j] = u
    out = []

    def rec(level, w, lw, depth):
        if level < 0:
            out.append((_dot(w, w), [t - x for t, x in zip(target, w)]))
            return
        num, den = lw[level], d[level + 1]
        q = (2 * num + den) // (2 * den)
        choices = [q]
        if depth < K:
            choices.append(q + 1 if 2 * num > 2 * q * den else q - 1)
        for qq in choices:
            h.charge(2 * n * words, 'lattice')
            if qq:
                w2 = [x - qq * y for x, y in zip(w, b[level])]
                lw2 = lw[:]
                lw2[level] -= qq * d[level + 1]
                for i in range(level):
                    lw2[i] -= qq * lam[level][i]
            else:
                w2, lw2 = w, lw
            rec(level - 1, w2, lw2, depth + 1)

    rec(n - 1, w0, lw0, 0)
    out.sort(key=lambda x: x[0])
    return out


def enumerate_cvp(h, red, target, rad2_units, words, cap, checks, r, rho0=0):
    """Schnorr-Euchner enumeration (float Gram-Schmidt of the LLL basis, scaled by 2^-S) of the lattice vectors v
    with |target - v|^2 <= rad2_units; every leaf's d = (sum c_i red_i[-1]) mod r is checked, at most `checks`;
    charged dim^3 words for the Gram-Schmidt data and dim x words per node, each charge made before the work it pays
    for (nodes in blocks of ENUM_BLOCK)"""
    dim = len(red)
    h.charge(dim ** 3 * words, 'lattice')
    S = max(0, max(max(abs(x) for x in row).bit_length() for row in red) - 60)

    def f(x):
        return float(x >> S) if x >= 0 else -float((-x) >> S)
    B = [[f(x) for x in row] for row in red]
    t = [f(x) for x in target]
    R2 = float(rad2_units >> (2 * S))
    Bs, bn, mu = [], [], [[0.0] * dim for _ in range(dim)]
    for i in range(dim):
        v = B[i][:]
        for j in range(i):
            mu[i][j] = sum(a * b for a, b in zip(B[i], Bs[j])) / bn[j]
            v = [a - mu[i][j] * b for a, b in zip(v, Bs[j])]
        Bs.append(v)
        bn.append(sum(a * a for a in v))
    y = [sum(a * b for a, b in zip(t, Bs[i])) / bn[i] for i in range(dim)]
    last = [row[-1] for row in red]
    c, ctr, dx, ddx = [0] * dim, [0.0] * dim, [0] * dim, [0] * dim
    part = [0.0] * (dim + 1)
    i = dim - 1
    ctr[i] = y[i]
    c[i] = round(ctr[i])
    dx[i] = ddx[i] = 1 if ctr[i] >= c[i] else -1
    nodes = nchk = paid = 0
    found = None
    while True:
        diff = ctr[i] - c[i]
        li = part[i + 1] + diff * diff * bn[i]
        nodes += 1
        if nodes > cap:
            break
        if nodes > paid:
            k = min(ENUM_BLOCK, cap - paid)
            h.charge(k * dim * words, 'lattice')
            paid += k
        if li <= R2:
            if i == 0:
                d = (sum(ci * x for ci, x in zip(c, last)) - rho0) % r     # h1: d0 - rho0
                nchk += 1
                if h.check(d):
                    found = d
                    break
                if nchk >= checks:
                    break
                c[0] += dx[0]
                ddx[0] = -ddx[0]
                dx[0] = ddx[0] - dx[0]
                continue
            part[i] = li
            i -= 1
            s = y[i]
            for j in range(i + 1, dim):
                s -= c[j] * mu[j][i]
            ctr[i] = s
            c[i] = round(s)
            dx[i] = ddx[i] = 1 if s >= c[i] else -1
            continue
        i += 1
        if i == dim:
            break
        c[i] += dx[i]
        ddx[i] = -ddx[i]
        dx[i] = ddx[i] - dx[i]
    return found


def postprocess(h, pairs, m, l, rho0=0):
    """L0 lattice divided by 2^m (multipliers z_i = 2 u_i, k_i / 2^l ~ -z_i d / r): LLL, Babai, then the
    bounded enumeration with the check as predicate"""
    r = h.r
    n = len(pairs)
    rows = [[z << l for z, _ in pairs] + [1]]
    for i in range(n):
        rows.append([0] * i + [r << l] + [0] * (n - i))
    target = [-(r * k) for _, k in pairs] + [0]
    red = h.lll(rows)
    words = max(1, max(abs(x) for row in red for x in row).bit_length() // 64)
    _, vec = h.babai(red, target)
    d = (vec[-1] - rho0) % r                 # h1: the lattice gives d0 = d + rho0
    if h.check(d):
        return d
    if ENUM_NODES <= 0:
        return None
    rad2 = int(ENUM_RAD * ENUM_RAD * (n * 0.25 + 1 / 12) * 2 ** 32) * r * r >> 32
    return enumerate_cvp(h, red, target, rad2, words, ENUM_NODES, ENUM_CHECKS, r, rho0)


def attempt_cap(dim, words):
    """LLL swap cap (lll_cap) + Babai + the enumeration's Gram-Schmidt and nodes + checks (h.check <= 2 x 256
    group operations each)"""
    return lll_cap(dim, words) + 2 * dim ** 3 * words + ENUM_NODES * dim * words + (ENUM_CHECKS + 1) * 512


def solve(h):
    m = h.r.bit_length()
    r = h.r
    l, spec, kinds, need, runs = params(m)
    pis = moduli(spec)
    Ptop = (1, 0)
    for z in pis[:-1]:
        Ptop = emul(Ptop, z)
    N = norm(emul(Ptop, pis[-1]))
    rho0 = h.rng.randrange(r)            # h1: random secret shift
    Q0 = h.Q + h.G * rho0                # d0 = d + rho0 is uniform for every d
    Q2 = -(Q0 + Q0)
    pairs = []
    h.charge(64 * m)
    for _ in range(runs):
        run = h.run()
        run.offset(global_scalar=GLOBAL)
        b = run.reg(l, sine=l)
        a = run.mixreg(spec)
        run.add(b, Q2)
        run.hexadd(a, h.G, kinds)
        k, (Y1, Y2) = run.measure()
        h.charge(64)
        if run.flagged:                  # Theorem G: subsets of unflagged runs only
            continue
        u = decode_mix(r, h.lam, Y1, Y2, N, Ptop)
        pairs.append((2 * u % r, k, run.id))
    if len(pairs) < need:
        return None
    dim = need + 1
    words = max(1, (m + l + 8) // 64)
    cap = attempt_cap(dim, words)
    tries = TRIES if TRIES is not None else max(1, int((2 ** 40 - 2 ** 34) // cap))
    tries = min(tries, int((2 ** 40 - h.classical_ops()) // cap))
    idx = list(range(len(pairs)))
    seen = set()
    total = math.comb(len(pairs), need)
    while len(seen) < min(tries, total):
        sub = tuple(sorted(h.rng.sample(idx, need)))
        if sub in seen:
            continue
        seen.add(sub)
        chosen = [pairs[i] for i in sub]
        d = h.try_subset([p[2] for p in chosen], lambda: postprocess(h, [(z, k) for z, k, _ in chosen], m, l, rho0), cap)
        if d is not None:
            return d
    return None
