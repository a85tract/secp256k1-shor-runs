"""every solve works on Q0 = Q + [rho0] G for one
random rho0 (one scalar multiplication, charged), so d0 = d + rho0 is uniform for every d and the V-A's success
rate holds for every d, not only on average; the decoder answers d0 - rho0.

(the V-A PASS of h1/hexcap/va_d270_s3_R79_final.txt) with the L0 lattice divided by 2^m: rows (2^l z_i, ..., 1),
2^l r e_i, target (-r k_i, 0), so entries are about m + l bits (4 words at n = 256) instead of 2m + l (8 words).
For secp256k1 the geometry is the same (last weight 1 instead of r / 2^m = 1 - 2^-128) and LLL is scale
invariant, so the swaps are the same and harness/hcore.py lll() charges half (it charges per 64-bit word).
lll_cap (calibrated by o1/s4/calib3.py on the 8-word lattice) is scaled by words / 8; d is the last coordinate.
Everything else is unchanged.

table of the run is built for [c] G with a fresh uniform c (the harness charges the per-run tables; with run.py
SPLIT = +-1 (n = 256 only) runs odd sigma: the sine box top gets one more bit on s1 (run.hexreg(..., split=1)), u is



Run (h1/candidates/stack_price.py shape, n = 256): Run.offset(); b = L bits in a sine state (run.reg(L, sine=L)),
added as [b](-2Q); a = run.hexreg(HEX, top='sine') (lowest window first; the top window is a sine-shaped box loaded
by the first lookup with b), added as [s1 + s2 lam] G with the kinds KINDS. Outcome (k, (Y1, Y2)).

the two coordinates' errors independent; h1/hex_outliers.py: 5.5 % wrong u at sigma = 2 against 8.2 % in the hexagonal
metric), z = 2u mod r, Ekera's L0 lattice on outcome-independent random NEED-subsets through h.try_subset, LLL, then
nearest plane with two choices at the first BRANCH levels (2^BRANCH leaves, ranked by residual, the best CHECKS leaves
checked with h.check; BRANCH = 0 is plain Babai). The branching's Gram-Schmidt and leaf work is charged with h.charge
in the same word units as h.babai (dim^3 words for the Gram-Schmidt data, dim^2 words per leaf level pass).
Defaults: D = 270 (L = 10, sigma = 4: a = 2 x 130 bits, a 4-bit sine box top window in an 18-bit first lookup,
9 W = 8 and 5 W = 9 windows and a W = 9 terminal), NEED = 32 with BRANCH = 8, R = RUNS. Kinds 'w8', 'w9', 'term' come
from run.py --kind (V-B files or declared:T:Q:PF:W).
"""
import math

L = 10
HEX = [9] + [8] * 9 + [9] * 5 + [4]  # lowest first; HEX[-1] is the sine box top window ('first', in the first lookup)
KINDS = ['term'] + ['w8'] * 9 + ['w9'] * 5 + ['first']
RUNS = 40
NEED = 32
TRIES = None                         # default: as many as 2^40 / attempt cap allows (minus 2^34 slack)
BRANCH = 8
CHECKS = 64
LLL_CAP = None                       # default 1.3 x the calibrated LLL charge at dimension NEED + 1 (o1/s4/calib*.py)
METRIC = 'euclid'                    # 'euclid' or 'hex'
GLOBAL = True                        # h1: one global scalar per run (Lemma G run law)
SPLIT = 0


def params(m):
    """layout per size: n = 256 uses the module defaults; toy sizes keep sigma = 4 (a = 2 x (m/2 + 2) bits), a sine box
    top window of 2-6 bits in the first lookup, a W = 9 terminal when it fits, then W = 8 / 9 windows"""
    if m >= 256:
        return L, list(HEX), list(KINDS), NEED, RUNS
    l = 10 if m >= 128 else (8 if m >= 64 else 6)
    E = m // 2 + 2
    for term in (True, False):
        rest0 = E - (9 if term else 0)
        for wt in (4, 5, 3, 6, 2):
            for k9 in range(0, 8):
                rest = rest0 - wt - 9 * k9
                if rest >= 0 and rest % 8 == 0 and l + 2 * wt <= 22:
                    k8 = rest // 8
                    hexw = ([9] if term else []) + [8] * k8 + [9] * k9 + [wt]
                    kinds = (['term'] if term else []) + ['w8'] * k8 + ['w9'] * k9 + ['first']
                    need = math.ceil(m / (l - 2.2)) + 1
                    return l, hexw, kinds, need, need + 10
    raise ValueError('no layout for m = %d' % m)


def lll_cap(dim):
    if LLL_CAP is not None:
        return LLL_CAP
    return int(1.3 * 2 ** (24.9 + 4.4 * math.log(dim / 20) / math.log(47 / 20)))


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


def scaled_basis(r, lam, N1, N2):
    """h1: Lagrange-reduced basis (Euclidean) of {(x1 N1, x2 N2): x2 = lam x1 mod r}, integer vectors"""
    b1, b2 = (N1, (lam % r) * N2), (0, r * N2)
    n = lambda v: v[0] * v[0] + v[1] * v[1]
    if n(b1) > n(b2):
        b1, b2 = b2, b1
    while True:
        mu = (2 * (b1[0] * b2[0] + b1[1] * b2[1]) + n(b1)) // (2 * n(b1))
        b2 = (b2[0] - mu * b1[0], b2[1] - mu * b1[1])
        if n(b2) >= n(b1):
            return b1, b2
        b1, b2 = b2, b1


def decode_u_split(r, basis, Y1, Y2, N1):
    """h1: u from (Y1, Y2) with Y1 / 2^P1 ~ u / r, Y2 / 2^P2 ~ u lam / r: the scaled lattice point nearest to
    (r Y1, r Y2) over a 5 x 5 neighbourhood of the Babai point (exact integers)"""
    (a1, a2), (c1, c2) = basis
    t1, t2 = r * Y1, r * Y2
    det = a1 * c2 - a2 * c1
    al0, be0 = _rnd(t1 * c2 - t2 * c1, det), _rnd(a1 * t2 - a2 * t1, det)
    best = None
    for da in range(-2, 3):
        for db in range(-2, 3):
            x1 = (al0 + da) * a1 + (be0 + db) * c1
            x2 = (al0 + da) * a2 + (be0 + db) * c2
            q = (x1 - t1) ** 2 + (x2 - t2) ** 2
            if best is None or q < best[0]:
                best = (q, x1 // N1)
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


def postprocess(h, pairs, m, l, rho0=0):
    """L0 lattice (Ekera) with the multipliers z_i = 2 u_i known: k_i / 2^l ~ -z_i d / r; LLL, then (branching)
    nearest plane, check-verified"""
    r = h.r
    n = len(pairs)
    big = 1 << l                     # 4-word lattice: the 8-word one divided by 2^m (last weight 1, not r / 2^m)
    rows = [[big * z for z, _ in pairs] + [1]]
    for i in range(n):
        rows.append([0] * i + [r * big] + [0] * (n - i))
    target = [-(r * k) for _, k in pairs] + [0]
    red = h.lll(rows)
    words = max(1, max(abs(x) for row in red for x in row).bit_length() // 64)
    if BRANCH <= 0:
        _, vec = h.babai(red, target)
        cands = [vec]
    else:
        cands = [v for _, v in branch_babai(h, red, target, BRANCH, words)[:CHECKS]]
    for vec in cands:
        d = (vec[-1] - rho0) % r               # the lattice gives d0 = d + rho0
        if h.check(d):
            return d
    return None


def attempt_cap(dim, words):
    """LLL cap + Gram-Schmidt + leaves + checks (h.check <= 2 x 256 group operations each)"""
    leaves = (2 ** BRANCH) * 4 * dim * dim * words if BRANCH > 0 else dim ** 3 * words
    lc = LLL_CAP if LLL_CAP is not None else lll_cap(dim) * words // 8   # calibrated at 8 words
    return lc + dim ** 3 * words + leaves + CHECKS * 512


def solve(h):
    m = h.r.bit_length()
    r = h.r
    l, hexw, kinds, need, runs = params(m)
    N = 1 << sum(hexw)
    split = SPLIT if m >= 256 else 0
    basis = glv_basis(r, h.lam)
    if split:
        N1, N2 = N << max(split, 0), N << max(-split, 0)
        sbasis = scaled_basis(r, h.lam, N1, N2)
    rho0 = h.rng.randrange(r)
    Q0 = h.Q + h.G * rho0                # candidate h1/candidates/ekera_first_b.py): d0 = d + rho0 is uniform for every d
    Q2 = -(Q0 + Q0)
    pairs = []
    h.charge(64 * m)
    for _ in range(runs):
        run = h.run()
        run.offset(global_scalar=GLOBAL)
        b = run.reg(l, sine=l)
        a = run.hexreg(hexw, top='sine', split=split)
        run.add(b, Q2)
        run.hexadd(a, h.G, kinds)
        k, (Y1, Y2) = run.measure()
        h.charge(64)
        if run.flagged:                  # h1: subsets of unflagged runs only (Theorem G's visible flags)
            continue
        u = decode_u_split(r, sbasis, Y1, Y2, N1) if split else decode_u(r, basis, Y1, Y2, N)
        pairs.append((2 * u % r, k, run.id))
    if len(pairs) < need:
        return None
    dim = need + 1
    words = max(1, (m + l + 8) // 64)
    cap = attempt_cap(dim, words)
    tries = TRIES if TRIES is not None else max(1, int((2 ** 40 - 2 ** 34) // cap))
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
