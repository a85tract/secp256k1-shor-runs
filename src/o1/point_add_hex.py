"""Unit-orbit ("hex") windowed point addition for j = 0 curves. V-B KIND 'hex_windowed_point_add'
(spec in o1/vbhex.py, registered at run time; not a harness-owned KIND).

Window digit delta = (-1)^s w^j rho_k in Z[w] (w acts as lam: (x, y) -> (beta x, y)), rho_k one representative per
mu_6-orbit of the minimal-norm residue system H_W of Z[w] / 2^W (o1/hexwin.py): 2W exponent bits per window from
16 bits from 32,768 points). Inputs x, y (accumulator), k (orbit index < K), s (sign), j (unit power, 0..2).

lookups changed: a truncated unary-iteration QROM over K entries loads (x2, beta x2, y2) (the extra payload is
CNOT fan-out, no Toffoli); the unit is applied by a controlled swap (j = 1) or x2 <- -(x2 + beta x2) = beta^2 x2
(j = 2), y2 <- -y2 (s = 1); undone before the measurement-based unlookup. The zero digit is not an input here
"""
import os, sys
_H1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'h1')
sys.path.insert(0, _H1)
import point_add as pa
import vbarith as va
import gcdmul

KIND = 'hex_windowed_point_add'
W = 3
EXACT = False
STEPS = None
REPLAY = 'exact'
SCHEDULE = None
KAPPA = 64
DELTA = 64
PACK = False
SQ_ANDS = None
CMP_BITS = None
MERGE = False


def window(n):
    return W


def declared_failure(n):
    pa.STEPS, pa.REPLAY, pa.SCHEDULE, pa.KAPPA, pa.DELTA = STEPS, REPLAY, SCHEDULE, KAPPA, DELTA
    pa.CMP_BITS = CMP_BITS
    return pa.declared_failure(n)


def _iter_t(c, ctl, bits, cb, L, idx=0):
    """unary iteration (point_add._iterate) truncated to addresses < L (valid inputs only)."""
    if not bits:
        cb(ctl, idx)
        return
    a, rest = bits[0], bits[1:]
    sh = len(rest)
    one = idx | (1 << sh)
    if one >= L:                                  # a = 1 only on invalid addresses: a = 0 on valid inputs
        _iter_t(c, ctl, rest, cb, L, idx)
        return
    if ctl is None:
        _iter_t(c, a, rest, cb, L, one)
        c.x(a)
        _iter_t(c, a, rest, cb, L, idx)
        c.x(a)
    else:
        t = c.and_(ctl, a)
        _iter_t(c, t, rest, cb, L, one)
        c.cx(ctl, t)
        _iter_t(c, t, rest, cb, L, idx)
        m = c.measure_x(t)
        c.x(a)
        c.cz(ctl, a, cond=m)
        c.x(a)


def qload_t(c, k, out, T):
    bits = [[b for b in range(len(out)) if (v >> b) & 1] for v in T]

    def leaf(f, i):
        for b in bits[i]:
            c.cx(f, out[b])
    _iter_t(c, None, k[::-1], leaf, len(T))


def qunload_t(c, k, out, T):
    """measurement-based unlookup (point_add.qunload) with the high-part iteration truncated."""
    L = len(T)
    Wk = len(k)
    ms = [c.measure_x(q) for q in out]
    l = (Wk + 1) // 2
    h = Wk - l
    lo, hi = k[:l][::-1], k[l:][::-1]
    onehot = c.alloc(1 << l)
    pa._iterate(c, None, lo, lambda f, j: c.cx(f, onehot[j]))

    def g_of(v):
        g, b = 0, 0
        while v:
            if v & 1:
                g ^= ms[b]
            v >>= 1
            b += 1
        return g

    def leaf(f, ih):
        for j in range(1 << l):
            i = (ih << l) | j
            if i >= L:
                continue
            g = g_of(T[i])
            if g:
                if f is None:
                    c.z(onehot[j], cond=g)
                else:
                    c.cz(f, onehot[j], cond=g)
    if h:
        _iter_t(c, None, hi, leaf, (L + (1 << l) - 1) >> l)
    else:
        leaf(None, 0)
    pa._iterate(c, None, lo, lambda f, j: c.cx(f, onehot[j]))
    c.free(onehot)


# ---------------------------------------------------------------- blocked carry chains: <= B + L/B live carries
BLOCK = 64
_orig = {}


def _blocked_add(c, X, Y, cin=None, cout=None):
    L = len(Y)
    X = va._pad(X, L)
    if L <= BLOCK:
        return _orig['add'](c, X, Y, cin, cout)
    blocks = [(i, min(i + BLOCK, L)) for i in range(0, L, BLOCK)]
    kept, cur = [], cin
    for lo, hi in blocks[:-1]:
        k = c.alloc(1)[0]
        _orig['add'](c, X[lo:hi], Y[lo:hi], cur, k)        # k ^= block carry-out
        kept.append((k, lo, hi, cur))
        cur = k
    lo, hi = blocks[-1]
    _orig['add'](c, X[lo:hi], Y[lo:hi], cur, cout)
    for k, lo, hi, kin in reversed(kept):                   # k = [Y_new < X + kin] = carry(~Y_new + X + kin)
        for q in Y[lo:hi]:
            c.x(q)
        _orig['carry_do'](c, X[lo:hi], Y[lo:hi], lambda cy, k=k: c.cx(cy, k) if cy is not None else None, kin)
        for q in Y[lo:hi]:
            c.x(q)
        c.free([k])


def _blocked_carry_do(c, X, Y, act, cin=None):
    L = len(Y)
    X = va._pad(X, L)
    if L <= BLOCK:
        return _orig['carry_do'](c, X, Y, act, cin)
    blocks = [(i, min(i + BLOCK, L)) for i in range(0, L, BLOCK)]
    kept, cur = [], cin
    for lo, hi in blocks[:-1]:
        k = c.alloc(1)[0]
        _orig['carry_do'](c, X[lo:hi], Y[lo:hi], lambda cy, k=k: c.cx(cy, k) if cy is not None else None, cur)
        kept.append((k, lo, hi, cur))
        cur = k
    lo, hi = blocks[-1]
    r = _orig['carry_do'](c, X[lo:hi], Y[lo:hi], act, cur)
    for k, lo, hi, kin in reversed(kept):
        _orig['carry_do'](c, X[lo:hi], Y[lo:hi], lambda cy, k=k: c.cx(cy, k) if cy is not None else None, kin)
        c.free([k])
    return r


def _blocked_allzero_do(c, Q, act):
    if len(Q) <= BLOCK:
        return _orig['allzero_do'](c, Q, act)
    blocks = [Q[i:i + BLOCK] for i in range(0, len(Q), BLOCK)]
    zs = []
    for blk in blocks:
        z = c.alloc(1)[0]
        _orig['allzero_do'](c, blk, lambda t, z=z: c.cx(t, z))
        zs.append(z)
    for z in zs:
        c.x(z)
    r = _orig['allzero_do'](c, zs, act)                     # [all block flags set]
    for z in zs:
        c.x(z)
    for blk, z in zip(blocks, zs):
        _orig['allzero_do'](c, blk, lambda t, z=z: c.cx(t, z))
    c.free(zs)
    return r


class _Blocked:
    def __enter__(self):
        for nm, f in (('add', _blocked_add), ('carry_do', _blocked_carry_do), ('allzero_do', _blocked_allzero_do)):
            _orig[nm] = getattr(va, nm)
            setattr(va, nm, f)

    def __exit__(self, *a):
        for nm in ('add', 'carry_do', 'allzero_do'):
            setattr(va, nm, _orig[nm])


def phase_fix(c, k, s, j, K, F, l=8):
    """(-1)^F(k, s, j) on the address registers (measurement-based unlookup of all payload slots at once).
    One-hot over the low l bits of k; unary iteration over (s, j0, j1, high bits of k), truncated at k < K."""
    lo = k[:l]
    hi_bits = ([s] if s is not None else []) + list(j) + list(k[l:])
    ns = 1 if s is not None else 0
    shift = ns + 2
    Lhi = ((K + (1 << l) - 1) >> l) << shift
    onehot = c.alloc(1 << l)
    pa._iterate(c, None, lo[::-1], lambda f, i: c.cx(f, onehot[i]))

    def leaf(f, ih):
        sv = ih & 1 if ns else 0
        jj = (ih >> ns) & 3
        kh = ih >> shift
        for i in range(1 << l):
            kk = (kh << l) | i
            if kk >= K or jj == 3:
                continue
            g = F(kk, sv, jj)                    # lane bit-vector (measurement outcomes differ per lane)
            if g:
                if f is None:
                    c.z(onehot[i], cond=g)
                else:
                    c.cz(f, onehot[i], cond=g)
    _iter_t(c, None, hi_bits[::-1], leaf, Lhi)
    pa._iterate(c, None, lo[::-1], lambda f, i: c.cx(f, onehot[i]))
    c.free(onehot)


def _dot(ms, v):
    g, b = 0, 0
    while v:
        if v & 1:
            g ^= ms[b]
        v >>= 1
        b += 1
    return g


def build(c, n, regs, ctx):
    p, T, beta = ctx['p'], ctx['table'], ctx['beta']
    K = len(T)
    gcdmul.STEPS, gcdmul.REPLAY, gcdmul.SCHEDULE, gcdmul.PACK = STEPS, REPLAY, SCHEDULE, PACK
    gcdmul.CMP_BITS, gcdmul.MERGE = CMP_BITS, MERGE
    va.KAPPA, va.DELTA = KAPPA, DELTA
    pa.SQ_ANDS = SQ_ANDS
    X, Y = list(regs['x']), regs['y']
    k, s, j = regs['k'], regs['s'][0], regs['j']
    j0, j1 = j
    xs = [P[0] for P in T]
    bx = [beta * x % p for x in xs]
    b2x = [beta * beta * x % p for x in xs]
    ys = [P[1] for P in T]
    pay3 = [x | (b << n) | (y << (2 * n)) for x, b, y in zip(xs, bx, ys)]
    t3p = [(3 * x % p) | ((3 * b % p) << n) for x, b in zip(xs, bx)]

    def sel_vals(kk, jj, mult):
        """(Dx, Db) after the selection for table index kk and unit power jj"""
        x, b, b2 = mult * xs[kk] % p, mult * bx[kk] % p, mult * b2x[kk] % p
        return [(x, b2), (b, b2), (b2, x)][jj]

    def select(Dx, Db, blocked):
        va.cswap(c, j0, Dx, Db)                      # j = 1: (beta x, x)
        if blocked:
            with _Blocked():
                va.modadd(c, Dx, Db, p)              # Db = x + beta x
                va.modneg(c, Db, p)                  # Db = beta^2 x (for every j)
        else:
            va.modadd(c, Dx, Db, p)
            va.modneg(c, Db, p)
        va.cswap(c, j1, Dx, Db)                      # j = 2: Dx = beta^2 x

    def lookup_xy(fn):
        D = c.alloc(3 * n)
        qload_t(c, k, D, pay3)
        Dx, Db, Dy = D[:n], D[n:2 * n], D[2 * n:]
        select(Dx, Db, True)
        mb = [c.measure_x(q) for q in Db]           # unused slot out before any full-width carry chain
        va.negate_nonzero(c, Dy, p, s)
        fn(Dx, Dy)
        mx = [c.measure_x(q) for q in Dx]
        my = [c.measure_x(q) for q in Dy]
        phase_fix(c, k, s, j, K, lambda kk, sv, jj: _dot(mb, sel_vals(kk, jj, 1)[1]) ^ _dot(mx, sel_vals(kk, jj, 1)[0])
                  ^ _dot(my, (p - ys[kk]) % p if sv else ys[kk]))

    def first(dx, dy):
        va.modsub(c, dx, X, p)
        va.modsub(c, dy, Y, p)
    lookup_xy(first)
    Y = gcdmul.inplace(c, X, Y, p, True)        # lambda = dy / dx
    D = c.alloc(2 * n)
    qload_t(c, k, D, t3p)
    Dx, Db = D[:n], D[n:]
    select(Dx, Db, False)
    mb = [c.measure_x(q) for q in Db]
    va.modadd(c, Dx, X, p)                      # x1 + 2 x2 (x2 unit-rotated)
    mx = [c.measure_x(q) for q in Dx]
    phase_fix(c, k, None, j, K, lambda kk, sv, jj: _dot(mb, sel_vals(kk, jj, 3)[1]) ^ _dot(mx, sel_vals(kk, jj, 3)[0]))
    pa.square_sub(c, Y, X, p, ctl=None)         # x2 - x3
    Y = gcdmul.inplace(c, X, Y, p, False)       # y3 + y2

    def last(dx, dy):
        va.modsub(c, dy, Y, p)
        va.modsub(c, dx, X, p)
    lookup_xy(last)
    va.modneg(c, X, p, ctl=None)
    return {'x': X, 'y': Y}
