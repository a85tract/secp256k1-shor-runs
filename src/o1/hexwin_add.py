"""Hex window on v10 core. V-B KIND 'hex_window_add' (spec registered by o1/vbhexw.py).

Input: accumulator (x, y) (affine, O = (0, 0)) and the window's residue register xr = a + 2^W b (a + b w in
(Z/2^W)^2, the register the semiclassical QFT measures). Output: accumulator + [digit(a, b)(lam)] P with
digit() of o1/hexdigits.py; xr restored. Inside:
  1. conversion (in place, reversible): x -> (x'', s, j), x'' the canonical orbit representative (hexdigits.convert);
     j and s are 3 new qubits (j0 = [j = 1], j1 = [j = 2]).
     (ping-pong division walk, chunked replay adder, Karatsuba squaring): three lookups by unary iteration over the
     caterpillar tree of canonical residues (K - 1 Toffoli each, K = (4^W - 4)/6 + 2), unit applied to the
     looked-up point (x -> beta^j x by two controlled swaps around one -(x + beta x); y -> (-1)^s y), every payload
     slot X-measured and ONE deferred phase oracle over (x'', s, j) at the end (one-hot over b''[1..W-1] x tree over
     (s, j, a'')).
  3. inverse conversion.
The correction-phase digit register of the semiclassical measurement (delta = (-1)^s w^j lift(x''), computed and
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'h1'))
sys.path.insert(0, _HERE)
import point_add as pa
import vbarith as va
import gcdmul
import pingpong
import point_add_hex as pah
from hexdigits import canonical_list

KIND = 'hex_window_add'
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
WALK = 'binary'
ROUNDS = None
PP_SCHED = None
CHUNK = None
CHUNK_CMP = 28
SQUARE = 'schoolbook'
SEL_BLOCK = 16          # carry block of the unit selection (runs while the 3n payload is live)
SELECT = 'blocked'      # 'mcarry': forward-only selection, carries X-measured (O(1) ancillas, select_mcarry)
HOLD = 'orbit'          # 'residue': hold the window register as the 18-qubit residue during the two core divisions
CUT_FLAGS = False
FIRST_L = 19            # the run's first window (the b-register), loaded by a direct lookup (first_lookup below)
FIRST_SS = False


def window(n):
    return W


def _set_core():
    pa.STEPS, pa.REPLAY, pa.SCHEDULE, pa.KAPPA, pa.DELTA = STEPS, REPLAY, SCHEDULE, KAPPA, DELTA
    pa.PACK, pa.SQ_ANDS, pa.CMP_BITS, pa.MERGE = PACK, SQ_ANDS, CMP_BITS, MERGE
    pa.WALK, pa.ROUNDS, pa.PP_SCHED, pa.CHUNK, pa.CHUNK_CMP = WALK, ROUNDS, PP_SCHED, CHUNK, CHUNK_CMP
    pa.SQUARE = SQUARE
    gcdmul.STEPS, gcdmul.REPLAY, gcdmul.SCHEDULE, gcdmul.PACK = STEPS, REPLAY, SCHEDULE, PACK
    gcdmul.CMP_BITS, gcdmul.MERGE = CMP_BITS, MERGE
    va.KAPPA, va.DELTA = KAPPA, DELTA
    va.CHUNK, va.CHUNK_CMP = CHUNK, CHUNK_CMP
    pingpong.ROUNDS, pingpong.SCHED = ROUNDS, PP_SCHED
    pa.CUT_FLAGS = CUT_FLAGS
    va.CUT_FLAGS = CUT_FLAGS


def declared_failure(n):
    _set_core()
    return pa.declared_failure(n)


def first_lookup(n, offset=False):
    """The run's first window (the b-register, FIRST_L bits) loaded into the identity accumulator by a direct
    with offset=True the unsigned 2^FIRST_L-entry table T0[k] + R0 of the harness's Run.offset()."""
    if offset and FIRST_SS:
        return first_lookup_ss(n)
    w0, s0 = pa.W, pa.SIGNED
    pa.SIGNED, pa.W, pa.OFFSET = True, FIRST_L, offset
    try:
        return pa.first_lookup(n)
    finally:
        pa.OFFSET, pa.W, pa.SIGNED = False, w0, s0


def first_lookup_ss(n):
    """Counts of the offset first lookup built as o1/first_ss.py (V-B PASS at 4-12 address bits; Toffoli count equal
    to its formula there): select-swap load of the 2^FIRST_L-entry table T0[k] + R0 into the 2n-qubit accumulator,
    then the phase fix-up of the select register. Counted with Counter (counts do not depend on the table)."""
    import random
    import count as _cnt
    import first_ss as _fs
    c = _cnt.Counter()
    k = c.alloc(FIRST_L)
    D = c.alloc(2 * n)
    rng = random.Random('first-ss-count')
    T = [rng.getrandbits(2 * n) for _ in range(1 << FIRST_L)]
    _fs.load(c, k, D, T)
    return dict(toffoli=c.toffoli, peak_qubits=c.peak, build='select-swap (o1/first_ss.py)')


# ------------------------------------------------------------------ small reversible arithmetic on W-bit registers
def _and_mbu(c, a, b, negb=False):
    """fresh t = a AND (b or NOT b); returns (t, undo) with undo = measurement-based uncomputation (0 Toffoli)."""
    if negb:
        c.x(b)
    t = c.and_(a, b)
    if negb:
        c.x(b)

    def undo():
        m = c.measure_x(t)
        if negb:
            c.x(b)
        c.cz(a, b, cond=m)
        if negb:
            c.x(b)
    return t, undo


def _csub(c, ctl, A, B):
    """B <- B - ctl * A mod 2^len (controlled operand as lazy AND terms)"""
    va.sub(c, [('and', ctl, q) for q in A], B)


def _cadd(c, ctl, A, B):
    va.add(c, [('and', ctl, q) for q in A], B)


def _cneg(c, ctl, B):
    """B <- -B mod 2^len if ctl"""
    for q in B:
        c.cx(ctl, q)
    va.add(c, [ctl], B)


def _j_bits(c, xa, xb, j0, j1, Wd):
    """j0 ^= [j = 1], j1 ^= [j = 2] of the residue (xa, xb) (hexdigits.convert); everything else restored."""
    undo = []
    z = None                                       # z_t = [a_u = b_u = 0 for all u < t] (None: constant 1)
    for t in range(Wd):
        a, b = xa[t], xb[t]
        g, ug = _and_mbu(c, b, a, negb=True)       # b AND NOT a  -> j = 1 at the valuation
        h, uh = _and_mbu(c, a, b)                  # a AND b      -> j = 2
        undo += [ug, uh]
        if z is None:
            c.cx(g, j0)
            c.cx(h, j1)
        else:
            c.ccx(z, g, j0)
            c.ccx(z, h, j1)
        if t < Wd - 1:
            c.x(a)
            c.x(b)
            nz, un = _and_mbu(c, a, b)            # NOT a AND NOT b
            c.x(a)
            c.x(b)
            fix = (a, b)
            undo.append(_wrap_neg_undo(c, un, fix))
            if z is None:
                z = nz
            else:
                z2, uz = _and_mbu(c, z, nz)
                undo.append(uz)
                z = z2
    for u in reversed(undo):
        u()


def _wrap_neg_undo(c, un, fix):
    a, b = fix

    def f():
        c.x(a)
        c.x(b)
        un()
        c.x(a)
        c.x(b)
    return f


def _s_bit(c, xa, s, Wd):
    """s ^= bit (v+1) of a', v = lowest set bit of a' (nothing if a' = 0 or v = W-1); everything else restored."""
    undo = []
    z = None                                       # z_t = [a'_u = 0 for all u < t] (None: constant 1)
    for t in range(Wd - 1):
        k, uk = _and_mbu(c, xa[t], xa[t + 1])
        undo.append(uk)
        if z is None:
            c.cx(k, s)
        else:
            c.ccx(z, k, s)
        if t < Wd - 2:
            if z is None:
                z1 = c.alloc(1)[0]
                c.cx(xa[0], z1)
                c.x(z1)

                def u1(z1=z1):
                    c.x(z1)
                    c.cx(xa[0], z1)
                    c.free([z1])
                undo.append(u1)
                z = z1
            else:
                z2, uz = _and_mbu(c, z, xa[t], negb=True)
                undo.append(uz)
                z = z2
    for u in reversed(undo):
        u()


def conv_fwd(c, xa, xb, Wd):
    s, j0, j1 = c.alloc(3)
    _j_bits(c, xa, xb, j0, j1, Wd)
    # x' = w^(-j) x: j = 1: (a, b) -> (b - a, -a) ; j = 2: (a, b) -> (-b, a - b)
    _csub(c, j0, xa, xb)
    va.cswap(c, j0, xa, xb)
    _cneg(c, j0, xb)
    _csub(c, j1, xb, xa)
    va.cswap(c, j1, xa, xb)
    _cneg(c, j1, xa)
    _s_bit(c, xa, s, Wd)
    _cneg(c, s, xa)
    _cneg(c, s, xb)
    return s, j0, j1


def conv_inv(c, xa, xb, s, j0, j1, Wd):
    _cneg(c, s, xb)
    _cneg(c, s, xa)
    _s_bit(c, xa, s, Wd)
    _cneg(c, j1, xa)
    va.cswap(c, j1, xa, xb)
    _cadd(c, j1, xb, xa)
    _cneg(c, j0, xb)
    va.cswap(c, j0, xa, xb)
    _cadd(c, j0, xa, xb)
    _j_bits(c, xa, xb, j0, j1, Wd)
    c.free([s, j0, j1])


# ------------------------------------------------------------------ caterpillar tree over canonical residues
def cat_tree(Wd, xa, xb):
    M = 1 << Wd

    def full(v):
        bits = [('a', t) for t in range(v + 2, Wd)] + [('b', t) for t in range(v + 1, Wd)]

        def rec(i, a, b):
            if i == len(bits):
                return ('leaf', (a, b))
            kd, t = bits[i]
            if kd == 'a':
                return ('br', xa[t], rec(i + 1, a, b), rec(i + 1, a | (1 << t), b))
            return ('br', xb[t], rec(i + 1, a, b), rec(i + 1, a, b | (1 << t)))
        return rec(0, 1 << v, 0)

    def spine(v):
        if v == Wd:
            return ('leaf', (0, 0))
        if v == Wd - 1:
            return ('br', xa[Wd - 1], spine(Wd), ('leaf', (M >> 1, 0)))
        return ('br', xa[v], spine(v + 1), full(v))
    return spine(0)


def iter_tree(c, ctl, node, cb):
    if node[0] == 'leaf':
        cb(ctl, node[1])
        return
    _, q, ch0, ch1 = node
    if ctl is None:
        iter_tree(c, q, ch1, cb)
        c.x(q)
        iter_tree(c, q, ch0, cb)
        c.x(q)
    else:
        t = c.and_(ctl, q)
        iter_tree(c, t, ch1, cb)
        c.cx(ctl, t)
        iter_tree(c, t, ch0, cb)
        m = c.measure_x(t)
        c.x(q)
        c.cz(ctl, q, cond=m)
        c.x(q)


def qload_tree(c, tree, out, pay):
    def leaf(f, key):
        v, b = pay[key], 0
        while v:
            if v & 1:
                c.cx(f, out[b])
            v >>= 1
            b += 1
    iter_tree(c, None, tree, leaf)


class _Dot:
    """lane-vector dot products <ms, v> (XOR of ms[b] over set bits b of v), by 8-bit chunk tables"""
    def __init__(self, ms):
        self.tabs = []
        for c0 in range(0, len(ms), 8):
            chunk = ms[c0:c0 + 8]
            T = [0] * (1 << len(chunk))
            for i in range(1, len(T)):
                low = i & -i
                T[i] = T[i ^ low] ^ chunk[low.bit_length() - 1]
            self.tabs.append(T)

    def __call__(self, v):
        g = 0
        for T in self.tabs:
            if not v:
                break
            g ^= T[v & (len(T) - 1)]
            v >>= 8
        return g


def phase_oracle(c, xa, xb, s, j0, j1, Wd, F):
    """(-1)^F(x'', s, j) for every valid (x'', s, j): one-hot over b''[1..W-1], unary iteration over (s, j, a'')."""
    M = 1 << Wd
    l = Wd - 1
    lo = xb[1:]
    onehot = c.alloc(1 << l)
    pa._iterate(c, None, lo[::-1], lambda f, i: c.cx(f, onehot[i]))

    def a_tree(tag, v):
        if v == Wd:
            return ('leaf', (tag, 0, Wd))
        if v == Wd - 1:
            return ('br', xa[Wd - 1], a_tree(tag, Wd), ('leaf', (tag, M >> 1, Wd - 1)))
        return ('br', xa[v], a_tree(tag, v + 1), a_full(tag, v))

    def a_full(tag, v):
        bits = list(range(v + 2, Wd))

        def rec(i, a):
            if i == len(bits):
                return ('leaf', (tag, a, v))
            t = bits[i]
            return ('br', xa[t], rec(i + 1, a), rec(i + 1, a | (1 << t)))
        return rec(0, 1 << v)

    def sj(sv):
        return ('br', j0, ('br', j1, a_tree((sv, 0), 0), a_tree((sv, 2), 0)), a_tree((sv, 1), 0))
    tree = ('br', s, sj(0), sj(1))

    def leaf(f, key):
        (sv, jj), a, v = key
        if v == Wd:
            cands = [0] if (sv, jj) == (0, 0) else []
        elif v == Wd - 1:
            cands = [0] if sv == 0 else []
        else:
            cands = range(0, 1 << l, 1 << v)
        for i in cands:
            g = F((a, 2 * i), sv, jj)
            if g:
                c.cz(f, onehot[i], cond=g)
    iter_tree(c, None, tree, leaf)
    pa._iterate(c, None, lo[::-1], lambda f, i: c.cx(f, onehot[i]))
    c.free(onehot)


# ------------------------------------------------------------------ the window
def build(c, n, regs, ctx):
    _set_core()
    pah.BLOCK = SEL_BLOCK
    p, T, beta = ctx['p'], ctx['table'], ctx['beta']
    inplace = pingpong.inplace if WALK == 'pingpong' else gcdmul.inplace
    X, Y = list(regs['x']), regs['y']
    xr = regs['xr']
    Wd = len(xr) // 2
    xa, xb = xr[:Wd], xr[Wd:]
    s, j0, j1 = conv_fwd(c, xa, xb, Wd)
    tree = cat_tree(Wd, xa, xb)
    xs = {k: P[0] for k, P in T.items()}
    bx = {k: beta * x % p for k, x in xs.items()}
    b2x = {k: (p - x - bx[k]) % p for k, x in xs.items()}
    ys = {k: P[1] for k, P in T.items()}
    pay3 = {k: xs[k] | (bx[k] << n) | (ys[k] << (2 * n)) for k in xs}
    t3p = {k: (3 * xs[k] % p) | ((3 * bx[k] % p) << n) for k in xs}
    recs = []                                         # (dot, value function (key, s, j) -> int)

    def sel_vals(kk, jj, mult):
        x, b, b2 = mult * xs[kk] % p, mult * bx[kk] % p, mult * b2x[kk] % p
        return [(x, b2), (b, b2), (b2, x)][jj]

    def select(Dx, Db):
        va.cswap(c, j0, Dx, Db)                      # j = 1: (beta x, x)
        with pah._Blocked():
            va.modadd(c, Dx, Db, p)                  # Db = x + beta x
            va.modneg(c, Db, p)                      # Db = beta^2 x (for every j)
        va.cswap(c, j1, Dx, Db)                      # j = 2: Dx = beta^2 x

    def lookup_xy(fn):
        D = c.alloc(3 * n)
        qload_tree(c, tree, D, pay3)
        Dx, Db, Dy = D[:n], D[n:2 * n], D[2 * n:]
        if SELECT == 'mcarry':
            select_mcarry(c, Dx, Db, j0, j1, p, n, xs, bx, recs, 1)
        else:
            select(Dx, Db)
        recs.append((_Dot([c.measure_x(q) for q in Db]), lambda kk, sv, jj: sel_vals(kk, jj, 1)[1]))
        va.negate_nonzero(c, Dy, p, s)
        fn(Dx, Dy)
        recs.append((_Dot([c.measure_x(q) for q in Dx]), lambda kk, sv, jj: sel_vals(kk, jj, 1)[0]))
        recs.append((_Dot([c.measure_x(q) for q in Dy]), lambda kk, sv, jj: (p - ys[kk]) % p if sv else ys[kk]))

    def first(dx, dy):
        va.modsub(c, dx, X, p)
        va.modsub(c, dy, Y, p)
    lookup_xy(first)
    if HOLD == 'residue':
        conv_inv(c, xa, xb, s, j0, j1, Wd)
    Y = inplace(c, X, Y, p, True)                    # lambda = dy / dx
    if HOLD == 'residue':
        s, j0, j1 = conv_fwd(c, xa, xb, Wd)
    D = c.alloc(2 * n)
    qload_tree(c, tree, D, t3p)
    Dx, Db = D[:n], D[n:]
    va.cswap(c, j0, Dx, Db)
    va.modadd(c, Dx, Db, p)
    va.modneg(c, Db, p)
    va.cswap(c, j1, Dx, Db)
    recs.append((_Dot([c.measure_x(q) for q in Db]), lambda kk, sv, jj: sel_vals(kk, jj, 3)[1]))
    va.modadd(c, Dx, X, p)                           # x1 + 2 x2 (x2 unit-rotated)
    recs.append((_Dot([c.measure_x(q) for q in Dx]), lambda kk, sv, jj: sel_vals(kk, jj, 3)[0]))
    if SQUARE == 'karatsuba':
        pa.square_sub_kara(c, Y, X, p)               # x2 - x3
    else:
        pa.square_sub(c, Y, X, p, ctl=None)
    if HOLD == 'residue':
        conv_inv(c, xa, xb, s, j0, j1, Wd)
    Y = inplace(c, X, Y, p, False)                   # y3 + y2
    if HOLD == 'residue':
        s, j0, j1 = conv_fwd(c, xa, xb, Wd)

    def last(dx, dy):
        va.modsub(c, dy, Y, p)
        va.modsub(c, dx, X, p)
    lookup_xy(last)
    va.modneg(c, X, p, ctl=None)

    def F(kk, sv, jj):
        g = 0
        for dot, fn in recs:
            g ^= dot(fn(kk, sv, jj))
        return g
    phase_oracle(c, xa, xb, s, j0, j1, Wd, F)
    conv_inv(c, xa, xb, s, j0, j1, Wd)
    return {'x': X, 'y': Y}


# ------------------------------------------------------------------ correction-phase digit register
def _sign_do(c, U, k, act):
    """act(qubit holding the sign bit of U + k) for an L-bit two's-complement register U and an int constant k
    (no overflow assumed); U restored."""
    L = len(U)
    kk = k % (1 << L)
    top_k = (kk >> (L - 1)) & 1

    def inner(cq):
        f = c.alloc(1)[0]
        if cq is not None:
            c.cx(cq, f)
        c.cx(U[L - 1], f)
        if top_k:
            c.x(f)
        r = act(f)
        if top_k:
            c.x(f)
        c.cx(U[L - 1], f)
        if cq is not None:
            c.cx(cq, f)
        c.free([f])
        return r
    return va.carry_do(c, va.cbits(kk & ((1 << (L - 1)) - 1), L - 1), U[:L - 1], inner)


def phase_digit(c, xa, xb, s, j0, j1, Wd):
    """delta = (-1)^s w^j lift(x'') in two L-bit two's-complement registers (L = W + 3), lift = hexreduce(centered
    (x'')) as in hexdigits.hexreduce, lift(0) = (2^W, 0). Returns (D1, D2, undo); undo() restores everything."""
    M = 1 << Wd
    L = Wd + 3
    A = list(xa) + [xa[Wd - 1]] * 3                      # centered a'' (sign-extended, read-only operand)
    B = list(xb) + [xb[Wd - 1]] * 3
    U, V, S = c.alloc(L), c.alloc(L), c.alloc(L)
    f = c.alloc(6)
    pq = c.alloc(2)
    qq = c.alloc(1)[0]
    e = c.alloc(4)
    z = c.alloc(1)[0]
    D1, D2 = c.alloc(L), c.alloc(L)
    hi1, hi2 = D1[Wd:], D2[Wd:]

    def flags(sign):
        """sign = +1: compute U, V, S, f, pq, qq, e, z ; sign = -1: the exact reverse"""
        steps = []
        steps.append(('uvs', None))
        for i, (reg, k, neg) in enumerate(((U, -(M + 1), True), (V, M, False), (U, M, False),
                                           (V, -(M + 1), True), (S, 0, False), (S, -1, False))):
            steps.append(('flag', (i, reg, k, neg)))
        steps.append(('or', 0))
        steps.append(('or', 1))
        steps.append(('qq', None))
        steps.append(('e', None))
        steps.append(('z', None))
        if sign < 0:
            steps.reverse()
        for kind, arg in steps:
            if kind == 'uvs':
                if sign > 0:
                    va.add(c, [va.K0] + A[:L - 1], U)
                    va.sub(c, B, U)
                    va.add(c, [va.K0] + B[:L - 1], V)
                    va.sub(c, A, V)
                    va.add(c, A, S)
                    va.add(c, B, S)
                else:
                    va.sub(c, B, S)
                    va.sub(c, A, S)
                    va.add(c, A, V)
                    va.sub(c, [va.K0] + B[:L - 1], V)
                    va.add(c, B, U)
                    va.sub(c, [va.K0] + A[:L - 1], U)
            elif kind == 'flag':
                i, reg, k, neg = arg
                if neg and sign < 0:
                    c.x(f[i])
                _sign_do(c, reg, k, lambda q, i=i: c.cx(q, f[i]))
                if neg and sign > 0:
                    c.x(f[i])
            elif kind == 'or':
                u, v = (f[0], f[1]) if arg == 0 else (f[2], f[3])
                if sign < 0:
                    c.x(pq[arg])
                c.x(u)
                c.x(v)
                c.ccx(u, v, pq[arg])
                c.x(u)
                c.x(v)
                if sign > 0:
                    c.x(pq[arg])                   # pq[arg] = u OR v
            elif kind == 'qq':
                c.x(pq[0])
                c.ccx(pq[0], pq[1], qq)            # q and not p (hexreduce: elif)
                c.x(pq[0])
            elif kind == 'e':
                c.x(f[4])
                c.ccx(pq[0], f[4], e[0])           # p and S >= 0: A -= M
                c.x(f[4])
                c.ccx(pq[0], f[4], e[1])           # p and S < 0: B += M
                c.ccx(qq, f[5], e[2])              # q and S <= 0: A += M
                c.x(f[5])
                c.ccx(qq, f[5], e[3])              # q and S > 0: B -= M
                c.x(f[5])
            elif kind == 'z':
                va.allzero_do(c, list(xa) + list(xb), lambda t: c.cx(t, z))

    def digit_ops(sign):
        if sign > 0:
            for i in range(L):
                c.cx(A[i], D1[i])
                c.cx(B[i], D2[i])
            va.sub(c, [e[0]], hi1)
            va.add(c, [e[2]], hi1)
            va.add(c, [z], hi1)
            va.add(c, [e[1]], hi2)
            va.sub(c, [e[3]], hi2)
            _csub(c, j0, D2, D1)                   # j = 1: (a, b) -> (-b, a - b)
            va.cswap(c, j0, D1, D2)
            _cneg(c, j0, D1)
            _csub(c, j1, D1, D2)                   # j = 2: (a, b) -> (b - a, -a)
            va.cswap(c, j1, D1, D2)
            _cneg(c, j1, D2)
            _cneg(c, s, D1)
            _cneg(c, s, D2)
        else:
            _cneg(c, s, D2)
            _cneg(c, s, D1)
            _cneg(c, j1, D2)
            va.cswap(c, j1, D1, D2)
            _cadd(c, j1, D1, D2)
            _cneg(c, j0, D1)
            va.cswap(c, j0, D1, D2)
            _cadd(c, j0, D2, D1)
            va.add(c, [e[3]], hi2)
            va.sub(c, [e[1]], hi2)
            va.sub(c, [z], hi1)
            va.sub(c, [e[2]], hi1)
            va.add(c, [e[0]], hi1)
            for i in range(L):
                c.cx(A[i], D1[i])
                c.cx(B[i], D2[i])
    flags(+1)
    digit_ops(+1)

    def undo():
        digit_ops(-1)
        flags(-1)
        c.free(D2)
        c.free(D1)
        c.free([z])
        c.free(e)
        c.free([qq])
        c.free(pq)
        c.free(f)
        c.free(S)
        c.free(V)
        c.free(U)
    return D1, D2, undo


# ------------------------------------------------------------------ low-ancilla unit selection (SELECT = 'mcarry')
def _fwd_add(c, A, B, recs, carries_fn):
    """B <- B + A mod 2^len(B) by a forward-only ripple; each carry qubit is X-measured right after its last use
    (O(1) live ancillas). The phases (-1)^(m_i c_i) are left to the window's deferred phase oracle, so the operands
    must be functions of the lookup address: carries_fn(key, s, j) returns the classical carry vector
    (bit i = carry into bit i) and is recorded with the outcomes in recs."""
    L = len(B)
    A = va._pad(A, L)
    carry, ms, bits = None, [], []
    for i in range(L):
        new = va._maj(c, A[i], B[i], carry) if i < L - 1 else None
        if carry is not None:
            c.cx(carry, B[i])
        a, done = va._x_get(c, A[i])
        if a is va.K1:
            c.x(B[i])
        elif a is not va.K0:
            c.cx(a, B[i])
        if done:
            done()
        if carry is not None:
            ms.append(c.measure_x(carry))
            bits.append(i)
        carry = new
    recs.append((_Dot(_spread(ms, bits)), carries_fn))


def _spread(ms, bits):
    """outcome list indexed by carry position (bit i of the carry vector), zeros elsewhere"""
    out = [0] * (max(bits) + 1 if bits else 0)
    for m, b in zip(ms, bits):
        out[b] = m
    return out


def _fwd_carry_out(c, A, B, target, recs, carries_fn):
    """target ^= carry-out of A + B over len(B) bits (A constants / qubits); B unchanged; carries X-measured."""
    L = len(B)
    A = va._pad(A, L)
    carry, ms, bits = None, [], []
    for i in range(L):
        new = va._maj(c, A[i], B[i], carry)
        if carry is not None:
            ms.append(c.measure_x(carry))
            bits.append(i)
        carry = new
    if carry is not None:
        c.cx(carry, target)
        ms.append(c.measure_x(carry))
        bits.append(L)
    recs.append((_Dot(_spread(ms, bits)), carries_fn))


def _carries(a, b, L):
    """bit i (1 <= i <= L) = carry into bit i of a + b (L-bit operands)"""
    return (((a + b) ^ a ^ b) >> 0) & ((1 << (L + 1)) - 1) & ~1


def select_mcarry(c, Dx, Db, j0, j1, p, n, xs, bx, recs, mult):
    """Dx <- beta^j x2, Db <- the other slot value, as select() of build(), with O(1) carry ancillas.
    Db0 = slot after the j0 swap; S = Dx0 + Db0 (n + 1 bits); f = [S >= p]; Db <- (1 + f) p - S = beta^2 x2."""
    e = c.alloc(1)[0]
    f = c.alloc(1)[0]
    va.cswap(c, j0, Dx, Db)

    def vals(kk, jj):
        x, b = mult * xs[kk] % p, mult * bx[kk] % p
        return (b, x) if jj == 1 else (x, b)
    S = list(Db) + [e]
    L = n + 1
    _fwd_add(c, list(Dx), S, recs, lambda kk, sv, jj: _carries(*vals(kk, jj), L))
    kneg = (1 << L) - p
    _fwd_carry_out(c, va.cbits(kneg, L), S, f, recs,
                   lambda kk, sv, jj: _carries(kneg, sum(vals(kk, jj)), L))
    for q in S:
        c.x(q)

    def sval(kk, jj):
        return sum(vals(kk, jj))
    _fwd_add(c, va.cbits(1, L), S, recs, lambda kk, sv, jj: _carries(1, ((1 << L) - 1) ^ sval(kk, jj), L))
    _fwd_add(c, va.cbits(p, L), S, recs, lambda kk, sv, jj: _carries(p, (-sval(kk, jj)) % (1 << L), L))
    _fwd_add(c, [f if (p >> i) & 1 else va.K0 for i in range(L)], S, recs,
             lambda kk, sv, jj: _carries(p if sval(kk, jj) >= p else 0, (p - sval(kk, jj)) % (1 << L), L))
    recs.append((_Dot([c.measure_x(f)]), lambda kk, sv, jj: 1 if sval(kk, jj) >= p else 0))
    c.free([e])
    va.cswap(c, j1, Dx, Db)
