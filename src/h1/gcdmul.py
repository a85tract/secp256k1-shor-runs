"""In-place modular multiplication / division by a quantum x via a recorded binary GCD.

Exact for every x in [0, p) (x = 0 is treated as x = 1, so z -> z) and every z in [0, p); p odd.

record(x):  binary GCD on (u, v) = (p, x) in Häner et al.'s (arXiv 2609.05625, Sec. VI) representation
            (u, v~ = v + (1 - a) u), so each step is one conditionally-inverted adder, one comparison and one
            controlled swap; registers shrink with the bound max(u, v) < 2^(2n - k) after k halvings.
            Transcript: per plain step j the bits (a_j = parity of v, s_j = a_j and v < u: swap), N = 2n steps
            (worst case of this GCD: u v drops by >= 2 per step, so v = 0 after <= log2(p x) + 1 <= 2n steps).
            The final state is (u, v~) = (1, 1) for every x != 0, so the GCD registers are freed.
replay:     the plain steps applied mod p to (R, S) = (0, z) give (z/x, 0)  [division]
inv-replay: the inverse steps applied to (R, S) = (w, 0) give (0, w x)    [multiplication]
unrecord:   the exact reverse of record (restores x, clears the transcript).
"""
import vbarith as va

STEPS = None        # GCD halvings per record: None = 2n (exact for every x); an int N < 2n is approximate:
                    # inputs whose GCD needs more than N halvings leave the GCD registers nonzero (garbage)


REPLAY = 'exact'    # 'exact': plain-form replay with exact modular gadgets (~9n Toffoli per step);
                    # 'auto': for a pseudo-Mersenne p (c = 2^n - p < 2^(n/2), e.g. secp256k1) the v~-form replay with
                    # Häner-style approximate gadgets (~2n + |c| + 2 KAPPA + DELTA per step), else 'exact'


def nsteps(n):
    return 2 * n if STEPS is None else int(STEPS)


SCHEDULE = None     # None: register widths from the worst-case bound max(u, v) < 2^(2n - k) (exact);
                    # (slope, margin): also cap at n + margin - slope k (probabilistic, Schrottenloher / Häner-style:
                    # inputs whose GCD state exceeds it leave garbage; rate from h1/gcd_sizes.py)


def _w(n, k):
    w = min(n, 2 * n - k)
    if SCHEDULE is not None:
        w = min(w, -int(-(n + SCHEDULE[1] - SCHEDULE[0] * k) // 1))
    return max(1, w)


PACK = False        # True: keep the transcript as 5 qubits per 3 steps (the 27 valid (a, m) triples, _pack3),
                    # unpacked around each use (8 Toffoli per pack or unpack); used with the v~-form ('auto') replay only


MERGE = False       # True: replay steps 2..N use vbarith.cinv_add_half_pm / dbl_cinv_add_pm (fold and halving
                    # corrections merged into one small addition)


CMP_BITS = None     # None: exact comparison [v~ < u] in record / unrecord; an int d: compare only the top d bits of
                    # the (w+1)-bit registers (probabilistic: a wrong decision derails the GCD; rate from
                    # h1/gcd_cmp.py, which replicates this record)


def _cmp(V, U, w):
    """the operands of the record's comparison [V < U] on w + 1 bits, truncated to the top CMP_BITS"""
    L = w + 1
    d = L if CMP_BITS is None else min(L, CMP_BITS)
    return V[L - d:L], U[L - d:L]


def _pack3(c, q, inverse=False):
    """q = [a1, m1, a2, m2, a3, m3], each (a_i, m_i) in {00, 10, 11}: in place, the 27 valid states -> m3 = 0
    (inverse=True: back). Codes (t_i = a_i + m_i): t3 < 2: (pair1, pair2, a3 = t3); t3 = 2 and t1 < 2:
    (0, 1, pair2, a1); t1 = t3 = 2, t2 < 2: (1, 0, 0, 1, a2); t1 = t2 = t3 = 2: (1, 1, 0, 1, 0). Every gate group
    below is its own inverse, so the inverse runs them in reverse order."""
    a1, m1, a2, m2, a3, m3 = q

    def swap_if(ctl, neg, x, y):            # if ctl and not neg: swap(x, y)
        c.x(neg)
        g = c.and_(ctl, neg)
        c.cx(y, x)
        c.ccx(g, x, y)
        c.cx(y, x)
        mm = c.measure_x(g)
        c.cz(ctl, neg, cond=mm)
        c.x(neg)

    def flip_if_01(x, y, t):                # if not x and y: t ^= 1
        c.x(x)
        c.ccx(x, y, t)
        c.x(x)

    def last():                             # if m1 and not a2 and m2: m3 ^= 1
        g = c.and_(m1, m2)
        flip_if_01(a2, g, m3)
        mm = c.measure_x(g)
        c.cz(m1, m2, cond=mm)

    ops = [lambda: c.cx(m3, a3),                              # pair 3 -> 00, 10, 01
           lambda: swap_if(m3, m1, a1, a3),                   # t3 = 2, t1 < 2: a3 <- a1, a1 <- 0
           lambda: c.cx(m3, m1),
           lambda: flip_if_01(a1, m1, m3),                    # pair 1 = 01 marks that case: m3 <- 0
           lambda: swap_if(m3, m2, a2, a3),                   # t1 = t3 = 2 (pair 1 now 10), t2 < 2
           lambda: c.cx(m3, m2),
           lambda: flip_if_01(a2, m2, m3),                    # pair 2 = 01 (pair 1 = 10) marks it
           lambda: (c.cx(m3, m1), c.cx(m3, a2), c.cx(m3, m2)),  # t1 = t2 = t3 = 2: (1, 0, 1, 0, 0, 1) -> (1, 1, 0, 1, 0, 1)
           last]
    for op in (reversed(ops) if inverse else ops):
        op()


def _group(tr, k):
    return [tr[k - 2][0], tr[k - 2][1], tr[k - 1][0], tr[k - 1][1], tr[k][0], tr[k][1]]


def _pack(c, tr, k):
    """pack the group of steps k-2, k-1, k (k % 3 == 0); frees m_k"""
    _pack3(c, _group(tr, k))
    c.free([tr[k][1]])
    tr[k] = (tr[k][0], None)


def _unpack(c, tr, k):
    tr[k] = (tr[k][0], c.alloc(1)[0])
    _pack3(c, _group(tr, k), inverse=True)


def record(c, X, p, pack=False):
    """consume X (n qubits, value in [1, p)) -> transcript [(a_j, s_j)] for j = 1..N+1, registers (U, V)."""
    n = len(X)
    N = nsteps(n)
    U = c.alloc(n + 1)                       # u < 2^n (+1 zero qubit)
    for i in range(n):
        if (p >> i) & 1:
            c.x(U[i])
    V = list(X) + c.alloc(2)                 # v~ < 2^(n+1) (+1 headroom)
    a0 = c.alloc(1)[0]
    c.cx(V[0], a0)
    va.cswap(c, a0, U[:n], V[:n])            # plain step 1: s_1 = a_1 (x < p)
    c.x(a0)
    va.add(c, va.ctl_cbits(a0, p, n + 2), V)  # v~ = v + (1 - a) u, u = p when a = 0
    c.x(a0)
    tr = [(a0, a0)]
    w = n
    for k in range(1, N + 1):
        a = c.alloc(1)[0]
        c.cx(V[1], a)
        c.cx(U[1], a)                        # a_{k+1} = v~[1] xor u[1]
        for q in V:
            c.cx(a, q)
        va.add(c, U, V)                      # v~ +- u   (conditionally inverted adder)
        for q in V:
            c.cx(a, q)
        V = V[1:] + V[:1]                    # halve: the zero lsb becomes the headroom bit
        m = c.alloc(1)[0]
        va.lt_do(c, *_cmp(V, U, w), lambda cy: c.ccx(a, cy, m))   # s_{k+1} = a and [v~ < u]
        va.cswap(c, m, U[:w], V[:w])
        tr.append((a, m))
        if pack and k % 3 == 0:
            _pack(c, tr, k)
        w2 = _w(n, k)
        if w2 < w:                           # shrink: the freed qubits are checked to be 0 by V-B
            c.free([U[w], V[w + 1]])
            U, V = U[:w], V[:w + 1]
            w = w2
    return tr, U, V


def unrecord(c, tr, U, V, p, n, pack=False):
    """exact reverse of record; returns X (n qubits)."""
    N = len(tr) - 1
    w = _w(n, N)
    for k in range(N, 0, -1):
        w0 = _w(n, k - 1)
        if w0 > w:
            U = U + c.alloc(1)
            V = V + c.alloc(1)
            w = w0
        if pack and k % 3 == 0:
            _unpack(c, tr, k)
        a, m = tr[k]
        va.cswap(c, m, U[:w], V[:w])
        va.lt_do(c, *_cmp(V, U, w), lambda cy: c.ccx(a, cy, m))
        c.free([m])
        V = V[-1:] + V[:-1]
        for q in V:
            c.cx(a, q)
        va.sub(c, U, V)
        for q in V:
            c.cx(a, q)
        c.cx(V[1], a)
        c.cx(U[1], a)
        c.free([a])
    a0 = tr[0][0]
    c.x(a0)
    va.sub(c, va.ctl_cbits(a0, p, n + 2), V)
    c.x(a0)
    va.cswap(c, a0, U[:n], V[:n])
    c.cx(V[0], a0)
    c.free([a0])
    for i in range(n):
        if (p >> i) & 1:
            c.x(U[i])
    c.free(U + V[n:])
    return V[:n]


def _plain_bits(tr, j):
    """plain step j (1-based): (a_j, s_j)"""
    return tr[j - 1]


def replay_div(c, tr, R, S, p):
    """(R, S) = (0, z) -> (z/x, 0); returns (R, S)."""
    N = len(tr) - 1
    for j in range(1, N + 1):
        a, s = _plain_bits(tr, j)
        va.cswap(c, s, R, S)
        va.modsub(c, R, S, p, ctl=a)
        S = va.modhalf(c, S, p)
    return R, S


def replay_mul(c, tr, R, S, p):
    """(R, S) = (w, 0) -> (0, w x); returns (R, S)."""
    N = len(tr) - 1
    for j in range(N, 0, -1):
        a, s = _plain_bits(tr, j)
        S = va.moddouble(c, S, p)
        va.modadd(c, R, S, p, ctl=a)
        va.cswap(c, s, R, S)
    return R, S


def replay_div_pm(c, tr, R, S, p, pack=False):
    """v~ form, approximate gadgets: (R, S) = (0, z) -> (z/x, z/x)."""
    va.cswap(c, tr[0][0], R, S)
    N = len(tr) - 1
    for k in range(1, N + 1):
        if pack and k % 3 == 1 and k + 2 <= N:
            _unpack(c, tr, k + 2)
        a, m = tr[k]
        if k == 1:
            _cinv_exact(c, a, R, S, p)            # S is exactly 0 here when x is odd: no approximation
            S = va.half_pm(c, S, p)
        elif MERGE:
            S = va.cinv_add_half_pm(c, a, R, S, p)    # (S + (-1)^a R) / 2 with one small correction
        else:
            va.cinv_modadd_pm(c, a, R, S, p)      # S + (-1)^a R
            S = va.half_pm(c, S, p)
        va.cswap(c, m, R, S)
        if pack and k % 3 == 0:
            _pack(c, tr, k)
    return R, S


def replay_mul_pm(c, tr, R, S, p, pack=False):
    """inverse of replay_div_pm: (R, S) = (w, w) -> (0, w x)."""
    N = len(tr) - 1
    for k in range(N, 0, -1):
        if pack and k % 3 == 0:
            _unpack(c, tr, k)
        a, m = tr[k]
        va.cswap(c, m, R, S)
        if k > 1 and MERGE:
            S = va.dbl_cinv_add_pm(c, a, R, S, p)     # 2 S - (-1)^a R with one small correction
        else:
            S = va.double_pm(c, S, p)
            c.x(a)
            if k == 1:
                _cinv_exact(c, a, R, S, p)        # result exactly 0 when x is odd
            else:
                va.cinv_modadd_pm(c, a, R, S, p)  # S - (-1)^a R
            c.x(a)
        if pack and k % 3 == 1 and k + 2 <= N:
            _pack(c, tr, k + 2)
    va.cswap(c, tr[0][0], R, S)
    return R, S


def _cinv_exact(c, a, R, S, p):
    """S <- S + (-1)^a R mod p with the exact gadgets (12n Toffoli)"""
    c.x(a)
    va.modadd(c, R, S, p, ctl=a)
    c.x(a)
    va.modsub(c, R, S, p, ctl=a)


def _zero_guard(c, X):
    """z = [X == 0]; if z: X = 1. Returns z."""
    z = c.alloc(1)[0]
    va.allzero_do(c, X, lambda q: c.cx(q, z))
    c.cx(z, X[0])
    return z


def _zero_unguard(c, X, z):
    c.cx(z, X[0])
    va.allzero_do(c, X, lambda q: c.cx(q, z))
    c.free([z])


def inplace(c, X, Z, p, divide):
    """Z <- Z / X (divide) or Z * X mod p; X unchanged (X = 0 acts as 1). Returns the new Z register."""
    n = len(X)
    z = _zero_guard(c, X)
    pm = REPLAY == 'auto' and va.pm_c(p, n) is not None
    pack = PACK and pm
    if pm:
        # force x odd: if x is even use (-x, -Z) (same quotient / product). Then the record's first step always
        # swaps, so the replay's only exact zero (S = 0 right after the first swap) sits at step 1, which uses
        # the exact gadget; every later replay value is Z * (GCD state) / x mod p (uniform for uniform Z).
        e = c.alloc(1)[0]
        c.cx(X[0], e)
        c.x(e)
        va.negate_nonzero(c, X, p, e)
        va.modneg(c, Z, p, ctl=e)
    tr, U, V = record(c, X, p, pack)
    c.x(U[0]); c.x(V[0])                     # (1, 1) -> (0, 0)
    wU, wV = len(U), len(V)
    c.free(U + V)
    other = c.alloc(n)
    if pm and divide:
        R, S = replay_div_pm(c, tr, other, Z, p, pack)
        for r, s in zip(R, S):
            c.cx(r, s)                            # (z/x, z/x) -> (z/x, 0)
        out, zero = R, S
    elif pm:
        for r, s in zip(Z, other):
            c.cx(r, s)                            # (w, 0) -> (w, w)
        R, S = replay_mul_pm(c, tr, Z, other, p, pack)
        out, zero = S, R
    elif divide:
        R, S = replay_div(c, tr, other, Z, p)
        out, zero = R, S
    else:
        R, S = replay_mul(c, tr, Z, other, p)
        out, zero = S, R
    c.free(zero)
    U, V = c.alloc(wU), c.alloc(wV)
    c.x(U[0]); c.x(V[0])
    X2 = unrecord(c, tr, U, V, p, n, pack)
    X[:] = X2
    if pm:
        va.negate_nonzero(c, X, p, e)
        c.x(e)
        c.cx(X[0], e)
        c.free([e])
    _zero_unguard(c, X, z)
    return out
