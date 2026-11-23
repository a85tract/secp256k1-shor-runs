"""In-place modular division / multiplication by a quantum x via the comparison-free ping-pong walk.

Walk (ECDSA.fail submission 3616dbf, arXiv 2609.09582 Sect. 5.3.3; written here from the paper's description):
  rho0 = p, rho1 = x* (x if x odd, x - p if x even), both odd; round i: (s, t) = (rho0, rho1) if i even else
  (rho1, rho0); e_i = t[1] xor s[1]; t <- (t + (-1)^e_i s) / 2 (stays odd).  After L rounds both are +-1 for all
  but a rare tail of x (h1/pingpong_stats.py). One transcript qubit per round; no comparison and no swap.
  Registers are two's complement with a shrinking width w_i (|rho| < 2^(w_i - 2)), one Gidney adder per round.
Replay (the same linear rounds mod p): (0, y) -> (eta0 y/x, eta1 y/x) [division]; its inverse maps
  (eta0 y, eta1 y) -> (0, y x) [multiplication]; eta = the walk's final signs.
Unrecord: the exact reverse of the walk (restores x, erases the transcript).
Replay cells: vbarith.cinv_add_half_pm / dbl_cinv_add_pm (approximate pseudo-Mersenne gadgets), whose n-bit add can
be split into chunks (vbarith.CHUNK) with approximate inter-chunk carry clearing, as in 3616dbf's replay.
"""
import math
import vbarith as va

ROUNDS = None       # walk rounds L; None: 2.43 n + 8.2 sqrt(n) (generous: mean 2.43 n, sd ~1.36 sqrt(n))
SCHED = None        # None: no shrinking (width n + 2); (slope, a, margin): width after k rounds
                    # min(n + 2, max(3, ceil(n - slope k + a sqrt(k) + margin) + 2)) (probabilistic)


def rounds(n):
    return int(ROUNDS) if ROUNDS is not None else int(math.ceil(2.43 * n + 8.2 * math.sqrt(n)))


def width(n, k):
    """register width during round k (0-based): values before round k satisfy |rho| < 2^(width - 2)"""
    if SCHED is None or k == 0:
        return n + 2
    s, a, m = SCHED
    return int(min(n + 2, max(3, math.ceil(n - s * k + a * math.sqrt(k) + m) + 2)))


def _shrink(c, R, w):
    """cut R to width w when the schedule drops. The next add needs |rho| < 2^(w - 2), i.e. the value fits in
    w - 1 bits, so the top bits are cut down to w - 1 with a flag check each (top bit = the sign below it) and one
    sign bit is grown back. (Checking only down to w bits, as before, let a value in [2^(w-2), 2^(w-1)) through: the
    next add then wrapped mod 2^w undetected, and a walk that still ended at +-1 left garbage in the replay.)"""
    if len(R) <= w:
        return R                    # width unchanged: |T'| <= (|T| + |S|) / 2 keeps the bound by itself
    while len(R) > w - 1:
        c.cx(R[-2], R[-1])
        va.drop(c, [R[-1]])         # flag measurement: a value above the width schedule is a detected failure
        R = R[:-1]
    return _grow(c, R, w)


def _grow(c, R, w):
    while len(R) < w:
        q = c.alloc(1)[0]
        c.cx(R[-1], q)
        R = R + [q]
    return R


def record(c, X, p):
    """consume X (n qubits, x in [1, p)) -> (tape of L qubits, sign qubits (sU, sV))."""
    n = len(X)
    L = rounds(n)
    w = width(n, 0)
    U = c.alloc(w)
    for i in range(n):
        if (p >> i) & 1:
            c.x(U[i])
    V = list(X) + c.alloc(w - n)
    g = c.alloc(1)[0]                        # odd lift: x* = x - p [x even]
    c.cx(V[0], g)
    c.x(g)
    va.sub(c, va.ctl_cbits(g, p, w), V)
    c.cx(V[-1], g)                           # g = [x* < 0]
    c.free([g])
    tape = []
    for k in range(L):
        S, T = (U, V) if k % 2 == 0 else (V, U)
        e = c.alloc(1)[0]
        c.cx(T[1], e)
        c.cx(S[1], e)
        for q in T:
            c.cx(e, q)
        va.add(c, S, T)                      # T + (-1)^e S (one's complement around the add)
        for q in T:
            c.cx(e, q)
        c.cx(T[-1], T[0])                    # halve (arithmetic shift): the zero lsb becomes the sign copy
        T = T[1:] + T[:1]
        if k % 2 == 0:
            V = T
        else:
            U = T
        tape.append(e)
        w2 = width(n, k + 1)
        U, V = _shrink(c, U, w2), _shrink(c, V, w2)
    signs = []
    for R in (U, V):                         # +-1 -> sign qubit
        for q in R[1:-1]:
            c.cx(R[-1], q)
        c.x(R[0])
        va.drop(c, R[:-1])                   # flag measurement: a walk not converged in L rounds is detected
        signs.append(R[-1])
    return tape, signs


def unrecord(c, tape, signs, p, n):
    """exact reverse of record; returns X (n qubits)."""
    L = len(tape)
    w = width(n, L)
    regs = []
    for s in signs:
        R = c.alloc(w - 1) + [s]
        c.x(R[0])
        for q in R[1:-1]:
            c.cx(s, q)
        regs.append(R)
    U, V = regs
    for k in range(L - 1, -1, -1):
        w0 = width(n, k)
        U, V = _grow(c, U, w0), _grow(c, V, w0)
        e = tape[k]
        if k % 2 == 0:
            S, T = U, V
        else:
            S, T = V, U
        c.cx(T[-2], T[-1])
        T = T[-1:] + T[:-1]
        for q in T:
            c.cx(e, q)
        va.sub(c, S, T)
        for q in T:
            c.cx(e, q)
        c.cx(T[1], e)
        c.cx(S[1], e)
        c.free([e])
        if k % 2 == 0:
            V = T
        else:
            U = T
    w = width(n, 0)
    g = c.alloc(1)[0]
    c.cx(V[-1], g)
    va.add(c, va.ctl_cbits(g, p, w), V)
    c.x(V[0])
    c.cx(V[0], g)
    c.x(V[0])
    c.free([g])
    for i in range(n):
        if (p >> i) & 1:
            c.x(U[i])
    c.free(U + V[n:])
    return V[:n]


ZBLOCK = 16         # zero tests: AND chains over blocks of ZBLOCK bits (<= n / ZBLOCK + ZBLOCK ancillas)


def _allzero_do(c, Q, act):
    """act([Q == 0]) with few ancillas: r_j = [block j == 0] (chain per block, uncomputed by measurement), then
    one chain over the r_j; the r_j are erased by recomputing them (~2 len(Q) Toffoli instead of len(Q))."""
    if len(Q) <= 2 * ZBLOCK:
        return va.allzero_do(c, Q, act)
    blocks = [Q[i:i + ZBLOCK] for i in range(0, len(Q), ZBLOCK)]
    rs = c.alloc(len(blocks))
    for r, b in zip(rs, blocks):
        va.allzero_do(c, b, lambda z, r=r: c.cx(z, r))
    for r in rs:
        c.x(r)
    out = va.allzero_do(c, rs, act)          # all r_j = 1  <=>  all complemented r_j = 0
    for r in rs:
        c.x(r)
    for r, b in zip(rs, blocks):
        va.allzero_do(c, b, lambda z, r=r: c.cx(z, r))
    c.free(rs)
    return out


def _cneg(c, Z, p, ctl):
    """Z <- -Z mod p if ctl (zero stays zero): g = ctl and [Z != 0], complement, subtract c - 1 on the low
    |c| + KAPPA bits (approximate pseudo-Mersenne chain), uncompute g (~4n + |c| + KAPPA Toffoli)"""
    n = len(Z)
    cc = (1 << n) - p
    g = c.alloc(1)[0]
    _allzero_do(c, Z, lambda z: (c.x(z), c.ccx(ctl, z, g), c.x(z)))
    for q in Z:
        c.cx(g, q)
    L = va._low(n, cc)
    (va._cut_sub if L < n else va.sub)(c, va.ctl_cbits(g, cc - 1, L), Z[:L])
    _allzero_do(c, Z, lambda z: (c.x(z), c.ccx(ctl, z, g), c.x(z)))
    c.free([g])


def replay_div(c, tape, Z1, p):
    """(0, Z1 = y) -> (eta0 y/x, eta1 y/x); returns (Z0, Z1)."""
    n = len(Z1)
    L = len(tape)
    Z1 = va.half_pm(c, Z1, p)                # round 0: Z0 = 0, so Z1 <- Z1 / 2 whatever e_0
    Z0 = c.alloc(n)                          # round 1: Z0 <- (0 + (-1)^e1 Z1) / 2
    for a, b in zip(Z1, Z0):
        c.cx(a, b)
    _cneg(c, Z0, p, tape[1])
    Z0 = va.half_pm(c, Z0, p)
    for k in range(2, L):
        if k % 2 == 0:
            Z1 = va.cinv_add_half_pm(c, tape[k], Z0, Z1, p)
        else:
            Z0 = va.cinv_add_half_pm(c, tape[k], Z1, Z0, p)
    return Z0, Z1


def replay_mul(c, tape, Z0, Z1, p):
    """inverse of replay_div: (eta0 w, eta1 w) -> (0, w x); returns Z1 (Z0 is freed)."""
    L = len(tape)
    for k in range(L - 1, 1, -1):
        if k % 2 == 0:
            Z1 = va.dbl_cinv_add_pm(c, tape[k], Z0, Z1, p)
        else:
            Z0 = va.dbl_cinv_add_pm(c, tape[k], Z1, Z0, p)
    Z0 = va.double_pm(c, Z0, p)
    _cneg(c, Z0, p, tape[1])
    for a, b in zip(Z1, Z0):
        c.cx(a, b)
    va.drop(c, Z0)                           # Z0 = Z1 is a checksum of the replay: flag-measure, do not free
    return va.double_pm(c, Z1, p)


# x = 0 is replaced by this constant (sha256('h1 pingpong zero guard 0') mod p): structured x such as 1 make
# atypical walks that leave the width schedule (x = 1 exceeds it at round 136; x = 3 needs > 704 rounds), while
# this one converges in 559 rounds inside it. Only exceptional inputs (dx = 0) reach it; their output is wrong
# anyway, but now without garbage.
X0 = 0x493c3029e089f7945255cdaafd5f6bfd37e5d3f3be136c78c3897f82c8ef4bce


def _guard(c, X):
    z = c.alloc(1)[0]
    va.allzero_do(c, X, lambda q: c.cx(q, z))
    for i in range(len(X)):
        if (X0 >> i) & 1:
            c.cx(z, X[i])
    return z


def _unguard(c, X, z):
    for i in range(len(X)):
        if (X0 >> i) & 1:
            c.cx(z, X[i])
    va.allzero_do(c, X, lambda q: c.cx(q, z))
    c.free([z])


def inplace(c, X, Z, p, divide):
    """Z <- Z / X (divide) or Z * X mod p; X unchanged (X = 0 acts as 1). Returns the new Z register.
    Needs a pseudo-Mersenne p (vbarith.pm_c)."""
    import gcdmul
    n = len(X)
    assert va.pm_c(p, n) is not None
    z = _guard(c, X) if n == 256 else gcdmul._zero_guard(c, X)
    tape, (sU, sV) = record(c, X, p)
    if divide:
        Z0, Z1 = replay_div(c, tape, Z, p)
        _cneg(c, Z0, p, sU)
        _cneg(c, Z1, p, sV)
        for a, b in zip(Z1, Z0):
            c.cx(a, b)                       # Z0 = Z1 = y/x -> Z0 = 0
        va.drop(c, Z0)                       # checksum: a silent replay error (a KAPPA-truncated carry chain)
                                             # makes Z0 != Z1; flag-measure it instead of freeing (detected)
        out = Z1
    else:
        Z0 = c.alloc(n)
        for a, b in zip(Z, Z0):
            c.cx(a, b)
        _cneg(c, Z0, p, sU)
        _cneg(c, Z, p, sV)
        out = replay_mul(c, tape, Z0, Z, p)
    X[:] = unrecord(c, tape, [sU, sV], p, n)
    if n == 256:
        _unguard(c, X, z)
    else:
        gcdmul._zero_unguard(c, X, z)
    return out
