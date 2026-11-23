"""Windowed affine point addition, for V-B KIND 'windowed_point_add' (SIGNED = False, table [k]P, k in
[0, 2^W)) or 'signed_windowed_point_add' (SIGNED = True: W exponent bits, odd digit d = 2k - 2^W + 1, table of
2^(W-1) points [2j+1]P plus a conditional negation of y). Structure after Häner et al. (arXiv 2609.05625,
Alg. 6) and Schrottenloher (arXiv 2606.02235): three table lookups (unary iteration, measurement-based unlookup:
code adapted from s2/vb/qrom.py), dx = x - x2, dy = y - y2, lambda = dy / dx by an in-place division, x += 3 x2,
x -= lambda^2 (= x2 - x3), y = lambda (x2 - x3) (= y3 + y2) by an in-place multiplication, then y -= y2,

Exactness: every gadget is exact on canonical inputs. Unsigned k = 0 (T[0] = O) returns the accumulator
(the square-subtract and the final negation are controlled on k != 0; dx = 0 acts as 1 in the GCD).
Wrong output, but no garbage and no phase, only on: accumulator = +-T (doubling / result O) and accumulator = O
with a nonzero addend. A Shor / Ekerå run feeds these with probability ~ 2^(W+1) / r per addition when the
first window loads a random offset point (Häner et al.'s random initial point; signed odd digits are never O);
hence EXACT = False (the harness then counts these exceptional failures instead of failing the circuit).
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vbarith as va
import gcdmul
import pingpong

KIND = 'windowed_point_add'
SIGNED = False
W = 2
EXACT = False
STEPS = None        # GCD halvings per record (None = 2n, exact); see gcdmul.STEPS
REPLAY = 'exact'    # 'auto': approximate pseudo-Mersenne replay when p allows it; see gcdmul.REPLAY
SCHEDULE = None     # (slope, margin) probabilistic GCD register widths; see gcdmul.SCHEDULE
KAPPA = 64          # approximate gadgets: carry-chain cut and flag-comparison width (vbarith)
DELTA = 64
PACK = False        # transcript packing, 5 qubits per 3 GCD steps (gcdmul.PACK; 'auto' replay only)
MERGE = False       # merged replay steps (gcdmul.MERGE; 'auto' replay)
CMP_BITS = None     # GCD record comparison on the top CMP_BITS bits only (gcdmul.CMP_BITS; probabilistic)
SQ_ANDS = None      # max AND terms materialized per squaring row (None = all); the rest are recomputed inside
                    # the adder (+1 Toffoli each), capping the square-subtract's qubit peak
SQUARE = 'schoolbook'   # 'karatsuba': square_sub_kara (pseudo-Mersenne p, signed windows / no control only)
WALK = 'binary'     # 'pingpong': in-place division / multiplication by the comparison-free ping-pong walk
                    # (pingpong.py; needs REPLAY = 'auto' and a pseudo-Mersenne p); 'binary': gcdmul.py
ROUNDS = None       # ping-pong walk rounds (pingpong.ROUNDS)
PP_SCHED = None     # ping-pong width schedule (slope, a, margin) (pingpong.SCHED)
CHUNK = None        # replay-cell adder chunk size (vbarith.CHUNK) and inter-chunk carry comparison width
CHUNK_CMP = 28
CUT_FLAGS = False   # v12: every truncated carry chain flags its carry out (vbarith.CUT_FLAGS): exact or flagged
OFFSET = False      # first_lookup(): the run's first window loads T0[k] + R0 for a random point R0 (a full unsigned


def window(n):
    return W


# Declared failure per addition of the approximations (V-B's 6,144 sampled lanes cannot see rates this small).
# Statistics for secp256k1, n = 256, from h1/results/gcd_sizes_200k.txt (200,000 random x):
_STEP_MEAN, _STEP_SD = 361.83, 9.52     # GCD halvings until v = 0; sampled tail above 400: 4.0e-5 (1.7x Gaussian)
_SCHED_20, _SCHED_DECAY = 4.5e-5, 29.0  # slope 0.68: fraction of x whose GCD sizes exceed the margin-20 line;
                                        # 1.3e-3 at margin 15 -> geometric decay 29x per 5 bits (extrapolated)


# ping-pong walk (h1/pingpong_stats.py, 100,000 random x, results/pingpong_stats_100k.json): rounds to converge
# mean 621.4, sd 21.7; P(> L) = 1.28e-3, 3.4e-4, 1.8e-4, 8e-5, 1e-5 at L = 690, 700, 704, 710, 720 (heavier than
# Gaussian). Declared: 1.8e-4 exp(0.14 (704 - L)) below 704 and 1.8e-4 exp(-0.118 (L - 704)) above (>= sampled).
# Width schedule (0.41, 1.5, m): 10 / 2 of 100,000 walks exceed it at m = 1.9 / 4.5; declared 3e-5 for m >= 5.
def _pp_failure(n):
    L = pingpong.rounds(n)
    f_conv = 1.8e-4 * math.exp(0.14 * (704 - L) if L < 704 else -0.118 * (L - 704))
    if PP_SCHED is None:
        f_w = 0.0
    elif tuple(PP_SCHED[:2]) == (0.41, 1.5) and PP_SCHED[2] >= 5:
        f_w = 3e-5
    else:
        raise SystemExit('declared_failure: width statistics exist for PP_SCHED = (0.41, 1.5, >= 5) only')
    k = 1 + (0 if CHUNK is None else -(-n // CHUNK) - 1)       # approximate ops per replay cell besides KAPPA
    d = min(KAPPA, DELTA, CHUNK_CMP if CHUNK is not None else KAPPA)
    f_gadget = (2 * L * (1 + k) + 200) * 2.0 ** -d
    return 2 * (2 * (f_conv + f_w) + f_gadget)


def declared_failure(n):
    """sum over the two GCD records (step-count tail x3 the Gaussian, width-schedule tail, truncated comparison)
    and the approximate gadgets (3 per replay step, 2 with MERGE but still counted as 3, and ~70 in the
    square-subtract, each failing with ~2^-min(KAPPA, DELTA) on uniform values), times 2 as margin.
    Only n = 256 (secp256k1) is declared; other sizes return 0."""
    pm = REPLAY == 'auto' and va.pm_c(_P256, 256) is not None
    if n == 256 and WALK == 'pingpong':
        return _pp_failure(n)
    if n != 256 or not (STEPS is not None or SCHEDULE is not None or pm):
        return 0.0
    N = 2 * n if STEPS is None else int(STEPS)
    f_steps = 3 * 0.5 * math.erfc((N + 0.5 - _STEP_MEAN) / _STEP_SD / math.sqrt(2))
    f_sched = 0.0
    if SCHEDULE is not None:
        slope, margin = SCHEDULE
        if slope > 0.68:
            raise SystemExit('declared_failure: width-schedule statistics exist for slope <= 0.68 only')
        f_sched = _SCHED_20 * _SCHED_DECAY ** (-(margin - 20) / 5.0)
    f_gadget = (3 * 2 * N + 70) * 2.0 ** -min(KAPPA, DELTA) if pm else 0.0
    f_cmp = 0.0
    if CMP_BITS is not None:
        # truncated record comparison (h1/results/gcd_cmp_200k.txt, schedule (0.68, 30)): the largest gap G
        # between the register top and the larger GCD value has P(G >= 65) = 1.5e-5, decaying >= 13x per 5 bits;
        # failures come from gaps within ~4 bits of CMP_BITS (at CMP_BITS = 64: 9.5e-4 sampled vs 10 P(G >= 60)
        # = 7.5e-3 modelled), so f_cmp = 10 P(G >= CMP_BITS - 4) with a margin shift for wider schedules
        if SCHEDULE is None:
            raise SystemExit('declared_failure: CMP_BITS needs a width schedule (gap statistics exist for one)')
        g = CMP_BITS - 4 - max(0.0, SCHEDULE[1] - 30)
        f_cmp = 10 * 1.5e-5 * 13.0 ** (-(g - 65) / 5.0) if g > 65 else 1.0
    return 2 * (2 * (f_steps + f_sched + f_cmp) + f_gadget)


_P256 = (1 << 256) - (1 << 32) - 977


# ---------------------------------------------------------------- QROM (adapted from s2/vb/qrom.py)
def _iterate(c, ctl, bits, cb, idx=0):
    if not bits:
        cb(ctl, idx)
        return
    a, rest = bits[0], bits[1:]
    sh = len(rest)
    if ctl is None:
        _iterate(c, a, rest, cb, idx | (1 << sh))
        c.x(a)
        _iterate(c, a, rest, cb, idx)
        c.x(a)
    else:
        t = c.and_(ctl, a)
        _iterate(c, t, rest, cb, idx | (1 << sh))
        c.cx(ctl, t)
        _iterate(c, t, rest, cb, idx)
        m = c.measure_x(t)
        c.x(a)
        c.cz(ctl, a, cond=m)
        c.x(a)


def qload(c, k, out, T):
    """out ^= T[k] (unary iteration, 2^W - 2 Toffoli)"""
    bits = [[b for b in range(len(out)) if (v >> b) & 1] for v in T]

    def leaf(f, i):
        for b in bits[i]:
            c.cx(f, out[b])
    _iterate(c, None, k[::-1], leaf)


def qunload(c, k, out, T):
    """measurement-based unlookup of out = T[k]: X-measure the payload, then fix (-1)^(m . T[k])"""
    Wk = len(k)
    ms = [c.measure_x(q) for q in out]
    l = (Wk + 1) // 2
    h = Wk - l
    lo, hi = k[:l][::-1], k[l:][::-1]
    onehot = c.alloc(1 << l)
    _iterate(c, None, lo, lambda f, j: c.cx(f, onehot[j]))

    def g_of(v):
        g = 0
        b = 0
        while v:
            if v & 1:
                g ^= ms[b]
            v >>= 1
            b += 1
        return g

    def leaf(f, ih):
        for j in range(1 << l):
            g = g_of(T[(ih << l) | j])
            if g:
                if f is None:
                    c.z(onehot[j], cond=g)
                else:
                    c.cz(f, onehot[j], cond=g)
    if h:
        _iterate(c, None, hi, leaf)
    else:
        leaf(None, 0)
    _iterate(c, None, lo, lambda f, j: c.cx(f, onehot[j]))
    c.free(onehot)


# ---------------------------------------------------------------- x <- x - lambda^2 mod p
def _square(c, L, Z, sign, sq_ands=-1):
    """Z <- Z + sign * L^2 (integer, len(Z) = 2n); triangular schoolbook, ~n^2 Toffoli"""
    n = len(L)
    sq = SQ_ANDS if sq_ands == -1 else sq_ands
    order = range(n) if sign > 0 else reversed(range(n))
    for i in order:
        cap = n - i - 1 if sq is None else min(n - i - 1, sq)
        T = [c.and_(L[i], L[i + j]) for j in range(1, cap + 1)]
        T2 = [('and', L[i], L[i + j]) for j in range(cap + 1, n - i)]
        hi = min(2 * n, n + i + 2)
        (va.add if sign > 0 else va.sub)(c, [L[i], va.K0] + T + T2, Z[2 * i:hi])
        for j, t in zip(range(1, cap + 1), T):
            m = c.measure_x(t)
            c.cz(L[i], L[i + j], cond=m)


def square_sub(c, L, X, p, ctl=None):
    """X <- X - ctl * L^2 mod p (L, X canonical n-bit)."""
    n = len(L)
    Z = c.alloc(2 * n)
    _square(c, L, Z, +1)
    ZL, W_ = Z[:n], Z[n:]                       # L^2 = ZH 2^n + ZL, ZH < p
    z0 = c.alloc(1)
    va.modsub(c, ZL[:n - 1] + z0, X, p, ctl=ctl)      # low n-1 bits (< p)
    c.free(z0)
    if ctl is None:
        va.modsub(c, 1 << (n - 1), X, p, ctl=ZL[n - 1])
    else:
        g = c.and_(ctl, ZL[n - 1])
        va.modsub(c, 1 << (n - 1), X, p, ctl=g)
        m = c.measure_x(g)
        c.cz(ctl, ZL[n - 1], cond=m)
    C = (1 << n) - p                            # 2^n mod p
    top = C.bit_length() - 1
    pm = gcdmul.REPLAY == 'auto' and va.pm_c(p, n) is not None
    dbl, hlf = (va.double_pm, va.half_pm) if pm else (va.moddouble, va.modhalf)
    for b in range(top + 1):
        if (C >> b) & 1:
            va.modsub(c, W_, X, p, ctl=ctl)
        if b < top:
            W_ = dbl(c, W_, p)
    for b in range(top):
        W_ = hlf(c, W_, p)
    _square(c, L, ZL + W_, -1)
    c.free(ZL + W_)


def _modadd(c, V, X, p, sub):
    """X <- X +- V mod p (approximate pseudo-Mersenne adder; V: operand list, constants allowed, value < p)"""
    if sub:
        for q in X:
            c.x(q)
    va.cinv_modadd_pm(c, None, V, X, p)
    if sub:
        for q in X:
            c.x(q)


def _times_c(c, V, cc, sign):
    """T += sign * cc * V for a fresh/used integer register T returned (cc = 2^32 + 977 style constant):
    copy V << top, then add V << b for the other set bits b of cc."""
    bits = [b for b in range(cc.bit_length()) if (cc >> b) & 1]
    top = bits[-1]
    T = c.alloc(len(V) + cc.bit_length())
    for i, q in enumerate(V):
        c.cx(q, T[top + i])
    for b in bits[:-1]:
        va.add(c, [va.K0] * b + list(V), T)
    return T, bits


def _untimes_c(c, V, T, bits):
    for b in reversed(bits[:-1]):
        va.sub(c, [va.K0] * b + list(V), T)
    for i, q in enumerate(V):
        c.cx(q, T[bits[-1] + i])
    c.free(T)


def _apply_beta(c, V, X, p, h, sub):
    """X <- X +- 2^h V mod p for V < 2^(n + 2): 2^h V = (V mod 2^(n-h)) 2^h + 2^n (V >> (n-h)),
    2^n = cc mod p, and cc * (V >> (n-h)) < p is formed as an integer."""
    n = len(X)
    cc = (1 << n) - p
    lo, hi = V[:n - h], V[n - h:]
    _modadd(c, [va.K0] * h + list(lo), X, p, sub)
    T, bits = _times_c(c, hi, cc, +1)
    _modadd(c, T, X, p, sub)
    _untimes_c(c, hi, T, bits)


def square_sub_kara(c, Lam, X, p):
    """X <- X - Lam^2 mod p by Karatsuba with direct accumulation (after ECDSA.fail 3616dbf, arXiv 2609.09582
    Sect. 5.3.4): Lam = L + 2^h H, A = L^2, B = H^2, C = (L + H)^2, and with 2^(2h) = 2^n = cc (mod p):
    Lam^2 = (1 - 2^h) A + 2^h C + (2^n - 2^h) B. Each of A, B, C is built in a clean register, applied to X and
    unbuilt; no 2n-bit product register. Pseudo-Mersenne p only (approximate modular adders)."""
    n = len(Lam)
    h = n // 2
    cc = (1 << n) - p
    L, H = Lam[:h], Lam[h:]
    A = c.alloc(2 * h)
    _square(c, L, A, +1, None)
    _modadd(c, A, X, p, True)                    # X -= A
    _apply_beta(c, A, X, p, h, False)            # X += 2^h A
    _square(c, L, A, -1, None)
    c.free(A)
    B = c.alloc(2 * (n - h))
    _square(c, H, B, +1, None)
    _apply_beta(c, B, X, p, h, False)            # X += 2^h B
    top = cc.bit_length() - 1                    # X -= cc B: B doubled in place, then halved back
    for b in range(top + 1):
        if (cc >> b) & 1:
            _modadd(c, B, X, p, True)
        if b < top:
            B = va.double_pm(c, B, p)
    for b in range(top):
        B = va.half_pm(c, B, p)
    _square(c, H, B, -1, None)
    c.free(B)

    def s_make():
        S = c.alloc(h + 1)
        for a, q in zip(L, S):
            c.cx(a, q)
        va.add(c, H, S)
        return S

    def s_undo(S):
        va.sub(c, H, S)
        for a, q in zip(L, S):
            c.cx(a, q)
        c.free(S)
    S = s_make()
    C = c.alloc(2 * (h + 1))
    _square(c, S, C, +1, None)
    s_undo(S)
    _apply_beta(c, C, X, p, h, True)             # X -= 2^h C
    S = s_make()
    _square(c, S, C, -1, None)
    s_undo(S)
    c.free(C)


def first_lookup(n):
    """Counts of a run's initial lookup (accumulator <- T[k] for the first window; signed digits: index
    transform, unary-iteration load of 2^(W-1) entries, conditional negation of y), built from the same gadgets
    as the addition (qload, negate_nonzero) and counted without simulation (counts do not depend on the table
    contents). The harness charges it once per run (cost.py)."""
    import count as _cnt
    c = _cnt.Counter()
    k = c.alloc(W)
    D = c.alloc(2 * n)
    p = _P256 if n == 256 else (1 << n) - 1
    m = (1 << (2 * n)) - 1
    if OFFSET:                              # random offset: every 2^W window value has its own entry
        pay = [(i * 0x9E3779B97F4A7C15F39CC0605CEDC834 + 1) & m for i in range(1 << W)]
        qload(c, k, D, pay)
        return dict(toffoli=c.toffoli, peak_qubits=c.peak)
    pay = [(i * 0x9E3779B97F4A7C15F39CC0605CEDC834 + 1) & m for i in range(1 << (W - 1 if SIGNED else W))]
    if SIGNED:
        s, idx = k[-1], k[:-1]
        for q in idx:
            c.cx(s, q)
            c.x(q)
        qload(c, idx, D, pay)
        c.x(s)
        va.negate_nonzero(c, D[n:], p, s)
        c.x(s)
        for q in idx:
            c.x(q)
            c.cx(s, q)
    else:
        qload(c, k, D, pay)
    return dict(toffoli=c.toffoli, peak_qubits=c.peak)


# ---------------------------------------------------------------- the point addition
def build(c, n, regs, ctx):
    p, T = ctx['p'], ctx['table']
    gcdmul.STEPS, gcdmul.REPLAY, gcdmul.SCHEDULE, gcdmul.PACK = STEPS, REPLAY, SCHEDULE, PACK
    gcdmul.CMP_BITS, gcdmul.MERGE = CMP_BITS, MERGE
    va.KAPPA, va.DELTA = KAPPA, DELTA
    va.CUT_FLAGS = CUT_FLAGS
    va.CHUNK, va.CHUNK_CMP = CHUNK, CHUNK_CMP
    pingpong.ROUNDS, pingpong.SCHED = ROUNDS, PP_SCHED
    inplace = pingpong.inplace if WALK == 'pingpong' else gcdmul.inplace
    X, Y, k = list(regs['x']), regs['y'], regs['k']
    enc = [(0, 0) if P is None else P for P in T]
    pay = [P[0] | (P[1] << n) for P in enc]
    t3 = [3 * P[0] % p for P in enc]
    if SIGNED:
        s, idx = k[-1], k[:-1]
        for q in idx:
            c.cx(s, q)
            c.x(q)                              # j = low bits, complemented when s = 0
        nf = None
    else:
        idx = k
        nf = c.alloc(1)[0]
        va.allzero_do(c, k, lambda z: c.cx(z, nf))
        c.x(nf)                                 # nf = [k != 0]

    def lookup_xy(fn):
        D = c.alloc(2 * n)
        qload(c, idx, D, pay)
        if SIGNED:
            c.x(s)
            va.negate_nonzero(c, D[n:], p, s)   # y2 -> -y2 for a negative digit
            c.x(s)
        fn(D[:n], D[n:])
        if SIGNED:
            c.x(s)
            va.negate_nonzero(c, D[n:], p, s)
            c.x(s)
        qunload(c, idx, D, pay)

    def first(dx, dy):
        va.modsub(c, dx, X, p)
        va.modsub(c, dy, Y, p)
    lookup_xy(first)
    Y = inplace(c, X, Y, p, True)        # lambda = dy / dx
    D = c.alloc(n)
    qload(c, idx, D, t3)
    va.modadd(c, D, X, p)                       # x1 + 2 x2
    qunload(c, idx, D, t3)
    if SQUARE == 'karatsuba' and nf is None:
        square_sub_kara(c, Y, X, p)             # x2 - x3
    else:
        square_sub(c, Y, X, p, ctl=nf)          # x2 - x3
    Y = inplace(c, X, Y, p, False)       # y3 + y2

    def last(dx, dy):
        va.modsub(c, dy, Y, p)
        va.modsub(c, dx, X, p)
    lookup_xy(last)
    va.modneg(c, X, p, ctl=nf)
    if SIGNED:
        for q in idx:
            c.x(q)
            c.cx(s, q)
    else:
        c.x(nf)
        va.allzero_do(c, k, lambda z: c.cx(z, nf))
        c.free([nf])
    return {'x': X, 'y': Y}
