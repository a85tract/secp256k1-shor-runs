"""Exact reversible arithmetic gadgets for the V-B builder (harness/vb.py `Circuit`), h1.

Registers are little-endian lists of qubit ids. Operand lists (the `X` of an adder) may mix
  qubits, the constants K0 / K1, and ('and', g, q) = the AND of two qubits, recomputed on the fly
  (3 Toffoli per bit, no extra register) for controlled additions of a register.
Carries follow Gidney (arXiv 1709.06648): one AND per bit, uncomputed by X measurement + CZ fix-up.
All modular gadgets assume canonical inputs in [0, p) and return canonical outputs; p odd, n = p.bit_length().
"""
K0, K1 = ('k', 0), ('k', 1)


def isconst(x):
    return x is K0 or x is K1


def drop(c, qs):
    """release qubits that are 0 on every input the approximate gadget handles: a flag measurement when the
    builder has one (V-B: an outcome 1 marks the run as a detected, declared failure), else a checked free"""
    (c.flag if hasattr(c, 'flag') else c.free)(qs)


def cbits(v, L):
    return [K1 if (v >> i) & 1 else K0 for i in range(L)]


def ctl_cbits(f, v, L):
    """operand f * v for a classical v and a control qubit f"""
    return [f if (v >> i) & 1 else K0 for i in range(L)]


# ---------------------------------------------------------------- carry ladder
def _x_get(c, x):
    """materialize an ('and', g, q) operand; returns (qubit or const, cleanup)"""
    if isinstance(x, tuple) and x[0] == 'and':
        t = c.and_(x[1], x[2])

        def done():
            m = c.measure_x(t)
            c.cz(x[1], x[2], cond=m)
        return t, done
    return x, None


def _maj(c, x, y, cin):
    """fresh qubit holding MAJ(x, y, cin) (None if known 0); x, y, cin unchanged afterwards"""
    x, done = _x_get(c, x)
    if cin is None:
        if x is K0:
            t = None
        elif x is K1:
            t = c.alloc(1)[0]
            c.cx(y, t)
        else:
            t = c.and_(x, y)
    elif x is K0:
        t = c.and_(y, cin)
    elif x is K1:
        c.x(y); c.x(cin)
        t = c.and_(y, cin)
        c.x(y); c.x(cin); c.x(t)
    else:
        c.cx(cin, x); c.cx(cin, y)
        t = c.and_(x, y)
        c.cx(cin, t)
        c.cx(cin, x); c.cx(cin, y)
    if done:
        done()
    return t


def _unmaj(c, x, y, cin, t):
    """uncompute t = MAJ(x, y, cin) (x already materialized)"""
    if t is None:
        return
    if cin is None:
        if x is K1:
            c.cx(y, t)
            c.free([t])
        else:
            m = c.measure_x(t)
            c.cz(x, y, cond=m)
    elif x is K0:
        m = c.measure_x(t)
        c.cz(y, cin, cond=m)
    elif x is K1:
        c.x(t)
        m = c.measure_x(t)
        c.x(y); c.x(cin)
        c.cz(y, cin, cond=m)
        c.x(y); c.x(cin)
    else:
        c.cx(cin, t)
        m = c.measure_x(t)
        c.cx(cin, x); c.cx(cin, y)
        c.cz(x, y, cond=m)
        c.cx(cin, x); c.cx(cin, y)


def _sumbit(c, x, y, ci):
    if ci is not None:
        c.cx(ci, y)
    if x is K1:
        c.x(y)
    elif x is not K0:
        c.cx(x, y)


def _pad(X, L):
    X = list(X)
    assert len(X) <= L, (len(X), L)
    return X + [K0] * (L - len(X))


def add(c, X, Y, cin=None, cout=None):
    """Y <- Y + X (+ cin) mod 2^len(Y); cout (a qubit) ^= carry out. X shorter than Y is zero-padded."""
    L = len(Y)
    X = _pad(X, L)
    top = L if cout is not None else L - 1
    t, cur = [], cin
    for i in range(top):
        cur = _maj(c, X[i], Y[i], cur)
        t.append(cur)
    if cout is not None and cur is not None:
        c.cx(cur, cout)
    for i in reversed(range(L)):
        ci = t[i - 1] if i > 0 else cin
        x, done = _x_get(c, X[i])
        if i < len(t):
            _unmaj(c, x, Y[i], ci, t[i])
        _sumbit(c, x, Y[i], ci)
        if done:
            done()


def sub(c, X, Y):
    """Y <- Y - X mod 2^len(Y)"""
    for q in Y:
        c.x(q)
    add(c, X, Y)
    for q in Y:
        c.x(q)


def carry_do(c, X, Y, act, cin=None):
    """act(carry-out of X + Y (+cin) over len(Y) bits) (a qubit, or None = known 0); everything restored."""
    L = len(Y)
    X = _pad(X, L)
    t, cur = [], cin
    for i in range(L):
        cur = _maj(c, X[i], Y[i], cur)
        t.append(cur)
    r = act(cur)
    for i in reversed(range(L)):
        x, done = _x_get(c, X[i])
        _unmaj(c, x, Y[i], t[i - 1] if i > 0 else cin, t[i])
        if done:
            done()
    return r


def lt_do(c, A, B, act):
    """act([A < B]); A, B qubit lists of equal length (A may also be an int constant)."""
    L = len(B)
    if isinstance(A, int):
        return carry_do(c, cbits(((1 << L) - 1) ^ A, L), B, act)     # carry(~A + B)
    assert len(A) == L
    for q in A:
        c.x(q)
    if any(isconst(b) for b in B):
        r = carry_do(c, B, A, act)              # carry(B + ~A): the operand with constants goes first
    else:
        r = carry_do(c, A, B, act)
    for q in A:
        c.x(q)
    return r


def ge_const_do(c, B, K, act):
    """act([B >= K]) for a classical 0 < K < 2^len(B): [B >= K] = carry(~(K-1) + B)."""
    L = len(B)
    return carry_do(c, cbits(((1 << L) - 1) ^ (K - 1), L), B, act)


def allzero_do(c, Q, act):
    """act([Q == 0]) via an AND chain of the complemented bits."""
    for q in Q:
        c.x(q)
    if len(Q) == 1:
        r = act(Q[0])
    else:
        chain, cur = [], Q[0]
        for q in Q[1:]:
            t = c.and_(cur, q)
            chain.append((t, cur, q))
            cur = t
        r = act(cur)
        for t, a, b in reversed(chain):
            m = c.measure_x(t)
            c.cz(a, b, cond=m)
    for q in Q:
        c.x(q)
    return r


def cswap(c, g, A, B):
    """if g: swap(A, B) (Fredkin per bit)"""
    for a, b in zip(A, B):
        c.cx(b, a)
        c.ccx(g, a, b)
        c.cx(b, a)


def flag(c):
    return c.alloc(1)[0]


def unflag(c, f):
    c.free([f])


# ---------------------------------------------------------------- modular arithmetic (canonical, exact)
def modadd(c, A, B, p, ctl=None):
    """B <- B + ctl*A mod p. A: qubit list (n bits) or int constant (< p). ctl: qubit or None."""
    n = len(B)
    hi = c.alloc(1)[0]
    Bx = B + [hi]
    if isinstance(A, int):
        add(c, ctl_cbits(ctl, A, n + 1) if ctl is not None else cbits(A, n + 1), Bx)
    elif ctl is None:
        add(c, A, Bx)
    else:
        add(c, [('and', ctl, a) for a in A], Bx)
    add(c, cbits((1 << (n + 1)) - p, n + 1), Bx)            # - p
    f = flag(c)
    c.cx(hi, f)                                             # f = [B + ctl A < p]
    add(c, ctl_cbits(f, p, n + 1), Bx)                      # + f p
    c.free([hi])
    # f == not (ctl and [B' < A])
    if isinstance(A, int):                                  # f == (not ctl) or [B' >= A]
        assert 0 < A < p
        if ctl is None:
            ge_const_do(c, B, A, lambda cy: c.cx(cy, f))
        else:
            ge_const_do(c, B, A, lambda cy: c.ccx(ctl, cy, f))
            c.cx(ctl, f)
            c.x(f)
    else:
        lt_do(c, B, A, (lambda cy: c.cx(cy, f)) if ctl is None else (lambda cy: c.ccx(ctl, cy, f)))
        c.x(f)
    unflag(c, f)


def modsub(c, A, B, p, ctl=None):
    """B <- B - ctl*A mod p (exact reverse of modadd)."""
    n = len(B)
    f = flag(c)
    if isinstance(A, int):
        assert 0 < A < p
        if ctl is None:
            ge_const_do(c, B, A, lambda cy: c.cx(cy, f))    # f = [B >= A] (no wrap)
        else:
            ge_const_do(c, B, A, lambda cy: c.ccx(ctl, cy, f))
            c.cx(ctl, f)
            c.x(f)                                          # f = not(ctl and not ge) = (not ctl) or ge
    else:
        lt_do(c, B, A, (lambda cy: c.cx(cy, f)) if ctl is None else (lambda cy: c.ccx(ctl, cy, f)))
        c.x(f)                                              # f = not(ctl and [B < A])
    hi = c.alloc(1)[0]
    Bx = B + [hi]
    add(c, ctl_cbits(f, (1 << (n + 1)) - p, n + 1), Bx)     # - f p ; sign bit hi == f
    c.cx(hi, f)
    unflag(c, f)
    add(c, cbits(p, n + 1), Bx)                             # + p
    if isinstance(A, int):
        sub(c, ctl_cbits(ctl, A, n + 1) if ctl is not None else cbits(A, n + 1), Bx)
    elif ctl is None:
        sub(c, A, Bx)
    else:
        sub(c, [('and', ctl, a) for a in A], Bx)
    c.free([hi])


def modhalf(c, B, p):
    """B <- B / 2 mod p; returns the new register (qubits rotate)."""
    n = len(B)
    f = flag(c)
    c.cx(B[0], f)
    hi = c.alloc(1)[0]
    Bx = B + [hi]
    add(c, ctl_cbits(f, p, n + 1), Bx)                      # even, < 2p
    c.free([Bx[0]])
    H = Bx[1:]
    ge_const_do(c, H, (p + 1) // 2, lambda cy: c.cx(cy, f))  # f == [h >= (p+1)/2]
    unflag(c, f)
    return H


def moddouble(c, B, p):
    """B <- 2 B mod p (exact reverse of modhalf); returns the new register."""
    n = len(B)
    f = flag(c)
    ge_const_do(c, B, (p + 1) // 2, lambda cy: c.cx(cy, f))  # f = [B >= (p+1)/2]
    lo = c.alloc(1)[0]
    Bx = [lo] + B                                           # 2B, n+1 bits
    add(c, ctl_cbits(f, (1 << (n + 1)) - p, n + 1), Bx)     # - f p
    c.cx(Bx[0], f)                                          # parity == f
    unflag(c, f)
    c.free([Bx[-1]])
    return Bx[:-1]


def modneg(c, B, p, ctl=None):
    """B <- -B mod p (if ctl). Zero stays zero."""
    n = len(B)
    g = flag(c)
    if ctl is None:
        allzero_do(c, B, lambda z: c.cx(z, g))
        c.x(g)                                              # g = [B != 0]
    else:
        allzero_do(c, B, lambda z: (c.x(z), c.ccx(ctl, z, g), c.x(z)))   # g = ctl and B != 0
    for q in B:
        c.cx(g, q)                                          # ~B
    add(c, ctl_cbits(g, (p + 1) % (1 << n), n), B)          # ~B + p + 1 = p - B (mod 2^n)
    if ctl is None:
        allzero_do(c, B, lambda z: c.cx(z, g))
        c.x(g)
    else:
        allzero_do(c, B, lambda z: (c.x(z), c.ccx(ctl, z, g), c.x(z)))
    unflag(c, g)


def negate_nonzero(c, B, p, ctl):
    """B <- p - B if ctl, for B known to be in [1, p) (n Toffoli); an involution."""
    n = len(B)
    for q in B:
        c.cx(ctl, q)
    add(c, ctl_cbits(ctl, (p + 1) % (1 << n), n), B)


# ---------------------------------------------------------------- approximate arithmetic for p = 2^n - c, c small
# (secp256k1: c = 2^32 + 977). Values are canonical in [0, p) except with probability ~ c / 2^n per operation;
# carry / borrow chains of the c-correction are cut KAPPA bits above c's top bit (a longer chain has probability
# 2^-KAPPA for uniform values); flag uncomputation compares only the top DELTA bits (wrong with probability
KAPPA = 64
DELTA = 64
CUT_FLAGS = False
                    # truncated chain is exact or flagged (before v12 a long chain gave a wrong value without a flag)


def _cut_add(c_, X, Y):
    """Y <- Y + X on the low bits Y of a longer register (a chain cut at len(Y) bits). With CUT_FLAGS the carry out
    of the cut is computed (one more Toffoli) and flag-measured: outcome 1 exactly when the full-width chain would
    have carried into the bits above the cut, i.e. when the truncated result differs from the untruncated one."""
    if CUT_FLAGS:
        co = c_.alloc(1)[0]
        add(c_, X, Y, cout=co)
        drop(c_, [co])
    else:
        add(c_, X, Y)


def _cut_sub(c_, X, Y):
    """Y <- Y - X on the low bits of a longer register; with CUT_FLAGS the borrow out is flag-measured (the carry
    out of ~Y + X is the borrow of Y - X)."""
    for q in Y:
        c_.x(q)
    _cut_add(c_, X, Y)
    for q in Y:
        c_.x(q)


def pm_c(p, n):
    """c = 2^n - p if p is pseudo-Mersenne enough for the approximate gadgets (c < 2^(n/2)), else None"""
    c = (1 << n) - p
    return c if 0 < c and c.bit_length() <= n // 2 else None


def _low(n, c):
    return min(n, c.bit_length() + KAPPA)


def cinv_modadd_pm(c_, a, A, B, p):
    """B <- B + (-1)^a A mod p (a: qubit or None = add), approximate, ~n + |c| + KAPPA + DELTA Toffoli.
    Häner et al. Sec. VI: one's complement before and after an approximate modular adder."""
    n = len(B)
    cc = (1 << n) - p
    A = _pad(A, n)
    if a is not None:
        for q in B:
            c_.cx(a, q)
    f = c_.alloc(1)[0]
    add_cout(c_, A, B, f)                                   # B + A = f 2^n + low
    L = _low(n, cc)
    (_cut_add if L < n else add)(c_, ctl_cbits(f, cc, L), B[:L])   # f: low + c  (= B + A - p), chain cut at L bits
    # f == [r < A] for the canonical result r: compare the top DELTA bits (exact when DELTA >= n)
    d = min(n, DELTA)
    lt_do(c_, B[n - d:], A[n - d:], lambda cy: cy is not None and c_.cx(cy, f))
    drop(c_, [f])
    if a is not None:
        for q in B:
            c_.cx(a, q)


def half_pm(c_, B, p):
    """B <- B / 2 mod p, approximate: rotate the parity bit to the top, then subtract (c - 1)/2 controlled by it
    (chain cut at |c| + KAPPA bits); no flag. Returns the new register."""
    n = len(B)
    cc = (1 << n) - p
    H = B[1:] + B[:1]                                       # q + b 2^(n-1)
    L = _low(n - 1, cc)
    _cut_sub(c_, ctl_cbits(H[-1], (cc - 1) // 2, L), H[:L])
    return H


def double_pm(c_, B, p):
    """B <- 2 B mod p (inverse of half_pm). Returns the new register."""
    n = len(B)
    cc = (1 << n) - p
    L = _low(n - 1, cc)
    _cut_add(c_, ctl_cbits(B[-1], (cc - 1) // 2, L), B[:L])
    return B[-1:] + B[:-1]


# ---------------------------------------------------------------- merged replay steps: one small correction
def _and(c_, x, y, negx=False, negy=False):
    """fresh qubit (x ^ negx) & (y ^ negy) (1 Toffoli); returns (qubit, undo) where undo() uncomputes it by
    X measurement + CZ fix-up (x, y must be unchanged by then)."""
    def lit():
        if negx:
            c_.x(x)
        if negy:
            c_.x(y)
    lit()
    g = c_.and_(x, y)
    lit()

    def undo():
        m = c_.measure_x(g)
        lit()
        c_.cz(x, y, cond=m)
        lit()
    return g, undo


def _sel_add(c_, H, L, terms, neg):
    """H[:L] += g K for the one set flag g of terms [(g, K >= 0)] (flags mutually exclusive), subtracted instead
    when the qubit neg is set (one's complement of H[:L] around the addition). The chain is cut at L bits."""
    pats = {}
    for i in range(L):
        key = tuple(j for j, (g, K) in enumerate(terms) if (K >> i) & 1)
        pats.setdefault(key, []).append(i)
    op, helpers = [K0] * L, []
    for key, bits in pats.items():
        if not key:
            continue
        if len(key) == 1:
            q = terms[key[0]][0]
        else:
            q = c_.alloc(1)[0]
            for j in key:
                c_.cx(terms[j][0], q)
            helpers.append((q, key))
        for i in bits:
            op[i] = q
    for q in H[:L]:
        c_.cx(neg, q)
    (_cut_add if L < len(H) else add)(c_, op, H[:L])
    for q in H[:L]:
        c_.cx(neg, q)
    for q, key in helpers:
        for j in key:
            c_.cx(terms[j][0], q)
        c_.free([q])


CHUNK = None        # replay cells: split the n-bit add into ceil(n / CHUNK) chunks (Gidney carries per chunk only);
CHUNK_CMP = 28      # the inter-chunk carries are cleared afterwards by comparing the top CHUNK_CMP bits of each chunk's
                    # sum with the addend's (approximate: wrong only if those bits are equal, ~2^-CHUNK_CMP on uniform
                    # values). After ECDSA.fail 3616dbf's add_chunked_measured (arXiv 2609.09582 Sect. 5.3.3).


def add_cout(c_, A, B, cout):
    """B <- B + A mod 2^len(B), cout ^= carry out; chunked when CHUNK is set (A a qubit list as long as B)."""
    L = len(B)
    if CHUNK is None or L <= CHUNK:
        return add(c_, A, B, cout=cout)
    A = _pad(A, L)
    k = -(-L // CHUNK)
    cut = [L * j // k for j in range(k + 1)]
    mids, cin = [], None
    for j in range(k):
        lo, hi = cut[j], cut[j + 1]
        co = cout if j == k - 1 else c_.alloc(1)[0]
        add(c_, A[lo:hi], B[lo:hi], cin=cin, cout=co)
        if j < k - 1:
            mids.append((lo, hi, co))
        cin = co
    for lo, hi, co in mids:                     # co = [S < A + cin] on the chunk, ~ [S_top < A_top]
        d = min(CHUNK_CMP, hi - lo)
        lt_do(c_, B[hi - d:hi], A[hi - d:hi], lambda cy: cy is not None and c_.cx(cy, co))
        drop(c_, [co])


def cinv_add_half_pm(c_, a, A, B, p):
    """B <- (B + (-1)^a A) / 2 mod p, approximate: cinv_modadd_pm followed by half_pm, with the 2^n -> c fold
    and the halving correction merged into one small addition (selected by a, the carry f and the parity o).
    ~n + |c| + KAPPA + DELTA + 5 Toffoli. Returns the new (rotated) register."""
    n = len(B)
    cc = (1 << n) - p
    for q in B:
        c_.cx(a, q)
    f = c_.alloc(1)[0]
    add_cout(c_, A, B, f)                       # raw sum T' = B~ + A - f 2^n (B~ = ~B if a)
    for q in B:
        c_.cx(a, q)                             # T' complemented back when a
    H = B[1:] + B[:1]                           # (T' >> 1) + T'[0] 2^(n-1)
    o = H[-1]
    c_.cx(f, o)                                 # o = parity of the folded value T' + (-1)^a f c
    # (T' + (-1)^a f c + o p) / 2 = rot + correction: -(c-1)/2 [~f o; a f ~o], -c [a f o], +(c+1)/2 [~a f ~o]
    u1, un1 = _and(c_, f, o, negx=True)
    v, unv = _and(c_, f, o, negy=True)
    u2, un2 = _and(c_, a, v)
    w, unw = _and(c_, f, o)
    s2, uns2 = _and(c_, a, w)
    s1, s3, neg = c_.alloc(3)
    c_.cx(u1, s1); c_.cx(u2, s1)
    c_.cx(v, s3); c_.cx(u2, s3)
    c_.cx(s1, neg); c_.cx(s2, neg)
    _sel_add(c_, H, _low(n - 1, cc), [(s1, (cc - 1) // 2), (s2, cc), (s3, (cc + 1) // 2)], neg)
    c_.cx(s1, neg); c_.cx(s2, neg)
    c_.cx(v, s3); c_.cx(u2, s3)
    c_.cx(u1, s1); c_.cx(u2, s1)
    c_.free([s1, s3, neg])
    uns2(); unw(); un2(); unv(); un1()
    # f == [F~ < A] for F = 2 H - o p (F~ = ~F if a); the top d bits of F are H[n-1-d : n-1]
    d = min(n - 1, DELTA)
    win = H[n - 1 - d:n - 1]
    for q in win:
        c_.cx(a, q)
    lt_do(c_, win, A[n - d:], lambda cy: c_.cx(cy, f))
    for q in win:
        c_.cx(a, q)
    drop(c_, [f])
    return H


def dbl_cinv_add_pm(c_, a, A, B, p):
    """B <- 2 B - (-1)^a A mod p (the inverse map of cinv_add_half_pm), approximate: double_pm followed by
    cinv_modadd_pm(not a), with the doubling correction and the fold merged into one small addition (selected by
    a, the carry f and the old top bit t). Returns the new (rotated) register."""
    n = len(B)
    cc = (1 << n) - p
    Y = B[-1:] + B[:-1]                         # 2 B - t 2^n + t, t = Y[0]
    c_.x(a)
    for q in Y:
        c_.cx(a, q)                             # complement when subtracting (a = 0)
    f = c_.alloc(1)[0]
    add_cout(c_, A, Y, f)
    for q in Y:
        c_.cx(a, q)
    c_.x(a)
    t = c_.alloc(1)[0]
    c_.cx(Y[0], t); c_.cx(A[0], t)              # bit 0 of the sum is t xor A[0]
    # correction t (c-1) + (-1)^(a+1) f c: c-1 [~f t], +-c [f ~t], 2c-1 [a f t], -1 [~a f t]; negative iff ~a f
    m1, un1 = _and(c_, f, t, negx=True)
    gc, un2 = _and(c_, f, t, negy=True)
    h, un3 = _and(c_, f, t)
    m3, un4 = _and(c_, a, h)
    neg, un5 = _and(c_, a, f, negx=True)
    m5 = c_.alloc(1)[0]
    c_.cx(h, m5); c_.cx(m3, m5)
    _sel_add(c_, Y, _low(n - 1, cc), [(m1, cc - 1), (gc, cc), (m3, 2 * cc - 1), (m5, 1)], neg)
    c_.cx(h, m5); c_.cx(m3, m5)
    c_.free([m5])
    un5(); un4(); un3(); un2(); un1()
    c_.cx(Y[0], t); c_.cx(A[0], t); c_.cx(f, t)  # the correction flips bit 0 exactly when f
    drop(c_, [t])
    # f == [X < A] for the sum X before the complement-back: top d bits of Y (complemented when a = 0)
    d = min(n, DELTA)
    win = Y[n - d:]
    c_.x(a)
    for q in win:
        c_.cx(a, q)
    lt_do(c_, win, A[n - d:], lambda cy: c_.cx(cy, f))
    for q in win:
        c_.cx(a, q)
    c_.x(a)
    drop(c_, [f])
    return Y
