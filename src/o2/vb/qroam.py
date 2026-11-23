"""V-B building block: select-swap table lookup (QROAM with lambda = 2, Low-Kliuchnikov-Schaeffer 2018,
Berry et al. 2019) plus measurement-based unlookup, for a classical table T[k] (W address bits, P payload bits).
Load: unary iteration over the high W-1 address bits writes T[2i] into out and T[2i] ^ T[2i+1] into a second
register G (CNOTs only, 2^(W-1) - 2 Toffoli), then out ^= k0 & G (P Toffoli), and G is X-measured at once
(outcomes mG). Unlookup: X-measure out (outcomes m), then remove (-1)^g(k),
g(k) = m.T[k] ^ mG.(T[k & ~1] ^ T[k | 1]), with the two-level phase lookup of s2/vb/qrom.py.
Toffoli: load 2^(W-1) - 2 + P, unlookup 2(2^l - 2) + (2^h - 2), l = ceil(W/2) (unary load: 2^W - 2).
Peak: 2P payload qubits during the load only (G is measured before out is used).
MODE = 'load'      : out ^= T[k]
MODE = 'roundtrip' : load, then unlookup; everything back to zero, no phase
BREAK = 'fixup'    : drop the mG part of the phase fix-up (must FAIL in roundtrip)
usage: python3 harness/vb.py o2/vb/qroam.py --n 4,6,8 -p MODE="'roundtrip'" -p P=16
"""
MODE = 'load'
P = 64
LAM = 2                 # 2: split k = 2i + k0; 3: k = 3q + r (r = k mod 3 in two qubits, k <- (k - r) / 3 in place)
UNLOAD = 'plain'
                        # iteration) or 'rec' (low floor(W/2) bits one-hot, X-measured, its phase removed by a
                        # recursive phase lookup on those bits)
BREAK = None
_T = {}


def table(n):
    if (n, P) not in _T:
        import random
        rng = random.Random('qroam-%d-%d' % (n, P))
        _T[(n, P)] = [rng.getrandbits(P) for _ in range(1 << n)]
    return _T[(n, P)]


def inputs(n):
    return [('k', n)]


def spec(n, v):
    T, k = table(n), v['k']
    if MODE == 'load':
        if LAM == 3:
            q = k // 3
            return {'out': T[k], 'G1': T[3 * q] ^ _t(T, 3 * q + 1), 'G2': T[3 * q] ^ _t(T, 3 * q + 2), 'k': k}
        return {'out': T[k], 'G': T[k & ~1] ^ T[k | 1], 'k': k}
    return {'k': k}


def sample(n, rng):
    return {'k': rng.getrandbits(n)}


def exceptional(n):
    return [{'k': 0}, {'k': (1 << n) - 1}, {'k': 1}, {'k': 2}]


def iterate(c, ctl, bits, cb, idx=0):
    """unary iteration over `bits` (msb first); cb(flag, index), flag None means 'always'"""
    if not bits:
        cb(ctl, idx)
        return
    a, rest = bits[0], bits[1:]
    sh = len(rest)
    if ctl is None:
        iterate(c, a, rest, cb, idx | (1 << sh))
        c.x(a)
        iterate(c, a, rest, cb, idx)
        c.x(a)
    else:
        t = c.and_(ctl, a)
        iterate(c, t, rest, cb, idx | (1 << sh))
        c.cx(ctl, t)
        iterate(c, t, rest, cb, idx)
        m = c.measure_x(t)
        c.x(a)
        c.cz(ctl, a, cond=m)
        c.x(a)


def _xor_const(c, f, reg, v):
    b = 0
    while v:
        if v & 1:
            if f is None:
                c.x(reg[b])
            else:
                c.cx(f, reg[b])
        v >>= 1
        b += 1


def load_ss(c, k, out, T, keep=False):
    """out ^= T[k] (out must be |0>); returns the X-outcomes of the select register for unload_ss
    (keep=True: return the select register G = T[k & ~1] ^ T[k | 1] itself, for the load-only check)"""
    P_ = len(out)
    G = c.alloc(P_)
    k0, hi = k[0], k[1:][::-1]

    def leaf(f, i):
        _xor_const(c, f, out, T[2 * i])
        _xor_const(c, f, G, T[2 * i] ^ T[2 * i + 1])
    iterate(c, None, hi, leaf)
    for b in range(P_):
        c.ccx(k0, G[b], out[b])
    if keep:
        return G
    return [c.measure_x(q) for q in G]


def unload_ss(c, k, out, T, mG):
    W = len(k)
    ms = [c.measure_x(q) for q in out]

    def par(v, mm):
        g, b = 0, 0
        while v:
            if v & 1:
                g ^= mm[b]
            v >>= 1
            b += 1
        return g

    def g_of(i):
        g = par(T[i], ms)
        if BREAK != 'fixup':
            g ^= par(T[i & ~1] ^ T[i | 1], mG)
        return g
    l = (W + 1) // 2
    h = W - l
    lo, hi = k[:l][::-1], k[l:][::-1]
    onehot = c.alloc(1 << l)
    iterate(c, None, lo, lambda f, j: c.cx(f, onehot[j]))

    def leaf(f, ih):
        for j in range(1 << l):
            g = g_of((ih << l) | j)
            if g:
                if f is None:
                    c.z(onehot[j], cond=g)
                else:
                    c.cz(f, onehot[j], cond=g)
    if h:
        iterate(c, None, hi, leaf)
    else:
        leaf(None, 0)
    iterate(c, None, lo, lambda f, j: c.cx(f, onehot[j]))
    c.free(onehot)


# ---------------------------------------------------------------- lambda = 3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'h1'))
import vbarith as _va


def _step3(c, q, r1, r2, up):
    """controlled (q) r <- r + 1 (up) or r - 1 mod 3 on the one-hot pair r1 = [r == 1], r2 = [r == 2]"""
    def f1():                                   # flip r1 if q and not r2
        c.x(r2); c.ccx(q, r2, r1); c.x(r2)

    def f2():                                   # flip r2 if q and not r1
        c.x(r1); c.ccx(q, r1, r2); c.x(r1)
    if up:
        f1(); f2()
    else:
        f2(); f1()


def _mod3(c, k, r1, r2, inverse=False):
    """r <- r + k mod 3 (bit i weighs 2^i = (-1)^i mod 3); inverse=True undoes it"""
    seq = list(enumerate(k))
    for i, q in (seq[::-1] if inverse else seq):
        _step3(c, q, r1, r2, (i % 2 == 0) != inverse)


def _mul_odd(c, k, a):
    """k <- a k mod 2^len(k) in place (a odd): top-down, bit i adds k_i (a - 1) 2^i to the bits above it"""
    m = len(k)
    for i in range(m - 2, -1, -1):
        v = ((a - 1) >> 1) % (1 << (m - i - 1))
        if v:
            _va.add(c, _va.ctl_cbits(k[i], v, m - i - 1), k[i + 1:])


def _iter_range(c, ctl, bits, cb, limit, idx=0):
    """unary iteration over indices < limit only (bits msb first; addresses >= limit never occur)"""
    if idx >= limit:
        return
    if not bits:
        cb(ctl, idx)
        return
    a, rest = bits[0], bits[1:]
    hi = idx | (1 << len(rest))
    if hi >= limit:                             # the 1-branch is empty: a = 0 on every valid input
        _iter_range(c, ctl, rest, cb, limit, idx)
        return
    if ctl is None:
        _iter_range(c, a, rest, cb, limit, hi)
        c.x(a)
        _iter_range(c, a, rest, cb, limit, idx)
        c.x(a)
    else:
        t = c.and_(ctl, a)
        _iter_range(c, t, rest, cb, limit, hi)
        c.cx(ctl, t)
        _iter_range(c, t, rest, cb, limit, idx)
        m = c.measure_x(t)
        c.x(a)
        c.cz(ctl, a, cond=m)
        c.x(a)


def _t(T, i):
    return T[i] if i < len(T) else 0


def load_ss3(c, k, out, T, keep=False):
    """out ^= T[k] by a three-way select; returns the X-outcomes of the two select registers (keep=True: the
    registers themselves, holding T[3q] ^ T[3q+1] and T[3q] ^ T[3q+2], q = k // 3)"""
    W, P_ = len(k), len(out)
    r1, r2 = c.alloc(2)
    _mod3(c, k, r1, r2)
    _va.sub(c, [r1, r2], k)                     # k - r, a multiple of 3
    _mul_odd(c, k, pow(3, -1, 1 << W))          # k = q = (k - r) / 3 < 2^(W-1)
    G1, G2 = c.alloc(P_), c.alloc(P_)

    def leaf(f, q):
        base = _t(T, 3 * q)
        _xor_const(c, f, out, base)
        _xor_const(c, f, G1, base ^ _t(T, 3 * q + 1))
        _xor_const(c, f, G2, base ^ _t(T, 3 * q + 2))
    _iter_range(c, None, k[:W - 1][::-1], leaf, -(-min(len(T), 1 << W) // 3))   # addresses >= len(T) never occur
    for b in range(P_):
        c.ccx(r1, G1[b], out[b])
        c.ccx(r2, G2[b], out[b])
    res = (G1, G2) if keep else ([c.measure_x(q) for q in G1], [c.measure_x(q) for q in G2])
    _mul_odd(c, k, 3)
    _va.add(c, [r1, r2], k)
    _mod3(c, k, r1, r2, inverse=True)
    c.free([r1, r2])
    return res


def unload_ss3(c, k, out, T, mG):
    mG1, mG2 = mG
    ms = [c.measure_x(q) for q in out]

    def par(v, mm):
        g, b = 0, 0
        while v:
            if v & 1:
                g ^= mm[b]
            v >>= 1
            b += 1
        return g

    def g_of(i):
        q = i // 3
        g = par(T[i], ms)
        if BREAK != 'fixup':
            g ^= par(_t(T, 3 * q) ^ _t(T, 3 * q + 1), mG1) ^ par(_t(T, 3 * q) ^ _t(T, 3 * q + 2), mG2)
        return g
    _phase_lookup(c, k, g_of)


def _phase_lookup(c, k, g_of):
    if UNLOAD == 'rec':
        return _phase_lookup_rec(c, k, g_of)
    W = len(k)
    l = (W + 1) // 2
    h = W - l
    lo, hi = k[:l][::-1], k[l:][::-1]
    onehot = c.alloc(1 << l)
    iterate(c, None, lo, lambda f, j: c.cx(f, onehot[j]))

    def leaf(f, ih):
        for j in range(1 << l):
            g = g_of((ih << l) | j)
            if g:
                if f is None:
                    c.z(onehot[j], cond=g)
                else:
                    c.cz(f, onehot[j], cond=g)
    if h:
        iterate(c, None, hi, leaf)
    else:
        leaf(None, 0)
    iterate(c, None, lo, lambda f, j: c.cx(f, onehot[j]))
    c.free(onehot)


def _phase_lookup_rec(c, k, g_of):
    """apply (-1)^g_of(k): W <= 2 by direct iteration; else one-hot O of the low floor(W/2) bits, unary iteration
    over the high bits with CZ to O, then X-measure O (phase (-1)^mo[lo]) and remove that phase recursively"""
    W = len(k)
    if W <= 2:
        def leaf0(f, i):
            g = g_of(i)
            if g:
                c.z(f, cond=g)
        iterate(c, None, k[::-1], leaf0)
        return
    l = W // 2
    lo, hi = k[:l][::-1], k[l:][::-1]
    onehot = c.alloc(1 << l)
    iterate(c, None, lo, lambda f, j: c.cx(f, onehot[j]))

    def leaf(f, ih):
        for j in range(1 << l):
            g = g_of((ih << l) | j)
            if g:
                c.cz(f, onehot[j], cond=g)
    iterate(c, None, hi, leaf)
    mo = [c.measure_x(q) for q in onehot]
    _phase_lookup_rec(c, k[:l], lambda j: mo[j])


def build(c, n, regs):
    k = regs['k']
    T = table(n)
    out = c.alloc(P)
    if LAM == 3:
        if MODE == 'load':
            G1, G2 = load_ss3(c, k, out, T, keep=True)
            return {'out': out, 'G1': G1, 'G2': G2}
        unload_ss3(c, k, out, T, load_ss3(c, k, out, T))
        return {}
    if MODE == 'load':
        return {'out': out, 'G': load_ss(c, k, out, T, keep=True)}
    unload_ss(c, k, out, T, load_ss(c, k, out, T))
    return {}
