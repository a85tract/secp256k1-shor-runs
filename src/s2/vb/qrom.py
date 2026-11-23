"""V-B building block: table lookup of a classical table T[k] (k = n address bits, P payload bits) with
unary iteration (Babbush et al. 2018: one AND per tree node, 2^n - 2 Toffoli), and measurement-based
UNLOOKUP (Gidney, windowed arithmetic): measure the payload in the X basis, then remove the resulting phase
(-1)^{g(k)}, g(k) = parity(m & T[k]), with a two-level phase lookup: one-hot of the low address half (2^l qubits)
x unary iteration over the high half; only Toffoli are the ANDs: ~ 2 (2^l - 2) + (2^h - 2), CZ are free.
MODE = 'load'      : out ^= T[k]                      (checks the lookup; counts the load)
MODE = 'roundtrip' : load, then unlookup; out all zero (checks the MBU fix-up on every outcome; counts both)
usage: python3 harness/vb.py s2/vb/qrom.py --n 4,6,8,10 -p MODE="'roundtrip'" [-p P=512]
"""
import random

MODE = 'load'
P = 64                  # payload bits (512 = an affine point at n = 256)
BREAK = None            # 'fixup': drop one CZ fix-up of the unlookup (must FAIL in roundtrip)

_T = {}


def table(n):
    if (n, P) not in _T:
        rng = random.Random('qrom-%d-%d' % (n, P))
        _T[(n, P)] = [rng.getrandbits(P) for _ in range(1 << n)]
    return _T[(n, P)]


def inputs(n):
    return [('k', n)]


def spec(n, v):
    if MODE == 'load':
        return {'out': table(n)[v['k']], 'k': v['k']}
    return {'k': v['k']}


def sample(n, rng):
    return {'k': rng.getrandbits(n)}


def exceptional(n):
    return [{'k': 0}, {'k': (1 << n) - 1}, {'k': 1}]


def iterate(c, ctl, bits, cb, idx=0):
    """Unary iteration over the address qubits `bits` (msb first): cb(flag, index) is called once per index with a
    qubit that is 1 exactly when the address equals index (relative to these bits)."""
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
        t = c.and_(ctl, a)                  # ctl & a
        iterate(c, t, rest, cb, idx | (1 << sh))
        c.cx(ctl, t)                        # ctl & !a
        iterate(c, t, rest, cb, idx)
        m = c.measure_x(t)                  # uncompute by measurement; fix the phase (-1)^(m ctl !a)
        c.x(a)
        c.cz(ctl, a, cond=m)
        c.x(a)


def load(c, k, out, T):
    iterate(c, None, k[::-1], lambda f, i: [c.cx(f, out[b]) for b in range(len(out)) if (T[i] >> b) & 1])


def unlookup(c, k, out, T):
    W = len(k)
    ms = [c.measure_x(q) for q in out]                  # releases the payload
    l = (W + 1) // 2
    h = W - l
    lo, hi = k[:l][::-1], k[l:][::-1]
    onehot = c.alloc(1 << l)
    if l:
        iterate(c, None, lo, lambda f, j: c.cx(f, onehot[j]))
    else:
        c.x(onehot[0])
    broken = [BREAK == 'fixup']

    def leaf(f, ih):
        for j in range(1 << l):
            v = T[(ih << l) | j]
            g = 0
            for b in range(len(ms)):
                if (v >> b) & 1:
                    g ^= ms[b]
            if broken[0] and ih == 1 and j == 1:
                broken[0] = False
                continue
            c.cz(f, onehot[j], cond=g)
    if h:
        iterate(c, None, hi, leaf)
    else:
        leaf(None, 0) if False else None
        for j in range(1 << l):
            v = T[j]
            g = 0
            for b in range(len(ms)):
                if (v >> b) & 1:
                    g ^= ms[b]
            c.z(onehot[j], cond=g)
    if l:
        iterate(c, None, lo, lambda f, j: c.cx(f, onehot[j]))
    else:
        c.x(onehot[0])
    c.free(onehot)


def build(c, n, regs):
    k = regs['k']
    T = table(n)
    out = c.alloc(P)
    load(c, k, out, T)
    if MODE == 'load':
        return {'out': out}
    unlookup(c, k, out, T)
    return {}
