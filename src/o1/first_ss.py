"""V-B building block: the run's first window with a random offset (Run.offset()) as a select-swap read.

out (P qubits; in the run P = 2n, the accumulator (x, y)) <- T[k] for an n-bit window k and a classical table T of
building blocks (o2/vb/qroam.py, imported unchanged):
  1. load_ss: unary iteration over the high n - 1 address bits writes T[2i] into out and T[2i] ^ T[2i+1] into a
     second P-qubit register G (CNOTs only, 2^(n-1) - 2 Toffoli), then out ^= k0 & G (P Toffoli), then G is
     X-measured (outcomes mG);
  2. the X-measurement leaves the phase (-1)^(mG . (T[2i] ^ T[2i+1])), i = k >> 1, a function of the high n - 1
     one-hot of the low half, unary iteration over the high half with CZ, X-measure the one-hot, recurse).
Toffoli: 2^(n-1) - 2 + P + f(n - 1), f(w) = 2^(w//2) - 2 + 2^(w - w//2) - 2 + f(w//2) (w > 2), f(w) = 2^w - 2 (w <= 2);
at n = 19, P = 512: 262,142 + 512 + 1,070 = 263,724 (unary read of the same table: 2^19 - 2 = 524,286).
Peak: n + 2P + (n - 2) during the load (G is measured before the phase lookup).
MODE 'load' (default): out = T[k], k unchanged, no garbage, no phase. BREAK = 'fixup' drops step 2 (must FAIL).
usage: python3 harness/vb.py o1/first_ss.py --n 4,6,8,10 -p P=64
"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'o2', 'vb'))
import qroam as _q      # noqa: E402

P = 64
BREAK = None
_T = {}


def table(n):
    if (n, P) not in _T:
        import random
        rng = random.Random('first-ss-%d-%d' % (n, P))
        _T[(n, P)] = [rng.getrandbits(P) for _ in range(1 << n)]
    return _T[(n, P)]


def _par(v, mm):
    g, b = 0, 0
    while v:
        if v & 1:
            g ^= mm[b]
        v >>= 1
        b += 1
    return g


def load(c, k, out, T):
    """out ^= T[k] (out |0>), k unchanged, no phase (steps 1 and 2 of the module doc)"""
    mG = _q.load_ss(c, k, out, T)
    if BREAK == 'fixup':
        return
    _q._phase_lookup_rec(c, k[1:], lambda i: _par(T[2 * i] ^ T[2 * i + 1], mG))


def build(c, n, regs):
    k = regs['k']
    out = c.alloc(P)
    load(c, k, out, table(n))
    return {'out': out}


def inputs(n):
    return [('k', n)]


def spec(n, v):
    return {'out': table(n)[v['k']], 'k': v['k']}


def sample(n, rng):
    return {'k': rng.getrandbits(n)}


def exceptional(n):
    return [{'k': 0}, {'k': (1 << n) - 1}, {'k': 1}, {'k': 2}]


def toffoli_formula(n, P_):
    def f(w):
        if w <= 2:
            return 2 ** w - 2
        lo = w // 2
        return 2 ** lo - 2 + 2 ** (w - lo) - 2 + f(lo)
    return 2 ** (n - 1) - 2 + P_ + f(n - 1)
