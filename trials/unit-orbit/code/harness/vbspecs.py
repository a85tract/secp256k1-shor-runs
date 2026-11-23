"""Harness-owned specifications for V-B. A circuit file that sets KIND = '<name>' is checked against the spec
here, never against its own spec(), so a "measured" cost in V-A always refers to the operation V-A prices.

KIND 'windowed_point_add' (one windowed point addition of a Shor run, lookup included):
  curve: secp256k1 at n = 256; at other n a fixed toy curve y^2 = x^3 + b of n bits (prime order).
  classical data: base point P = [s]G (fixed per check) and the table T[k] = [k]P, k in [0, 2^W),
  passed to build(c, n, regs, ctx) as ctx = dict(p, a, b, W, table) with table[k] = (x, y) or None for O.
  inputs: x, y (n bits each: the accumulator in affine coordinates; O is encoded as (0, 0)), k (W bits).
  outputs: x, y = accumulator + T[k]; k unchanged; no garbage, no phase.
  random inputs: accumulator = [t]G for random t, random k. Exceptional inputs: k = 0 (T[0] = O),
  accumulator = T[k] (doubling), accumulator = -T[k] (result O), accumulator = O, accumulator = -T[k] with
  k = 1, x = 0 points where they exist.
"""
import random

from ecc import Curve, SECP256K1, gen_j0_curve

_CURVES = {}


def curve_for(n):
    if n not in _CURVES:
        if n == 256:
            inst = SECP256K1
        else:
            inst = gen_j0_curve(n, random.Random('vb-curve-%d' % n), mov=False)
        _CURVES[n] = (Curve(inst['p'], inst['a'], inst['b']), (inst['Gx'], inst['Gy']), inst['r'])
    return _CURVES[n]


def _enc(P):
    return (0, 0) if P is None else P


class WindowedPointAdd:
    def __init__(self, n, W, seed='0'):
        self.n, self.W = n, W
        self.E, self.G, self.r = curve_for(n)
        rng = random.Random('vb-base-%s-%d-%d' % (seed, n, W))
        self.P = self.E.mul(rng.randrange(1, self.r), self.G)
        T, acc = [], None
        for _ in range(1 << W):
            T.append(acc)
            acc = self.E.add(acc, self.P)
        self.table = T
        self.ctx = dict(p=self.E.p, a=self.E.a, b=self.E.b, W=W, table=T)

    def inputs(self):
        return [('x', self.n), ('y', self.n), ('k', self.W)]

    def spec(self, v):
        A = None if (v['x'], v['y']) == (0, 0) else (v['x'], v['y'])
        R = _enc(self.E.add(A, self.table[v['k']]))
        return {'x': R[0], 'y': R[1], 'k': v['k']}

    def sample(self, rng):
        A = self.E.mul(rng.randrange(1, self.r), self.G)
        return {'x': A[0], 'y': A[1], 'k': rng.randrange(1 << self.W)}

    def exceptional(self, rng):
        out = []
        ks = [0, 1, (1 << self.W) - 1] + [rng.randrange(1, 1 << self.W) for _ in range(5)]
        for k in ks:
            T = self.table[k]
            for A in (T, self.E.neg(T), None, self.E.mul(rng.randrange(1, self.r), self.G),
                      self.E.neg(self.E.add(T, T))):
                A = _enc(A)
                out.append({'x': A[0], 'y': A[1], 'k': k})
        return out


class SignedWindowedPointAdd(WindowedPointAdd):
    """KIND 'signed_windowed_point_add' (signed odd digits, F4 in odd-digit form; added by h1):
    W = exponent bits per addition; the window register k (W bits) selects the odd digit d = 2k - 2^W + 1
    (|d| < 2^W, never 0) and the output is accumulator + [d]P. The classical table has 2^(W-1) entries
    T[j] = [2j + 1]P, j = |d| >> 1, passed as ctx['table'] (the circuit forms j and the sign itself).
    Why this prices the same run as V-A's add(): [d]P = [k](2P) + C with the classical C = [1 - 2^W]P, so a run
    of such windows adds [x](2P) per window plus a known constant, which only relabels the discarded point
    register (the exponent registers' outcome law is unchanged). The base point of the window is 2P.
    Exceptional inputs: accumulator = [d]P (doubling), = -[d]P (result O), = O, = -[2d]P (result -[d]P), and
    k = 0 / 2^W - 1 / 2^(W-1) (d = -(2^W - 1), 2^W - 1, 1)."""

    def __init__(self, n, W, seed='0'):
        self.n, self.W = n, W
        self.E, self.G, self.r = curve_for(n)
        rng = random.Random('vb-base-signed-%s-%d-%d' % (seed, n, W))
        self.P = self.E.mul(rng.randrange(1, self.r), self.G)
        self.table = [self.E.mul(2 * j + 1, self.P) for j in range(1 << (W - 1))]
        self.ctx = dict(p=self.E.p, a=self.E.a, b=self.E.b, W=W, table=self.table, signed=True)

    def point(self, k):
        d = 2 * k - (1 << self.W) + 1
        T = self.table[abs(d) >> 1]
        return T if d > 0 else self.E.neg(T)

    def spec(self, v):
        A = None if (v['x'], v['y']) == (0, 0) else (v['x'], v['y'])
        R = _enc(self.E.add(A, self.point(v['k'])))
        return {'x': R[0], 'y': R[1], 'k': v['k']}

    def exceptional(self, rng):
        out = []
        ks = [0, (1 << self.W) - 1, 1 << (self.W - 1), (1 << (self.W - 1)) - 1] + \
            [rng.randrange(1 << self.W) for _ in range(4)]
        for k in ks:
            T = self.point(k)
            for A in (T, self.E.neg(T), None, self.E.mul(rng.randrange(1, self.r), self.G),
                      self.E.neg(self.E.add(T, T))):        # A = -2T: see WindowedPointAdd.exceptional
                A = _enc(A)
                out.append({'x': A[0], 'y': A[1], 'k': k})
        # small denominators: x1 - x2 = delta and x2 - x3 = delta for the smallest
        # delta >= 1 that is a valid x offset and for delta = 3 when valid (in-place inversion of a small value)
        p = self.E.p
        for k in ks[:4]:
            T = self.point(k)
            for sgn in (1, -1):                     # sgn = 1: x(A) = x(T) + d; sgn = -1: x(A + T) = x(T) - d
                found = []
                for d in range(1, 64):
                    xr = (T[0] + sgn * d) % p
                    yr = _sqrt_or_none(xr ** 3 + self.E.a * xr + self.E.b, p)
                    if yr is not None and (d == 3 or not found):
                        found.append((d, (xr, yr)))
                    if len(found) == 2 or (found and d > 3):
                        break
                for d, Rp in found:
                    A = Rp if sgn == 1 else self.E.add(Rp, self.E.neg(T))
                    A = _enc(A)
                    out.append({'x': A[0], 'y': A[1], 'k': k})
        return out


def _sqrt_or_none(a, p):
    from ecc import sqrt_mod
    a %= p
    if a == 0:
        return 0
    if pow(a, (p - 1) // 2, p) != 1:
        return None
    return sqrt_mod(a, p)


KINDS = {'windowed_point_add': WindowedPointAdd, 'signed_windowed_point_add': SignedWindowedPointAdd}
