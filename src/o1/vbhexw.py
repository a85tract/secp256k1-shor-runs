"""Run harness/vb.py with the spec KIND 'hex_window_add' registered at run time (harness files untouched).

Spec: curve = vbspecs.curve_for(n) (secp256k1 at n = 256); base point P = [t]G (seeded); inputs x, y (affine
accumulator, O = (0, 0)) and xr (2W bits: residue a = xr mod 2^W, b = xr >> W); output accumulator +
[d1 + d2 lam] P with (d1, d2) = hexdigits.digit(a, b, W); xr unchanged. The circuit gets the table of one point
per canonical residue x'' (hexdigits.canonical_list): T[x''] = [lift(x'')(lam)] P, lift(0) = 2^W.
Exceptional inputs: accumulator = +-(the digit's point), O; residues 0 (digit 2^W), 2^(W-1) (orbit of size 3).
usage: python3 o1/vbhexw.py <vb.py arguments>
"""
import os, sys, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'harness'))
sys.path.insert(0, HERE)
import vbspecs
from vbhex import _units
from hexdigits import canonical_list, lift, digit


class HexWindowAdd:
    def __init__(self, n, W, seed='0'):
        self.n, self.W = n, W
        self.E, self.G, self.r = vbspecs.curve_for(n)
        self.beta, self.lam = _units(self.E, self.r, self.G)
        rng = random.Random('vb-base-hexw-%s-%d-%d' % (seed, n, W))
        self.P = self.E.mul(rng.randrange(1, self.r), self.G)
        M = 1 << W
        E, p = self.E, self.E.p
        mult = {0: None, 1: self.P}
        for i in range(2, 2 * M + 1):
            mult[i] = E.add(mult[i - 1], self.P)
        for i in range(1, 2 * M + 1):
            mult[-i] = E.neg(mult[i])

        def phi(Q):
            return None if Q is None else (self.beta * Q[0] % p, Q[1])
        self._mult, self._phi = mult, phi
        self.table = {}
        for key in canonical_list(W):
            d1, d2 = lift(key, W)
            self.table[key] = E.add(mult[d1], phi(mult[d2]))
        assert all(T is not None and T[0] != 0 for T in self.table.values())
        self.ctx = dict(p=p, a=E.a, b=E.b, W=W, table=self.table, beta=self.beta)

    def inputs(self):
        return [('x', self.n), ('y', self.n), ('xr', 2 * self.W)]

    def point(self, xr):
        M = 1 << self.W
        d1, d2 = digit(xr & (M - 1), xr >> self.W, self.W)
        return self.E.add(self._mult[d1], self._phi(self._mult[d2]))

    def spec(self, v):
        A = None if (v['x'], v['y']) == (0, 0) else (v['x'], v['y'])
        R = vbspecs._enc(self.E.add(A, self.point(v['xr'])))
        return {'x': R[0], 'y': R[1], 'xr': v['xr']}

    def sample(self, rng):
        A = self.E.mul(rng.randrange(1, self.r), self.G)
        return {'x': A[0], 'y': A[1], 'xr': rng.randrange(1 << (2 * self.W))}

    def exceptional(self, rng):
        W, M = self.W, 1 << self.W
        out = []
        for xr in (0, M >> 1, (M >> 1) << W, rng.randrange(1, M * M), rng.randrange(1, M * M)):
            T = self.point(xr)
            for A in (T, self.E.neg(T), None, self.E.mul(rng.randrange(1, self.r), self.G)):
                A = vbspecs._enc(A)
                out.append(dict(xr=xr, x=A[0], y=A[1]))
        return out


vbspecs.KINDS['hex_window_add'] = HexWindowAdd

if __name__ == '__main__':
    import vb
    vb.main()
