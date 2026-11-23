"""V-B spec KIND 'n7_window_add': one norm-7 x 4^w window addition (o1/mixrad.py N7Window).

Inputs x, y (affine accumulator, O = (0, 0)) and xr = c | a << 3 | b << (3 + w): the residue (c, a + b w) of
Z[w] / ((3 + w) 2^w) = F_7 x (Z / 2^w)^2, c in [0, 7) (the value 7 is not a valid input). Output accumulator +
[d1 + d2 lam] P with (d1, d2) = N7Window.digit(residue); xr unchanged. The circuit gets the table of one point per
orbit representative: T[(zf, x)] = [rep_digit(zf, x)(lam)] P (zf = 0: x in (Z / 2^w)^2, the class c = 1; zf = 1: the
hex-canonical x'' of the class c = 0; the zero residue's digit is pi itself, so no entry is O).
Exceptional inputs: accumulator = +-(the digit's point), O; residues 0, (0, 2^(w-1)) (hex orbit of size 3),
(c != 0, x = 0), random.
usage: python3 o1/vbn7w.py <vb.py arguments>   (o1/vbx2n7.py for approximate circuits)
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'harness'))
sys.path.insert(0, HERE)
import vbspecs                     # noqa: E402
from vbhex import _units           # noqa: E402
import mixrad as mr                # noqa: E402

BAR = False


def residue_rep(win, c, a, b):
    """an Eisenstein integer z with phi7(z) = c and z = (a, b) mod 2^w"""
    w = win.w
    k = (c - win.phi7((a, b))) * win.inv2w % 7
    return (a + (k << w), b)


class N7WindowAdd:
    def __init__(self, n, w, seed='0'):
        self.n, self.w = n, w
        self.win = mr.N7Window(w, BAR)
        self.E, self.G, self.r = vbspecs.curve_for(n)
        self.beta, self.lam = _units(self.E, self.r, self.G)
        rng = random.Random('vb-base-n7w-%s-%d-%d' % (seed, n, w))
        self.P = self.E.mul(rng.randrange(1, self.r), self.G)
        E, p = self.E, self.E.p
        self._p = p
        self.table = {}
        for zf, x in self.win.reps():
            d = self.win.rep_digit(zf, x)
            self.table[(zf, x)] = self._pt(d)
        assert all(T is not None and T[0] != 0 for T in self.table.values())
        self.ctx = dict(p=p, a=E.a, b=E.b, W=w, table=self.table, beta=self.beta, bar=BAR)

    def _pt(self, d):
        k = (d[0] + d[1] * self.lam) % self.r
        return self.E.mul(k, self.P) if k else None

    def inputs(self):
        return [('x', self.n), ('y', self.n), ('xr', 3 + 2 * self.w)]

    def split(self, xr):
        M = 1 << self.w
        return xr & 7, (xr >> 3) & (M - 1), xr >> (3 + self.w)

    def point(self, xr):
        c, a, b = self.split(xr)
        d = self.win.digit(residue_rep(self.win, c, a, b))
        return self._pt(d)

    def spec(self, v):
        A = None if (v['x'], v['y']) == (0, 0) else (v['x'], v['y'])
        R = vbspecs._enc(self.E.add(A, self.point(v['xr'])))
        return {'x': R[0], 'y': R[1], 'xr': v['xr']}

    def enc(self, c, a, b):
        return c | (a << 3) | (b << (3 + self.w))

    def sample(self, rng):
        A = self.E.mul(rng.randrange(1, self.r), self.G)
        M = 1 << self.w
        return {'x': A[0], 'y': A[1], 'xr': self.enc(rng.randrange(7), rng.randrange(M), rng.randrange(M))}

    def exceptional(self, rng):
        M = 1 << self.w
        out = []
        for c, a, b in ((0, 0, 0), (0, M >> 1, 0), (0, 0, M >> 1), (3, 0, 0), (1, 0, 0),
                        (rng.randrange(1, 7), rng.randrange(M), rng.randrange(M)), (0, rng.randrange(1, M), rng.randrange(M))):
            xr = self.enc(c, a, b)
            T = self.point(xr)
            for A in (T, self.E.neg(T), None, self.E.mul(rng.randrange(1, self.r), self.G)):
                A = vbspecs._enc(A)
                out.append(dict(xr=xr, x=A[0], y=A[1]))
        return out


vbspecs.KINDS['n7_window_add'] = N7WindowAdd

if __name__ == '__main__':
    import vb
    vb.main()
