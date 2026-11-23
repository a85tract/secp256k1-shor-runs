"""Run harness/vb.py on o1/point_add_hex.py with the spec below registered at run time (harness files untouched).

Spec 'hex_windowed_point_add' (proposed): curve = vbspecs.curve_for(n) (secp256k1 at n = 256, j = 0 toy curves
otherwise); base point P = [s]G; table T[k] = [rho_k(lam)] P for the K mu_6-orbit representatives rho_k != 0 of
H_W (o1/hexwin.py); inputs x, y (affine accumulator, O = (0, 0)), k (< K), s (1 bit), j (0..2);
output accumulator + [(-1)^s w^j rho_k (lam)] P. Exceptional: doubling, opposite, O accumulator.
Usage: python3 o1/vbhex.py <vb.py arguments>
"""
import os, sys, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'harness'))
sys.path.insert(0, HERE)
import vbspecs
from hexwin import hexdigits, mulw


def _units(E, r, G):
    p = E.p
    assert p % 3 == 1 and r % 3 == 1
    beta = next(pow(g, (p - 1) // 3, p) for g in range(2, p) if pow(g, (p - 1) // 3, p) != 1)
    lam = next(pow(g, (r - 1) // 3, r) for g in range(2, r) if pow(g, (r - 1) // 3, r) != 1)
    if E.mul(lam, G) != (beta * G[0] % p, G[1]):
        lam = lam * lam % r
    assert E.mul(lam, G) == (beta * G[0] % p, G[1])
    return beta, lam


class HexWindowedPointAdd:
    def __init__(self, n, W, seed='0'):
        self.n, self.W = n, W
        self.E, self.G, self.r = vbspecs.curve_for(n)
        self.beta, self.lam = _units(self.E, self.r, self.G)
        rng = random.Random('vb-base-hex-%s-%d-%d' % (seed, n, W))
        self.P = self.E.mul(rng.randrange(1, self.r), self.G)
        _, _, orbits = hexdigits(W)
        self.reps = [o[0] for o in orbits if o[0] != (0, 0)]
        self.K = len(self.reps)
        self.kbits = max(1, (self.K - 1).bit_length())
        self.table = [self.E.mul((u + v * self.lam) % self.r, self.P) for (u, v) in self.reps]
        self.ctx = dict(p=self.E.p, a=self.E.a, b=self.E.b, W=W, table=self.table, beta=self.beta)

    def inputs(self):
        return [('x', self.n), ('y', self.n), ('k', self.kbits), ('s', 1), ('j', 2)]

    def digit(self, k, s, j):
        d = self.reps[k]
        for _ in range(j):
            d = mulw(*d)
        return (-d[0], -d[1]) if s else d

    def point(self, v):
        u, w = self.digit(v['k'], v['s'], v['j'])
        return self.E.mul((u + w * self.lam) % self.r, self.P)

    def spec(self, v):
        A = None if (v['x'], v['y']) == (0, 0) else (v['x'], v['y'])
        R = vbspecs._enc(self.E.add(A, self.point(v)))
        return {'x': R[0], 'y': R[1], 'k': v['k'], 's': v['s'], 'j': v['j']}

    def sample(self, rng):
        A = self.E.mul(rng.randrange(1, self.r), self.G)
        return {'x': A[0], 'y': A[1], 'k': rng.randrange(self.K), 's': rng.randrange(2), 'j': rng.randrange(3)}

    def exceptional(self, rng):
        out = []
        for (k, s, j) in [(0, 0, 0), (self.K - 1, 1, 2), (rng.randrange(self.K), 0, 1), (rng.randrange(self.K), 1, 1)]:
            v = {'k': k, 's': s, 'j': j}
            T = self.point(v)
            for A in (T, self.E.neg(T), None, self.E.mul(rng.randrange(1, self.r), self.G)):
                A = vbspecs._enc(A)
                out.append(dict(v, x=A[0], y=A[1]))
        return out


vbspecs.KINDS['hex_windowed_point_add'] = HexWindowedPointAdd

if __name__ == '__main__':
    import vb
    vb.main()
