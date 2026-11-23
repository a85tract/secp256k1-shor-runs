"""Number theory, short-Weierstrass curve arithmetic and the instance set (toy prime-order curves + secp256k1).

Only the harness uses this module; candidates never see coordinates or discrete logs.
"""
import json, os, random

# ---------------------------------------------------------------- number theory

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)


def is_prime(n):
    """Miller-Rabin; deterministic below 3.3e24, 13 fixed + 8 random bases above (error < 4^-21)."""
    if n < 2:
        return False
    for q in _MR_BASES:
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    bases = list(_MR_BASES)
    if n >= 3317044064679887385961981:
        rng = random.Random(n)
        bases += [rng.randrange(2, n - 1) for _ in range(8)]
    for a in bases:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def sqrt_mod(a, p):
    """Square root mod an odd prime p (Tonelli-Shanks); None if a is a non-residue."""
    a %= p
    if a == 0:
        return 0
    if pow(a, (p - 1) // 2, p) != 1:
        return None
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    q, s = p - 1, 0
    while q % 2 == 0:
        q, s = q // 2, s + 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2, i = t2 * t2 % p, i + 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c, t, r = i, b * b % p, t * b * b % p, r * b % p
    return r


def isqrt(n):
    import math
    return math.isqrt(n)


def cornacchia3(p):
    """x, y with x^2 + 3 y^2 = p for a prime p = 1 mod 3."""
    r0 = sqrt_mod(-3, p)
    a, b = p, r0
    lim = isqrt(p)
    while b > lim:
        a, b = b, a % b
    x = b
    rest = p - x * x
    assert rest % 3 == 0
    y = isqrt(rest // 3)
    assert x * x + 3 * y * y == p
    return x, y


# ---------------------------------------------------------------- curve y^2 = x^3 + a x + b, affine, None = infinity

class Curve:
    def __init__(self, p, a, b):
        self.p, self.a, self.b = p, a, b

    def on_curve(self, P):
        if P is None:
            return True
        x, y = P
        return (y * y - (x * x * x + self.a * x + self.b)) % self.p == 0

    def neg(self, P):
        return None if P is None else (P[0], (-P[1]) % self.p)

    def add(self, P, Q):
        if P is None:
            return Q
        if Q is None:
            return P
        p = self.p
        x1, y1 = P[0] % p, P[1] % p
        x2, y2 = Q[0] % p, Q[1] % p
        if x1 == x2:
            if (y1 + y2) % p == 0:
                return None
            lam = (3 * x1 * x1 + self.a) * pow(2 * y1, -1, p) % p
        else:
            lam = (y2 - y1) * pow(x2 - x1, -1, p) % p
        x3 = (lam * lam - x1 - x2) % p
        return (x3, (lam * (x1 - x3) - y1) % p)

    def mul(self, k, P):
        R = None
        if k < 0:
            k, P = -k, self.neg(P)
        while k:
            if k & 1:
                R = self.add(R, P)
            P = self.add(P, P)
            k >>= 1
        return R

    def random_point(self, rng):
        p = self.p
        while True:
            x = rng.randrange(p)
            y = sqrt_mod(x * x * x + self.a * x + self.b, p)
            if y is not None:
                return (x, y if rng.random() < 0.5 else (-y) % p)


# ---------------------------------------------------------------- instances

SECP256K1 = dict(
    name='secp256k1', n=256,
    p=2**256 - 2**32 - 977, a=0, b=7,
    r=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141,
    Gx=0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    Gy=0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)


def embedding_degree_ok(p, r, kmax=100):
    x = 1
    for _ in range(kmax):
        x = x * p % r
        if x == 1:
            return False
    return True


def endo_constants(curve, r, G):
    """beta (cube root of 1 mod p) and lambda (cube root of 1 mod r) with (beta x, y) = [lambda](x, y)."""
    p = curve.p
    beta = None
    h = 2
    while beta is None:
        c = pow(h, (p - 1) // 3, p)
        if c != 1:
            beta = c
        h += 1
    s = sqrt_mod(-3, r)
    lam = (-1 + s) * pow(2, -1, r) % r
    phiG = (beta * G[0] % p, G[1])
    if curve.mul(lam, G) != phiG:
        lam = (-1 - s) * pow(2, -1, r) % r
        assert curve.mul(lam, G) == phiG
    return beta, lam


def gen_j0_curve(nbits, rng, mov=True):
    """Random prime-order curve y^2 = x^3 + b over an nbits-bit prime p = 1 mod 3 (not anomalous, embedding
    degree > 100 unless mov=False, which V-B uses for arithmetic-only toy curves of a few bits)."""
    while True:
        p = rng.randrange(2 ** (nbits - 1), 2 ** nbits) | 1
        if p % 3 != 1 or not is_prime(p):
            continue
        x, y = cornacchia3(p)
        traces = {2 * x, -2 * x, x + 3 * y, -(x + 3 * y), x - 3 * y, -(x - 3 * y)}
        orders = sorted(p + 1 - t for t in traces if is_prime(p + 1 - t) and p + 1 - t != p)
        if not orders:
            continue
        N = orders[0] if len(orders) == 1 else rng.choice(orders)
        if mov and (N.bit_length() != nbits or not embedding_degree_ok(p, N)):
            continue      # (mov=False: arithmetic-only toy curve, any prime order; none of 5-6 bits has r ~ p)
        for _ in range(200):
            b = rng.randrange(1, p)
            E = Curve(p, 0, b)
            P = E.random_point(rng)
            if E.mul(N, P) is None:
                break
        else:
            continue
        G = P  # any non-identity point generates a group of prime order
        return dict(name=f'j0_{nbits}_{p:x}_{b:x}', n=nbits, p=p, a=0, b=b, r=N, Gx=G[0], Gy=G[1])


def complete(inst):
    """Validate an instance and add endomorphism constants. Raises on any inconsistency."""
    E = Curve(inst['p'], inst['a'], inst['b'])
    G = (inst['Gx'], inst['Gy'])
    r, p = inst['r'], inst['p']
    assert is_prime(p) and is_prime(r) and r != p, inst['name']
    assert E.on_curve(G) and E.mul(r, G) is None, inst['name']
    assert abs(p + 1 - r) <= 2 * isqrt(p) + 2, inst['name']
    assert embedding_degree_ok(p, r), inst['name']
    out = dict(inst)
    if inst['a'] == 0 and p % 3 == 1:
        out['beta'], out['lam'] = endo_constants(E, r, G)
    return out


_STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instances.json')
TOY_SIZES = (12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192)


def build_store(per_size=3, seed=20261112):
    rng = random.Random(seed)
    insts = [SECP256K1]
    for nb in TOY_SIZES:
        insts += [gen_j0_curve(nb, rng) for _ in range(per_size)]
    for nb in (256,):
        insts += [gen_j0_curve(nb, rng) for _ in range(2)]
    insts = [complete(i) for i in insts]
    with open(_STORE, 'w') as f:
        json.dump(insts, f, indent=0)
    return insts


def load_instances():
    if not os.path.exists(_STORE):
        return build_store()
    with open(_STORE) as f:
        return json.load(f)


def instances_for(n):
    return [i for i in load_instances() if i['n'] == n]


if __name__ == '__main__':
    import time
    t = time.time()
    insts = build_store()
    for i in insts:
        print(i['n'], i['name'][:40], 'lam' in i)
    print(len(insts), 'instances, %.1f s' % (time.time() - t))
