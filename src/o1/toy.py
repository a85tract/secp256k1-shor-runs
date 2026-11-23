"""Small exact toy curves for checks: prime-order j = 0 curves y^2 = x^3 + b over F_p.

"""
import numpy as np


def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def count_points(p, b):
    sq = np.zeros(p, dtype=np.int64)
    for y in range(p):
        sq[(y * y) % p] += 1
    xs = np.arange(p, dtype=np.int64)
    rhs = (xs * xs % p * xs + b) % p
    return 1 + int(sq[rhs].sum())


def find_curve(pmin, skip=0):
    """Smallest prime p >= pmin, p = 1 mod 3, with some b making #E prime (not anomalous, r != p)."""
    p = pmin
    found = 0
    while True:
        if is_prime(p) and p % 3 == 1:
            for b in range(1, p):
                r = count_points(p, b)
                if is_prime(r) and r != p:
                    # embedding degree > 6 check
                    if all(pow(p, k, r) != 1 for k in range(1, 7)):
                        if found == skip:
                            return p, b, r
                        found += 1
                        break
        p += 1


class Curve:
    def __init__(self, p, b, r):
        self.p, self.b, self.r = p, b, r
        # generator: first point found (prime order, so any non-O point generates)
        for x in range(p):
            rhs = (x * x * x + b) % p
            for y in range(p):
                if (y * y) % p == rhs:
                    self.G = (x, y)
                    break
            else:
                continue
            break
        # multiples table: pts[k] = kG (None = O), and inverse map
        self.pts = [None] * r
        P = None
        for k in range(r):
            self.pts[k] = P
            P = self.add(P, self.G)
        assert P is None, "order mismatch"
        self.idx = {P: k for k, P in enumerate(self.pts)}
        # j = 0 endomorphism: beta in F_p with beta^3 = 1, lam mod r with (beta x, y) = lam (x, y)
        self.beta = next(c for c in range(2, p) if pow(c, 3, p) == 1)
        bx = (self.beta * self.G[0] % p, self.G[1])
        self.lam = self.idx[bx]
        self.X = np.array([-1 if P is None else P[0] for P in self.pts], dtype=np.int64)
        self.Y = np.array([-1 if P is None else P[1] for P in self.pts], dtype=np.int64)

    def add(self, P, Q):
        p = self.p
        if P is None:
            return Q
        if Q is None:
            return P
        x1, y1 = P
        x2, y2 = Q
        if x1 == x2:
            if (y1 + y2) % p == 0:
                return None
            lam = (3 * x1 * x1) * pow(2 * y1, -1, p) % p
        else:
            lam = (y2 - y1) * pow(x2 - x1, -1, p) % p
        x3 = (lam * lam - x1 - x2) % p
        return (x3, (lam * (x1 - x3) - y1) % p)


if __name__ == "__main__":
    for pmin in (200, 400, 600, 1000):
        p, b, r = find_curve(pmin)
        C = Curve(p, b, r)
        print(f"p={p} b={b} r={r} G={C.G} beta={C.beta} lam={C.lam} lam^3 mod r={pow(C.lam,3,r)}")
