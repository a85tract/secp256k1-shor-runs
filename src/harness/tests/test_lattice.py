"""Checks of lattice.py: LLL output spans the same lattice (same Gram determinant, unimodular change checked by
expressing old rows in the new basis), is size-reduced and satisfies Lovász; integer Babai equals a reference
Babai with exact Fractions. Run: python3 harness/tests/test_lattice.py"""
import os, random, sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import lattice  # noqa: E402


def gso(rows):
    bs, norms, mu = [], [], []
    for r in rows:
        v = [Fraction(x) for x in r]
        m = []
        for u, nu in zip(bs, norms):
            c = sum(Fraction(a) * b for a, b in zip(r, u)) / nu
            m.append(c)
            v = [a - c * x for a, x in zip(v, u)]
        bs.append(v)
        norms.append(sum(x * x for x in v))
        mu.append(m)
    return bs, norms, mu


def ref_babai(rows, target):
    bs, norms, _ = gso(rows)
    w = list(target)
    coeffs = [0] * len(rows)
    for i in reversed(range(len(rows))):
        c = round(sum(Fraction(a) * b for a, b in zip(w, bs[i])) / norms[i])
        coeffs[i] = c
        w = [a - c * x for a, x in zip(w, rows[i])]
    return coeffs, [t - x for t, x in zip(target, w)]


def solve_int(rows, v):
    """Coefficients x with x B = v (exact), or None if not integral."""
    n = len(rows)
    M = [[Fraction(rows[i][j]) for i in range(n)] + [Fraction(v[j])] for j in range(len(v))]
    # least-squares-free: B is square here
    for col in range(n):
        piv = next(r for r in range(col, n) if M[r][col] != 0)
        M[col], M[piv] = M[piv], M[col]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col] / M[col][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    x = [M[i][n] / M[i][i] for i in range(n)]
    return x if all(xi.denominator == 1 for xi in x) else None


def main():
    rng = random.Random(7)
    ok = True
    for trial in range(40):
        n = rng.randrange(2, 9)
        bits = rng.choice((8, 40, 120, 300))
        rows = [[rng.randrange(-2 ** bits, 2 ** bits) for _ in range(n)] for _ in range(n)]
        if trial % 2:   # HNP-shaped
            q = rng.randrange(2 ** bits, 2 ** (bits + 1))
            rows = [[rng.randrange(q) for _ in range(n - 1)] + [1]] + \
                   [[q if i == j else 0 for j in range(n)] for i in range(n - 1)]
        red, _ = lattice.lll(rows)
        bs, norms, mu = gso(red)
        size_red = all(abs(m) <= Fraction(1, 2) for row in mu for m in row)
        lovasz = all(norms[k] >= (Fraction(99, 100) - mu[k][k - 1] ** 2) * norms[k - 1] for k in range(1, n))
        same = all(solve_int(red, r) is not None for r in rows) and all(solve_int(rows, r) is not None for r in red)
        target = [rng.randrange(-2 ** (bits + 2), 2 ** (bits + 2)) for _ in range(n)]
        bab = lattice.babai(red, target)
        ref = ref_babai(red, target)
        good = size_red and lovasz and same and bab == ref
        ok &= good
        if not good:
            print('FAIL trial', trial, n, bits, size_red, lovasz, same, bab == ref)
    print('lattice checks:', 'ALL OK' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
