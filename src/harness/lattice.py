"""Exact lattice tools for post-processing: integral LLL (Cohen, Alg. 2.6.7) and Babai's nearest plane."""
from fractions import Fraction


def _dot(x, y):
    return sum(p * q for p, q in zip(x, y))


def lll(rows, delta=0.99, max_swaps=None):
    """LLL-reduce linearly independent integer row vectors. Returns (reduced rows, number of swaps).
    A floating-point pass (exact basis, float Gram-Schmidt from exact dot products) does the bulk of the work;
    the exact integral LLL then finishes and certifies the result, so the output is exactly LLL-reduced.
    max_swaps: the reduction stops inside its loops as soon as one more swap would pass this many (both passes
    together) and returns (None, swaps done); the harness derives it from a decoding attempt's work cap."""
    pre, s1 = _fp_lll([list(map(int, r)) for r in rows], delta, max_swaps)
    if pre is None:
        return None, s1
    out, s2 = _exact_lll(pre, delta, None if max_swaps is None else max_swaps - s1)
    return out, s1 + s2


def _fp_lll(b, delta, max_swaps=None):
    n = len(b)
    if n < 2:
        return b, 0
    G = [[_dot(x, y) for y in b] for x in b]          # exact Gram matrix, updated with the basis
    # Gram-Schmidt data scale uniformly with the dot products: shift them so floats cannot overflow
    sh = max(0, max(abs(G[i][i]) for i in range(n)).bit_length() + n.bit_length() - 600)
    mu = [[0.0] * n for _ in range(n)]
    rr = [[0.0] * n for _ in range(n)]

    def row(k):
        Gk = G[k]
        for j in range(k):
            v = float(Gk[j] >> sh) - sum(mu[j][i] * rr[k][i] for i in range(j))
            rr[k][j] = v
            mu[k][j] = v / rr[j][j]
        rr[k][k] = float(Gk[k] >> sh) - sum(mu[k][j] * rr[k][j] for j in range(k))

    def sub(k, j, q):                                  # b_k -= q b_j, with the Gram matrix
        b[k] = [x - q * y for x, y in zip(b[k], b[j])]
        Gk, Gj = G[k], G[j]
        gkk = Gk[k] - 2 * q * Gk[j] + q * q * Gj[j]
        for i in range(n):
            if i != k:
                Gk[i] -= q * Gj[i]
                G[i][k] = Gk[i]
        Gk[k] = gkk

    row(0)
    k, swaps, guard = 1, 0, 0
    while k < n:
        guard += 1
        if guard > 200000:
            break
        for _ in range(60):
            row(k)
            if all(abs(mu[k][j]) <= 0.51 for j in range(k)):
                break
            for j in reversed(range(k)):
                q = round(mu[k][j])
                if q:
                    sub(k, j, q)
                    for i in range(j):
                        mu[k][i] -= q * mu[j][i]
                    mu[k][j] -= q
        if rr[k][k] <= 0 or rr[k][k] < (delta - mu[k][k - 1] ** 2) * rr[k - 1][k - 1]:
            if max_swaps is not None and swaps >= max_swaps:
                return None, swaps                      # work budget reached: stop here
            b[k], b[k - 1] = b[k - 1], b[k]
            G[k], G[k - 1] = G[k - 1], G[k]
            for Gi in G:
                Gi[k], Gi[k - 1] = Gi[k - 1], Gi[k]
            swaps += 1
            k = max(k - 1, 1)
            row(k - 1)
        else:
            k += 1
    return b, swaps


def _exact_lll(rows, delta=0.99, max_swaps=None):
    """Integral LLL (Cohen, Alg. 2.6.7), exact."""
    b = [None] + [list(map(int, r)) for r in rows]          # 1-indexed as in Cohen
    n = len(rows)
    num, den = Fraction(delta).limit_denominator(1000).as_integer_ratio()
    d = [1] + [0] * n
    lam = [[0] * (n + 1) for _ in range(n + 1)]
    d[1] = _dot(b[1], b[1])
    k, kmax, swaps = 2, 1, 0

    def redi(k, l):
        if 2 * abs(lam[k][l]) <= d[l]:
            return
        q = (2 * lam[k][l] + d[l]) // (2 * d[l])
        bl, bk = b[l], b[k]
        b[k] = [x - q * y for x, y in zip(bk, bl)]
        lam[k][l] -= q * d[l]
        for i in range(1, l):
            lam[k][i] -= q * lam[l][i]

    def swapi(k):
        b[k], b[k - 1] = b[k - 1], b[k]
        for j in range(1, k - 1):
            lam[k][j], lam[k - 1][j] = lam[k - 1][j], lam[k][j]
        lm = lam[k][k - 1]
        B = (d[k - 2] * d[k] + lm * lm) // d[k - 1]
        for i in range(k + 1, kmax + 1):
            t = lam[i][k]
            lam[i][k] = (d[k] * lam[i][k - 1] - lm * t) // d[k - 1]
            lam[i][k - 1] = (B * t + lm * lam[i][k]) // d[k]
        d[k - 1] = B

    while k <= n:
        if k > kmax:
            kmax = k
            for j in range(1, k + 1):
                u = _dot(b[k], b[j])
                for i in range(1, j):
                    u = (d[i] * u - lam[k][i] * lam[j][i]) // d[i - 1]
                if j < k:
                    lam[k][j] = u
                else:
                    d[k] = u
                    if u == 0:
                        raise ValueError('rows are linearly dependent')
        while True:
            redi(k, k - 1)
            if den * d[k] * d[k - 2] < num * d[k - 1] ** 2 - den * lam[k][k - 1] ** 2:
                if max_swaps is not None and swaps >= max_swaps:
                    return None, swaps                  # work budget reached: stop here
                swapi(k)
                swaps += 1
                k = max(2, k - 1)
            else:
                for l in range(k - 2, 0, -1):
                    redi(k, l)
                k += 1
                break
    return [row for row in b[1:]], swaps


def _integral_gso(b):
    """Cohen's integral Gram-Schmidt data for integer rows b (0-indexed): d[0..n] with d[0] = 1 and
    d[k] = prod_{j<=k} |b*_j|^2 * ..., lam[k][j] = d[j+1] mu_kj (all integers)."""
    n = len(b)
    d = [1] + [0] * n
    lam = [[0] * n for _ in range(n)]
    for k in range(n):
        for j in range(k + 1):
            u = _dot(b[k], b[j])
            for i in range(j):
                u = (d[i + 1] * u - lam[k][i] * lam[j][i]) // d[i]
            if j < k:
                lam[k][j] = u
            else:
                d[k + 1] = u
                if u == 0:
                    raise ValueError('rows are linearly dependent')
    return d, lam


def babai(rows, target):
    """Babai's nearest-plane algorithm, exact: size-reduce the target against the rows using integral
    Gram-Schmidt data (c_i = round(<w, b*_i> / |b*_i|^2) = round(lam_wi / d_i)).
    Returns (integer coefficients c, lattice vector sum_i c_i rows_i)."""
    b = [list(map(int, r)) for r in rows]
    n = len(b)
    d, lam = _integral_gso(b)
    w = list(map(int, target))
    lw = [0] * n                                   # lam_{w, j} = d_j <w, b*_j>  (Cohen's recurrence)
    for j in range(n):
        u = _dot(w, b[j])
        for i in range(j):
            u = (d[i + 1] * u - lw[i] * lam[j][i]) // d[i]
        lw[j] = u
    coeffs = [0] * n
    for l in reversed(range(n)):
        q = (2 * lw[l] + d[l + 1]) // (2 * d[l + 1])     # round(lw_l / d_{l+1})
        if q:
            coeffs[l] = q
            w = [x - q * y for x, y in zip(w, b[l])]
            lw[l] -= q * d[l + 1]
            for i in range(l):
                lw[i] -= q * lam[l][i]
    return coeffs, [t - x for t, x in zip(target, w)]
