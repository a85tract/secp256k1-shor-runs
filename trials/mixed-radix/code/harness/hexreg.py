"""Hex (unit-orbit) exponent registers for j = 0 curves: digits and the exact sampler, after o1/hexva.py and

Window i (lowest first) has W_i and holds a residue x_i in (Z / 2^W_i)^2; it adds the digit delta_i = digit(x_i,
W_i), the Eisenstein integer of minimal norm congruent to x_i mod 2^W_i (w = omega acts on points as [lam]), at
position P_i = W_0 + ... + W_(i-1). The register adds [s1 + s2 lam] P with s1 + s2 w = sum_i delta_i 2^P_i.

Exact sampling (o1/hexva.py module doc, with per-window sizes). Given the Fourier index u, window i is in the state
(1 / M_i) sum_x e(theta_i(delta(x))) |x>, M_i = 2^W_i, theta_i(d) = u c (d1 + d2 lam) 2^P_i / r. The semiclassical
measurement takes the windows from the top down; before measuring window i (step t) it applies the correction
e(-<Y_low, delta(x)> / 2^(Q_t + W_i)), Y_low = sum_{s < t} y_s 2^(Q_s) the outcomes so far, Q_t = sum of the sizes
of the windows above i, and measures in the Fourier basis of (Z / M_i)^2 (orthonormal: the digits are a complete
residue system mod M_i). So P(y_t | y_<t) = |FFT2_x[e(theta_i(delta(x)) - <Y_low, delta(x)> / 2^(Q_t + W_i))](y_t)|^2
/ M_i^4, and the outcome is Y = sum_t y_t 2^(Q_t) in [0, 2^P)^2, P = sum W_i, peaked at Y / 2^P = (u c / r,
"""
from fractions import Fraction
import numpy as np


# ---- digits (o1/hexdigits.py, unchanged)
def mulw(u, v):
    """w (u + v w) = -v + (u - v) w"""
    return (-v, u - v)


def unit(s, j, d):
    for _ in range(j):
        d = mulw(*d)
    return (-d[0], -d[1]) if s else d


def centered(a, M):
    return a - M if a >= M // 2 else a


def hexreduce(A, B, M):
    p = 2 * A - B > M or 2 * B - A < -M
    q = 2 * A - B < -M or 2 * B - A > M
    if p:
        if A + B >= 0:
            A -= M
        else:
            B += M
    elif q:
        if A + B <= 0:
            A += M
        else:
            B -= M
    return A, B


def convert(a, b, W):
    M = 1 << W
    if a == 0 and b == 0:
        return W, 0, 0, (0, 0)
    v = 0
    while not ((a >> v) & 1 or (b >> v) & 1):
        v += 1
    j = {(1, 0): 0, (0, 1): 1, (1, 1): 2}[((a >> v) & 1, (b >> v) & 1)]
    x1 = (a, b)
    for _ in range((3 - j) % 3):
        x1 = mulw(*x1)
    x1 = (x1[0] % M, x1[1] % M)
    s = (x1[0] >> (v + 1)) & 1 if v + 1 < W else 0
    x2 = ((-x1[0]) % M, (-x1[1]) % M) if s else x1
    return v, s, j, x2


def digit(a, b, W):
    M = 1 << W
    v, s, j, x2 = convert(a, b, W)
    lift = (M, 0) if x2 == (0, 0) else hexreduce(centered(x2[0], M), centered(x2[1], M), M)
    return unit(s, j, lift)


_CACHE = {}


def digit_arrays(W):
    """d1, d2: int64 arrays (M, M), digit of residue (a, b) = d1[a, b] + d2[a, b] w"""
    if W not in _CACHE:
        M = 1 << W
        d1 = np.zeros((M, M), dtype=np.int64)
        d2 = np.zeros((M, M), dtype=np.int64)
        for a in range(M):
            for b in range(M):
                d1[a, b], d2[a, b] = digit(a, b, W)
        _CACHE[W] = (d1, d2)
    return _CACHE[W]


# ---- sampler
def sine(W):
    """the top window's envelope with top='sine': sin(pi (s + 1/2) / 2^W), s < 2^W (harness/sineprep.py)"""
    return np.sin(np.pi * (np.arange(1 << W) + 0.5) / (1 << W))


def cond(Ws, t, uc, r, lam, Y1, Y2, top=None, split=0):
    """P(y_t | y_<t) as an (M, M) array [y1, y2] for window i = len(Ws) - 1 - t (module doc). top='sine': the top
    window (t = 0) holds box digits s + t w, s, t < 2^W, with amplitudes sine(W)[s] sine(W)[t] (a complete residue
    system too, so the lower windows are unchanged; only an exact first lookup can read such a window).
    split = P1 - P2 (top='sine' only; h1/hex_sigma3.py): the top box has 2^(W + max(split, 0)) x 2^(W + max(-split, 0))
    digits, so the register has 2 sum(Ws) + |split| bits (odd sigma); each coordinate's denominator counts its own
    bits above window i, and the top window's array is (M1, M2)."""
    if split and top != 'sine':
        raise ValueError('split needs the sine box top window')
    i = len(Ws) - 1 - t
    W = Ws[i]
    e1, e2 = max(split, 0), max(-split, 0)
    L1 = uc * pow(2, sum(Ws[:i]), r) % r
    L2 = L1 * lam % r
    den = 2 ** (sum(Ws[i + 1:]) + W)
    g1 = float(Fraction(L1, r) - Fraction(Y1, den << e1))      # exact rational, rounded once to float64
    g2 = float(Fraction(L2, r) - Fraction(Y2, den << e2))
    g1 -= round(g1)
    g2 -= round(g2)
    if top == 'sine' and t == 0:
        p1, p2 = (np.abs(np.fft.fft(a * np.exp(2j * np.pi * ((np.arange(a.size) * g) % 1.0)))) ** 2 /
                  (a.size * (a * a).sum()) for a, g in ((sine(W + e1), g1), (sine(W + e2), g2)))
        return p1[:, None] * p2[None, :]
    d1, d2 = digit_arrays(W)
    ph = d1 * g1 + d2 * g2
    ph -= np.round(ph)
    return np.abs(np.fft.fft2(np.exp(2j * np.pi * ph))) ** 2 / float(1 << W) ** 4


def sample(rng, Ws, c, u, r, lam, top=None, split=0):
    """Outcome (Y1, Y2) of a hex register with window sizes Ws (lowest first) that added [s] (point with dlog c),
    given the Fourier index u, and its exact probability (top, split: see cond; Y1 < 2^(sum Ws + max(split, 0)),
    Y2 < 2^(sum Ws + max(-split, 0)))."""
    Y1 = Y2 = 0
    Q1 = Q2 = 0
    uc = u * c % r
    prob = 1.0
    for t in range(len(Ws)):
        P = cond(Ws, t, uc, r, lam, Y1, Y2, top, split)
        cdf = np.cumsum(P.ravel())
        idx = min(cdf.size - 1, int(np.searchsorted(cdf, rng.random() * cdf[-1], side='right')))
        prob *= float(P.ravel()[idx]) / float(cdf[-1])
        y1, y2 = divmod(idx, P.shape[1])
        Y1 += y1 << Q1
        Y2 += y2 << Q2
        Q1 += P.shape[0].bit_length() - 1
        Q2 += P.shape[1].bit_length() - 1
    return (Y1, Y2), prob


# ---- the first lookup of a run that starts at the identity (estimate grade)
def first_ss_toffoli(bits, n):
    """o1/first_ss.py (V-B PASS at 4-12 address bits, count = this formula there): the random-offset first lookup
    as a select-swap load of the 2^bits-entry table T0[k] + R0 into the 2n-qubit accumulator plus the phase fix-up
    of the select register: 2^(bits-1) - 2 + 2n + f(bits - 1), f(w) = 2^(w//2) - 2 + 2^(w - w//2) - 2 + f(w//2)
    for w > 2, else 2^w - 2. Peak bits + 4n + (bits - 2)."""
    def f(w):
        return 2 ** w - 2 if w <= 2 else 2 ** (w // 2) - 2 + 2 ** (w - w // 2) - 2 + f(w // 2)
    return dict(toffoli=2 ** (bits - 1) - 2 + 2 * n + f(bits - 1), peak_qubits=bits + 4 * n + bits - 2)
