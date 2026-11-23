"""Mixed-radix Eisenstein exponent registers for j = 0 curves (Run.mixreg): construction, o1/mixrad.py (o1

A window i holds a residue of Z[w] modulo an Eisenstein integer pi_i (w = omega acts on points as [lam]); it adds the
digit d_i = digit_i(residue), an element of a fixed complete residue system mod pi_i, at position Pi_i = pi_0 ...
pi_(i-1) (lowest window first), so the register adds [s] P with s = sum_i Pi_i d_i, and the register's states form a
complete residue system of Z[w] / (Pi_m). The hex windows are pi = 2^W. The norm-7 window type is pi = q 2^w with
q = 3 + w (norm 7) or its conjugate 2 - w: Z[w] / (pi) = F_7 x (Z / 2^w)^2 (CRT), residue (c, x), c = phi7(z),
x = z mod 2^w. The six units map onto F_7^* bijectively, so for c != 0 the unit of the orbit is fixed by c alone:
the orbit of (c, x) has the representative (1, u^-1 x), u the unit with phi7(u) = c, and x' = u^-1 x is free
(4^w representatives). For c = 0 the orbits are those of the hex window W = w. Representatives:
K = 4^w + (4^w - 4) / 6 + 2 (19,116 at w = 7, against 10,924 for W = 8 and 43,692 for W = 9), about N / 6.

Digits ("rosette" lift): rep(c0, x) = h(x) + 2^w kappa, h the hex Voronoi lift of x mod 2^w (hexreduce of the centred
residue, h(0) = 2^w as in the circuit's digit register) and kappa in {0} u units the element with
phi7(h + 2^w kappa) = c0; then digit(c, x) = u rep(1, x') for c != 0, digit(0, x) = u rep(0, x'') with
(x'', s, j) = hexreg.convert(x) and u = (-1)^s w^j, and the zero residue gets the digit pi itself (so the adder never
adds O). Every digit is congruent to its residue.

Exact sampler. Characters of Z[w] / (pi) are e(<E, d>) with E in M_pi^(-T) Z^2 / Z^2, M_z the matrix of
multiplication by z in the basis (1, w). With the Smith form U M_pi V = diag(e1, e2) a residue d has the index
rho = U d mod (e1, e2), and E = U^T diag(1 / e) y gives <E, d> = y . rho / e, so each conditional is one (e1, e2)
FFT. The overall outcome eta (functional on s, mod Z^2) peaks at uc (1, lam) / r; eta_(i) = M_(Pi_i)^T eta satisfies
M_(pi_i)^T eta_(i) = eta_(i+1), so window i (taken from the top down) sees base_i = M_(pi_i)^(-T) eta_(i+1), gets the
correction e(-<base_i, d>), and is measured in the Fourier basis of Z[w] / (pi_i):
P(y | earlier) = |FFT_rho[amp e(<g0_i - base_i, d(rho)>)](y)|^2 / N_i, g0_i = uc phi_r(Pi_i) (1, lam) / r.
The top window may be the sine-shaped box (digits s + t w, s, t < 2^W, amplitudes sine(s) sine(t)), pi = 2^W.

Layout syntax (parse_layout, lowest first): hW = hex window W, nw / mw = norm-7 window (3 + w) 2^w / (2 - w) 2^w,
sW = sine box top (last only); 'x k' repeats a token, e.g. 'h9,n7,m7x10,h8x3,s4'.
"""
import math

import numpy as np

import hexreg
from fractions import Fraction


# ------------------------------------------------------------------ Eisenstein integers z = a + b w
def emul(z, t):
    a, b = z
    c, d = t
    return (a * c - b * d, a * d + b * c - b * d)


def epow(z, k):
    r = (1, 0)
    for _ in range(k):
        r = emul(r, z)
    return r


def norm(z):
    a, b = z
    return a * a - a * b + b * b


def mat(z):
    """multiplication by z on column vectors (c, d) = c + d w"""
    a, b = z
    return ((a, -b), (b, a - b))


UNITS = [(1, 0), (0, 1), (-1, -1), (-1, 0), (0, -1), (1, 1)]     # (-1)^s w^j at index 3 s + j


def unit_sj(s, j):
    return UNITS[3 * s + j]


def einv_unit(u):
    return {(1, 0): (1, 0), (0, 1): (-1, -1), (-1, -1): (0, 1), (-1, 0): (-1, 0), (0, -1): (1, 1),
            (1, 1): (0, -1)}[u]


def phi_r(z, r, lam):
    return (z[0] + z[1] * lam) % r


# ------------------------------------------------------------------ Smith normal form of a 2 x 2 integer matrix
def snf2(M):
    """U M V = diag(e1, e2), e1 | e2, U, V unimodular (lists of lists); returns U, V, (e1, e2)"""
    A = [list(M[0]), list(M[1])]
    U = [[1, 0], [0, 1]]
    V = [[1, 0], [0, 1]]

    def rowop(i, k, f):                  # row i -= f row k
        A[i] = [A[i][t] - f * A[k][t] for t in range(2)]
        U[i] = [U[i][t] - f * U[k][t] for t in range(2)]

    def colop(j, k, f):                  # col j -= f col k
        for t in range(2):
            A[t][j] -= f * A[t][k]
            V[t][j] -= f * V[t][k]

    def rowswap():
        A[0], A[1] = A[1], A[0]
        U[0], U[1] = U[1], U[0]

    def colswap():
        for t in range(2):
            A[t][0], A[t][1] = A[t][1], A[t][0]
            V[t][0], V[t][1] = V[t][1], V[t][0]
    while True:
        # bring the smallest nonzero entry to (0, 0)
        best = None
        for i in range(2):
            for j in range(2):
                if A[i][j] and (best is None or abs(A[i][j]) < abs(A[best[0]][best[1]])):
                    best = (i, j)
        if best is None:
            break
        if best[0] == 1:
            rowswap()
        if best[1] == 1:
            colswap()
        p = A[0][0]
        done = True
        if A[1][0]:
            rowop(1, 0, A[1][0] // p)
            done = done and A[1][0] == 0
        if A[0][1]:
            colop(1, 0, A[0][1] // p)
            done = done and A[0][1] == 0
        if not done:
            continue
        if A[1][1] % p:
            # add row 1 to row 0 and continue
            A[0] = [A[0][t] + A[1][t] for t in range(2)]
            U[0] = [U[0][t] + U[1][t] for t in range(2)]
            continue
        break
    if A[0][0] < 0:
        A[0] = [-v for v in A[0]]
        U[0] = [-v for v in U[0]]
    if A[1][1] < 0:
        A[1] = [-v for v in A[1]]
        U[1] = [-v for v in U[1]]
    assert A[0][1] == 0 and A[1][0] == 0 and A[1][1] % A[0][0] == 0
    return U, V, (A[0][0], A[1][1])


def matvec(M, v):
    return (M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1])


def inv2(M):
    """inverse of a unimodular 2 x 2 integer matrix"""
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    assert det in (1, -1)
    return [[M[1][1] * det, -M[0][1] * det], [-M[1][0] * det, M[0][0] * det]]


# ------------------------------------------------------------------ windows
class Window:
    """pi (Eisenstein integer), digit(z) for a representative z of a residue, amp (None = uniform)"""
    kind = 'generic'

    def setup(self):
        self.N = norm(self.pi)
        self.U, self.V, self.e = snf2(mat(self.pi))
        assert self.e[0] * self.e[1] == self.N
        self.Uinv = inv2(self.U)
        self._arr = None

    def rho(self, d):
        v = matvec(self.U, d)
        return (v[0] % self.e[0], v[1] % self.e[1])

    def arrays(self):
        """d1, d2 (int64, shape e) of the digit of residue rho, and the amplitude array (or None)"""
        if self._arr is None:
            e1, e2 = self.e
            d1 = np.zeros((e1, e2), dtype=np.int64)
            d2 = np.zeros((e1, e2), dtype=np.int64)
            for a in range(e1):
                for b in range(e2):
                    z = matvec(self.Uinv, (a, b))
                    d = self.digit(z)
                    d1[a, b], d2[a, b] = d
            self._arr = (d1, d2, self.amp_array())
        return self._arr

    def amp_array(self):
        return None

    def check(self):
        """complete residue system: every digit is congruent to its residue (and so all are distinct mod pi)"""
        d1, d2, _ = self.arrays()
        e1, e2 = self.e
        bad = 0
        for a in range(e1):
            for b in range(e2):
                if self.rho((int(d1[a, b]), int(d2[a, b]))) != (a, b):
                    bad += 1
        return bad


class HexWindow(Window):
    kind = 'hex'

    def __init__(self, W):
        self.W = W
        self.pi = (1 << W, 0)
        self.setup()

    def digit(self, z):
        M = 1 << self.W
        return hexreg.digit(z[0] % M, z[1] % M, self.W)

    def nreps(self):
        return (4 ** self.W - 4) // 6 + 2


class SineTop(Window):
    """the top window as a box s + t w (s, t < 2^W) with sine amplitudes (read only by the exact first lookup)"""
    kind = 'sine'

    def __init__(self, W):
        self.W = W
        self.pi = (1 << W, 0)
        self.setup()

    def digit(self, z):
        M = 1 << self.W
        return (z[0] % M, z[1] % M)

    def amp_array(self):
        a = hexreg.sine(self.W)
        A = a[:, None] * a[None, :]
        return A / math.sqrt((A * A).sum())


class N7Window(Window):
    """pi = q 2^w, q = 3 + w (bar = False) or 2 - w (bar = True), norm 7 4^w; rosette digits (module doc)"""
    kind = 'n7'

    def __init__(self, w, bar=False):
        self.w, self.bar = w, bar
        self.q = (2, -1) if bar else (3, 1)
        self.pi = emul(self.q, (1 << w, 0))
        self.om = 2 if bar else 4                 # image of w in F_7 = Z[w] / (q)
        self.setup()
        self.uof = {}
        for idx, u in enumerate(UNITS):
            self.uof[self.phi7(u)] = idx
        assert sorted(self.uof) == [1, 2, 3, 4, 5, 6]
        self.inv2w = pow(1 << w, -1, 7)

    def phi7(self, z):
        return (z[0] + self.om * z[1]) % 7

    def split(self, z):
        M = 1 << self.w
        return self.phi7(z), (z[0] % M, z[1] % M)

    def hlift(self, x):
        """hex lift of x mod 2^w as the circuit's digit register computes it (o1/hexwin_add.py phase_digit):
        hexreduce of the centred residue, and (2^w, 0) for x = 0"""
        M = 1 << self.w
        if x[0] % M == 0 and x[1] % M == 0:
            return (M, 0)
        return hexreg.hexreduce(hexreg.centered(x[0] % M, M), hexreg.centered(x[1] % M, M), M)

    def rep(self, c0, x):
        """rosette lift of the residue (c0, x): h(x) + 2^w kappa"""
        h = self.hlift(x)
        need = (c0 - self.phi7(h)) * self.inv2w % 7
        if need == 0:
            return h
        k = UNITS[self.uof[need]]
        return (h[0] + (k[0] << self.w), h[1] + (k[1] << self.w))

    def orbit(self, z):
        """orbit form of the residue of z: (zflag, s, j, xrep) with residue = (-1)^s w^j (zflag ? (0, xrep) : (1, xrep))"""
        M = 1 << self.w
        c, x = self.split(z)
        if c:
            idx = self.uof[c]
            s, j = divmod(idx, 3)
            ui = einv_unit(UNITS[idx])
            xp = emul(ui, x)
            return 0, s, j, (xp[0] % M, xp[1] % M)
        v, s, j, x2 = hexreg.convert(x[0], x[1], self.w)
        return 1, s, j, x2

    def rep_digit(self, zflag, xrep):
        """digit of the orbit representative (the table entry is [Pi_i rep_digit] P)"""
        if zflag:
            if xrep == (0, 0):
                return self.pi
            return self.rep(0, xrep)
        return self.rep(1, xrep)

    def digit(self, z):
        zf, s, j, xr = self.orbit(z)
        return emul(unit_sj(s, j), self.rep_digit(zf, xr))

    def reps(self):
        M = 1 << self.w
        out = [(0, (a, b)) for a in range(M) for b in range(M)]
        hx = sorted({hexreg.convert(a, b, self.w)[3] for a in range(M) for b in range(M)})
        out += [(1, x) for x in hx]
        return out

    def nreps(self):
        return 4 ** self.w + (4 ** self.w - 4) // 6 + 2


def parse_layout(spec):
    """'h9,n7x13,h8,s4' (lowest first): hN = hex W = N, n7 / m7 = N7Window(7, bar False / True) (an 'x' count
    repeats), sN = sine box top W = N (must be last)"""
    out = []
    for tok in spec.split(','):
        tok = tok.strip()
        cnt = 1
        if 'x' in tok:
            tok, cnt = tok.split('x')
            cnt = int(cnt)
        kind, val = tok[0], int(tok[1:])
        for _ in range(cnt):
            if kind == 'h':
                out.append(HexWindow(val))
            elif kind == 'n':
                out.append(N7Window(val, False))
            elif kind == 'm':
                out.append(N7Window(val, True))
            elif kind == 's':
                out.append(SineTop(val))
            else:
                raise ValueError(tok)
    return out


_WCACHE = {}


def cached(spec):
    """windows with shared digit arrays (same type and size share one instance)"""
    out = []
    for w in parse_layout(spec):
        key = (w.kind, getattr(w, 'W', None), getattr(w, 'w', None), getattr(w, 'bar', None))
        if key not in _WCACHE:
            _WCACHE[key] = w
        out.append(_WCACHE[key])
    return out


def positions(wins):
    P = [(1, 0)]
    for w in wins:
        P.append(emul(P[-1], w.pi))
    return P            # P[i] = Pi_i, P[-1] = Pi_m


def log2norm(wins):
    return sum(math.log2(w.N) for w in wins)


# ------------------------------------------------------------------ exact sampler
def _mT_inv_apply(z, v):
    """M_z^(-T) v for a rational 2-vector v (Fractions)"""
    (a, b), (c, d) = mat(z)                 # M = [[a, b], [c, d]]; M^T = [[a, c], [b, d]]
    det = a * d - b * c
    # (M^T)^-1 = 1/det [[d, -c], [-b, a]]
    return (Fraction(d * v[0] - c * v[1], det), Fraction(-b * v[0] + a * v[1], det))


def _mT_apply(z, v):
    (a, b), (c, d) = mat(z)
    return (a * v[0] + c * v[1], b * v[0] + d * v[1])


def _frac1(x):
    return x - math.floor(x)


def cond(win, g):
    """P(y) over the (e1, e2) outcome grid for phase functional g (floats mod 1)"""
    d1, d2, amp = win.arrays()
    ph = d1 * g[0] + d2 * g[1]
    ph -= np.round(ph)
    A = np.exp(2j * np.pi * ph)
    if amp is not None:
        A = A * amp
        return np.abs(np.fft.fft2(A)) ** 2 / float(win.N)
    return np.abs(np.fft.fft2(A)) ** 2 / float(win.N) ** 2


def sample(rng, wins, uc, r, lam, want_logp=False):
    """outcome eta (pair of Fractions in [0, 1)) of a register with windows `wins` (lowest first) given uc"""
    P = positions(wins)
    eta_up = (Fraction(0), Fraction(0))
    logp = 0.0
    for i in range(len(wins) - 1, -1, -1):
        w = wins[i]
        base = _mT_inv_apply(w.pi, eta_up)
        L = uc * phi_r(P[i], r, lam) % r
        g = (float(_frac1(Fraction(L, r) - base[0])), float(_frac1(Fraction(L * lam % r, r) - base[1])))
        Pr = cond(w, g).ravel()
        cdf = np.cumsum(Pr)
        idx = min(cdf.size - 1, int(np.searchsorted(cdf, rng.random() * cdf[-1], side='right')))
        if want_logp:
            logp += math.log(float(Pr[idx]) / float(cdf[-1]))
        y = divmod(idx, w.e[1])
        # E = U^T diag(1/e) y
        f = (Fraction(y[0], w.e[0]), Fraction(y[1], w.e[1]))
        E = (w.U[0][0] * f[0] + w.U[1][0] * f[1], w.U[0][1] * f[0] + w.U[1][1] * f[1])
        eta_up = (_frac1(base[0] + E[0]), _frac1(base[1] + E[1]))
    return (eta_up, logp) if want_logp else eta_up


def logprob(eta, wins, uc, r, lam):
    """log P(eta) under the sampler (the product of its conditionals)"""
    P = positions(wins)
    m = len(wins)
    etas = [None] * (m + 1)
    for i in range(m + 1):
        v = _mT_apply(P[i], eta)
        etas[i] = (_frac1(Fraction(v[0])), _frac1(Fraction(v[1])))
    lp = 0.0
    for i in range(m - 1, -1, -1):
        w = wins[i]
        base = _mT_inv_apply(w.pi, etas[i + 1])
        E = (_frac1(etas[i][0] - base[0]), _frac1(etas[i][1] - base[1]))
        # y = diag(e) U^(-T) E mod e
        Ui = w.Uinv
        f = (Ui[0][0] * E[0] + Ui[1][0] * E[1], Ui[0][1] * E[0] + Ui[1][1] * E[1])
        y = (f[0] * w.e[0], f[1] * w.e[1])
        assert y[0].denominator == 1 and y[1].denominator == 1, 'eta is not a character of the register'
        y = (int(y[0]) % w.e[0], int(y[1]) % w.e[1])
        L = uc * phi_r(P[i], r, lam) % r
        g = (float(_frac1(Fraction(L, r) - base[0])), float(_frac1(Fraction(L * lam % r, r) - base[1])))
        p = cond(w, g)[y]
        if p <= 0:
            return -math.inf
        lp += math.log(p)
    return lp


# ------------------------------------------------------------------ h1: labels and counts for the cost model
def label(win):
    """window label of kinds and of --global-mc window lists: W for hex and sine windows (as Run.hexreg), 'nw' / 'mw'
    for the norm-7 windows (3 + w) 2^w / (2 - w) 2^w"""
    if win.kind in ('hex', 'sine'):
        return win.W
    return ('m' if win.bar else 'n') + str(win.w)


def win_qubits(lab):
    """register qubits of a window with label lab: 2W for hex and sine windows, 2w + 3 for norm-7 windows (residue
    (c, x): c in 3 qubits, x in 2w)"""
    return 2 * lab if isinstance(lab, int) else 2 * int(lab[1:]) + 3


def win_entries(lab):
    """table entries (orbit representatives) of a window that an addition reads: (4^W - 4) / 6 + 2 for hex,
    4^w + (4^w - 4) / 6 + 2 for norm-7"""
    if isinstance(lab, int):
        return (4 ** lab - 4) // 6 + 2
    w = int(lab[1:])
    return 4 ** w + (4 ** w - 4) // 6 + 2


def check_label(lab):
    """a valid window label: an int W in 1..10, or 'nw' / 'mw' with w in 1..8"""
    if isinstance(lab, int):
        return 1 <= lab <= 10
    return isinstance(lab, str) and len(lab) >= 2 and lab[0] in 'nm' and lab[1:].isdigit() and 1 <= int(lab[1:]) <= 8


def labels(spec):
    """the labels of a layout's windows that are additions (lowest first; a sine top is in the first lookup)"""
    wins = parse_layout(spec)
    if wins and wins[-1].kind == 'sine':
        wins = wins[:-1]
    return [label(w) for w in wins]


def register_norm(wins):
    """N = norm(Pi_m) of the whole register: outcomes are (Y1, Y2) = N eta"""
    return norm(positions(wins)[-1])


def sample_Y(rng, wins, c, u, r, lam):
    """outcome (Y1, Y2) = N eta of a register with windows wins that added [s] (point with dlog c), given the Fourier
    index u, and its exact probability"""
    eta, lp = sample(rng, wins, u * c % r, r, lam, want_logp=True)
    N = register_norm(wins)
    Y = (eta[0] * N, eta[1] * N)
    assert Y[0].denominator == 1 and Y[1].denominator == 1
    return (int(Y[0]), int(Y[1])), math.exp(lp)
