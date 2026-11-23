"""the seven-state preparation and the mixed-radix Fourier measurement of a

A norm-7 window (harness/mixreg.py N7Window, pi = q 2^w, q = 3 + w or 2 - w) holds a residue of Z[w] / (pi) in the
residue form (c, xa, xb) of o1/n7win.py: c = phi7(z) in [0, 7) on 3 qubits (c0 c1 c2, LSB first), x = z mod 2^w on
2w qubits. Z[w] / (pi) = F_7 x (Z / 2^w)^2 (CRT), so its characters are
    chi_(k, a)(c, x) = e(k c / 7 + (a1 xa + a2 xb) / 2^w),   k in F_7, a in (Z / 2^w)^2,
and as a functional on digits d = d1 + d2 w (c = d1 + om d2 mod 7, om = 4 for 3 + w, 2 for 2 - w; x = d mod 2^w)
    E(k, a) = (k / 7 + a1 / 2^w, k om / 7 + a2 / 2^w) mod 1.
Circuits (gates: ry, rz, h, cx, cp = controlled phase; no Toffoli, no ancilla):
  PREP   = PREP7 on c (uniform over 0..6; Mottonen's tree of uniformly controlled Ry, real amplitudes) and H on x.
  MEASURE = QFT7^dagger on c (any unitary V with V chi_k = phase |k> will do, since c is measured right after:
           Givens rotations on index pairs differing in one bit, each a doubly controlled 2 x 2 unitary built from
           uniformly controlled Rz / Ry and a controlled phase, and the leftover diagonal dropped before the measurement)
           and the inverse QFT of Z / 2^w on xa and on xb (textbook circuit; its semiclassical form, controlled phases
           replaced by classically controlled ones after each measurement, gives the same outcome law).
Check (window w = 1, 2, 3 of both types): the prepared state is exactly uniform on the 7 4^w residues (and 0 on c = 7),
and for random phase functionals g the outcome law of MEASURE on sum_rho e(<g, d(rho)>) |c(rho), x(rho)> equals the
harness's law mixreg.cond(win, g) mapped to E by sample()'s E = U^T diag(1 / e) y, entry by entry.
usage: python3 -B h1/n7/n7reg_circuits.py [seed]"""
import cmath
import math
import os
import random
import sys
from fractions import Fraction

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'harness'))
import mixreg                              # noqa: E402

TOL = 1e-13


# ------------------------------------------------------------------ gates and a state-vector simulator
def ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(t):
    return np.diag([cmath.exp(-0.5j * t), cmath.exp(0.5j * t)])


H2 = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def apply(state, nq, gates):
    """apply a gate list (time order) to a state of nq qubits (qubit t = bit t of the index)"""
    psi = state.reshape([2] * nq)                 # axis nq - 1 - t is qubit t
    for g in gates:
        op = g[0]
        if op in ('ry', 'rz', 'h'):
            t = g[1]
            U = H2 if op == 'h' else (ry(g[2]) if op == 'ry' else rz(g[2]))
            psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [nq - 1 - t])), 0, nq - 1 - t)
        elif op == 'cx':
            c, t = g[1], g[2]
            sl = [slice(None)] * nq
            sl[nq - 1 - c] = 1
            sub = psi[tuple(sl)]
            ax = (nq - 1 - t) if t > c else (nq - 1 - t) - 1    # sub lost the control's axis
            psi = psi.copy()
            psi[tuple(sl)] = np.flip(sub, axis=ax)
        elif op == 'cp':                               # diag(1, 1, 1, e^(i theta)) on qubits (a, b)
            a, b, th = g[1], g[2], g[3]
            sl = [slice(None)] * nq
            sl[nq - 1 - a] = 1
            sl[nq - 1 - b] = 1
            psi = psi.copy()
            psi[tuple(sl)] *= cmath.exp(1j * th)
        else:
            raise ValueError(op)
    return psi.reshape(-1)


def unitary(nq, gates):
    return np.stack([apply(np.eye(2 ** nq, dtype=complex)[:, i].copy(), nq, gates) for i in range(2 ** nq)], axis=1)


def counts(gates):
    out = {}
    for g in gates:
        out[g[0]] = out.get(g[0], 0) + 1
    return out


# ------------------------------------------------------------------ building blocks (no Toffoli)
def ucr(axis, angles, controls, target):
    """uniformly controlled rotation R_axis(angles[j]) on target, j = sum_t bit(controls[t]) 2^t: R(a) CX R(b) CX on
    the last control with a = (alpha0 + alpha1) / 2, b = (alpha0 - alpha1) / 2 (X R(b) X = R(-b) for Ry and Rz)"""
    angles = np.asarray(angles, dtype=float)
    if not controls:
        return [(axis, target, float(angles[0]))] if abs(angles[0]) > TOL else []
    half = len(angles) // 2
    a0, a1 = angles[:half], angles[half:]
    if np.all(np.abs(a0 - a1) <= TOL):           # independent of the last control
        return ucr(axis, a0, controls[:-1], target)
    return (ucr(axis, (a0 + a1) / 2, controls[:-1], target) + [('cx', controls[-1], target)] +
            ucr(axis, (a0 - a1) / 2, controls[:-1], target) + [('cx', controls[-1], target)])


def diag_gate(phases, qubits):
    """diagonal gate e^(i phases[j]) on qubits (j = sum_t bit(qubits[t]) 2^t), up to a global phase: a uniformly
    controlled Rz on the last qubit, then the same on the rest"""
    phases = np.asarray(phases, dtype=float)
    if len(qubits) == 0:
        return []
    half = len(phases) // 2
    lo, hi = phases[:half], phases[half:]          # last qubit 0 / 1, indexed by the others
    # last qubit 0 / 1 -> e^(i lo), e^(i hi) = e^(i (lo + hi) / 2) Rz(hi - lo)
    return ucr('rz', hi - lo, qubits[:-1], qubits[-1]) + diag_gate((lo + hi) / 2, qubits[:-1])


def zyz(U):
    """U = e^(i alpha) Rz(beta) Ry(gamma) Rz(delta)"""
    det = np.linalg.det(U)
    alpha = cmath.phase(det) / 2
    V = U * cmath.exp(-1j * alpha)                 # det V = 1: V = [[a, -b*], [b, a*]]
    a, b = V[0, 0], V[1, 0]
    gamma = 2 * math.atan2(abs(b), abs(a))
    s, d = cmath.phase(a) if abs(a) > TOL else 0.0, cmath.phase(b) if abs(b) > TOL else 0.0
    # Rz(beta) Ry(gamma) Rz(delta) [0, 0] = e^(-i (beta + delta) / 2) cos, [1, 0] = e^(i (beta - delta) / 2) sin
    beta, delta = d - s, -(d + s)
    if abs(a) <= TOL:
        beta, delta = 2 * d, 0.0
    if abs(b) <= TOL:
        beta, delta = -2 * s, 0.0
    W = cmath.exp(1j * alpha) * rz(beta) @ ry(gamma) @ rz(delta)
    assert np.max(np.abs(W - U)) < 1e-11, (U, W)
    return alpha, beta, gamma, delta


def two_level(G, i, j, nq):
    """the 2 x 2 unitary G on basis states (|i>, |j>) (i, j differ in one bit t), identity elsewhere: G on qubit t,
    controlled by the other qubits holding i's bits"""
    t = (i ^ j).bit_length() - 1
    assert i ^ j == 1 << t
    if i & (1 << t):                               # i holds bit t = 1: order the pair as (|0>, |1>) of qubit t
        G = np.array([[G[1, 1], G[1, 0]], [G[0, 1], G[0, 0]]])
        i, j = j, i
    alpha, beta, gamma, delta = zyz(G)
    ctl = [q for q in range(nq) if q != t]
    pat = sum(((i >> q) & 1) << k for k, q in enumerate(ctl))
    sel = lambda x: [x if j2 == pat else 0.0 for j2 in range(2 ** len(ctl))]
    # time order: Rz(delta), Ry(gamma), Rz(beta), then the phase e^(i alpha) on the controls' pattern
    return (ucr('rz', sel(delta), ctl, t) + ucr('ry', sel(gamma), ctl, t) + ucr('rz', sel(beta), ctl, t) +
            diag_gate(sel(alpha), ctl))


def prep7():
    """|000> -> 7^(-1/2) sum_(c < 7) |c> on qubits (0, 1, 2) = (c0, c1, c2): P(c2 = 1) = 3 / 7; c1 given c2 (1 / 2, then
    1 / 3: only 6 of {4, 5, 6}); c0 given (c1, c2) (1 / 2, except 0 for c = 6)"""
    th = lambda p1: 2 * math.asin(math.sqrt(p1))       # Ry angle with P(1) = p1
    return ([('ry', 2, th(3 / 7))] + ucr('ry', [th(1 / 2), th(1 / 3)], [2], 1) +
            ucr('ry', [th(1 / 2), th(1 / 2), th(1 / 2), 0.0], [1, 2], 0))


def qft7_measure():
    """gates V with V chi_k = phase_k |k> (chi_k = 7^(-1/2) sum_c e(k c / 7) |c>), |7> -> |7>: Givens elimination of
    W = DFT7 (+) 1 by two-level rotations on one-bit pairs, G_m ... G_1 W = D, so W^dagger = D^dagger G_m ... G_1 and
    the circuit is G_1 .. G_m; D^dagger, a diagonal applied last, does not change a computational-basis measurement"""
    W = np.zeros((8, 8), dtype=complex)
    for c in range(7):
        for k in range(7):
            W[c, k] = cmath.exp(2j * math.pi * k * c / 7) / math.sqrt(7)    # column k = chi_k
    W[7, 7] = 1
    A = W.copy()
    gates = []
    for col in range(7):
        live = list(range(col, 8))
        order = _tree(col, live)
        for src, dst in order:
            a, b = A[dst, col], A[src, col]
            r = math.hypot(abs(a), abs(b))
            if abs(b) <= TOL or r <= TOL:
                continue
            G = np.array([[np.conj(a), np.conj(b)], [-b, a]]) / r              # G [a, b]^T = [r, 0]^T
            A[[dst, src], :] = G @ A[[dst, src], :]
            gates += two_level(G, dst, src, 3)
    off = A - np.diag(np.diag(A))
    assert np.max(np.abs(off)) < 1e-10, 'Givens elimination left off-diagonal %g' % np.max(np.abs(off))
    return gates


def _tree(col, live):
    """(src, dst) pairs differing in one bit, all inside `live`, that move every live row's weight to row col:
    breadth-first tree of the hypercube restricted to live, processed from the leaves"""
    live = set(live)
    parent, order, frontier = {col: None}, [col], [col]
    while frontier:
        nxt = []
        for v in frontier:
            for t in range(3):
                u = v ^ (1 << t)
                if u in live and u not in parent:
                    parent[u] = v
                    order.append(u)
                    nxt.append(u)
        frontier = nxt
    assert len(parent) == len(live), 'live rows not connected'
    return [(u, parent[u]) for u in reversed(order) if parent[u] is not None]


def qft2(qubits):
    """QFT of Z / 2^w: |a> -> 2^(-w/2) sum_x e(a x / 2^w) |x>, qubits LSB first (textbook circuit, then reversal)"""
    w = len(qubits)
    g = []
    for i in range(w - 1, -1, -1):
        g.append(('h', qubits[i]))
        for m in range(i - 1, -1, -1):
            g.append(('cp', qubits[m], qubits[i], 2 * math.pi / 2 ** (i - m + 1)))
    for t in range(w // 2):
        a, b = qubits[t], qubits[w - 1 - t]
        g += [('cx', a, b), ('cx', b, a), ('cx', a, b)]
    return g


def inverse(gates):
    out = []
    for g in reversed(gates):
        if g[0] in ('ry', 'rz'):
            out.append((g[0], g[1], -g[2]))
        elif g[0] == 'cp':
            out.append(('cp', g[1], g[2], -g[3]))
        else:
            out.append(g)
    return out


# ------------------------------------------------------------------ checks
def check_blocks():
    P = prep7()
    st = apply(np.eye(8, dtype=complex)[:, 0].copy(), 3, P)
    want = np.array([1 / math.sqrt(7)] * 7 + [0.0])
    e_prep = float(np.max(np.abs(st - want)))
    V = qft7_measure()
    U = unitary(3, V)
    chi = np.zeros((8, 8), dtype=complex)
    for k in range(7):
        chi[:7, k] = [cmath.exp(2j * math.pi * k * c / 7) / math.sqrt(7) for c in range(7)]
    chi[7, 7] = 1
    M = U @ chi                                     # column k must be phase_k e_k
    e_meas = float(np.max(np.abs(np.abs(M) - np.eye(8))))
    for w in (1, 2, 3):
        q = list(range(w))
        F = unitary(w, qft2(q))
        Mw = 2 ** w
        want = np.array([[cmath.exp(2j * math.pi * a * x / Mw) for a in range(Mw)] for x in range(Mw)]) / math.sqrt(Mw)
        assert np.max(np.abs(F - want)) < 1e-12, 'qft2 w = %d' % w
    return P, V, e_prep, e_meas


def check_window(w, bar, V, P, rng, ntests=6):
    """state vector of the whole residue-form register (3 + 2w qubits) through PREP and MEASURE against mixreg.cond"""
    win = mixreg.N7Window(w, bar)
    nq = 3 + 2 * w
    M = 1 << w
    d1, d2, _ = win.arrays()
    e1, e2 = win.e
    idx = {}                                        # residue rho -> basis index of (c, xa, xb)
    for a in range(e1):
        for b in range(e2):
            d = (int(d1[a, b]), int(d2[a, b]))
            c, x = win.phi7(d), (d[0] % M, d[1] % M)
            idx[(a, b)] = c + 8 * (x[0] + M * x[1])
    assert len(set(idx.values())) == 7 * 4 ** w == win.N
    xa, xb = list(range(3, 3 + w)), list(range(3 + w, 3 + 2 * w))
    prep = P + [('h', t) for t in xa + xb]
    st = apply(np.eye(2 ** nq, dtype=complex)[:, 0].copy(), nq, prep)
    ok_idx = np.zeros(2 ** nq, dtype=bool)
    ok_idx[list(idx.values())] = True
    e_prep = max(float(np.max(np.abs(st[ok_idx] - 1 / math.sqrt(win.N)))), float(np.max(np.abs(st[~ok_idx]))))
    meas = V + inverse(qft2(xa)) + inverse(qft2(xb))
    worst = 0.0
    for tno in range(ntests):
        g = (0.0, 0.0) if tno == 0 else (rng.random(), rng.random())
        psi = np.zeros(2 ** nq, dtype=complex)
        for (a, b), i in idx.items():
            ph = int(d1[a, b]) * g[0] + int(d2[a, b]) * g[1]
            psi[i] = cmath.exp(2j * math.pi * (ph - round(ph))) / math.sqrt(win.N)
        out = np.abs(apply(psi, nq, meas)) ** 2
        Pc = {}
        for i in range(2 ** nq):
            k, a1, a2 = i % 8, (i >> 3) % M, i >> (3 + w)
            if k == 7:
                assert out[i] < 1e-24, 'outcome c = 7 has weight %g' % out[i]
                continue
            E = ((Fraction(k, 7) + Fraction(a1, M)) % 1, (Fraction(k * win.om, 7) + Fraction(a2, M)) % 1)
            Pc[E] = Pc.get(E, 0.0) + float(out[i])
        Ph = {}
        law = mixreg.cond(win, g)
        U = win.U
        for y0 in range(e1):
            for y1 in range(e2):
                f = (Fraction(y0, e1), Fraction(y1, e2))
                E = ((U[0][0] * f[0] + U[1][0] * f[1]) % 1, (U[0][1] * f[0] + U[1][1] * f[1]) % 1)
                Ph[E] = Ph.get(E, 0.0) + float(law[y0, y1])
        assert set(Pc) == set(Ph) and len(Pc) == win.N, 'outcome sets differ'
        worst = max(worst, max(abs(Pc[E] - Ph[E]) for E in Pc))
    return e_prep, worst


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rng = random.Random(seed)
    P, V, e_prep, e_meas = check_blocks()
    cp, cv = counts(P), counts(V)
    print('PREP7 (|000> -> uniform over c < 7): %s; max |amplitude error| %.2e' % (cp, e_prep))
    print('QFT7 measurement (V chi_k = phase |k>, |7> fixed): %s; max ||(V chi)_k| - delta| %.2e' % (cv, e_meas))
    print('Toffoli in both: %d; ancilla qubits: 0' % sum(1 for g in P + V if g[0] in ('ccx', 'ccz', 'toffoli')))
    worst_all = 0.0
    for w in (1, 2, 3):
        for bar in (False, True):
            ep, wd = check_window(w, bar, V, P, rng)
            worst_all = max(worst_all, ep, wd)
            print('window %s w = %d (%2d qubits, %4d residues): prepared state max error %.2e; outcome law vs '
                  'mixreg.cond, max |difference| %.2e over 6 phase functionals'
                  % ('2 - w' if bar else '3 + w', w, 3 + 2 * w, 7 * 4 ** w, ep, wd))
    ok = max(e_prep, e_meas, worst_all) < 1e-10
    q7 = counts(inverse(qft2(list(range(7)))))
    print('per norm-7 window at w = 7 (17 register qubits): PREP = PREP7 + 14 H; MEASURE = QFT7 measurement + two '
          'inverse QFTs of Z / 2^7 (each %s as a coherent circuit; semiclassically 7 H and 21 classically controlled '
          'phases, no CNOT); Toffoli 0, ancilla 0, so the window kinds\' counts and peak are unchanged' % q7)
    print('ALL OK' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
