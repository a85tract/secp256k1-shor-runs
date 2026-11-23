"""Brute-force check of the coupling bound behind `--approx`.

A run with exponent values x in [0, N), point label t(x) = c x mod r (cyclic group Z_r), eigenvalue index u and
QFT outcome o has ideal amplitude A(o | u) = (1/N) sum_x e(u t(x)/r - o x/N), kappa = |A|. A faulty circuit has
flagged inputs X_f (dropped by the flag measurement), dirty-ancilla inputs X_d (orthogonal ancilla state: they
leave the clean coherent sum) and silent inputs X_s (wrong label t'(x), clean ancillas). With
e = (|X_f| + |X_d| + 2 |X_s|) / N the harness keeps the exact outcome with probability (1 - e / kappa)^2, i.e. it
claims  P(no flag, outcome o | u) >= kappa^2 (1 - e / kappa)^2 = (kappa - e)^2  whenever kappa >= e.
This test computes the true P(no flag, o, clean ancillas | u) for random small instances and fault sets and checks
Amplitude registers (state sum_x c_x |x>, sum |c_x|^2 = 1): A(o | u) = sum_x c_x / sqrt(N) e(...), so a bad input
weighs |c_x| / sqrt(N) and e = sum over bad inputs of |c_x| / sqrt(N) (silent twice); main_amps() checks
(kappa - e)^2 the same way, and the averaging step behind the harness's factor: for any per-window failure
rates g(k) and an envelope phi on some of the window's bits, sum_x |c_x| g(k(x)) / sqrt(N) <= max|phi| / RMS(phi)
x mean g, and <= mean g when the window reads no enveloped bit.
behind Run.offset(per_window=True): with a fresh random offset in every window, P(no flag, o | u) >=
((1 - pi) kappa - e_S)^2, pi = max over x of P(some addition misbehaves on its correct operands), e_S = max over x
of P(the first misbehaviour is unflagged), all offsets enumerated exactly on a toy group.
Run: python3 harness/tests/test_coupling.py
"""
import cmath, math, random


def amp(N, r, labels, keep, u, o):
    s = 0j
    for x in range(N):
        if keep[x]:
            s += cmath.exp(2j * math.pi * (u * labels[x] / r - o * x / N))
    return s / N


def main(trials=300, seed=1):
    rng = random.Random(seed)
    worst_new, old_viol, checked = 1e9, 0, 0
    for _ in range(trials):
        r = rng.choice([53, 101, 197, 389])
        N = rng.choice([8, 16, 32])
        c = rng.randrange(1, r)
        ideal = [c * x % r for x in range(N)]
        nf, nd, ns = rng.randrange(0, 3), rng.randrange(0, 2), rng.randrange(0, 3)
        xs = rng.sample(range(N), nf + nd + ns)
        Xf, Xd, Xs = set(xs[:nf]), set(xs[nf:nf + nd]), set(xs[nf + nd:])
        bad = [ideal[x] if x not in Xs else (ideal[x] + rng.randrange(1, r)) % r for x in range(N)]
        keep_clean = [x not in Xf and x not in Xd for x in range(N)]
        e = (len(Xf) + len(Xd) + 2 * len(Xs)) / N
        p_noflag = 1 - len(Xf) / N
        for u in range(r):
            for o in range(N):
                kap = abs(amp(N, r, ideal, [True] * N, u, o))
                p_true = abs(amp(N, r, bad, keep_clean, u, o)) ** 2      # no flag, outcome o, clean ancillas
                if kap > e:
                    checked += 1
                    worst_new = min(worst_new, p_true - (kap - e) ** 2)
                    if len(Xs) and len(Xd) == 0 and p_true + 1e-12 < (kap - e) ** 2 / (1 - e) * p_noflag:
                        old_viol += 1
    print('checked %d (u, o) pairs with kappa > e over %d random fault sets: min of true - claimed = %.3g '
          '(must be >= 0); normalised form violated in %d pairs' % (checked, trials, worst_new, old_viol))
    assert worst_new >= -1e-12, 'coupling bound violated'


def amp_c(N, r, labels, keep, c, u, o):
    s = 0j
    for x in range(N):
        if keep[x]:
            s += c[x] * cmath.exp(2j * math.pi * (u * labels[x] / r - o * x / N))
    return s / math.sqrt(N)


def main_amps(trials=200, seed=2):
    rng = random.Random(seed)
    worst, checked = 1e9, 0
    for _ in range(trials):
        r = rng.choice([53, 101, 197])
        N = rng.choice([8, 16, 32])
        cl = rng.randrange(1, r)
        ideal = [cl * x % r for x in range(N)]
        c = [rng.random() * cmath.exp(2j * math.pi * rng.random()) for _ in range(N)]
        nrm = math.sqrt(sum(abs(v) ** 2 for v in c))
        c = [v / nrm for v in c]
        nf, nd, ns = rng.randrange(0, 3), rng.randrange(0, 2), rng.randrange(0, 3)
        xs = rng.sample(range(N), nf + nd + ns)
        Xf, Xd, Xs = set(xs[:nf]), set(xs[nf:nf + nd]), set(xs[nf + nd:])
        bad = [ideal[x] if x not in Xs else (ideal[x] + rng.randrange(1, r)) % r for x in range(N)]
        keep_clean = [x not in Xf and x not in Xd for x in range(N)]
        e = (sum(abs(c[x]) for x in Xf | Xd) + 2 * sum(abs(c[x]) for x in Xs)) / math.sqrt(N)
        for u in range(r):
            for o in range(N):
                kap = abs(amp_c(N, r, ideal, [True] * N, c, u, o))
                if kap > e:
                    checked += 1
                    worst = min(worst, abs(amp_c(N, r, bad, keep_clean, c, u, o)) ** 2 - (kap - e) ** 2)
    print('amplitude registers: checked %d (u, o) pairs with kappa > e over %d random fault sets: min of true - '
          'claimed = %.3g (must be >= 0)' % (checked, trials, worst))
    assert worst >= -1e-12, 'coupling bound violated for amplitude registers'
    # averaging step: x = (window bits k, other bits); envelope phi on E = some bits of x
    worst_in, worst_out = 1e9, 1e9
    for _ in range(2000):
        nb = rng.randrange(2, 7)
        N = 1 << nb
        wbits = rng.sample(range(nb), rng.randrange(1, nb + 1))          # bits the window reads
        ebits = rng.sample(range(nb), rng.randrange(1, nb + 1))          # enveloped bits
        def sub(x, bits):
            return sum(((x >> b) & 1) << i for i, b in enumerate(bits))
        phi = [rng.random() + (0.0 if rng.random() < 0.5 else 1.0) for _ in range(1 << len(ebits))]
        cx = [phi[sub(x, ebits)] for x in range(N)]
        nrm = math.sqrt(sum(v * v for v in cx))
        cx = [v / nrm for v in cx]
        g = [rng.random() ** 3 for _ in range(1 << len(wbits))]
        lhs = sum(cx[x] * g[sub(x, wbits)] for x in range(N)) / math.sqrt(N)
        gbar = sum(g) / len(g)
        rms = math.sqrt(sum(v * v for v in phi) / len(phi))
        if set(wbits) & set(ebits):
            worst_in = min(worst_in, max(phi) / rms * gbar - lhs)
        else:
            worst_out = min(worst_out, gbar - lhs)
    print('averaging step over 2,000 random envelopes and rates: min of bound - weight = %.3g (window reads enveloped '
          'bits), %.3g (it does not); both must be >= 0' % (worst_in, worst_out))
    assert worst_in >= -1e-12 and worst_out >= -1e-12, 'envelope factor violated'


def main_window_offsets(seed=5, r=13):
    """Z_r; a register of 4 qubits (window a[0:2] in the direct lookup with R0, window a[2:4] = addition 2), b
    register of 2 qubits (addition 3, table -(d + rho) k). Each addition has a fixed failure set of operand pairs
    (A, B) (flagged, or silent: output + 1); every offset triple (R0, R2, R3) is enumerated for every rho."""
    import numpy as np
    rng = np.random.default_rng(seed)
    d = int(rng.integers(1, r))
    Na, Nb = 16, 4
    N = Na * Nb
    XA, XB = np.repeat(np.arange(Na), Nb), np.tile(np.arange(Nb), Na)
    Q = np.exp(-2j * np.pi * (np.outer(XA, XA) / Na + np.outer(XB, XB) / Nb)) / math.sqrt(N)    # [o, x]
    amp = 1 / math.sqrt(N)
    R0, R2, R3 = [g.ravel() for g in np.meshgrid(np.arange(r), np.arange(r), np.arange(r), indexing='ij')]
    C = R0.size
    cidx, U = np.arange(C), np.arange(r)
    worst = 1e9
    for fl, sl in ((0.10, 0.0), (0.06, 0.03), (0.15, 0.06)):
        F = {}
        for w in (2, 3):
            z = rng.random((r, r))
            F[w] = np.where(z < fl, 1, np.where(z < fl + sl, 2, 0))
        for rho in range(r):
            dp = (d + rho) % r
            T1, T2, T3 = (XA & 3) % r, (4 * (XA >> 2)) % r, (-dp * XB) % r
            lab = (XA - XB * dp) % r
            ideal = np.abs(Q @ (amp * np.exp(2j * np.pi * np.outer(lab, U) / r))) ** 2       # [o, u] = P_I(o | u)
            kap = np.sqrt(ideal)
            A = (R0[:, None] + T1[None, :]) % r                                              # [config, x]
            B2 = (T2[None, :] + R2[:, None]) % r
            c2 = F[2][A, B2]
            out = (A + B2 + (c2 == 2)) % r
            B3 = (T3[None, :] + R3[:, None]) % r
            f3 = np.where(c2 == 1, 0, F[3][out, B3])                                          # actual operands
            final = (out + B3 + (f3 == 2)) % r
            flagged = (c2 == 1) | (f3 == 1)
            labr = (final - (R0 + R2 + R3)[:, None]) % r
            c3 = F[3][(A + B2) % r, B3]                                                       # correct operands
            pi = float(((c2 > 0) | (c3 > 0)).mean(axis=0).max())
            eS = float(np.where(c2 > 0, c2 == 2, c3 == 2).mean(axis=0).max())
            H = np.zeros((C, r, N), complex)
            v = np.where(flagged, 0.0, amp)
            for x in range(N):
                np.add.at(H, (cidx, labr[:, x]), v[:, x, None] * Q[None, :, x])
            Pavg = (np.abs(np.fft.ifft(H, axis=1) * r) ** 2).mean(axis=0).T                  # [o, u]
            worst = min(worst, float((Pavg - np.clip((1 - pi) * kap - eS, 0, None) ** 2).min()))
    print('per-window offsets (Lemma 1*): Z_%d, all %d offset triples per rho, 3 fault densities: min over '
          '(rho, o, u) of P(no flag, o | u) - ((1 - pi) kappa - e_S)^2 = %.3g (must be >= 0)' % (r, C, worst))
    assert worst >= -1e-12, 'Lemma 1* bound violated'


if __name__ == '__main__':
    main()
    main_amps()
    main_window_offsets()
