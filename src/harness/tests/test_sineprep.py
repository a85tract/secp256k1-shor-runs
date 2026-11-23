"""harness/sineprep.py checked on a state vector: for k = 1..5 and every combination of its k + 1 measurement
outcomes, the gate list leaves the k register qubits in the normalised sine state and every other qubit (f, the
phase-gradient register, the AND ancillas, the carry) in |0>, up to a global phase; each outcome has probability
1/2; the Toffoli count of the list equals sineprep.toffoli(k).
Run: python3 harness/tests/test_sineprep.py
"""
import itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sineprep                          # noqa: E402

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def one(psi, n, q, U):
    v = psi.reshape(1 << (n - q - 1), 2, 1 << q)
    return np.einsum('ab,ibj->iaj', U, v).reshape(-1)


def run(k, outcomes):
    """Apply sineprep.gates(k) with the given outcome per measurement key; returns (state, probability)."""
    n = 3 * k + 2
    idx = np.arange(1 << n)
    bit = lambda q: (idx >> q) & 1
    psi = np.zeros(1 << n, complex)
    psi[0] = 1.0
    prob, res = 1.0, {}

    def apply(g):
        nonlocal psi, prob
        op = g[0]
        if op == 'h':
            psi = one(psi, n, g[1], H)
        elif op == 'p':
            psi = one(psi, n, g[1], np.diag([1, np.exp(1j * g[2])]))
        elif op == 'x':
            psi = psi[idx ^ (1 << g[1])]
        elif op == 'z':
            psi = psi * (1 - 2 * bit(g[1]))
        elif op == 'cx':
            psi = psi[idx ^ (bit(g[1]) << g[2])]
        elif op == 'cz':
            psi = psi * (1 - 2 * (bit(g[1]) & bit(g[2])))
        elif op == 'ccx':
            psi = psi[idx ^ ((bit(g[1]) & bit(g[2])) << g[3])]
        elif op == 'mx':
            m = outcomes[g[2]]
            psi = one(psi, n, g[1], H)
            psi = np.where(bit(g[1]) == m, psi, 0)
            p = float(np.vdot(psi, psi).real)
            prob *= p
            psi = psi / np.sqrt(p)
            if m:
                psi = psi[idx ^ (1 << g[1])]
            res[g[2]] = m
        elif op == 'if':
            if res[g[1]]:
                apply(g[2])
        else:
            raise ValueError(op)

    for g in sineprep.gates(k):
        apply(g)
    return psi, prob


def main():
    worst, ok = 1.0, True
    for k in range(1, 6):
        keys = ['a%d' % q for q in range(k)] + ['f']
        s = sineprep.amplitudes(k).astype(complex)
        target = np.zeros(1 << (3 * k + 2), complex)
        target[:1 << k] = s / np.linalg.norm(s)          # register = low k qubits, everything else |0>
        probs = []
        for outs in itertools.product((0, 1), repeat=k + 1):
            psi, p = run(k, dict(zip(keys, outs)))
            fid = abs(np.vdot(target, psi)) ** 2
            worst = min(worst, fid)
            probs.append(p)
        nccx = sum(1 for g in sineprep.gates(k) if g[0] == 'ccx')
        ok &= abs(min(probs) - 2.0 ** -(k + 1)) < 1e-12 and abs(max(probs) - 2.0 ** -(k + 1)) < 1e-12
        ok &= nccx == sineprep.toffoli(k)
        print('k = %d: %d outcome combinations, each with probability %.6g, fidelity >= %.15f, Toffoli %d (toffoli(k) '
              '= %d)' % (k, len(probs), min(probs), worst, nccx, sineprep.toffoli(k)))
    ok &= worst > 1 - 1e-12
    # the identity behind the gate list at the sizes the 16-addition layout uses (no ancillas): ramp^-1 of
    # w_0 chi_0 +- w_1 chi_1, the minus branch followed by X and Z on the top qubit
    for k in (8, 14):
        N = 1 << k
        j = np.arange(N)
        w1 = -np.exp(1j * np.pi / N)
        s = sineprep.amplitudes(k) / np.linalg.norm(sineprep.amplitudes(k))
        for sign in (1, -1):
            v = np.exp(-1j * np.pi * j / N) * (1 + sign * w1 * np.exp(2j * np.pi * j / N))
            if sign < 0:
                v = v[j ^ (N // 2)] * (1 - 2 * (j >> (k - 1)))
            fid = abs(np.vdot(s, v / np.linalg.norm(v))) ** 2
            ok &= fid > 1 - 1e-12
            print('k = %d, f outcome %d: identity fidelity %.15f' % (k, sign < 0, fid))
    print('Toffoli at k = 8 / 14 (the 16-addition layout): %d / %d, against the declared 2^(k+2) = %d / %d'
          % (sineprep.toffoli(8), sineprep.toffoli(14), 2 ** 10, 2 ** 16))
    print('ALL OK' if ok else 'SOME CHECKS FAILED')
    assert ok


if __name__ == '__main__':
    main()
