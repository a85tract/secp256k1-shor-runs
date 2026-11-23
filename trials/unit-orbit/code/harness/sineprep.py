"""Counted preparation of the sine state sum_j sin(pi (j + 1/2) / N) |j> / sqrt(N / 2), N = 2^k, on k qubits (j
little-endian), the shaped register of Run.reg(bits, sine=k). Checked gate by gate on a state vector in
harness/tests/test_sineprep.py for every measurement outcome.

Why it is cheap: with D = diag(e^{i pi j / N}), D |sine> is proportional to e^{i pi/(2N)} chi_1 - e^{-i pi/(2N)} chi_0,
chi_f = sum_j e^{2 pi i f j / N} |j>, a superposition of two Fourier states. So
  1. an ancilla f in (|0> - e^{i pi/N} |1>) / sqrt(2); the k qubits in |+> (chi_0);
  2. phase e^{2 pi i f j / N}: a_q = f AND x_q (k Toffoli), then a_q added into a k-qubit phase-gradient register
     G = sum_y e^{-2 pi i y / N} |y> (adding c multiplies it by e^{2 pi i c / N}; Cuccaro ripple, 2 (k - 1)
     Toffoli), then each a_q X-measured with CZ(f, x_q) on outcome 1 (measurement-based uncomputation, no Toffoli);
  3. f X-measured: outcome 0 leaves w_0 chi_0 + w_1 chi_1, outcome 1 the same with a minus sign;
  4. the ramp D^-1 (one phase per qubit), which gives the sine state, or for f's outcome 1 the cosine state, which
     X then Z on the top qubit turn into the sine state.
Cost convention (as for the run's final semiclassical QFT, which is not charged): Toffoli are counted, single-qubit
rotations (f's phase, G's preparation and removal, the ramp: 2k + 1 of them) are not. Extra qubits while it runs:
2k + 2 (f, G, the a_q, one carry), far below a run's arithmetic peak, released before the arithmetic starts.
"""
import math

import numpy as np


def amplitudes(k):
    """The 2^k sine amplitudes (not normalised; the harness normalises)."""
    return np.sin(np.pi * (np.arange(1 << k) + 0.5) / (1 << k))


def toffoli(k):
    """Toffoli count of the preparation of a k-qubit sine state: k ANDs plus a 2 (k - 1)-Toffoli ripple addition."""
    return 3 * k - 2


def gates(k):
    """Gate list on qubits x_0..x_{k-1} (the register, little-endian) = 0..k-1, f = k, G = k+1..2k, a = 2k+1..3k,
    carry = 3k + 1. Entries: ('h', q), ('p', q, phi) (diag(1, e^{i phi})), ('x', q), ('z', q), ('cx', c, t),
    ('cz', a, b), ('ccx', a, b, t), ('mx', q, key) (X-basis measurement, the qubit is then reset to |0>), and
    ('if', key, gate) (gate applied when the outcome under key was 1)."""
    N = 1 << k
    X = list(range(k))
    f = k
    G = list(range(k + 1, 2 * k + 1))
    A = list(range(2 * k + 1, 3 * k + 1))
    c0 = 3 * k + 1
    g = []
    for q, t in enumerate(G):                       # phase-gradient register: bit q carries e^{-2 pi i 2^q / N}
        g += [('h', t), ('p', t, -2 * math.pi * (1 << q) / N)]
    g += [('h', f), ('p', f, math.pi + math.pi / N)]
    g += [('h', x) for x in X]
    g += [('ccx', f, x, a) for x, a in zip(X, A)]
    g += _ripple_add(A, G, c0)                      # G += A mod 2^k: phase e^{2 pi i f j / N}
    for q, (x, a) in enumerate(zip(X, A)):
        g += [('mx', a, 'a%d' % q), ('if', 'a%d' % q, ('cz', f, x))]
    g += [('mx', f, 'f')]
    g += [('p', x, -math.pi * (1 << q) / N) for q, x in enumerate(X)]
    g += [('if', 'f', ('x', X[-1])), ('if', 'f', ('z', X[-1]))]
    for q, t in enumerate(G):                       # remove the gradient register
        g += [('p', t, 2 * math.pi * (1 << q) / N), ('h', t)]
    return g


def _ripple_add(A, B, c):
    """B <- B + A mod 2^len(B) (Cuccaro et al. MAJ / UMA ladder, carry qubit c in |0> and returned to |0>):
    2 (n - 1) Toffoli for n bits."""
    n = len(B)
    if n == 1:
        return [('cx', A[0], B[0])]
    g = []
    cin = c
    for i in range(n - 1):                          # MAJ(cin, B_i, A_i): A_i becomes the carry out of bit i
        g += [('cx', A[i], B[i]), ('cx', A[i], cin), ('ccx', cin, B[i], A[i])]
        cin = A[i]
    g += [('cx', A[n - 1], B[n - 1]), ('cx', cin, B[n - 1])]     # top sum bit, no carry out
    for i in reversed(range(n - 1)):                # UMA(cin, B_i, A_i)
        cin = A[i - 1] if i > 0 else c
        g += [('ccx', cin, B[i], A[i]), ('cx', A[i], cin), ('cx', cin, B[i])]
    return g
