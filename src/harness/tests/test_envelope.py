"""Envelope registers (Run.reg with 2^k amplitudes, k < bits) in the cost model and the coupling model.

Hand-built runs on a toy instance with cost.MODEL set by hand (packed windows of w = 16, offset first lookup):
(a) shaped qubits covered by one add() call that lies in the first window (the exact first lookup) and reaches
    its register's top bit add nothing to the peak (they are measured right after the lookup); shaped qubits
    anywhere else add their number;
(b) the coupling weight (eps = pf x weight) counts max|phi| / RMS(phi) only for approximate windows that read
    shaped qubits, so shaped qubits that only the first lookup reads give eps = pf x additions;
(c) measure() samples such runs (a 260-bit register with a 4-qubit envelope) without error.
The sampler itself is checked against brute force in test_sampler.py (env_case), the envelope factor and the
"window reads no shaped bit" bound in test_coupling.py (main_amps).
Run: python3 harness/tests/test_envelope.py
"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cost                              # noqa: E402
from ecc import instances_for           # noqa: E402
from hcore import Context                # noqa: E402


def sine(k):
    return np.sin(np.pi * (np.arange(1 << k) + 0.5) / (1 << k))


def build(h, order):
    """b: 12-qubit sine register (all shaped), a: 260 qubits with a 4-qubit sine envelope on top; add() calls in
    the given order of 'b', 'atop' (a's shaped bits 256..259), 'alow' (a's bits 0..255)."""
    run = h.run()
    b = run.reg(12, amplitudes=sine(12), prep_toffoli=1000)
    a = run.reg(260, amplitudes=sine(4), prep_toffoli=100)
    run.offset()
    calls = dict(b=(b, -h.Q, 0, 12), atop=(a, h.G * (1 << 256), 256, 260), alow=(a, h.G, 0, 256))
    for name in order:
        reg, P, lo, hi = calls[name]
        run.add(reg, P, lo, hi)
    return run


def main():
    cost.MODEL.update(w=16, pack=True, qubits=1000, toffoli_per_add=1e6, first_lookup=2 ** 16,
                      first_lookup_offset=2 ** 16, grade='estimate')
    inst = instances_for(16)[0]
    h = Context(inst, 12345 % inst['r'], 7, fail_per_add=1e-4, kappa=True)
    m = float(sine(4).max() / math.sqrt((sine(4) ** 2).mean()))       # envelope factor max|phi| / RMS(phi)
    oks = []
    # (1) b and a's shaped top bits in the first window, a's low 256 bits in 16 windows
    run = build(h, ['b', 'atop', 'alow'])
    w1 = run._shaped_weight(17)
    run.measure()
    shape, c = h.runs[-1]
    ok = c['windows'] == 17 and c['additions'] == 16 and c['qubits'] == 1000 and abs(w1 - 16) < 1e-12
    oks.append(ok)
    print('shaped qubits only in the first lookup: windows %d, additions %d, peak %d (expect 1000), weight %.4f '
          '(expect 16) %s' % (c['windows'], c['additions'], c['qubits'], w1, 'OK' if ok else 'FAIL'))
    # (2) a's low bits right after b, its shaped bits last: they sit in window 16 (approximate), and a's bits 0..3
    # fill the first window without being a top part (held by first_window_held)
    run = build(h, ['b', 'alow', 'atop'])
    w2 = run._shaped_weight(17)
    run.measure()
    shape, c = h.runs[-1]
    ok = c['qubits'] == 1000 + 4 + 4 and abs(w2 - (15 + m)) < 1e-12
    oks.append(ok)
    print('shaped qubits in an approximate window: peak %d (expect 1008: 4 shaped + 4 low first-window bits held), '
          'weight %.4f (expect 15 + %.4f) %s' % (c['qubits'], w2, m, 'OK' if ok else 'FAIL'))
    # (3) the old rule (every addition times the product of the factors over all amplitude registers) as a bound
    old = 16 * m * float(sine(12).max() / math.sqrt((sine(12) ** 2).mean()))
    ok = w1 <= old and w2 <= old
    oks.append(ok)
    print('new weights %.2f, %.2f <= old rule %.2f %s' % (w1, w2, old, 'OK' if ok else 'FAIL'))
    # (4) sampling: many measured runs, outcomes in range
    for _ in range(200):
        out = build(h, ['b', 'atop', 'alow']).measure()
        assert 0 <= out[0] < 1 << 12 and 0 <= out[1] < 1 << 260
    print('200 runs sampled (envelope on a 260-qubit register): OK')
    print('ALL OK' if all(oks) else 'SOME CHECKS FAILED')
    assert all(oks)


if __name__ == '__main__':
    main()
