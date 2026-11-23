"""Per-window offsets in the harness (Run.offset(per_window=True), Lemma 1*; the bound itself is checked by
brute force in test_coupling.py, main_window_offsets).

Hand-built Ekera-type runs on a toy instance, cost.MODEL at the estimate grade (w = 16, packed windows), approximate
arithmetic with a split failure rate (pi, e_S) per addition:
(a) the b register (built from Q) sits in an approximate window, and the runs still use the coupling model (no
    heuristic runs), since with per-window offsets the tables do not matter;
(b) with e_S = 0 the keep factor is (1 - pi)^2 for every outcome, so the bad-run fraction is 1 - (1 - pi)^2;
(c) the estimate grade adds 2^(w-1) Toffoli per addition for the unsigned tables, and each run charges 2^w group
    operations per approximate window for its tables;
(d) without the split rate (no Monte Carlo on uniform operand pairs) such runs fall back to the heuristic model.
Run: python3 harness/tests/test_window_offsets.py
"""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cost                              # noqa: E402
from ecc import instances_for           # noqa: E402
from hcore import Context                # noqa: E402


def run_once(h, m=64, l=16):
    run = h.run()
    run.offset(per_window=True)
    a, b = run.reg(m + 16), run.reg(l)
    run.add(a, h.G)                      # a first: b (Q-built) lands in the last, approximate window
    run.add(b, -h.Q)
    return run.measure()


def main():
    cost.MODEL.update(cost.HANER)
    cost.MODEL.update(w=16, pack=True)
    inst = instances_for(64)[0]
    pi_add = 0.004
    oks = []
    h = Context(inst, 1234567 % inst['r'], 3, fail_per_add=pi_add, kappa=True, fail_split=(pi_add, 0.0))
    before = h.classical_ops()
    R = 4000
    for _ in range(R):
        run_once(h)
    shape, c = h.runs[-1]
    adds = c['additions']
    pi = adds * pi_add
    exp_bad = 1 - (1 - pi) ** 2
    sd = math.sqrt(exp_bad * (1 - exp_bad) / R)
    frac = h.garbage_runs / R
    ok = h.heuristic_runs == 0 and abs(frac - exp_bad) < 4 * sd
    oks.append(ok)
    print('(a, b) %d runs, %d additions each, q_tables %s: heuristic runs %d (expect 0), bad fraction %.4f, expected '
          '1 - (1 - pi)^2 = %.4f +- %.4f %s' % (R, adds, c['q_tables'], h.heuristic_runs, frac, exp_bad, sd,
                                                'OK' if ok else 'FAIL'))
    per_add = cost.HANER['toffoli_per_add'] + 2 ** 15
    ok = abs(c['toffoli'] - (adds * per_add + c['first_lookup'])) < 1e-6 and 'unsigned tables' in c['grade']
    oks.append(ok)
    print('(c) Toffoli per run %.6g = %d x (%d + 2^15) + first lookup %d; grade "%s" %s' % (
        c['toffoli'], adds, cost.HANER['toffoli_per_add'], c['first_lookup'], c['grade'], 'OK' if ok else 'FAIL'))
    charged = h.classical_ops() - before
    ok = charged >= R * adds * 2 ** 16
    oks.append(ok)
    print('(c) classical work charged %.4g >= %d runs x %d windows x 2^16 = %.4g %s' % (
        charged, R, adds, R * adds * 2 ** 16, 'OK' if ok else 'FAIL'))
    h2 = Context(inst, 7654321 % inst['r'], 4, fail_per_add=pi_add, kappa=True)
    for _ in range(50):
        run_once(h2)
    ok = h2.heuristic_runs == 50
    oks.append(ok)
    print('(d) without the split rate: %d of 50 runs on the heuristic model %s' % (h2.heuristic_runs,
                                                                                  'OK' if ok else 'FAIL'))
    print('ALL OK' if all(oks) else 'SOME CHECKS FAILED')
    assert all(oks)


if __name__ == '__main__':
    main()
