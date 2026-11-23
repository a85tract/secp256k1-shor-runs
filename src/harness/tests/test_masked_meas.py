"""Lane-masked measurement in harness/vb.py: measure_x(q, cond)
and flag([q], cond) must give the same outcomes, phases, flag lanes and lane states as the builders' borrowed-ancilla
emulation (alloc anc, cx(q, anc, cond), cx(anc, q, cond), measure_x(anc) / flag([anc])), with a peak one qubit lower.
Random circuits on 64 lanes (CNOT / Toffoli / phase gates between conditional measurements), both forms from one seed.
Run: python3 harness/tests/test_masked_meas.py
"""
import os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import vb                                # noqa: E402


def build(c, rng, masked):
    qs = c.alloc(6)
    for i, q in enumerate(qs):
        c.s[q] = rng.getrandbits(c.L)               # random lane values
    outs = []
    for _ in range(60):
        r = rng.random()
        a, b, t = rng.sample(qs, 3)
        if r < 0.3:
            c.cx(a, t, rng.choice([None, rng.getrandbits(c.L)]))
        elif r < 0.5:
            c.ccx(a, b, t, rng.choice([None, rng.getrandbits(c.L)]))
        elif r < 0.6:
            c.cz(a, b)
        else:
            cond = rng.getrandbits(c.L)
            fl = rng.random() < 0.3
            if masked:
                if fl:
                    c.flag([t], cond)
                else:
                    outs.append(c.measure_x(t, cond))
            else:
                anc = c.alloc(1)[0]
                c.cx(t, anc, cond)
                c.cx(anc, t, cond)
                if fl:
                    c.flag([anc])
                else:
                    outs.append(c.measure_x(anc))
    return outs, [c.s[q] for q in qs]


def main():
    ok = True
    for trial in range(30):
        res = []
        for masked in (False, True):
            c = vb.Circuit(64, random.Random(trial))
            outs, states = build(c, random.Random(1000 + trial), masked)
            res.append((outs, states, c.phase, c.flag_mask, c.peak, c.live))
        (o0, s0, p0, f0, pk0, l0), (o1, s1, p1, f1, pk1, l1) = res
        ok &= o0 == o1 and s0 == s1 and p0 == p1 and f0 == f1 and l0 == l1 and pk1 == pk0 - 1
    print('masked measurement vs borrowed-ancilla emulation, 30 random circuits: outcomes, phases, flags and lane '
          'states equal, peak one lower: %s' % ('OK' if ok else 'FAIL'))
    print('ALL OK' if ok else 'SOME FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
