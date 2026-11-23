"""Work cap of a decoding attempt (h.try_subset(run_ids, fn, cap)) is enforced inside the lattice reduction
.

NEED + 1 = 20, built from random outcome values (what a bad run could feed it), reduced inside an attempt for a
range of caps. Checks: (a) the work charged by every attempt is <= its cap; (b) with a cap of 1 nothing is done
(the reduction does not start); (c) a cap at or above the uncapped work gives the uncapped result, charged the
same; (d) a cap below it stops the reduction (the attempt fails) and charges no more than the cap.
Run: python3 harness/tests/test_attempt_cap.py
"""
import os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ecc import instances_for           # noqa: E402
from hcore import Context                # noqa: E402


def lattice_rows(r, m, l, sigma, n, rng):
    A, big = 1 << (m + sigma), 1 << (m + l)
    zs = [(2 * r * rng.randrange(A) + A) // (2 * A) for _ in range(n)]
    rows = [[big * z for z in zs] + [r]]
    for i in range(n):
        rows.append([0] * i + [r * big] + [0] * (n - i))
    target = [-(r * rng.randrange(1 << l) << m) for _ in range(n)] + [0]
    return rows, target


def attempt(h, rows, target, cap):
    out = {}

    def fn():
        red = h.lll(rows)
        out['red'] = red
        h.babai(red, target)
        return None
    before = h.classical_ops()
    capped = h.capped_attempts
    h.try_subset([], fn, cap)
    return h.classical_ops() - before, h.capped_attempts > capped, out.get('red')


def main(trials=12, seed=4):
    inst = [i for i in instances_for(256) if i['name'] == 'secp256k1'][0]
    rng = random.Random(seed)
    worst = 0.0
    for t in range(trials):
        h = Context(inst, rng.randrange(1, inst['r']), rng.getrandbits(64))
        rows, target = lattice_rows(inst['r'], 256, 16, 16, 19, rng)
        full, capped, red_full = attempt(h, rows, target, 2 ** 40)
        assert not capped and red_full is not None
        for cap in (1, 10 ** 5, full // 4, full // 2, full - 1, full, 2 * full):
            spent, capped, red = attempt(h, rows, target, cap)
            assert spent <= cap, (cap, spent)
            worst = max(worst, spent / cap)
            if cap == 1:
                assert spent == 0 and capped
            if cap >= full:
                assert not capped and red == red_full and spent == full, (cap, full, spent)
            elif cap < full:
                assert capped, (cap, full)
        print('trial %2d: uncapped attempt %d units; every capped attempt stopped at or below its cap' % (t, full))
    print('largest charged / cap over all capped attempts: %.4f (must be <= 1); ALL OK' % worst)


if __name__ == '__main__':
    main()
