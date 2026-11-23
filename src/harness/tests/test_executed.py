"""Per-lane executed Toffoli in harness/vb.py: a Toffoli (ccx / ccz) with a classical condition runs only on the
lanes where the condition holds. Random gate sequences on 64 lanes, compared with a direct per-lane count; then
a V-B run of a circuit without conditional Toffoli reports executed = emitted.
Run: python3 harness/tests/test_executed.py
"""
import os, random, subprocess, sys, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import vb                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rng = random.Random(5)
    ok = True
    for trial in range(20):
        L = 64
        c = vb.Circuit(L, random.Random(trial))
        q = c.alloc(3)
        direct = [0] * L
        for _ in range(rng.randrange(1, 400)):
            cond = rng.choice([None, rng.getrandbits(L), 0, c.ALL])
            if rng.random() < 0.5:
                c.ccx(q[0], q[1], q[2], cond)
            else:
                c.ccz(q[0], q[1], q[2], cond)
            for lane in range(L):
                direct[lane] += 1 if cond is None or (cond >> lane) & 1 else 0
        ok &= c.executed() == direct
    print('executed per lane against a direct count, 20 random sequences: %s' % ('OK' if ok else 'FAIL'))
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'a.json')
        subprocess.run([sys.executable, os.path.join(HERE, '..', 'vb.py'), os.path.join(HERE, '..', 'vb_examples',
                        'adders.py'), '--n', '8,64', '--json', out], check=True, capture_output=True)
        for s in json.load(open(out))['sizes']:
            cnt = s['counts']
            good = cnt['toffoli_executed_mean'] == cnt['toffoli_executed_max'] == cnt['toffoli']
            ok &= good
            print('V-B adders n = %d: Toffoli %d, executed mean %.1f, max %d %s' % (
                s['n'], cnt['toffoli'], cnt['toffoli_executed_mean'], cnt['toffoli_executed_max'],
                'OK' if good else 'FAIL'))
    print('ALL OK' if ok else 'SOME CHECKS FAILED')
    assert ok


if __name__ == '__main__':
    main()
