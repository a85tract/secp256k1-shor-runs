"""Hex registers with windows of different sizes (harness/hexreg.py, Run.hexreg / hexadd, cost.hex_run_cost).

(a) Brute force (o1/tests_hexva.py's formula, mixed sizes): the group is Z_r with a cube root of unity lam; the
    register holds x = (x_0 .. x_(m-1)), sigma(x) = sum_i digit(x_i, W_i) 2^(P_i), the accumulator g(x) =
    c (s1 + s2 lam) mod r. Measured in the orthonormal basis |phi_Y> = N^-1 sum_x e(<Y, sigma(x)> / N)|x> (N = 2^P,
    the sigma form a complete residue system mod N), with the point register traced out:
    P(Y) = sum_g |N^-2 sum_{x: g(x) = g} e(-<Y, sigma(x)> / N)|^2. Sampler: r^-1 sum_u prod_t cond(...)[y_t].
(c) Pricing of a hex run window by window (kinds, first lookup by formula, failure weights summed), and the
    errors for a wrong kind / size.
(d) End to end on a 24-bit toy curve: a hex run's outcome peaks at Y / 2^P = (u c / r, u c lam / r).
Run: python3 harness/tests/test_hexreg.py
"""
import itertools, os, random, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import hexreg, cost, hcore                       # noqa: E402


def cube_root(r):
    for g in range(2, r):
        lam = pow(g, (r - 1) // 3, r)
        if lam != 1:
            return lam


def brute(Ws, r, lam, c, top=None, split=0):
    """top='sine': the top window has box digits (s, t) with amplitude sine[s] sine[t]; P(Y) = sum_g
    |sum_{x: g(x) = g} phi(x) e(-<Y, sigma(x)> / N)|^2 / (N^2 sum_x |phi(x)|^2)"""
    e1, e2 = max(split, 0), max(-split, 0)
    N1, N2, P = 1 << (sum(Ws) + e1), 1 << (sum(Ws) + e2), [sum(Ws[:i]) for i in range(len(Ws))]
    digs = [[hexreg.digit(a, b, W) for a in range(1 << W) for b in range(1 << W)] for W in Ws]
    amps = [[1.0] * len(d) for d in digs]
    if top == 'sine':
        W = Ws[-1]
        a1, a2 = hexreg.sine(W + e1), hexreg.sine(W + e2)
        digs[-1] = [(s, t) for s in range(1 << (W + e1)) for t in range(1 << (W + e2))]
        amps[-1] = [a1[s] * a2[t] for s in range(1 << (W + e1)) for t in range(1 << (W + e2))]
    groups = {}
    norm = 0.0
    for xs in itertools.product(*[range(len(d)) for d in digs]):
        s1 = sum(digs[i][x][0] << P[i] for i, x in enumerate(xs))
        s2 = sum(digs[i][x][1] << P[i] for i, x in enumerate(xs))
        ph = float(np.prod([amps[i][x] for i, x in enumerate(xs)]))
        norm += ph * ph
        groups.setdefault(c * (s1 + s2 * lam) % r, []).append((s1, s2, ph))
    Y1, Y2 = np.meshgrid(np.arange(N1), np.arange(N2), indexing='ij')
    out = np.zeros((N1, N2))
    for sig in groups.values():
        amp = sum(f * np.exp(-2j * np.pi * (((Y1 * s1) % N1) / N1 + ((Y2 * s2) % N2) / N2)) for s1, s2, f in sig)
        out += np.abs(amp) ** 2 / (N1 * N2 * norm)
    return out


def sampler_dist(Ws, r, lam, c, top=None, split=0):
    e1, e2 = max(split, 0), max(-split, 0)
    N1, N2 = 1 << (sum(Ws) + e1), 1 << (sum(Ws) + e2)
    out = np.zeros((N1, N2))
    for u in range(r):
        for Y1 in range(N1):
            for Y2 in range(N2):
                pr, Q1, Q2, lo1, lo2 = 1.0, 0, 0, 0, 0
                for t in range(len(Ws)):
                    W = Ws[len(Ws) - 1 - t]
                    w1, w2 = (W + e1, W + e2) if t == 0 else (W, W)
                    y1, y2 = (Y1 >> Q1) & ((1 << w1) - 1), (Y2 >> Q2) & ((1 << w2) - 1)
                    pr *= hexreg.cond(Ws, t, u * c % r, r, lam, lo1, lo2, top, split)[y1, y2]
                    lo1, lo2, Q1, Q2 = lo1 + (y1 << Q1), lo2 + (y2 << Q2), Q1 + w1, Q2 + w2
                out[Y1, Y2] += pr / r
    return out


def main():
    ok = True
    for Ws, r, top, split in (([1, 2], 13, None, 0), ([2, 1], 31, None, 0), ([2, 1, 1], 19, None, 0),
                              ([1, 1, 2], 7, None, 0), ([1, 2], 13, 'sine', 0), ([2, 1, 2], 19, 'sine', 0),
                              ([1, 2], 13, 'sine', 1), ([2, 1, 2], 19, 'sine', -1), ([1, 1, 1], 7, 'sine', 1)):
        lam = cube_root(r)
        for c in (1, 5):
            b, s = brute(Ws, r, lam, c, top, split), sampler_dist(Ws, r, lam, c, top, split)
            err = float(np.abs(b - s).max())
            good = err < 1e-12 and abs(s.sum() - 1) < 1e-12
            ok &= good
            print('(a) Ws %s%s, r %d, c %d: max |brute - sampler| %.1e, total %.15f %s' % (
                Ws, (' top sine' + (' split %+d' % split if split else '')) if top else '', r, c, err, s.sum(),
                'OK' if good else 'FAIL'))
    try:
        sys.path.insert(0, os.path.join(HERE, '..', '..', 'o1'))
        saved = {k: getattr(hcore.Run, k) for k in ('hexreg', 'hexadd', 'reg', 'measure')}
        import hexva                        # patches hcore.Run at import: undone right away
        for k, v in saved.items():
            setattr(hcore.Run, k, v)
        rng = random.Random(3)
        worst = 0.0
        for _ in range(200):
            W, m, r = rng.choice([1, 2, 3]), rng.choice([2, 3]), rng.choice([7, 13, 19, 31, 1009])
            lam, t = cube_root(r), rng.randrange(m)
            uc, Y1, Y2 = rng.randrange(r), rng.randrange(1 << (W * t)), rng.randrange(1 << (W * t))
            worst = max(worst, float(np.abs(hexreg.cond([W] * m, t, uc, r, lam, Y1, Y2) -
                                            hexva.hex_cond(W, m, t, uc, r, lam, Y1, Y2)).max()))
        ok &= worst == 0.0
        print('(b) equal sizes against o1/hexva.hex_cond, 200 random steps: max |diff| %.1e %s'
              % (worst, 'OK' if worst == 0.0 else 'FAIL'))
    except ImportError:
        print('(b) o1/hexva.py not found: skipped')
    # (c) pricing
    cost.KINDS.clear()
    cost.KINDS['a8'] = cost.declared_kind('declared:800000:1247:0.001:8')
    cost.KINDS['a9'] = cost.declared_kind('declared:900000:1249:0.002:9')
    shape = dict(reg_bits=[6, 2 * (6 + 8 * 3 + 9)], add_bits=[6], add_q=[True], start_zero=True, offset=True,
                 hex=[dict(Ws=[9, 8, 8, 8, 6], kinds=['a9', 'a8', 'a8', 'a8', 'first'], q=False)])
    c = cost.hex_run_cost(shape, 256)
    f = hexreg.first_ss_toffoli(18, 256)
    exp_tof = 900000 + 3 * 800000 + f['toffoli']
    good = (c['toffoli'] == exp_tof and c['qubits'] == 1249 and abs(c['fail_weight'] - 0.005) < 1e-15 and
            c['additions'] == 4 and c['first_bits'] == 18 and not c['q_tables'] and not c['measured'])
    ok &= good
    print('(c) priced run: %d Toffoli (expect %d: first lookup %d by formula), %d qubits, fail weight %.4g, %d '
          'additions %s' % (c['toffoli'], exp_tof, f['toffoli'], c['qubits'], c['fail_weight'], c['additions'],
                            'OK' if good else 'FAIL'))
    for bad, why in ((dict(kinds=['a8', 'a8', 'a8', 'a8', 'first']), 'size 9 window priced by a W = 8 kind'),
                     (dict(kinds=['zz', 'a8', 'a8', 'a8', 'first']), 'kind not loaded')):
        sh = dict(shape, hex=[dict(shape['hex'][0], **bad)])
        try:
            cost.hex_run_cost(sh, 256)
            ok = False
            print('(c) %s: accepted FAIL' % why)
        except ValueError as e:
            print('(c) %s: rejected (%s) OK' % (why, e))
    # (d) end to end on a toy curve: the run API, its recorded cost, and the outcome peak of the sampler
    import json
    inst = [i for i in json.load(open(os.path.join(HERE, '..', 'instances.json'))) if i['n'] == 24][0]
    cost.KINDS['k2'] = cost.declared_kind('declared:1000:100:0:2')
    cost.KINDS['k3'] = cost.declared_kind('declared:2000:120:0:3')
    h = hcore.Context(inst, 12345, 7)
    Ws = [2, 3, 3, 2, 2]
    for _ in range(20):
        run = h.run()
        run.offset()
        b = run.reg(4)
        a = run.hexreg(Ws)
        run.add(b, h.Q)
        run.hexadd(a, h.G, ['k2', 'k3', 'k3', 'k2', 'first'])
        k, (Y1, Y2) = run.measure()
        good = 0 <= k < 16 and 0 <= Y1 < 1 << 12 and 0 <= Y2 < 1 << 12
    run = h.run()
    run.offset()
    a = run.hexreg(Ws, top='sine')
    try:
        run.hexadd(a, h.G, 'k2')
        good = False
        print("(d) top='sine' without a 'first' window: accepted FAIL")
    except hcore.HarnessError:
        pass
    run = h.run()
    run.offset()
    b = run.reg(4)
    a = run.hexreg(Ws, top='sine')
    run.add(b, h.Q)
    run.hexadd(a, h.G, ['k2', 'k3', 'k3', 'k2', 'first'])
    run.measure()
    cs = h.runs[-1][1]
    good &= cs['toffoli'] == 2 * 1000 + 2 * 2000 + hexreg.first_ss_toffoli(8, 24)['toffoli'] + 2 * (3 * 2 - 2) and \
        'counted sine-state preparation' in cs['grade']
    run = h.run()                        # odd sigma: split = +1 puts one more top bit on s1
    run.offset()
    b = run.reg(4)
    a = run.hexreg(Ws, top='sine', split=1)
    run.add(b, h.Q)
    run.hexadd(a, h.G, ['k2', 'k3', 'k3', 'k2', 'first'])
    _k, (Y1s, Y2s) = run.measure()
    cs1 = h.runs[-1][1]
    good &= cs1['first_bits'] == 4 + 2 * 2 + 1 and Y1s < 1 << 13 and Y2s < 1 << 12 and cs1['toffoli'] == \
        2 * 1000 + 2 * 2000 + hexreg.first_ss_toffoli(9, 24)['toffoli'] + (3 * 3 - 2) + (3 * 2 - 2)
    try:
        h.run().hexreg(Ws, split=1)
        good = False
        print('(d) split without a sine top: accepted FAIL')
    except hcore.HarnessError:
        pass
    c = h.runs[-3][1]
    good &= c['kinds'] == ['k2', 'k3'] and c['first_bits'] == 8 and c['toffoli'] == 2 * 1000 + 2 * 2000 + \
        hexreg.first_ss_toffoli(8, 24)['toffoli'] and c['qubits'] == max(120, hexreg.first_ss_toffoli(8, 24)['peak_qubits'])
    ok &= good
    print('(d) 20 toy runs through Run.hexreg / hexadd / measure, recorded cost %s %s' % (
        {k: c[k] for k in ('toffoli', 'qubits', 'first_bits', 'kinds')}, 'OK' if good else 'FAIL'))
    rng = random.Random(1)
    r, lam, errs = inst['r'], inst['lam'], []
    for _ in range(40):
        u, cc = rng.randrange(r), rng.randrange(1, r)
        (Y1, Y2), _p = hexreg.sample(rng, Ws, cc, u, r, lam)
        N = 1 << sum(Ws)
        e1 = (Y1 / N - u * cc / r) % 1
        e2 = (Y2 / N - u * cc * lam % r / r) % 1
        errs.append(max(min(e1, 1 - e1), min(e2, 1 - e2)) * N)
    good = float(np.median(errs)) < 2
    ok &= good
    print('(d) toy curve %s, Ws %s: median distance of Y from (u c, u c lam) N / r: %.2f cells %s'
          % (inst['name'], Ws, float(np.median(errs)), 'OK' if good else 'FAIL'))
    print('ALL OK' if ok else 'SOME FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
