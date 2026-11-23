"""Mixed-radix Eisenstein registers (Run.mixreg, harness/mixreg.py, construction).

    register's outcome law P(eta) = |sum_s amp(s) e(uc phi(s) / r - <eta, s>)|^2 over all characters of Z[w] / (Pi_m)
    equals the product of the sampler's conditionals (logprob), and the sampler's draws follow it.
(b) a mixed register of hex windows only gives the outcome law of Run.hexreg's sampler (harness/hexreg.py), sine top
    included: P(Y) of hexreg.sample equals exp(logprob(Y / 2^P)) of mixreg.
(c) labels, register qubits and table entries; a run with a mixed register is priced window by window with kinds of
    matching labels (W, 'nw', 'mw'); a kind of another label is refused; outcomes and bad-run outcomes lie in [0, N)^2.
(d) Lemma G: a path Monte Carlo file with a 'layout' gives that layout's labels; runs whose additions differ are refused.
Run: python3 harness/tests/test_mixreg.py
"""
import itertools, json, math, os, random, sys, tempfile
from fractions import Fraction
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import numpy as np                          # noqa: E402
import cost, hcore, hexreg, mixreg as mr    # noqa: E402


def cube_root(r):
    for g in range(2, r):
        l = pow(g, (r - 1) // 3, r)
        if l != 1:
            return l


def brute(wins, uc, r, lam):
    """o1/tests_mixrad.py brute(), unchanged"""
    P = mr.positions(wins)
    lists = []
    for i, w in enumerate(wins):
        d1, d2, amp = w.arrays()
        items = []
        for a in range(w.e[0]):
            for b in range(w.e[1]):
                d = (int(d1[a, b]), int(d2[a, b]))
                am = 1.0 / math.sqrt(w.N) if amp is None else float(amp[a, b])
                items.append((mr.emul(P[i], d), am))
        lists.append(items)
    states = []
    for combo in itertools.product(*lists):
        s = (sum(c[0][0] for c in combo), sum(c[0][1] for c in combo))
        am = 1.0
        for c in combo:
            am *= c[1]
        states.append((s, am))
    U, V, e = mr.snf2(mr.mat(P[-1]))
    A = np.array([am for _, am in states])
    ph0 = np.array([float(Fraction(uc * mr.phi_r(s, r, lam) % r, r)) for s, _ in states])
    out = {}
    for y0 in range(e[0]):
        for y1 in range(e[1]):
            f = (Fraction(y0, e[0]), Fraction(y1, e[1]))
            eta = (mr._frac1(U[0][0] * f[0] + U[1][0] * f[1]), mr._frac1(U[0][1] * f[0] + U[1][1] * f[1]))
            ph = np.array([float(mr._frac1(eta[0] * s[0] + eta[1] * s[1])) for s, _ in states])
            out[eta] = abs((A * np.exp(2j * np.pi * (ph0 - ph))).sum()) ** 2
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}


def part_a():
    worst, tvs = 0.0, []
    for spec, r in [('n1,s1', 13), ('m1,h1,s1', 31), ('n1,m1', 37), ('h1,n1,s2', 43), ('n2,s1', 61), ('m1,n1,s1', 67)]:
        lam = cube_root(r)
        wins = mr.parse_layout(spec)
        for uc in (1, 5 % r, r - 2):
            bf = brute(wins, uc, r, lam)
            for eta, p in bf.items():
                q = math.exp(mr.logprob(eta, wins, uc, r, lam)) if p > 1e-300 else 0.0
                worst = max(worst, abs(p - q))
        rng = random.Random(1)
        cnt, T = {}, 2000
        for _ in range(T):
            eta = mr.sample(rng, wins, 3 % r, r, lam)
            cnt[eta] = cnt.get(eta, 0) + 1
        bf = brute(wins, 3 % r, r, lam)
        assert set(cnt) <= set(bf)
        tv = 0.5 * sum(abs(cnt.get(k, 0) / T - v) for k, v in bf.items())
        # expected TV of an exact sampler ~ sqrt(support / (2 pi T)); allow 1.5x
        tvs.append((spec, tv, 1.5 * math.sqrt(len(bf) / (2 * math.pi * T))))
    ok = worst < 1e-12 and all(tv < lim for _, tv, lim in tvs)
    print('(a) brute force: max |P - sampler| = %.2e over 6 layouts x 3 uc; draws TV %s %s'
          % (worst, ', '.join('%s %.3f (< %.3f)' % t for t in tvs), 'OK' if ok else 'FAIL'))
    return ok


def part_b():
    """hex windows only: hexreg's law (Y in [0, 2^P)^2, eta = Y / 2^P) against mixreg's logprob"""
    inst = [i for i in json.load(open(os.path.join(HERE, '..', 'instances.json'))) if i['n'] == 24][0]
    r = int(inst['r']) if isinstance(inst['r'], (int, str)) else inst['r']
    h = hcore.Context(inst, 7, 7)
    lam = h.lam
    worst, n = 0.0, 0
    rng = random.Random(5)
    for Ws, top, spec in (([3, 2, 4], None, 'h3,h2,h4'), ([2, 3, 2], 'sine', 'h2,h3,s2'), ([4, 3], 'sine', 'h4,s3')):
        wins = mr.parse_layout(spec)
        P = sum(Ws)
        for _ in range(40):
            uc = rng.randrange(1, h.r)
            Y, p = hexreg.sample(rng, Ws, uc, 1, h.r, lam, top)
            q = math.exp(mr.logprob((Fraction(Y[0], 1 << P), Fraction(Y[1], 1 << P)), wins, uc, h.r, lam))
            worst = max(worst, abs(p - q) / p)
            n += 1
    ok = worst < 1e-9
    print('(b) hex windows: hexreg.sample against mixreg.logprob on %d outcomes, max relative diff %.2e %s'
          % (n, worst, 'OK' if ok else 'FAIL'))
    return ok


def part_c():
    ok = (mr.labels('h9,n7,m7x10,h8x3,s4') == [9, 'n7'] + ['m7'] * 10 + [8] * 3 and mr.win_entries('n7') == 19116
          and mr.win_entries(8) == 10924 and mr.win_entries(9) == 43692 and mr.win_qubits('m7') == 17
          and mr.win_qubits(9) == 18 and not mr.check_label('k7') and not mr.check_label(11)
          and cost.kind_label('m7_window_add', 7) == 'm7' and cost.kind_label('hex_window_add', 8) == 8
          and cost.kind_label('hex_terminal_add', 9) == 9 and 'hex_terminal_add' in cost.HEX_KINDS)
    inst = [i for i in json.load(open(os.path.join(HERE, '..', 'instances.json'))) if i['n'] == 24][0]
    cost.KINDS.clear()
    cost.KINDS.update(h2=cost.declared_kind('declared:1000:100:0:2'), n2=cost.declared_kind('declared:1200:101:0:n2'),
                      m2=cost.declared_kind('declared:1300:102:0:m2'))
    spec = 'h2,n2,m2,s2'
    h = hcore.Context(inst, 11, 11)
    N = mr.register_norm(mr.parse_layout(spec))
    outs = []
    for _ in range(20):
        run = h.run()
        run.offset()
        b = run.reg(4)
        a = run.mixreg(spec)
        run.add(b, h.Q)
        run.hexadd(a, h.G, ['h2', 'n2', 'm2', 'first'])
        k, Y = run.measure()
        outs.append(Y)
    c = h.runs[-1][1]
    # peak: the largest part (the 8-bit first lookup's 110 qubits at n = 24 here); Toffoli: the kinds, the first lookup
    # and the two counted sine preparations of the 2-bit top (2 x (3 x 2 - 2))
    ok &= (c['add_ws'] == [2, 'n2', 'm2'] and c['first_bits'] == 4 + 4 and c['additions'] == 3
           and c['qubits'] == max(102, hexreg.first_ss_toffoli(8, 24)['peak_qubits'])
           and c['toffoli'] == 3500 + c['first_lookup'] + 8 and c['table_entries'] == 4 + 20 + 20 + 2 ** 8
           and h.runs[-1][0]['reg_bits'] == [4, 4 + 7 + 7 + 4] and all(0 <= y < N for Y in outs for y in Y))
    bad = 0
    for kinds in (['h2', 'm2', 'n2', 'first'], ['h2', 'n2', 'h2', 'first'], ['n2', 'n2', 'm2', 'first']):
        run = h.run()
        run.offset()
        a = run.mixreg(spec)
        run.hexadd(a, h.G, kinds)
        try:
            run.measure()
        except ValueError:
            bad += 1
    ok &= bad == 3
    for spec2 in ('h2,s2,n2', 'h11', 'n9', 'q2', ''):
        try:
            h.run().mixreg(spec2)
            ok = False
        except hcore.HarnessError:
            pass
    # a bad run's outcome: harness _hex_garbage stays in range
    rng = random.Random(3)
    e = dict(Ws=[2, 'n2', 'm2', 2], mix=mr.parse_layout(spec), split=0)
    ok &= all(0 <= y < N for y in hcore._hex_garbage(rng, e))
    print('(c) labels %s, pricing: windows %s, first lookup %d bits, %d table entries, wrong-label kinds refused %d / 3, '
          'bad layouts refused %s' % (mr.labels(spec), c['add_ws'], c['first_bits'], c['table_entries'], bad,
                                      'OK' if ok else 'FAIL'))
    return ok


def part_d():
    inst = [i for i in json.load(open(os.path.join(HERE, '..', 'instances.json'))) if i['n'] == 24][0]
    pm = dict(paths=1000, J=3, layout='h2,n2,m2,s2', windows=[2, 3, 3], per_addition=dict(lanes=3000, DO=6, P=0),
              end=dict(rflag=5, offcurve=0, oncurve_wrong=0, recovered=0), counts_DO_P={'0,0': 994, '1,0': 6})
    ok = True
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 'pm.json')
        json.dump(pm, open(f, 'w'))
        g = cost.global_mc_constants(f)
        ok &= g['windows'] == [2, 'n2', 'm2']
        cost.KINDS.clear()
        cost.KINDS.update(h2=cost.declared_kind('declared:1000:100:0:2'), n2=cost.declared_kind('declared:1200:101:0:n2'),
                          m2=cost.declared_kind('declared:1300:102:0:m2'))
        for spec, kinds, want in (('h2,n2,m2,s2', ['h2', 'n2', 'm2', 'first'], True),
                                  ('h2,m2,n2,s2', ['h2', 'm2', 'n2', 'first'], False)):
            h = hcore.Context(inst, 13, 13, approx=True, kappa=True, global_mc=g)
            run = h.run()
            run.offset(global_scalar=True)
            b = run.reg(4, sine=4)
            a = run.mixreg(spec)
            run.add(b, h.Q)
            run.hexadd(a, h.G, kinds)
            try:
                run.measure()
                got = True
            except hcore.HarnessError:
                got = False
            ok &= got == want
    print('(d) Lemma G: Monte Carlo layout -> windows %s; matching run accepted, swapped n / m refused %s'
          % (g['windows'], 'OK' if ok else 'FAIL'))
    return ok


if __name__ == '__main__':
    res = [part_a(), part_b(), part_c(), part_d()]
    print('ALL OK' if all(res) else 'SOME FAILED')
    sys.exit(0 if all(res) else 1)
