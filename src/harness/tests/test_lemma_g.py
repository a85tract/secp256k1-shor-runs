"""Lemma G runs (paper Sec. V.D and App. B; harness: offset(global_scalar=True) + run.py --global-mc) on hex runs.

(a) global_mc_constants: path-level bounds (E[N_DO + 2 N_P], E[corr] over fixed blocks, phi) on a small synthetic
    path Monte Carlo, against direct computation.
(b) keep rule on a 24-bit toy curve: P(visible flag) = phi (binomial check), flagged runs carry run.flagged and
    make try_subset fail; with g0 = 1, e0 = 0, phi = 0 every run is kept; with e0 large every run is an unflagged
    bad run; a run whose additions differ from the Monte Carlo's J is refused; a kind charged at an abort cap is refused
    against a Monte Carlo that did not count aborts under the same stream cap.
(a2) aborted paths of a capped Monte Carlo count in phi; batches under different caps do not pool.
(c) the e0 weight Lambda of a sine b register and a sine top window: the product of their L1 factors.
(d) the per-run table charge: 2 group operations per table entry.
    with and without per-addition lists do not pool, addition i's kind must carry entry i's cap, and a root-sign
    terminal kind (KIND hex_terminal_add) charges its tables, cost.terminal_classical_units(W), every run.
(f) run-level abort cap (o1/s7/termva7.py folded in): constants, pooling, pricing at run_cap, no double application.
Run: python3 harness/tests/test_lemma_g.py
"""
import json, math, os, random, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import cost, hcore, hexreg, sineprep        # noqa: E402
import numpy as np                          # noqa: E402


def main():
    ok = True
    pm = dict(paths=1000, J=3, per_addition=dict(lanes=10000, DO=20, P=3, wrong_oncurve=0),
              end=dict(rflag=30, offcurve=5, oncurve_wrong=0, recovered=1),
              counts_DO_P={'0,0': 960, '1,0': 30, '2,0': 6, '1,1': 3, '0,2': 1})
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 'pm.json')
        json.dump(pm, open(f, 'w'))
        g = cost.global_mc_constants(f, 0.001)
        g2 = cost.global_mc_constants(f + ',' + f, 0.001)     # two batches pooled: every count doubled
        f3 = os.path.join(d, 'pm3.json')
        json.dump(dict(pm, J=2), open(f3, 'w'))
        try:
            cost.global_mc_constants(f + ',' + f3, 0.001)
            pooled_ok = False
        except ValueError:
            pooled_ok = True
    # path-level variables, J = 3, vmax = 6, blocks 1, 2, 3, 4-6 at a / 4 each:
    # S = N_DO + 2 N_P: (1,0) -> 1, (2,0) -> 2, (1,1) -> 3, (0,2) -> 4; corr: (2,0) -> 1, (1,1) -> 2, (0,2) -> 4
    ub = lambda k, n, a: cost.cp_upper(k, n, a)
    q = 0.00025
    exp_S = ub(40, 1000, q) + ub(10, 1000, q) + ub(4, 1000, q) + 3 * ub(1, 1000, q)
    exp_corr = ub(10, 1000, q) + ub(4, 1000, q) + ub(1, 1000, q) + 3 * ub(1, 1000, q)
    exp_S2 = ub(80, 2000, q) + ub(20, 2000, q) + ub(8, 2000, q) + 3 * ub(2, 2000, q)
    pooled_ok &= abs(g2['eS'] - exp_S2) < 1e-15 and abs(g2['phi'] - (1 - cost.cp_upper(1930, 2000, 0.001))) < 1e-15
    good = (pooled_ok and g['windows'] == [8, 8, 8] and abs(g['ecorr'] - exp_corr) < 1e-15 and
            abs(g['eS'] - exp_S) < 1e-15 and abs(g['fDO'] - 0.002) < 1e-15 and
            abs(g['phi'] - (1 - ub(965, 1000, 0.001))) < 1e-15 and abs(g['alpha'] - 0.005) < 1e-15 and g['J'] == 3)
    good &= cost.mean_blocks(30) == [(1, 1), (2, 2), (3, 3), (4, 7), (8, 15), (16, 30)] and cost.mean_blocks(6) == [
        (1, 1), (2, 2), (3, 3), (4, 6)]
    try:                                     # the histogram must count every path
        cost.ub_mean({0: 10, 1: 2}, 13, 6, 0.001)
        good = False
    except ValueError:
        pass
    # sanity of the CP bounds against the binomial tail
    p = cost.cp_upper(20, 10000, 0.001)
    tail = math.exp(cost._log_binom_cdf(20, 10000, p))
    good &= abs(tail - 0.001) < 1e-6 and cost.cp_lower(0, 10, 0.1) == 0.0
    ok &= good
    print('(a) constants: E[N_DO + 2 N_P] UB %.4g (expect %.4g), E[corr] UB %.4g (expect %.4g), phi LB %.4g, alpha %.3g, '
          'pooling %s %s' % (g['eS'], exp_S, g['ecorr'], exp_corr, g['phi'], g['alpha'], 'OK' if pooled_ok else 'FAIL',
             'OK' if good else 'FAIL'))
    # (a2) abort cap (h1/hexcap/path_mc_cap.py): aborted paths ('aflag') are visibly flagged ends; files counted
    # under another stream cap (or none) do not pool
    with tempfile.TemporaryDirectory() as d:
        f, f4 = os.path.join(d, 'pm.json'), os.path.join(d, 'pm4.json')
        json.dump(pm, open(f, 'w'))
        json.dump(dict(pm, stream_cap=726000, stream_toffoli=761147, end=dict(pm['end'], aflag=4)), open(f4, 'w'))
        g4 = cost.global_mc_constants(f4, 0.001)
        good = (abs(g4['phi'] - (1 - ub(961, 1000, 0.001))) < 1e-15 and g4['stream_cap'] == 726000
                and g4['stream_toffoli'] == 761147 and g['stream_cap'] is None)
        try:
            cost.global_mc_constants(f + ',' + f4, 0.001)
            good = False
        except ValueError:
            pass
    ok &= good
    print('(a2) abort cap: phi LB %.4g with 4 aborted paths, cap %s, mixed caps refused %s'
          % (g4['phi'], g4['stream_cap'], 'OK' if good else 'FAIL'))

    inst = [i for i in json.load(open(os.path.join(HERE, '..', 'instances.json'))) if i['n'] == 24][0]
    cost.KINDS.clear()
    cost.KINDS['k2'] = cost.declared_kind('declared:1000:100:0.01:2')
    cost.KINDS['k3'] = cost.declared_kind('declared:1500:100:0.01:3')
    Ws = [2, 2, 2, 3]                         # three approximate windows, the top one in the first lookup

    def runs(gmc, n, seed, top=None, adds=None):
        h = hcore.Context(inst, 4242, seed, fail_per_add=0.01, kappa=True, approx=True, global_mc=gmc)
        outs = []
        for _ in range(n):
            run = h.run()
            run.offset(global_scalar=True)
            b = run.reg(4, sine=4)
            a = run.hexreg(adds or Ws, top=top)
            run.add(b, h.Q)
            run.hexadd(a, h.G, ['k%d' % w for w in (adds or Ws)[:-1]] + ['first'])
            run.measure()
            outs.append(run)
        return h, outs

    base = dict(J=3, windows=[2, 2, 2], eS=0.0, e0_paths=0.0, phi=0.0, alpha=0.0, src='test')
    h, rs = runs(dict(base), 300, 1)
    good = h.garbage_runs == 0 and not any(r.flagged for r in rs) and h.lemma_g_runs == 300
    h, rs = runs(dict(base, e0_paths=10.0), 300, 2)
    good &= h.garbage_runs == 300 and not any(r.flagged for r in rs)
    phi = 0.25
    h, rs = runs(dict(base, phi=phi, eS=0.15), 2000, 3)    # g0 = 0.85: g0^2 <= 1 - phi as Theorem G needs
    nf = sum(r.flagged for r in rs)
    z = (nf - 2000 * phi) / math.sqrt(2000 * phi * (1 - phi))
    good &= abs(z) < 4 and h.garbage_runs >= nf
    try:
        runs(dict(base, phi=phi), 1, 6)          # g0 ~ 1: g0^2 > 1 - phi, refused
        good = False
    except hcore.HarnessError:
        pass
    fl = [r.id for r in rs if r.flagged][:1] + [r.id for r in rs if not r.flagged][:3]
    good &= h.try_subset(fl, lambda: 1, 1000) is None and h.try_subset(fl[1:], lambda: 1, 1000) == 1
    for adds in ([2, 2, 3], [2, 2, 3, 3]):    # 2 additions against J = 3; 3 additions with a W = 3 window
        try:
            runs(dict(base), 1, 4, adds=adds)
            good = False
        except hcore.HarnessError:
            pass
    # a kind charged at an abort cap needs a Monte Carlo that counted aborts under the same stream cap and stream
    capped = dict(cost.KINDS['k2'], abort_cap=1000, stream_cap=726000, stream_toffoli=761147)
    k2 = cost.KINDS['k2']
    cost.KINDS['k2'] = capped
    for gm, want in ((dict(base), False), (dict(base, stream_cap=726000, stream_toffoli=761147), True),
                     (dict(base, stream_cap=725000, stream_toffoli=761147), False)):
        try:
            runs(gm, 1, 7)
            good &= want
        except hcore.HarnessError:
            good &= not want
    cost.KINDS['k2'] = k2
    ok &= good
    print('(b) keep rule: all kept at e0 = 0, all bad at e0 = 10, %d / 2000 flagged at phi = 0.25 (z = %.2f), '
          'try_subset fails on a flagged run, runs refused against windows [2, 2, 2]: 2 additions, a W = 3 window %s'
          % (nf, z, 'OK' if good else 'FAIL'))
    h, rs = runs(dict(base), 1, 5, top='sine')
    lam = rs[0]._lemma_g_weight()
    l1 = lambda a: float(np.abs(a).sum()) / math.sqrt(a.size * float((np.abs(a) ** 2).sum()))
    exp = l1(np.asarray(sineprep.amplitudes(4))) * l1(hexreg.sine(3)) ** 2
    good = abs(lam - exp) < 1e-12 and lam < 1
    ok &= good
    print('(c) Lambda of a sine b (4 bits) and a sine top window (W = 3): %.6f (expect %.6f) %s'
          % (lam, exp, 'OK' if good else 'FAIL'))
    entries = 3 * ((4 ** 2 - 4) // 6 + 2) + 2 ** (4 + 6)
    good = h.ops['group'] >= 2 * entries and h.runs[-1][1]['table_entries'] == entries
    ok &= good
    print('(d) per-run tables: %d entries charged twice (%d group ops in the trial) %s'
          % (entries, h.ops['group'], 'OK' if good else 'FAIL'))
    # phi; addition i's kind must carry entry i's cap; a terminal kind's tables are charged every run
    TERM, LEAD = (379100, 392944), (726000, 761147)
    lists = dict(stream_caps=[TERM[0], LEAD[0], LEAD[0]], stream_toffolis=[TERM[1], LEAD[1], LEAD[1]])
    with tempfile.TemporaryDirectory() as d:
        f, f5 = os.path.join(d, 'pm.json'), os.path.join(d, 'pm5.json')
        json.dump(dict(pm, stream_cap=LEAD[0], stream_toffoli=LEAD[1]), open(f, 'w'))
        json.dump(dict(pm, stream_cap=LEAD[0], stream_toffoli=LEAD[1], end=dict(pm['end'], tflag=7), **lists),
                  open(f5, 'w'))
        g5 = cost.global_mc_constants(f5, 0.001)
        good = (abs(g5['phi'] - (1 - ub(958, 1000, 0.001))) < 1e-15 and g5['stream_caps'] == lists['stream_caps']
                and g5['stream_toffolis'] == lists['stream_toffolis'])
        try:
            cost.global_mc_constants(f + ',' + f5, 0.001)       # with and without per-addition lists: refused
            good = False
        except ValueError:
            pass
    cost.KINDS['t2'] = dict(cost.KINDS['k2'], abort_cap=1000, stream_cap=TERM[0], stream_toffoli=TERM[1],
                            vb_kind='hex_terminal_add')
    cost.KINDS['c2'] = dict(cost.KINDS['k2'], abort_cap=1000, stream_cap=LEAD[0], stream_toffoli=LEAD[1])
    gm = dict(base, stream_cap=LEAD[0], stream_toffoli=LEAD[1], **lists)

    def run_kinds(kinds, seed):
        h = hcore.Context(inst, 4242, seed, fail_per_add=0.01, kappa=True, approx=True, global_mc=gm)
        run = h.run()
        run.offset(global_scalar=True)
        b = run.reg(4, sine=4)
        a = run.hexreg(Ws)
        run.add(b, h.Q)
        run.hexadd(a, h.G, kinds + ['first'])
        before = h.ops['other']
        run.measure()
        return h.ops['other'] - before
    units = run_kinds(['t2', 'c2', 'c2'], 8)
    good &= units == cost.terminal_classical_units(2) == 310 * 16 + 1032
    for kinds in (['c2', 't2', 'c2'], ['c2', 'c2', 'c2'], ['t2', 'c2', 'k2']):   # terminal misplaced; uncapped kind
        try:
            run_kinds(kinds, 9)
            good = False
        except hcore.HarnessError:
            pass
    ok &= good
    print('(e) per-addition caps: phi LB %.4g with 7 terminal flags, lists pool only with lists, terminal at the '
          'lowest addition accepted and elsewhere refused, %d terminal units charged per run %s'
          % (g5['phi'], units, 'OK' if good else 'FAIL'))
    # (f) run-level abort cap (o1/s7/termva7.py folded in): constants carry run_cap; a Lemma G run's streams are
    # charged run_cap instead of sum(stream_caps); files with other run caps do not pool; a cap at or above the sum
    # is refused; a second application (an old wrapper) is refused
    RC = sum(lists['stream_caps']) - 5000
    with tempfile.TemporaryDirectory() as d:
        f6, f7, f8 = (os.path.join(d, x) for x in ('pm6.json', 'pm7.json', 'pm8.json'))
        base5 = dict(pm, stream_cap=LEAD[0], stream_toffoli=LEAD[1], **lists)
        json.dump(dict(base5, run_cap=RC), open(f6, 'w'))
        json.dump(dict(base5, run_cap=RC - 1), open(f7, 'w'))
        json.dump(dict(base5, run_cap=sum(lists['stream_caps'])), open(f8, 'w'))
        g6 = cost.global_mc_constants(f6, 0.001)
        good = g6['run_cap'] == RC
        for bad in (f6 + ',' + f7, f8):
            try:
                cost.global_mc_constants(bad, 0.001)
                good = False
            except ValueError:
                pass
    gm_plain, gm = gm, dict(gm, run_cap=RC)
    h = hcore.Context(inst, 4242, 10, fail_per_add=0.01, kappa=True, approx=True, global_mc=gm)
    run = h.run()
    run.offset(global_scalar=True)
    b = run.reg(4, sine=4)
    a = run.hexreg(Ws)
    run.add(b, h.Q)
    run.hexadd(a, h.G, ['t2', 'c2', 'c2', 'first'])
    run.measure()
    c6 = h.runs[-1][1]
    h0 = hcore.Context(inst, 4242, 10, fail_per_add=0.01, kappa=True, approx=True, global_mc=gm_plain)
    run = h0.run()
    run.offset(global_scalar=True)
    b = run.reg(4, sine=4)
    a = run.hexreg(Ws)
    run.add(b, h0.Q)
    run.hexadd(a, h0.G, ['t2', 'c2', 'c2', 'first'])
    run.measure()
    c0 = h0.runs[-1][1]
    good &= c0['toffoli'] - c6['toffoli'] == 5000 and c6['cost'] == c6['qubits'] * c6['toffoli']
    hrc = cost.hex_run_cost
    cost.hex_run_cost = lambda shape, n: dict(hrc(shape, n), run_cap=RC)       # what an old wrapper would leave
    try:
        h = hcore.Context(inst, 4242, 11, fail_per_add=0.01, kappa=True, approx=True, global_mc=gm)
        run = h.run()
        run.offset(global_scalar=True)
        b = run.reg(4, sine=4)
        a = run.hexreg(Ws)
        run.add(b, h.Q)
        run.hexadd(a, h.G, ['t2', 'c2', 'c2', 'first'])
        run.measure()
        good = False
    except hcore.HarnessError:
        pass
    finally:
        cost.hex_run_cost = hrc
    ok &= good
    print('(f) run-level cap: constants carry it, other caps do not pool, a cap >= the sum is refused, the run is '
          'charged %d Toffoli less (sum of stream caps - run cap), a second application refused %s'
          % (c0['toffoli'] - c6['toffoli'], 'OK' if good else 'FAIL'))
    print('ALL OK' if ok else 'SOME FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
