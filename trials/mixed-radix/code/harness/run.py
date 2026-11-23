#!/usr/bin/env python3
"""Run a candidate through V-A and print its report.

usage: python3 harness/run.py CANDIDATE.py [--sizes 16,24,32,64,256] [--trials 200] [--trials256 100]
                              [-p KEY=VALUE ...] [--approx] [--measured VB.json | --window W --extra-toffoli X]
                              [--seed S] [--timeout SEC] [--json OUT] [--jobs K] [--trial-log OUT.jsonl]
Trials run in K worker processes (--jobs; pin them with taskset), print progress lines as they finish and, with
--trial-log, append one JSON line per finished trial. A trial that hits --timeout counts as a failed trial.
--resume (with --trial-log): trials already in the log under the same configuration fingerprint (candidate, parameters,
measured / kind / Monte Carlo / first-lookup files, harness sources, seed and options; --print-fingerprint shows it)
are taken from the log instead of rerun; every trial's randomness comes from (seed, n, t), so this equals an
uninterrupted run. --resume-verify K (default 2) first reruns K logged trials per size and stops unless they reproduce.
"""
import argparse, ast, importlib.util, json, math, multiprocessing, numbers, os, random, signal, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost                                   # noqa: E402
from ecc import instances_for                 # noqa: E402
from hcore import Context, BudgetExceeded     # noqa: E402
from scan import scan                         # noqa: E402


class TrialTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise TrialTimeout()


def load_candidate(path, params):
    spec = importlib.util.spec_from_file_location('candidate_' + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for kv in params:
        k, v = kv.split('=', 1)
        if not hasattr(mod, k):
            raise SystemExit('candidate has no parameter %s' % k)
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            pass
        setattr(mod, k, v)
    return mod


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - hw), min(1.0, c + hw))


def exact_lower(k, n, alpha):
    """One-sided exact (Clopper-Pearson) lower confidence bound on a success probability from k successes in n
    trials: the p with P(X >= k | n, p) = alpha (alpha^(1/n) for k = n)."""
    if k <= 0:
        return 0.0
    if k >= n:
        return alpha ** (1.0 / n)

    def tail(p):                              # P(X >= k | n, p)
        lp, lq = math.log(p), math.log1p(-p)
        terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
                 for i in range(k, n + 1)]
        mx = max(terms)
        return math.exp(mx) * sum(math.exp(t - mx) for t in terms)
    lo, hi = 0.0, k / n
    for _ in range(200):
        mid = (lo + hi) / 2
        if tail(mid) < alpha:
            lo = mid
        else:
            hi = mid
    return lo


def lg(x):
    return math.log2(x) if x > 0 else float('-inf')


def run_trial(mod, n, t, inst, args):
    """one trial: a fresh instance secret and context, the candidate's solve(), its counts"""
    rng = random.Random('%s/%d/%d' % (args.seed, n, t))
    d = rng.randrange(1, inst['r'])
    cap = mod.budget(inst['r'].bit_length()) if hasattr(mod, 'budget') else 10000
    h = Context(inst, d, rng.getrandbits(64), max_classical=cost.CLASSICAL_CEILING, max_runs=cap,
                fail_per_add=cost.MODEL['fail_per_add'] if args.approx else 0.0,
                kappa=cost.MODEL.get('flagged', False) if not cost.KINDS else
                all(m['flagged'] or m['exact'] for m in cost.KINDS.values()),
                fail_split=cost.MODEL.get('fail_split') if args.approx else None, approx=args.approx,
                global_mc=cost.GLOBAL_MC_BY_N.get(n, cost.GLOBAL_MC))
    t0 = time.process_time()
    err, ok = None, False
    signal.alarm(args.timeout)
    try:
        out = mod.solve(h)
        ok = isinstance(out, numbers.Integral) and int(out) % inst['r'] == d
    except BudgetExceeded as e:
        err = 'budget: ' + str(e)
    except TrialTimeout:
        err = 'timeout %ds (counted as a failed trial)' % args.timeout
    except Exception as e:  # candidate bug: count and report
        err = '%s: %s' % (type(e).__name__, str(e)[:80])
    finally:
        signal.alarm(0)
    return dict(n=n, t=t, curve=inst.get('name'), ok=bool(ok), err=err, secs=time.process_time() - t0,
                runs=len(h.runs), ops=h.classical_ops(), garbage=h.garbage_runs, heuristic=h.heuristic_runs,
                lemma_g=h.lemma_g_runs,
                unsim=h.unsimulated_runs, capped=h.capped_attempts, max_attempt=h.max_attempt,
                run_costs=[c for _, c in h.runs],
                trial_cost=sum(c['cost'] for _, c in h.runs),
                shapes=sorted({(tuple(s['reg_bits']), c['additions']) for s, c in h.runs}))


_W = {}


def _worker(task):
    n, t, i = task
    return run_trial(_W['mod'], n, t, _W['insts'][n][i], _W['args'])


_TRIAL_KEYS = ('ok', 'runs', 'ops', 'garbage', 'lemma_g', 'heuristic', 'capped', 'max_attempt')


def fingerprint(args):
    """sha256 of everything a trial's result depends on besides (n, t): candidate and harness sources, every input
    file, parameters, seed and options (not --sizes / --trials / --jobs / --progress / output paths)"""
    import hashlib

    def fh(path):
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    here = os.path.dirname(os.path.abspath(__file__))
    files = {'candidate': fh(args.candidate)}
    for name in ('hcore', 'hexreg', 'cost', 'lattice', 'ecc', 'sineprep', 'scan', 'vbspecs'):
        f = os.path.join(here, name + '.py')
        if os.path.exists(f):
            files[name] = fh(f)
    files['instances'] = fh(os.path.join(here, 'instances.json'))
    for spec in args.kind:
        name, src = spec.split('=', 1)
        files['kind:' + name] = src if src.startswith('declared:') else fh(src)
    for spec in args.global_mc:
        nn, fl = spec.split('=', 1) if '=' in spec.split(',')[0] and spec.split('=', 1)[0].isdigit() else ('*', spec)
        files['global_mc:' + nn] = [fh(f) for f in fl.split(',')]
    for key in ('measured', 'first_lookup'):
        if getattr(args, key, None):
            files[key] = fh(getattr(args, key))
    opts = dict(param=sorted(args.param), seed=args.seed, approx=args.approx, timeout=args.timeout,
                all_curves=args.all_curves, pf_scale=args.pf_scale, global_mc_alpha=args.global_mc_alpha,
                measured_n=args.measured_n, window=args.window, extra_toffoli=args.extra_toffoli,
                pack_windows=getattr(args, 'pack_windows', None))
    return hashlib.sha256(json.dumps(dict(files=files, opts=opts), sort_keys=True, default=str).encode()).hexdigest()


def logged_trials(path, fp, n):
    """trials of size n in a trial log under fingerprint fp (a line without 'fp' takes the last header's)"""
    done, cur = {}, None
    if not path or not os.path.exists(path):
        return done
    for line in open(path):
        rec = json.loads(line)
        if rec.get('header'):
            cur = rec['fp']
            continue
        if rec.get('fp', cur) == fp and rec['n'] == n:
            done[rec['t']] = rec
    return done


def run_size(mod, n, trials, args):
    insts = instances_for(n)
    if n == 256 and not args.all_curves:
        insts = [i for i in insts if i.get('name') == 'secp256k1']    # targets are judged on secp256k1 itself
    if not insts:
        raise SystemExit('no instances of %d bits (see ecc.TOY_SIZES)' % n)
    res = dict(n=n, trials=0, ok=0, errors={}, runs=[], ops=[], secs=[], run_costs=[], trial_costs=[],
               shapes=set(), garbage=0, instances=len(insts), secp_trials=0, secp_ok=0)
    _W.update(mod=mod, args=args)
    _W.setdefault('insts', {})[n] = insts
    done = {t: r for t, r in logged_trials(args.trial_log, args.fp, n).items() if t < trials} if args.resume else {}
    if done and args.resume_verify:
        for t in random.Random('verify/%s/%d' % (args.seed, n)).sample(sorted(done), min(args.resume_verify, len(done))):
            r = run_trial(mod, n, t, insts[t % len(insts)], args)
            if any(r[k] != done[t].get(k, r[k]) for k in _TRIAL_KEYS):
                raise SystemExit('--resume: trial %d at n = %d does not reproduce its log line (%s against %s)'
                                 % (t, n, {k: r[k] for k in _TRIAL_KEYS}, {k: done[t].get(k) for k in _TRIAL_KEYS}))
        print('  [n = %d] %d trials taken from %s; %d of them rerun and reproduced' % (
            n, len(done), args.trial_log, min(args.resume_verify, len(done))), flush=True)
    tasks = [(n, t, t % len(insts)) for t in range(trials) if t not in done]
    log = open(args.trial_log, 'a') if args.trial_log else None
    t_start, every = time.time(), max(1, args.progress)
    if args.jobs > 1:
        pool = multiprocessing.get_context('fork').Pool(args.jobs)
        it = pool.imap_unordered(_worker, tasks)
    else:
        pool, it = None, map(_worker, tasks)
    resumed = [dict(r, run_costs=[r['worst_run']] if r.get('worst_run') else [], shapes=r.get('shapes', []),
                    resumed=True) for _, r in sorted(done.items())]
    try:
        for r in __import__('itertools').chain(resumed, it):
            res['trials'] += 1
            res['ok'] += r['ok']
            if r['curve'] == 'secp256k1':
                res['secp_trials'] += 1
                res['secp_ok'] += r['ok']
            if r['err']:
                res['errors'][r['err']] = res['errors'].get(r['err'], 0) + 1
            res['secs'].append(r['secs'])
            res['runs'].append(r['runs'])
            res['ops'].append(r['ops'])
            res['garbage'] += r['garbage']
            res['heuristic'] = res.get('heuristic', 0) + r['heuristic']
            res['lemma_g'] = res.get('lemma_g', 0) + r.get('lemma_g', 0)
            res['unsim'] = res.get('unsim', 0) + r['unsim']
            res['capped'] = res.get('capped', 0) + r.get('capped', 0)
            res['max_attempt'] = max(res.get('max_attempt', 0), r.get('max_attempt', 0))
            res['run_costs'] += r['run_costs']
            res['trial_costs'].append(r['trial_cost'])
            res['shapes'] |= set((tuple(b), a) for b, a in r['shapes'])
            if log and not r.get('resumed'):
                rec = dict((k, v) for k, v in r.items() if k not in ('run_costs', 'shapes'))
                rec.update(fp=args.fp, shapes=r['shapes'],
                           worst_run=max(r['run_costs'], key=lambda c: c['cost']) if r['run_costs'] else None)
                log.write(json.dumps(rec, default=str) + '\n')
                log.flush()
            if args.progress and (res['trials'] % every == 0 or not r['ok']):
                print('  [n = %d] %d / %d trials, %d ok, %d failed (last failure: trial %d %s), %d bad runs, '
                      '%.0f s' % (n, res['trials'], trials, res['ok'], res['trials'] - res['ok'],
                                  r['t'] if not r['ok'] else -1, (r['err'] or 'decoder') if not r['ok'] else '',
                                  res['garbage'], time.time() - t_start), flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
        if log:
            log.close()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('candidate')
    ap.add_argument('--sizes', default='16,24,32,64,128,256')
    ap.add_argument('--trials', type=int, default=200, help='trials per toy size')
    ap.add_argument('--trials256', type=int, default=100, help='trials at n = 256')
    ap.add_argument('-p', '--param', action='append', default=[], help='override a candidate constant KEY=VALUE')
    ap.add_argument('--approx', action='store_true',
                    help='model Häner-style approximate arithmetic: a run returns garbage with probability '
                         '4 pf_add x (additions), pf_add = 0.0335 / 28')
    ap.add_argument('--measured', help='price windowed additions with this V-B result (vb.py --json) of a '
                                       'KIND windowed_point_add circuit at n = 256 (grade "measured")')
    ap.add_argument('--measured-n', type=int, default=256, help=argparse.SUPPRESS)
    ap.add_argument('--kind', action='append', default=[], metavar='NAME=FILE',
                    help='one more V-B-checked circuit for hex windows (Run.hexadd kinds), e.g. w8=hexlead_w8.json; '
                         'NAME=declared:TOFFOLI:QUBITS:PF:W declares an unbuilt one (estimate grade)')
    ap.add_argument('--global-mc', action='append', default=[],
                    help="path Monte Carlo of the run's additions under one global scalar per run "
                         "(o3/sim/path_mc.py or h1/hexcap/path_mc_cap.py JSON; several batches comma-separated): "
                         "Lemma G constants for hex runs with offset(global_scalar=True) (paper Sec. V.D and App. B). N=FILES applies "
                         "to size N only (each size's runs have their own window sequence); FILES to every other size")
    ap.add_argument('--global-mc-alpha', type=float, default=0.001, help='one-sided level of each of its five bounds')
    ap.add_argument('--first-lookup', help='V-B result of the first-lookup circuit of hex runs (o1/first_ss.py run '
                                           'at the address bits with -p P=512); otherwise priced by formula')
    ap.add_argument('--window', type=int, help='estimate grade only: window size w (e.g. 17 for signed 16-bit '
                                               'tables), each window still priced at Häner\'s per-window figure')
    ap.add_argument('--extra-toffoli', type=float, default=0.0,
                    help='estimate grade only: declared extra Toffoli per windowed addition (e.g. a conditional '
                         'negation for signed windows)')
    ap.add_argument('--pack-windows', action='store_true',
                    help='windows may span several add() calls of a run (joint lookup tables over qubits of '
                         'different registers): windows = ceil(total added bits / w) instead of per call (README)')
    ap.add_argument('--all-curves', action='store_true',
                    help='n = 256: rotate trials over all 256-bit instances (default: secp256k1 only; the PASS '
                         'rule at n = 256 is the 95 %% Wilson lower bound >= 0.99 on secp256k1 itself)')
    ap.add_argument('--seed', default='0')
    ap.add_argument('--timeout', type=int, default=600, help='seconds per trial (a timeout is a failed trial)')
    ap.add_argument('--json')
    ap.add_argument('--jobs', type=int, default=1, help='worker processes (pin them with taskset)')
    ap.add_argument('--trial-log', help='append one JSON line per finished trial')
    ap.add_argument('--resume', action='store_true', help='take the trials already in --trial-log under this '
                                                          'configuration fingerprint instead of rerunning them')
    ap.add_argument('--resume-verify', type=int, default=2, help='rerun this many logged trials per size first; '
                                                                  'stop unless they reproduce')
    ap.add_argument('--print-fingerprint', action='store_true', help='print the configuration fingerprint and exit')
    ap.add_argument('--progress', type=int, default=0, help='print a progress line every K trials (and at failures)')
    ap.add_argument('--pf-scale', type=float, default=1.0,
                    help='what-if: multiply the failure weight per addition (robustness checks; never a target claim)')
    args = ap.parse_args()
    args.fp = fingerprint(args)
    if args.print_fingerprint:
        print(args.fp)
        return
    if args.resume and not args.trial_log:
        raise SystemExit('--resume needs --trial-log')

    signal.signal(signal.SIGALRM, _alarm)
    if args.measured:
        if args.window or args.extra_toffoli:
            raise SystemExit('--window / --extra-toffoli modify the estimate grade; a measured circuit has its own')
        cost.use_measured(args.measured, args.measured_n)
    elif args.window or args.extra_toffoli:
        w = args.window or cost.W
        cost.MODEL.update(w=w, toffoli_per_add=cost.HANER_TOFFOLI_PER_ADD + args.extra_toffoli,
                          grade='estimate (declared: window %d, %+.4g Toffoli per addition)' % (w, args.extra_toffoli))
    cost.MODEL['pack'] = bool(args.pack_windows)
    for kv in args.kind:
        name, spec = kv.split('=', 1)
        if name == 'first' or name in cost.KINDS:
            raise SystemExit("--kind %s: the name 'first' is reserved and every name is loaded once" % name)
        cost.KINDS[name] = (cost.declared_kind(spec) if spec.startswith('declared:')
                            else cost.model_from_vb(spec, args.measured_n))
    if args.first_lookup:
        cost.use_first_lookup(args.first_lookup, args.measured_n)
    for spec in args.global_mc:
        nn, files = spec.split('=', 1) if '=' in spec.split(',')[0] and spec.split('=', 1)[0].isdigit() else (None, spec)
        g = cost.global_mc_constants(files, args.global_mc_alpha)
        if nn is None:
            cost.GLOBAL_MC = g
        else:
            cost.GLOBAL_MC_BY_N[int(nn)] = g
        print('Lemma G constants%s (%s): per path UB E[N_DO + 2 N_P] %.4g (observed per addition: f_DO %.3g, f_P '
              '%.3g), UB E[corr] %.4g, UB recovered %.3g, UB class (iii) %.3g, LB phi %.4g; alpha %.3g%s'
              % ('' if nn is None else ' at n = %s' % nn, g['src'], g['eS'], g['fDO'], g['fP'], g['ecorr'], g['rec'], g['iii'],
                 g['phi'], g['alpha'], '' if g.get('stream_cap') is None else
                 '; aborts counted at a stream cap of %d of %d CCX' % (g['stream_cap'], g['stream_toffoli'])))
    GMC = cost.GLOBAL_MC_BY_N.get(256, cost.GLOBAL_MC)      # the n = 256 constants enter the verdict's confidence
    if args.pf_scale != 1.0:
        for m in [cost.MODEL] + list(cost.KINDS.values()):
            m['fail_per_add'] *= args.pf_scale
            m['pf_source'] = 'what-if x %g: %s' % (args.pf_scale, m.get('pf_source', '?'))
    # joint confidence: union bound over every Monte Carlo bound in use
    alphas = [m['alpha_mc'] for m in ([cost.MODEL] if args.measured else []) + list(cost.KINDS.values())
              if m.get('alpha_mc') is not None]
    if GMC:
        alphas.append(GMC['alpha'])
    alpha_mc = sum(alphas) if alphas else None
    mod = load_candidate(args.candidate, args.param)
    issues = scan(args.candidate)
    sizes = [int(s) for s in args.sizes.split(',')]
    print('candidate %s %s' % (args.candidate, ' '.join(args.param)))
    print('source scan: %s' % ('clean' if not issues else '; '.join('line %d %s' % i for i in issues)))
    if cost.KINDS:      # hex runs are priced by their kinds; their failure law is printed per size (Lemma G or coupling)
        print('hex runs: priced per kind (%s); runs under Lemma G (offset(global_scalar=True) with --global-mc) keep '
              'outcomes by its rule, counted per size below' % ', '.join(sorted(cost.KINDS)))
    print('arithmetic model%s: %s; additions priced %s%s' % (' of runs without hex kinds' if cost.KINDS else '',
        ('approximate with flagged failures, coupling model (bad weight %.3g per addition from the %s, summed '
         'over a run; flagged runs are bad runs, no rerun; runs without offset(), runs whose approximate '
         'additions read tables built from Q, hash runs: old heuristic garbage model)' % (cost.MODEL['fail_per_add'], cost.MODEL.get('pf_source', '?'))
         if cost.MODEL.get('flagged') else 'approximate, HEURISTIC independent-garbage-run model (failure %.3g per '
         'addition, 4 pf per run; the circuit has no flags)' % cost.MODEL['fail_per_add']) if args.approx
        else 'exact', cost.MODEL['grade'], '; windows packed across add() calls' if args.pack_windows else ''))
    print('%5s %6s %8s %17s %6s %9s %9s %8s  %s' % ('n', 'trials', 'success', '95% interval', 'runs',
                                                     'lg ops', 'lg opsmax', 's/trial', 'per-run shape'))
    summary = dict(candidate=args.candidate, params=args.param, approx=args.approx, scan=issues, sizes=[])
    for n in sizes:
        trials = args.trials256 if n == 256 else args.trials
        res = run_size(mod, n, trials, args)
        lo, hi = wilson(res['ok'], res['trials'])
        if n == 256:                    # the n = 256 verdict uses secp256k1 trials only, also with --all-curves
            lo, hi = wilson(res['secp_ok'], res['secp_trials'])
        shapes = sorted(res['shapes'])
        shape_txt = ', '.join('bits %s -> %d add.' % ('+'.join(map(str, b)), a) for b, a in shapes[:3])
        if len(shapes) > 3:
            shape_txt += ' (+%d more)' % (len(shapes) - 3)
        print('%5d %6d %8.4f   [%.4f, %.4f] %6.2f %9.2f %9.2f %8.3f  %s' % (
            n, res['trials'], res['ok'] / res['trials'], lo, hi, sum(res['runs']) / res['trials'],
            lg(sum(res['ops']) / res['trials']), lg(max(res['ops'])), sum(res['secs']) / res['trials'], shape_txt))
        if n == 256 and res['secp_trials'] != res['trials']:
            print('      interval and verdict from the %d secp256k1 trials only (%d ok); pooled over %d curves: %d / %d'
                  % (res['secp_trials'], res['secp_ok'], res['instances'], res['ok'], res['trials']))
        exact = None
        if n == 256 and args.approx and alpha_mc is not None:
            # the success bound and the Monte Carlo bound must hold jointly at >= 95 %:
            # exact one-sided bound at alpha = 0.05 - alpha_mc, union bound with the Monte Carlo's alpha_mc
            a_s = 0.05 - alpha_mc
            exact = exact_lower(res['secp_ok'], res['secp_trials'], a_s)
            print('      success >= %.6f: exact one-sided bound at alpha = %.3g; with the Monte Carlo bound '
                  '(alpha = %.3g) the joint confidence is >= 0.95 (union bound); the verdict needs this bound and '
                  'the Wilson bound >= 0.99'
                  % (exact, a_s, alpha_mc))
        for e, c in res['errors'].items():
            print('      %d x %s' % (c, e))
        if res.get('max_attempt'):
            print('      decoding attempts: largest executed attempt %.4g units (2^%.2f); %d attempts stopped at '
                  'their cap' % (res['max_attempt'], math.log2(res['max_attempt']), res.get('capped', 0)))
        if res['garbage']:
            print('      %d runs returned garbage (approximate-arithmetic model%s)' % (
                res['garbage'], ', %d runs under the old heuristic model' % res['heuristic'] if res.get('heuristic')
                else ''))
        if res.get('lemma_g'):      # the arithmetic-model line above does not describe these runs
            print('      %d runs under Lemma G\'s keep rule (paper Sec. V.D and App. B; --global-mc %s), visibly flagged runs counted as '
                  'garbage' % (res['lemma_g'], cost.GLOBAL_MC_BY_N.get(n, cost.GLOBAL_MC)['src']))
        summary['sizes'].append(dict(n=n, trials=res['trials'], ok=res['ok'], interval=[lo, hi], exact_lower=exact,
                                     secp_trials=res['secp_trials'], secp_ok=res['secp_ok'],
                                     mean_runs=sum(res['runs']) / res['trials'], max_runs=max(res['runs']),
                                     mean_ops=sum(res['ops']) / res['trials'], max_ops=max(res['ops']),
                                     errors=res['errors'], secs=sum(res['secs']) / res['trials'],
                                     unsimulated=res.get('unsim', 0), garbage_runs=res['garbage'],
                                     heuristic_runs=res.get('heuristic', 0), capped_attempts=res.get('capped', 0),
                                     lemma_g_runs=res.get('lemma_g', 0),
                                     max_attempt=res.get('max_attempt', 0),
                                     shapes=[list(s) for s in shapes]))
        if n == 256:
            report_cost(mod, res, summary)
    flags = flag_lines(summary)
    for f in flags:
        print('FLAG: ' + f)
    summary['flags'] = flags
    low = [s['n'] for s in summary['sizes']
           if (s['interval'][0] < 0.99 or (s.get('exact_lower') is not None and s['exact_lower'] < 0.99)
               or not s['secp_trials'] if s['n'] == 256 else s['ok'] < 0.99 * s['trials'])]
    unsim = [s['n'] for s in summary['sizes'] if s.get('unsimulated')]
    no256 = all(s['n'] != 256 for s in summary['sizes'])     # the verdict is about n = 256; toy-only runs cannot PASS
    passed = not low and not flags and not unsim and not no256
    why = []
    if no256:
        why.append('n = 256 not run')
    if low:
        why.append('success below 0.99 at n = %s' % ','.join(map(str, low)))
    if unsim:
        why.append('runs not simulated at n = %s' % ','.join(map(str, unsim)))
    if flags:
        why.append('%d flag(s)' % len(flags))
    print('VERDICT: %s' % ('PASS (95 % lower bound >= 0.99 at n = 256 on secp256k1, point estimate >= 0.99 at toy '
                           'sizes, no flags)' if passed else 'FAIL (' + '; '.join(why) + ')'))
    if 'cost256' in summary:
        tg = summary['cost256']['targets']
        largest = summary['cost256']['largest_run']
        g_runs = sum(s.get('lemma_g_runs', 0) for s in summary['sizes'])
        if 'kinds' in largest:      # hex run: every part of the largest run must be measured, every kind qualify
            used = [cost.KINDS[k] for k in largest['kinds']]
            meas = largest['measured'] and args.measured_n == 256
            exact = all(m['exact'] for m in used) and not args.approx
            flagged = all(m['flagged'] or m['exact'] for m in used) or bool(g_runs)
            src_ok = all(m['exact'] or m.get('pf_source', '').startswith('run-input') for m in used) or bool(g_runs)
            pf_txt = ('Lemma G (paper Sec. V.D and App. B) with %s' % GMC['src'] if g_runs else
                      'summed failure weight per run <= %.3g (%s)' % (largest['fail_weight'], '; '.join(
                          '%s: %.3g per addition, %s' % (k, cost.KINDS[k]['fail_per_add'],
                                                         cost.KINDS[k].get('pf_source')) for k in largest['kinds'])))
        else:
            meas = cost.MODEL['grade'].startswith('measured') and args.measured_n == 256
            exact = cost.MODEL.get('exact', False) and not args.approx
            flagged = cost.MODEL.get('flagged')
            src_ok = cost.MODEL.get('pf_source', '').startswith('run-input')
            pf_txt = 'bad weight per addition <= %.3g, %s' % (cost.MODEL['fail_per_add'], cost.MODEL.get('pf_source'))
        heur = not exact and (not args.approx or not flagged or
                              any(s.get('heuristic_runs') for s in summary['sizes']))

        def word(ok):
            if not (ok and passed):
                return 'not met'
            if not meas:
                return 'met by estimate only'
            if heur:
                return 'met at grade heuristic only (approximate arithmetic without a circuit-specific bound)'
            if args.pf_scale != 1.0:
                return 'what-if run (--pf-scale %g), no target claim' % args.pf_scale
            if not exact and not src_ok:
                return 'met at grade heuristic only (failure weight per addition is a declared estimate)'
            prep = (' with a declared (not V-B checked) state preparation of the amplitude registers'
                    if 'declared state preparation' in summary['cost256']['largest_run']['grade'] else '')
            return ('MET' + prep) if exact else ('met under the coupling bound (approximate arithmetic; %s)%s'
                                                 % (pf_txt, prep))
        print('targets (PASS, a measured grade and exact arithmetic or a circuit-specific bound are needed): '
              'intermediate %s, main %s' % (word(tg['intermediate']), word(tg['main'])))
    summary['verdict'] = 'PASS' if passed else 'FAIL'
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(summary, f, indent=1, default=str)


def report_cost(mod, res, summary):
    if not res['run_costs']:
        print('cost at n = 256: no runs executed')
        return
    B = max(c['cost'] for c in res['run_costs'])
    worst = max(res['run_costs'], key=lambda c: c['cost'])
    budget = mod.budget(256) if hasattr(mod, 'budget') else None
    T_obs = max(res['trial_costs'])
    T = budget * B if budget is not None else T_obs
    tg = cost.targets(B, T)            # finite budget: flagged runs are bad runs inside it, never rerun
    if 'kinds' in worst:
        print('cost at n = 256 (hex run, grade by part: %s):' % worst['grade'])
    else:
        print('cost at n = 256 (grade %s: %d qubits, %.4g Toffoli per windowed addition, window %d):' % (
            worst['grade'], cost.MODEL['qubits'], cost.MODEL['toffoli_per_add'], cost.MODEL['w']))
    print('  largest run: %d exponent bits, %d windows, %d additions + initial lookup (%d Toffoli), %.4g Toffoli, '
          '%d qubits' % (worst['exp_bits'], worst['windows'], worst['additions'], worst.get('first_lookup', 0),
                         worst['toffoli'], worst['qubits']))
    print('  B = %.4g = %.4f B0 (B0 = %.5g = 1,457 x 39,255,168, Häner et al. body; %.4f of the rounded 5.8e10)' % (
        B, B / cost.B0, cost.B0, B / cost.B0_ROUNDED))
    print('  T = %.4g = %.3f B0 = %.3f T0 (T0 = %d x B0; %s; mean over trials %.3f B0, max %.3f B0)' % (
        T, T / cost.B0, T / cost.T0, cost.T0_ATTEMPTS,
        ('budget %d runs x B' % budget if budget is not None else 'no budget() declared: max observed'),
        sum(res['trial_costs']) / len(res['trial_costs']) / cost.B0, T_obs / cost.B0))
    print('  cost vs targets: intermediate (B <= B0/2, T <= 4 T0) %s; main (B <= B0/4, T <= 10 T0) %s' % (
        'within' if tg['intermediate'] else 'outside', 'within' if tg['main'] else 'outside'))
    if res.get('unsim'):
        print('  %d hash runs above the grid limit were priced, not simulated: success at n = 256 not measured'
              % res['unsim'])
    summary['cost256'] = dict(B=B, B_over_B0=B / cost.B0, B_over_B0_rounded=B / cost.B0_ROUNDED, T=T,
                              T_over_T0=T / cost.T0, budget_runs=budget,
                              largest_run=worst, targets=tg)


def flag_lines(summary):
    flags = []
    if summary['scan']:
        flags.append('source scan found forbidden constructs: results untrusted')
    toy = [s for s in summary['sizes'] if s['n'] != 256]
    for s in summary['sizes']:
        if any(k.startswith('budget') for k in s['errors']):
            flags.append('n = %d: classical or run budget exceeded in %d trials' % (
                s['n'], sum(v for k, v in s['errors'].items() if k.startswith('budget'))))
        if s['max_ops'] > cost.CLASSICAL_CEILING:
            flags.append('n = %d: classical work above 2^40' % s['n'])
    ran256 = any(s['n'] == 256 and s['trials'] for s in summary['sizes'])
    if len(toy) >= 2:
        a, b = toy[-2], toy[-1]
        if a['max_ops'] > 0 and b['max_ops'] > 0:
            slope = (lg(b['max_ops']) - lg(a['max_ops'])) / (b['n'] - a['n'])
            proj = lg(b['max_ops']) + slope * (256 - b['n'])
            # the projection stands in for n = 256 only when n = 256 was not simulated; when it was, its measured
            # maximum (checked against 2^40 above, and by BudgetExceeded in every trial) is the evidence
            if slope > 0.2 or (proj > 40 and not ran256):
                flags.append('classical work grows %.2f bits per bit of n between n = %d and %d '
                             '(projects to 2^%.1f at n = 256)' % (slope, a['n'], b['n'], proj))
    return flags


if __name__ == '__main__':
    main()
