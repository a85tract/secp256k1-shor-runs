#!/usr/bin/env python3
"""Quick check: recompute a set of the paper's numbers and print PASS or FAIL for each.

usage: python3 quickcheck.py [--jobs 3]

It reruns the frozen trial code of the three configurations at 16, 24 and 32 bits (all 200 registered trials per
size) and the first two registered trials at 256 bits, recomputes the floors of Section III, the per-kind and per-run
Toffoli counts and the certificate constants, and compares them with the paper and with the saved trial logs.
About 5 minutes on 3 cores. Outputs go to quickcheck_out/.
"""
import argparse, json, math, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'quickcheck_out')
results = []


def check(name, ok, detail=''):
    results.append(ok)
    print('%s  %s%s' % ('PASS' if ok else 'FAIL', name, ('  (' + detail + ')') if detail else ''), flush=True)


def run(cmd, cwd, log):
    with open(log, 'w') as f:
        return subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1')).returncode


def close(a, b, tol):
    return abs(a - b) <= tol


# Table VII rows at 16, 24, 32 bits: solved, runs, bad or flagged runs, work mean, work largest.
TABLE7 = {
    'unit-orbit': {16: (200, 16, 61, 18.8, 20.8), 24: (200, 18, 48, 21.9, 22.2), 32: (200, 20, 119, 21.4, 25.9)},
    'mixed-radix': {16: (200, 16, 183, 21.5, 25.48), 24: (200, 18, 268, 24.2, 30.42), 32: (200, 20, 225, 23.6, 28.95)},
    'root-sign': {16: (200, 16, 194, 21.4, 26.31), 24: (200, 18, 246, 23.9, 29.21), 32: (200, 20, 249, 24.0, 29.24)},
}
# Table III / Table VI: runs, Toffoli per run, peak qubits, B/B0, T/T0; App. B.5: 1 - g0 <=, path term <=, phi_LB >=.
RUNS = {
    'unit-orbit': (79, 12626051, 1246, 0.2751, 7.24, 0.04284, 2.10e-3, 0.03145),
    'mixed-radix': (110, 12264316, 1246, 0.2672, 9.80, 0.04289, 1.64e-3, 0.03256),
    'root-sign': (102, 11866224, 1246, 0.2585, 8.79, 0.04070, 1.41e-3, 0.03115),
}
LOGS = {'unit-orbit': 'va_d270fix_4w_R79_trials.jsonl', 'mixed-radix': 'va_n7fix_R110_trials.jsonl',
        'root-sign': 'va_n7term_fix_R102_trials.jsonl'}


def section3():
    src = os.path.join(ROOT, 'src')
    log = os.path.join(OUT, 'numbers_n256.txt')
    run([sys.executable, '-B', 's3/theory/numbers_n256.py'], src, log)
    rows = {}
    for line in open(log):
        m = re.match(r'\s*(\d+)\s+80 \|\s+([\d.]+)\s+([\d.]+) \|\s+([\d.]+)\s+([\d.]+) \|', line)
        if m:
            rows[int(m.group(1))] = tuple(float(x) for x in m.groups()[1:])
    table1 = {4: (299.31, 0.628, 253.99, 0.528), 16: (266.83, 0.556, 251.99, 0.523), 32: (261.38, 0.544, 250.99, 0.521),
              64: (258.47, 0.538, 249.99, 0.519), 128: (256.64, 0.534, 248.99, 0.517), 240: (255.38, 0.531, 248.08, 0.515)}
    ok = all(rows.get(r) == v for r, v in table1.items())
    check('Table I floors, R = 4, 16, 32, 64, 128, 240 (bits, B/B0, adaptive bits, adaptive B/B0)', ok)
    log = os.path.join(OUT, 'frontier.txt')
    run([sys.executable, '-B', 'base/frontier.py'], src, log)
    txt = open(log).read()
    def frontier(tag):
        m = re.search(r'%s\s+budget\s+[\d.]+ B0: D\* = ([\d.]+) bits from R = (\d+) runs on, B >= ([\d.]+) B0, X <= ([\d.]+)'
                      % re.escape(tag), txt)
        return (float(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))) if m else None
    f10, f4, fu1, fad = frontier('10T0'), frontier('4T0'), frontier('10T0u1'), frontier('ad10T0')
    check('Floor inside 10 T0: 55 runs of 258.977 bits, B >= 0.5387 B0, splitting alone at most 1.86 times',
          f10 is not None and f10[1] == 55 and close(f10[0], 258.977, 5e-4) and close(f10[2], 0.5387, 5e-5)
          and close(f10[3], 1.856, 5e-4), str(f10))
    check('Floor inside 4 T0: 21 runs of 264.24 bits, at most 1.82 times',
          f4 is not None and f4[1] == 21 and close(f4[0], 264.245, 5e-4) and close(f4[3], 1.817, 5e-4), str(f4))
    check('Adaptive floor inside 10 T0: 57 runs of 250.15 bits, at most 1.93 times',
          fad is not None and fad[1] == 57 and close(fad[0], 250.153, 5e-4) and close(fad[3], 1.926, 5e-4), str(fad))
    check('Floor at the borrowed core\'s window cost: 112 runs of 256.94 bits, at most 1.87 times',
          fu1 is not None and fu1[1] == 112 and close(fu1[0], 256.94, 5e-3) and close(fu1[3], 1.872, 5e-4), str(fu1))
    log = os.path.join(OUT, 'floors.txt')
    run([sys.executable, '-B', 'paper/figures/floors.py'], src, log)
    gaps = {}
    for line in open(log):
        m = re.match(r'\s*R =\s*(\d+): floor ([\d.]+) bits, configuration ([\d.]+) bits, gap ([\d.]+) bits', line)
        if m:
            gaps[int(m.group(1))] = (float(m.group(2)), float(m.group(4)))
    want = {79: (257.84, 12.16), 102: (257.17, 10.71), 110: (256.98, 10.90)}
    ok = all(r in gaps and close(gaps[r][0], f, 0.005) and close(gaps[r][1], g, 0.005) for r, (f, g) in want.items())
    check('Sec. VII.C floors 257.84 / 257.17 / 256.98 bits at 79 / 102 / 110 runs, gaps 12.16 / 10.71 / 10.90 bits',
          ok, ', '.join('%d: %s' % (r, gaps.get(r)) for r in want))


def kinds():
    def kind(path):
        d = json.load(open(os.path.join(ROOT, path)))
        return d['sizes'][0]['abort_cap'] + d.get('extra_toffoli', 0)
    t = 'trials/root-sign/inputs/'
    got = {'signed window on the core': kind('src/base/lead_w16_cap.json'),
           'unit-orbit W = 8': kind('trials/unit-orbit/inputs/hexlead_w8_peak_cap_n256.json'),
           'unit-orbit W = 9': kind('trials/unit-orbit/inputs/hexlead_w9_peak_cap_n256.json'),
           'mixed-radix (3+w)2^7': kind(t + 'n7lead_n7_peak_cap_n256.json'),
           'mixed-radix (2-w)2^7': kind(t + 'n7lead_m7_peak_cap_n256.json'),
           'root-sign last addition': kind(t + 'hexterm_w9_n256.json')}
    want = {'signed window on the core': 816538, 'unit-orbit W = 8': 780001, 'unit-orbit W = 9': 912269,
            'mixed-radix (3+w)2^7': 813221, 'mixed-radix (2-w)2^7': 813219, 'root-sign last addition': 514177}
    check('Table II Toffoli per addition at the abort cap with digit register (6 kinds)', got == want,
          ', '.join('%s %d' % kv for kv in got.items()))


def trial_records(path, sizes, upto256):
    recs = {}
    for line in open(path):
        r = json.loads(line)
        if r['n'] in sizes and (r['n'] != 256 or r['t'] < upto256):
            recs[(r['n'], r['t'])] = (r['ok'], r['runs'], r['garbage'], r['ops'], r.get('lemma_g'), r.get('trial_cost'))
    return recs


def trials(jobs):
    procs = {}
    for name in TABLE7:
        prefix = os.path.join(OUT, name.replace('-', '_'))
        log = open(prefix + '.log', 'w')
        procs[name] = (subprocess.Popen(['bash', os.path.join(ROOT, 'trials', name, 'run.sh'), '16,24,32,256', '200', '2',
                                         prefix], stdout=log, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                                                 PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ['PATH'])),
                       prefix, log)
        if len(procs) % jobs == 0:          # at most `jobs` processes at a time
            for p, _, _ in procs.values():
                p.wait()
    for name, (p, prefix, log) in procs.items():
        p.wait(); log.close()
        txt = open(prefix + '.txt').read()
        rows = {}
        for n in (16, 24, 32):
            m = re.search(r'^\s+%d\s+(\d+)\s+([\d.]+)\s+\[[^\]]*\]\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)' % n, txt, re.M)
            g = re.search(r'^\s+%d\s+\d+ .*\n(?:.*\n)*?\s+(\d+) runs returned garbage' % n, txt, re.M)
            if m and g:
                trials_n = int(m.group(1)); solved = round(float(m.group(2)) * trials_n)
                rows[n] = (solved, round(float(m.group(3))), int(g.group(1)), float(m.group(4)), float(m.group(5)))
        for n, want in TABLE7[name].items():
            got = rows.get(n)
            dec = 1 if want[4] == round(want[4], 1) and name == 'unit-orbit' else 2
            ok = got is not None and got[:3] == want[:3] and round(got[3], 1) == want[3] and round(got[4], dec) == want[4]
            check('Table VII %s, %d bits: %d of 200 solved, %d runs, %d bad or flagged, work %.1f / %s'
                  % (name, n, want[0], want[1], want[2], want[3], want[4]), ok, 'got %s' % (got,))
        saved = trial_records(os.path.join(ROOT, 'trials', name, 'results', LOGS[name]), (16, 24, 32, 256), 2)
        new = trial_records(prefix + '_trials.jsonl', (16, 24, 32, 256), 2)
        same = saved == new and len(new) == 602
        check('%s: 600 trials at 16-32 bits and the first 2 at 256 bits equal the registered trial log record by record'
              % name, same, '%d of %d records equal' % (sum(new.get(k) == v for k, v in saved.items()), len(saved)))
        d = json.load(open(prefix + '.json'))
        c = d['cost256']; lr = c['largest_run']
        want = RUNS[name]
        ok = (c['budget_runs'] == want[0] and int(lr['toffoli']) == want[1] and lr['qubits'] == want[2]
              and round(c['B_over_B0'], 4) == want[3] and round(c['T_over_T0'], 2) == want[4])
        check('Table III %s: %d runs of %s Toffoli on %d qubits, B = %.4f B0, T = %.2f T0'
              % (name, want[0], format(want[1], ','), want[2], want[3], want[4]), ok,
              'got %d runs, %d Toffoli, %d qubits, %.6f B0, %.4f T0' % (c['budget_runs'], lr['toffoli'], lr['qubits'],
                                                                      c['B_over_B0'], c['T_over_T0']))
        m = re.search(r'^Lemma G constants at n = 256 .*?UB E\[N_DO \+ 2 N_P\] ([\d.e-]+) .*?UB E\[corr\] ([\d.e-]+), '
                      r'UB recovered ([\d.e-]+), UB class \(iii\) ([\d.e-]+), LB phi ([\d.e-]+)', txt, re.M)
        if m:
            eS, corr, rec, cl3, phi = (float(x) for x in m.groups())
            path = corr + rec + cl3
            ok = eS <= want[5] and path <= want[6] + 5e-6 and phi >= want[7]
            detail = '1 - g0 <= %.5f, path term <= %.3e, phi_LB >= %.5f' % (eS, path, phi)
        else:
            ok, detail = False, 'constants line not found'
        check('App. B.5 %s constants: 1 - g0 <= %.5f, path term <= %.2e, phi_LB >= %.5f'
              % (name, want[5], want[6], want[7]), ok, detail)


def chain():
    log = os.path.join(OUT, 'chain.txt')
    run([sys.executable, '-B', 'base/chain.py'], os.path.join(ROOT, 'src'), log)
    txt = open(log).read()
    want = {'Signed windows, unsplit': ('U1', '22,896,081', '1,244', '0.497995'),
            'Unit-orbit windows, unsplit': ('U2-h', '22,151,891', '1,246', '0.482583'),
            'Mixed-radix windows, unsplit': ('U2-n', '21,823,943', '1,246', '0.475439'),
            'Mixed-radix and root-sign, unsplit': ('U2-nt', '21,425,851', '1,246', '0.466767'),
            'Ekera\'s tradeoff on the core': ('Ekera control', '16,363,777', '1,244', '0.355916')}
    for label, (row, tof, q, b) in want.items():
        m = re.search(r'^\| %s[^|]*\| [^|]* \| [^|]* \| ([\d,]+) \| ([\d,]+) \| ([\d.]+) \|' % re.escape(row), txt, re.M)
        ok = m is not None and m.groups() == (q, tof, b)
        check('Table III %s: %s Toffoli on %s qubits, B = %.4f B0' % (label, tof, q, float(b)), ok,
              'got %s' % (m.groups() if m else None,))
    ok = ('core Haener -> U1 = 2.008' in txt and 'windows 1.032 -> R3 split 1.754' in txt
          and 'windows 1.047 -> B\' split 1.779' in txt and 'windows 1.067 -> C\' split 1.806' in txt
          and 'B0 / B(Ekera control) = 2.810' in txt and '1.294 / 1.332 / 1.377' in txt)
    check('Sec. VII.B gain factors: core 2.01, windows 1.03-1.07, splitting 1.75-1.81; Ekera control 2.81, ours 1.29-1.38 further',
          ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--jobs', type=int, default=3, help='trial processes at a time (default 3)')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    section3()
    kinds()
    check('Sec. VI: no failure in 563 trials gives success >= 0.994506 at confidence 0.955', 0.045 ** (1 / 563) > 0.994506,
          '0.045^(1/563) = %.8f' % 0.045 ** (1 / 563))
    chain()
    trials(max(1, a.jobs))
    print('%d of %d checks passed in %.0f s: %s' % (sum(results), len(results), time.time() - t0,
                                                   'PASS' if all(results) else 'FAIL'))
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
