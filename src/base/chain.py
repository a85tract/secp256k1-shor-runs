#!/usr/bin/env python3
"""The table and the chain of factors of Table III and Sec. VII.B of the paper, read from saved files (no simulation).

Rows: Haener et al. (B0, T0 from harness/cost.py); U1 and the textbook 2m-bit U1 (harness/run.py JSON of
base/u1_shor.py); U2-h / U2-n / U2-nt (harness pricing JSON of base/u2_price.py, single-run decoding JSON of
base/u2_decode.py); the Ekera control (base/ekera_final.json: harness/run.py on base/ekera_ctrl.py at the s and
1 - (1 - p)^R >= 0.99 for p = (1 - 4 eps) P_dec, the harness's garbage-run model (4 pf per addition, the model of U1's
harness runs and of Haener's per-attempt success): eps = the run's summed failure weight (leader windows 2.486e-3
L309), P_dec = the sampled decoder success at the largest cap whose classical work per run is <= 2^37.5 (<= 2^39.5
for 4 runs).
usage: python3 base/chain.py
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(REPO, 'harness'))
import cost

PF_WIN, PF_TERM, WORK_MAX = 2.260e-3 + 2.257e-4, 1.6e-3, 37.5


def J(path):
    return json.load(open(os.path.join(REPO, path)))


def harness_row(path):
    d = J(path)
    c = d['cost256']
    lr = c['largest_run']
    s = [x for x in d['sizes'] if x['n'] == 256][0]
    return dict(bits=lr['exp_bits'], adds=lr['additions'], first=lr.get('first_bits') or 16, q=lr['qubits'],
                tof=lr['toffoli'], B=c['B_over_B0'], runs=c['budget_runs'], T=c['T_over_T0'],
                succ='%d / %d' % (s['ok'], s['trials']))


def runs_for(p, target=0.99):
    R = 1
    while 1 - (1 - p) ** R < target:
        R += 1
    return R


def main():
    rows = [('Haener et al. (B0)', dict(bits=466, adds=28, first=18, q=cost.HANER_QUBITS, tof=cost.HANER_TOFFOLI, B=1.0,
                                        runs=cost.T0_ATTEMPTS, T=1.0, succ='0.861 per attempt'))]
    u1 = harness_row('base/u1_R4.json')
    single = J('base/u1_single_approx.json')['sizes'][0]
    u1['succ'] = '%s with 4 runs; %.4f per run' % (u1['succ'], single['ok'] / single['trials'])
    rows.append(('U1', u1))
    t512 = harness_row('base/u1_t512.json')
    rows.append(('U1, textbook 2m = 512 bits', t512))
    names = dict(h="U2-h (R3's kinds)", n="U2-n (B''s kinds)", nt="U2-nt (C''s kinds)")
    for v in ('h', 'n', 'nt'):
        r = harness_row('base/u2_price_%s.json' % v)
        dec = J('base/u2_decode_%s.json' % ('h' if v == 'h' else 'nt'))
        ok = [x for x in dec['caps'] if x['log2work'] <= WORK_MAX]
        best = max(ok, key=lambda x: x['P'])
        eps = (r['adds'] - 1) * PF_WIN + (PF_TERM if v == 'nt' else PF_WIN)
        p = (1 - 4 * eps) * best['P']
        r['runs'] = runs_for(p)
        r['T'] = r['runs'] * r['B'] / cost.T0_ATTEMPTS
        r['succ'] = 'decoder %.4f (%d candidates, b at q %.3f, 2^%.1f per run, %d samples); per run %.4f' % (
            best['P'], best['CA'], best['qb'], best['log2work'], dec['samples'], p)
        rows.append((names[v], r))
    e4 = harness_row('base/ekera_final.json')
    par = dict(x.split('=') for x in J('base/ekera_final.json')['params'])
    one = [x for x in J('base/ekera/scan_s%s.json' % par['S'])['sizes'] if x['n'] == 256][0]
    e4['succ'] = '%s with %s batches; per batch %.4f' % (e4['succ'], par['ATTEMPTS'], one['ok'] / one['trials'])
    rows.append(('Ekera control, s = %s, plain decoder' % par['S'], e4))
    for name, path in (('R3 D = 270 hex stack', 'h1/hexcap/va_d270fix_4w_R79.json'),
                       ("B' norm-7, W9 terminal", 'h1/n7/va_n7fix_R110.json'),
                       ("C' norm-7, root-sign terminal", 'h1/n7/va_n7term_fix_R102.json')):
        rows.append((name, harness_row(path)))
    print('| Row | bits | additions + first | qubits | Toffoli per run | B / B0 | runs | T / T0 | success |')
    print('|---|---|---|---|---|---|---|---|---|')
    for name, r in rows:
        print('| %s | %d | %d + %d | %s | %s | %.6f | %d | %.3f | %s |' % (name, r['bits'], r['adds'], r['first'],
              format(r['q'], ','), format(int(r['tof']), ','), r['B'], r['runs'], r['T'], r['succ']))
    B = {name.split(' ')[0]: r['B'] for name, r in rows}
    core = 1 / B['U1']
    print('\nchain (B0 / B): core Haener -> U1 = %.3f' % core)
    for u2, st in (('U2-h', 'R3'), ('U2-n', "B'"), ('U2-nt', "C'")):
        w, s = B['U1'] / B[u2], B[u2] / B[st]
        print('  -> %s windows %.3f -> %s split %.3f; product %.3f = B0 / B(%s) %.3f; ours (windows x split) %.3f'
              % (u2, w, st, s, core * w * s, st, 1 / B[st], w * s))
    print('inside the window step: U1 -> U2-h (hex windows, terminal = W9 window, sine-top first lookup) %.3f; '
          'U2-h -> U2-n (norm-7 windows) %.3f; U2-n -> U2-nt (root-sign terminal) %.3f'
          % (B['U1'] / B['U2-h'], B['U2-h'] / B['U2-n'], B['U2-n'] / B['U2-nt']))
    ek = B['Ekera']
    print("Ekera path: U1 -> Ekera control (his tradeoff, plain decoding) %.3f; -> R3 / B' / C' (our windows, "
          "decoder and certificate) %.3f / %.3f / %.3f; B0 / B(Ekera control) = %.3f"
          % (B['U1'] / ek, ek / B['R3'], ek / B["B'"], ek / B["C'"], 1 / ek))
    print("domain step (exponent bits per run): U1 464 -> Ekera control %d = %.3f; ours 464.5 -> 267.9 (log2 norms "
          "of U2-nt and C') = %.3f, 464 -> 270 harness bits = %.3f"
          % (e4['bits'], 464 / e4['bits'], 464.533 / 267.88, 464 / 270))


if __name__ == '__main__':
    main()
