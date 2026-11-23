#!/usr/bin/env python3
"""Third control: pick Ekera's tradeoff factor s for the plain decoder on U1's windows.

Reads the one-batch harness runs of base/run_all.sh ekscan (base/ekera_ctrl.py, ATTEMPTS = 1, n = 256): scan_s<s>.json
with the garbage-run model (--approx: the batch's success q, sampled) and ideal_s<s>.json without it (the decoder's
success on clean runs). Plain decoding retries whole batches, so A batches succeed with 1 - (1 - q)^A; A(s) is the
smallest A with 1 - (1 - q)^A >= 0.99 at the point estimate of q (A95: at the 95 % lower end of q), T = A n B. The
model column checks the garbage model's prediction q = P_dec x (1 - 4 pf x additions)^n (a bad run spoils its batch).
s is chosen to minimize B with T <= 10 T0 (ties: smaller T); classical work is A batches at the sampled maximum per
batch (one LLL and one Babai call in dimension n + 1, plus the check). Window check: B of each s at W = 12-17 from
base/u1_sweep.txt's per-window cost at the cap and the first lookups of base/u1_sweep.py. Sensitivity line: the same
choice if a batch needed only n = s runs and its decoder never failed on clean runs (garbage model only).
Writes base/ekera_tradeoff.csv (one row per s: the leader-core Ekera curve, B and T, for the paper's Figure 5).
usage: python3 base/ekera_pick.py
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import u1_sweep                 # first_lookup(); imports cost (B0) and the counting code

SS = (1, 2, 3, 4, 5, 6, 7, 8, 10)
M = 256


def load(name):
    d = json.load(open(os.path.join(HERE, 'ekera', name)))
    return [e for e in d['sizes'] if e['n'] == M][0], d['cost256'], d['params']


def batches(q):
    return math.inf if q <= 0 else (1 if q >= 0.99 else math.ceil(math.log(0.01) / math.log(1 - q)))


def window_check(ma, l, e16):
    """(B / B0, W, f) of the run shape a = ma, b = l bits, minimized over W <= 17 (u1_sweep.txt's cost at the cap and
    peak) and the first lookup's f bits (u1_sweep.first_lookup: the port's own at f = W = 16, else formula)"""
    best = (math.inf, 0, 0)
    for line in open(os.path.join(HERE, 'u1_sweep.txt')):
        t = line.split()
        if len(t) > 4 and t[0].isdigit() and 12 <= int(t[0]) <= 17:
            W, cap, q = int(t[0]), int(t[2]), int(t[3])
            for f in range(10, 21):
                ft, fq = u1_sweep.first_lookup(f, W, e16)[:2]
                B = max(q, fq) * ((math.ceil((ma - f) / W) + math.ceil(l / W)) * cap + ft) / u1_sweep.cost.B0
                best = min(best, (B, W, f))
    return best


def main():
    vb = json.load(open(os.path.join(HERE, 'lead_w16_cap.json')))
    e16 = [e for e in vb['sizes'] if e['n'] == M][0]
    pf = e16['declared_failure']
    u1 = json.load(open(os.path.join(HERE, 'u1_R4.json')))['cost256']['B_over_B0']
    print('per-addition failure pf = %.4e (base/lead_w16_cap.json); garbage model: a run is bad with 4 pf x additions'
          % pf)
    print('%3s %3s %5s %4s %8s | %22s | %22s %6s | %3s %7s | %3s %7s | %6s | %s' % (
        's', 'n', 'bits', 'add', 'B/B0', 'clean decoder [95 %]', 'batch q [95 %]', 'model', 'A', 'T/T0', 'A95',
        'T/T0', 'lg cl', 'best W, first lookup f: B/B0'))
    rows = []
    for s in SS:
        e, c, p = load('scan_s%d.json' % s)
        ei, _, _ = load('ideal_s%d.json' % s)
        lr = c['largest_run']
        n, B, adds = c['budget_runs'], c['B_over_B0'], lr['additions']
        q, ql = e['ok'] / e['trials'], e['interval'][0]
        pd = ei['ok'] / ei['trials']
        qm = pd * (1 - 4 * pf * adds) ** n
        A, A95 = batches(q), batches(ql)
        T, T95 = A * n * B / 3, A95 * n * B / 3
        lgc = math.log2(A * e['max_ops']) if A < math.inf else math.inf
        l = -(-M // s)
        bw, W, f = window_check(lr['exp_bits'] - l, l, e16)
        rows.append((s, n, B, A, T, A95, T95, lgc, adds))
        print('%3d %3d %5d %4d %8.6f | %.4f [%.4f, %.4f] | %.4f [%.4f, %.4f] %6.4f | %3s %7.3f | %3s %7.3f | %6.2f | '
              'W = %d, f = %d: %.4f' % (s, n, lr['exp_bits'], adds, B, pd, ei['interval'][0], ei['interval'][1], q, ql,
                                       e['interval'][1], qm, A, T, A95, T95, lgc, W, f, bw))
    with open(os.path.join(HERE, 'ekera_tradeoff.csv'), 'w') as f:
        f.write('s,runs_per_batch,B_B0,batches,T_T0,batches_95,T_T0_95,log2_classical\n')
        for r in rows:
            f.write('%d,%d,%.6f,%s,%.4f,%s,%.4f,%.2f\n' % r[:8])
    ok = [r for r in rows if r[4] <= 10 and r[7] <= 40]
    s, n, B, A, T, A95, T95, lgc, _ = min(ok, key=lambda r: (r[2], r[4]))
    print('choice: s = %d, B = %.6f B0 (U1 / B = %.4f), %d batches of %d runs = %d runs, T = %.3f T0 (%d batches, '
          'T = %.3f T0 under the 95 %% rule), classical <= 2^%.1f per solve' % (s, B, u1 / B, A, n, A * n, T,
                                                                               A95, T95, lgc))
    opt = []                    # sensitivity, model only: n = s runs per batch, a decoder never wrong on clean runs
    for r in rows:
        A = batches((1 - 4 * pf * r[8]) ** r[0])
        opt.append((r[2], A * r[0] * r[2] / 3, r[0], A))
    B, T, s, A = min(o for o in opt if o[1] <= 10)
    print('optimistic decoder (model: n = s runs per batch, always right on clean runs): s = %d, B = %.6f B0, %d '
          'batches, T = %.3f T0' % (s, B, A, T))


if __name__ == '__main__':
    main()
