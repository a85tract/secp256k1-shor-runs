"""o3 s6: path-level Monte Carlo of the leader stream under the global-scalar run law (paper Sec. V.D and App. B; , Lemma G) for a
mixed-radix (norm-7 / hex) window sequence, with the executed-Toffoli abort cap. This is h1/hexcap/path_mc_cap.py
syntax, lowest first; a sine top, if given, is in the exact first lookup and is dropped) adds the digit d of a uniform
residue of Z[w] / (pi_i) at the Eisenstein position Pi_i = pi_0 ... pi_(i-1): t_i = phi(d) phi(Pi_i) mod r, phi(z) =
stand-in sizes (hex W, w + 1 for a norm-7 x 4^w window), which is what the patched harness compares, and 'layout' the
SPEC. usage: python3 -I o3/s6/path_mc_n7.py OPS_RAW PATHS SPEC SEED OUT_JSON [STREAM_CAP]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path[:0] = [os.path.join(ROOT, 'h1', 'hexcap'), os.path.join(ROOT, 'o1')]
from path_mc_cap import *  # noqa: E402,F401,F403  (h1: simulate, R, LAM, batch_add, fixed_base*, on_curve, ...)
from path_mc_cap import json, random, time, np, offdep, lanes_value, simulate  # noqa: E402
import mixrad as mr  # noqa: E402


def main(raw, paths, spec, seed, out_json, stream_cap):
    wins = mr.parse_layout(spec)
    if wins[-1].kind == 'sine':
        wins = wins[:-1]                             # the top is in the exact first lookup
    PI = mr.positions(wins)[:-1]
    DIG = [(w.arrays()[0].ravel(), w.arrays()[1].ravel()) for w in wins]
    windows = [w.W if w.kind == 'hex' else w.w + 1 for w in wins]   # o1/s5/mixva.py stand-in sizes
    print('layout %s: %d additions, stand-in sizes %s, residues %s' % (spec, len(wins), windows,
          [len(d[0]) for d in DIG]), file=sys.stderr, flush=True)
    t0 = time.time()
    ops = np.memmap(raw, dtype=offdep.DT, mode='r')
    kinds = np.array(ops['kind'])
    n_tof = int(((kinds == 13) | (kinds == 14)).sum())       # the stream's emitted CCX + CCZ (761,147)
    del kinds
    rng = random.Random(seed)
    tabs = fixed_base_tables()
    J = len(windows)
    pos = [sum(windows[:i]) for i in range(J)]
    order = list(range(J - 1, -1, -1))                       # top window first, the terminal (lowest) last
    cs = [rng.randrange(1, R) for _ in range(paths)]
    t = [[0] * paths for _ in range(J)]
    for w in range(J):
        sh = (PI[w][0] + PI[w][1] * LAM) % R            # o3: the window's Eisenstein place value Pi_w
        D1, D2 = DIG[w]                                 # o3: the digits of all residues of the window (mixrad)
        for k in range(paths):
            ii = rng.randrange(len(D1))                 # a uniform residue
            t[w][k] = (int(D1[ii]) + int(D2[ii]) * LAM) * sh % R
    A = fixed_base(tabs, [rng.randrange(1, R) for _ in range(paths)])
    print('setup %.0f s' % (time.time() - t0), file=sys.stderr, flush=True)
    nDO = [0] * paths
    nP = [0] * paths
    real = [None] * paths            # None: on the correct path; else the deviated accumulator (x, y)
    state = ['ok'] * paths           # ok / dev / rflag / aflag / dirty
    per_add = dict(lanes=0, DO=0, P=0, wrong=0, rhit=0, dirty=0, aborted=0, wrong_oncurve=0, phase_any=0)
    garb = dict(lanes=0, rhit=0, dirty=0, aborted=0, out_oncurve=0)
    ex_n, ex_sum, ex_sq, ex_min, ex_max = 0, 0, 0, None, None
    for step, w in enumerate(order):
        B = fixed_base(tabs, [cs[k] * t[w][k] % R for k in range(paths)])
        Anext = batch_add(A, B)
        dev = [k for k in range(paths) if state[k] == 'dev']
        P_in = A + [real[k] for k in dev]
        A_in = B + [B[k] for k in dev]
        L = len(P_in)
        q, ph, rhit, sk = simulate(ops, L, P_in, A_in, seed * 1000 + step + 1)
        ex = n_tof - sk
        ab = ex > stream_cap
        ex_n += paths
        ex_sum += int(ex[:paths].sum())
        ex_sq += int((ex[:paths] * ex[:paths]).sum())
        ex_min = int(ex[:paths].min()) if ex_min is None else min(ex_min, int(ex[:paths].min()))
        ex_max = int(ex[:paths].max()) if ex_max is None else max(ex_max, int(ex[:paths].max()))
        xo, yo = lanes_value(q, 0, L), lanes_value(q, 256, L)
        dirty = 0
        for tq in range(512, 1227):
            dirty |= q[tq]
        for k in range(paths):
            ok_out = Anext[k] is not None and (xo[k], yo[k]) == Anext[k]
            fr, fp, fd, fa = (rhit >> k) & 1, (ph >> k) & 1, (dirty >> k) & 1, bool(ab[k])
            do = (not ok_out) or fr or fd or fa
            per_add['lanes'] += 1
            per_add['wrong'] += not ok_out
            per_add['rhit'] += fr
            per_add['dirty'] += fd
            per_add['aborted'] += fa
            per_add['phase_any'] += fp
            if not ok_out and on_curve((xo[k], yo[k])):
                per_add['wrong_oncurve'] += 1
            if do:
                per_add['DO'] += 1
                nDO[k] += 1
            elif fp:
                per_add['P'] += 1
                nP[k] += 1
            if state[k] == 'ok' and do:          # first deviation of the real path happens on the correct operands
                if fa:
                    state[k] = 'aflag'
                elif fr:
                    state[k] = 'rflag'
                elif fd:
                    state[k] = 'dirty'
                else:
                    state[k] = 'dev'
                    real[k] = (xo[k], yo[k])
        for i, k in enumerate(dev):              # chained garbage of paths that deviated earlier
            lane = paths + i
            garb['lanes'] += 1
            fr, fd, fa = (rhit >> lane) & 1, (dirty >> lane) & 1, bool(ab[lane])
            out = (xo[lane], yo[lane])
            garb['rhit'] += fr
            garb['dirty'] += fd
            garb['aborted'] += fa
            garb['out_oncurve'] += on_curve(out)
            if fa:
                state[k] = 'aflag'
            elif fr:
                state[k] = 'rflag'
            elif fd:
                state[k] = 'dirty'
            else:
                real[k] = out
        A = Anext
        print('window %d/%d (%s): %d lanes, %d deviated, %d aborted, executed mean %.1f max %d, %.0f s'
              % (step + 1, J, wins[w].kind, L, len(dev), int(ab.sum()), float(ex[:paths].mean()),
                 int(ex[:paths].max()), time.time() - t0), file=sys.stderr, flush=True)
    end = dict(correct_plus=0, correct_minus=0, rflag=0, aflag=0, dirty=0, offcurve=0, oncurve_wrong=0, recovered=0)
    corr_sum = 0
    hist = {}
    for k in range(paths):
        n = nDO[k] + nP[k]
        hist['%d,%d' % (nDO[k], nP[k])] = hist.get('%d,%d' % (nDO[k], nP[k]), 0) + 1
        if n >= 2:
            corr_sum += nDO[k] + 2 * nP[k] - 1 + ((-1) ** nP[k] if nDO[k] == 0 else 0)
        s = state[k]
        if s == 'ok':
            end['correct_plus' if nP[k] % 2 == 0 else 'correct_minus'] += 1
        elif s in ('rflag', 'aflag', 'dirty'):
            end[s] += 1
        else:
            X = real[k]
            if X == A[k]:
                end['recovered'] += 1
            elif on_curve(X):
                end['oncurve_wrong'] += 1
            else:
                end['offcurve'] += 1
    mean = ex_sum / ex_n
    res = dict(paths=paths, J=J, windows=windows, layout=spec, seed=seed, raw=os.path.basename(raw), stream_cap=stream_cap,
               stream_toffoli=n_tof, per_addition=per_add, garbage_chain=garb, end=end, corr_sum=corr_sum,
               corr_mean=corr_sum / paths, counts_DO_P=hist,
               executed=dict(lanes=ex_n, mean=mean, std=(ex_sq / ex_n - mean * mean) ** 0.5, min=ex_min, max=ex_max),
               secs=time.time() - t0)
    print(json.dumps(res), flush=True)
    json.dump(res, open(out_json, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5],
         int(sys.argv[6]) if len(sys.argv) > 6 else 726000)
