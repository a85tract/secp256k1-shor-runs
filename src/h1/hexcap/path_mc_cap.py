"""h1: path-level Monte Carlo of the leader stream under the global-scalar run law (paper Sec. V.D and App. B; , Lemma G), for a run
with its own window sequence and the executed-Toffoli abort cap. o3/sim/path_mc.py with three changes:

1. Windows. WINDOWS lists the run's hex additions lowest first (the harness's order, e.g. the D = 270 stack:
   9,8,8,8,8,8,8,8,8,8,9,9,9,9,9; the top window belongs to the exact first lookup and is not listed). Window i sits
   at bit P_i = W_0 + ... + W_(i-1) of each coordinate; its addend scalar is t_i = (d1 + d2 lam) 2^(P_i) mod r with
   (d1, d2) = o1/hexdigits.digit(a, b, W_i) of a uniform residue (a, b) in (Z / 2^W_i)^2. The additions run top
   window first, so the lowest one (the terminal) is the last; the statistics do not depend on the order.
2. Abort cap. The replay counts, per lane, the conditional CCX / CCZ it skips (bit-sliced counter, as o2/port/vbx2.py);
   executed = the stream's emitted CCX + CCZ - skipped. A lane whose executed count passes STREAM_CAP is aborted by
   the controller: a visible flag, counted as a deviation (DO) of that addition on the correct operands, and a
   visibly flagged end ('aflag') for the path (whichever addition aborts first). The window circuit's other Toffoli
   are unconditional (h1/hexcap/hexlead_cap.py), so this is the V-B circuit's abort rule.
3. The replay is s2/port/clsim_ph.simulate (phase tracking, fresh X-measurement outcomes per call) with the counter.

on each path's CORRECT pair and on the REAL pair of every path that already deviates (chained garbage), per-path
N_DO / N_P and corr, end classes correct (+ / -), R-flagged, aborted, dirty, off-curve, on-curve wrong, recovered.
usage: python3 -I h1/hexcap/path_mc_cap.py OPS_RAW PATHS WINDOWS SEED OUT_JSON [STREAM_CAP]"""
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path[:0] = [os.path.join(ROOT, 'o3', 'sim'), os.path.join(ROOT, 's2', 'port'), os.path.join(ROOT, 'o2', 'port'),
                os.path.join(ROOT, 'o1')]
import offdep  # noqa: E402
from clsim import to_lanes, lanes_value, NO, CHUNK  # noqa: E402
from hexdigits import digit  # noqa: E402
from path_mc import R, LAM, batch_add, fixed_base_tables, fixed_base, on_curve  # noqa: E402


def simulate(ops, L, P_in, A_in, seed):
    """s2/port/clsim_ph.simulate plus a per-lane count of skipped conditional CCX / CCZ (returned as an int array)"""
    ALL = (1 << L) - 1
    q = [0] * 1227
    bits = [0] * 1595
    xs, ys = to_lanes([p[0] for p in P_in], 256), to_lanes([p[1] for p in P_in], 256)
    ax, ay = to_lanes([a[0] for a in A_in], 256), to_lanes([a[1] for a in A_in], 256)
    q[0:256], q[256:512] = xs, ys
    bits[0:256], bits[256:512] = ax, ay
    rh = random.Random(seed)
    base, stack = ALL, []
    ph = 0
    rhit = 0
    planes = []                     # bit-sliced count of skipped conditional CCX / CCZ per lane

    def skip(m):
        i = 0
        while m:
            if i == len(planes):
                planes.append(0)
            planes[i], m = planes[i] ^ m, planes[i] & m
            i += 1
    N = len(ops)
    for c0 in range(0, N, CHUNK):
        ch = ops[c0:c0 + CHUNK]
        K, C1, C2, T = ch['kind'].tolist(), ch['c1'].tolist(), ch['c2'].tolist(), ch['t'].tolist()
        CT, CC = ch['ct'].tolist(), ch['cc'].tolist()
        for j in range(len(K)):
            kd = K[j]
            cc = CC[j]
            if kd == 8:
                if cc == NO and base == ALL:
                    q[T[j]] ^= q[C1[j]]
                else:
                    q[T[j]] ^= q[C1[j]] & (base if cc == NO else base & bits[cc])
            elif kd == 13:
                if cc == NO and base == ALL:
                    q[T[j]] ^= q[C1[j]] & q[C2[j]]
                else:
                    cond = base if cc == NO else base & bits[cc]
                    q[T[j]] ^= q[C1[j]] & q[C2[j]] & cond
                    if cond != ALL:
                        skip(ALL & ~cond)
            elif kd == 6:
                q[T[j]] ^= base if cc == NO else base & bits[cc]
            elif kd == 11 or kd == 12:
                cond = base if cc == NO else base & bits[cc]
                t = T[j]
                v = q[t] & cond
                if v:
                    m = rh.getrandbits(L)
                    ph ^= m & v
                    if kd == 11:
                        rhit |= v
                    q[t] &= ~cond
                else:
                    m = rh.getrandbits(L) if kd == 12 else 0
                if kd == 12:
                    ct = CT[j]
                    bits[ct] = (bits[ct] & ~cond) | ((m if v else rh.getrandbits(L)) & cond)
            elif kd == 10:
                cond = base if cc == NO else base & bits[cc]
                a, b = T[j], C1[j]
                d = (q[a] ^ q[b]) & cond
                q[a] ^= d
                q[b] ^= d
            elif kd == 15:
                stack.append(base)
                base = base if cc == NO else base & bits[cc]
            elif kd == 16:
                base = stack.pop()
            elif kd == 3:
                bits[CT[j]] ^= base if cc == NO else base & bits[cc]
            elif kd == 4:
                bits[CT[j]] &= ~(base if cc == NO else base & bits[cc])
            elif kd == 5:
                bits[CT[j]] |= base if cc == NO else base & bits[cc]
            elif kd == 7:
                ph ^= q[T[j]] & (base if cc == NO else base & bits[cc])
            elif kd == 9:
                ph ^= q[T[j]] & q[C1[j]] & (base if cc == NO else base & bits[cc])
            elif kd == 14:
                cond = base if cc == NO else base & bits[cc]
                ph ^= q[T[j]] & q[C1[j]] & q[C2[j]] & cond
                if cond != ALL:
                    skip(ALL & ~cond)
            elif kd == 0:
                ph ^= base if cc == NO else base & bits[cc]
    nby = (L + 7) // 8
    sk = np.zeros(L, dtype=np.int64)
    for i, pl in enumerate(planes):
        b = np.unpackbits(np.frombuffer(pl.to_bytes(nby, 'little'), dtype=np.uint8), bitorder='little')[:L]
        sk += b.astype(np.int64) << i
    return q, ph, rhit, sk


def main(raw, paths, windows, seed, out_json, stream_cap):
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
        sh = pow(2, pos[w], R)
        M = 1 << windows[w]
        for k in range(paths):
            d1, d2 = digit(rng.randrange(M), rng.randrange(M), windows[w])
            t[w][k] = (d1 + d2 * LAM) * sh % R
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
        print('window %d/%d (W = %d at bit %d): %d lanes, %d deviated, %d aborted, executed mean %.1f max %d, %.0f s'
              % (step + 1, J, windows[w], pos[w], L, len(dev), int(ab.sum()), float(ex[:paths].mean()),
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
    res = dict(paths=paths, J=J, windows=windows, seed=seed, raw=os.path.basename(raw), stream_cap=stream_cap,
               stream_toffoli=n_tof, per_addition=per_add, garbage_chain=garb, end=end, corr_sum=corr_sum,
               corr_mean=corr_sum / paths, counts_DO_P=hist,
               executed=dict(lanes=ex_n, mean=mean, std=(ex_sq / ex_n - mean * mean) ** 0.5, min=ex_min, max=ex_max),
               secs=time.time() - t0)
    print(json.dumps(res), flush=True)
    json.dump(res, open(out_json, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(',')], int(sys.argv[4]), sys.argv[5],
         int(sys.argv[6]) if len(sys.argv) > 6 else 726000)
