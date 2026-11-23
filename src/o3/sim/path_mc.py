"""o3: path-level Monte Carlo of the leader stream under the global-scalar run law (paper Sec. V.D and App. B; , Lemma G).

Each path is one input x of a run with J hex W = 8 windows: per window a uniform residue (a, b) in (Z/256)^2 and its
digit (d1, d2) = o1/hexdigits.digit(a, b, 8), scalar t_w = (d1 + d2 lam) 2^(8 (w + 1)) mod r (lam^2 + lam + 1 = 0,
the endomorphism's eigenvalue). Run randomness: c uniform in [1, r - 1] and R0 uniform, so A_1 = [s0] G with s0
phase-tracking replay, s2/port/clsim_ph.py, fresh X-measurement outcomes per call) on every path's CORRECT pair
(A_w, B_w) and, in the same call, on the REAL pair of every path whose accumulator already deviates, so wrong
outputs are chained through the later windows as the circuit would.

Per path, on the correct path: N_DO (wrong output, an R op meeting a nonzero qubit, or a dirty ancilla) and N_P
(right output, clean, phase flipped); corr = (N_DO + 2 N_P - 1 + (-1)^N_P [N_DO = 0]) [N_DO + N_P >= 2], the
second-order term of Lemma G. Real end class: correct (+ or - sign), R-flagged, dirty, off-curve at the end (the
on-curve test of the measured accumulator flags it), on-curve wrong, or recovered (deviated, back at the correct
point, unflagged and clean).
usage: python3 -I o3/sim/path_mc.py OPS_RAW PATHS J SEED OUT_JSON"""
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path[:0] = [os.path.join(ROOT, 's2', 'port'), os.path.join(ROOT, 'o2', 'port'), os.path.join(ROOT, 'o1')]
import offdep  # noqa: E402
from clsim import to_lanes, lanes_value, P  # noqa: E402
from clsim_ph import simulate  # noqa: E402
from hexdigits import digit  # noqa: E402

R = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
LAM = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
assert (LAM * LAM + LAM + 1) % R == 0


def dbl(Pt):
    x, y = Pt
    lam = 3 * x * x * pow(2 * y, -1, P) % P
    x3 = (lam * lam - 2 * x) % P
    return x3, (lam * (x - x3) - y) % P


def add1(A, B):
    if A is None:
        return B
    if B is None:
        return A
    if A[0] == B[0]:
        return dbl(A) if A[1] == B[1] else None
    lam = (B[1] - A[1]) * pow(B[0] - A[0], -1, P) % P
    x3 = (lam * lam - A[0] - B[0]) % P
    return x3, (lam * (A[0] - x3) - A[1]) % P


def batch_add(As, Bs):
    res = list(As)
    idx, den = [], []
    for k, (a, b) in enumerate(zip(As, Bs)):
        if b is None:
            continue
        if a is None:
            res[k] = b
            continue
        if a[0] == b[0]:
            res[k] = add1(a, b)
            continue
        idx.append(k)
        den.append((b[0] - a[0]) % P)
    if den:
        n = len(den)
        pref = [1] * (n + 1)
        for i, v in enumerate(den):
            pref[i + 1] = pref[i] * v % P
        inv = pow(pref[n], -1, P)
        for i in range(n - 1, -1, -1):
            iv = inv * pref[i] % P
            inv = inv * den[i] % P
            k = idx[i]
            (x1, y1), (x2, y2) = As[k], Bs[k]
            lam = (y2 - y1) * iv % P
            x3 = (lam * lam - x1 - x2) % P
            res[k] = (x3, (lam * (x1 - x3) - y1) % P)
    return res


def fixed_base_tables():
    tabs = []
    base = (GX, GY)
    for j in range(32):
        row = [None, base]
        for v in range(2, 256):
            row.append(add1(row[-1], base))
        tabs.append(row)
        for _ in range(8):
            base = dbl(base)
    return tabs


def fixed_base(tabs, scalars):
    acc = [None] * len(scalars)
    for j in range(32):
        acc = batch_add(acc, [tabs[j][(s >> (8 * j)) & 255] for s in scalars])
    return acc


def on_curve(Pt):
    x, y = Pt
    return x < P and y < P and (y * y - x * x * x - 7) % P == 0


def main(raw, paths, J, seed, out_json):
    t0 = time.time()
    ops = np.memmap(raw, dtype=offdep.DT, mode='r')
    rng = random.Random(seed)
    tabs = fixed_base_tables()
    cs = [rng.randrange(1, R) for _ in range(paths)]
    t = [[0] * paths for _ in range(J)]
    for w in range(J):
        sh = pow(2, 8 * (w + 1), R)
        for k in range(paths):
            d1, d2 = digit(rng.randrange(256), rng.randrange(256), 8)
            t[w][k] = (d1 + d2 * LAM) * sh % R
    A = fixed_base(tabs, [rng.randrange(1, R) for _ in range(paths)])
    print('setup %.0f s' % (time.time() - t0), file=sys.stderr, flush=True)
    nDO = [0] * paths
    nP = [0] * paths
    real = [None] * paths            # None: on the correct path; else the deviated accumulator (x, y)
    state = ['ok'] * paths           # ok / dev / rflag / dirty
    per_add = dict(lanes=0, DO=0, P=0, wrong=0, rhit=0, dirty=0, wrong_oncurve=0, phase_any=0)
    garb = dict(lanes=0, rhit=0, dirty=0, out_oncurve=0)
    for w in range(J):
        B = fixed_base(tabs, [cs[k] * t[w][k] % R for k in range(paths)])
        Anext = batch_add(A, B)
        dev = [k for k in range(paths) if state[k] == 'dev']
        P_in = A + [real[k] for k in dev]
        A_in = B + [B[k] for k in dev]
        L = len(P_in)
        q, ph, rhit = simulate(ops, L, P_in, A_in, seed * 1000 + w + 1)
        xo, yo = lanes_value(q, 0, L), lanes_value(q, 256, L)
        dirty = 0
        for tq in range(512, 1227):
            dirty |= q[tq]
        for k in range(paths):
            ok_out = Anext[k] is not None and (xo[k], yo[k]) == Anext[k]
            fr, fp, fd = (rhit >> k) & 1, (ph >> k) & 1, (dirty >> k) & 1
            do = (not ok_out) or fr or fd
            per_add['lanes'] += 1
            per_add['wrong'] += not ok_out
            per_add['rhit'] += fr
            per_add['dirty'] += fd
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
                if fr:
                    state[k] = 'rflag'
                elif fd:
                    state[k] = 'dirty'
                else:
                    state[k] = 'dev'
                    real[k] = (xo[k], yo[k])
        for i, k in enumerate(dev):              # chained garbage of paths that deviated earlier
            lane = paths + i
            garb['lanes'] += 1
            fr, fd = (rhit >> lane) & 1, (dirty >> lane) & 1
            out = (xo[lane], yo[lane])
            garb['rhit'] += fr
            garb['dirty'] += fd
            garb['out_oncurve'] += on_curve(out)
            if fr:
                state[k] = 'rflag'
            elif fd:
                state[k] = 'dirty'
            else:
                real[k] = out
        A = Anext
        print('window %d/%d: %d lanes, %d deviated, %.0f s' % (w + 1, J, L, len(dev), time.time() - t0),
              file=sys.stderr, flush=True)
    end = dict(correct_plus=0, correct_minus=0, rflag=0, dirty=0, offcurve=0, oncurve_wrong=0, recovered=0)
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
        elif s in ('rflag', 'dirty'):
            end[s] += 1
        else:
            X = real[k]
            if X == A[k]:
                end['recovered'] += 1
            elif on_curve(X):
                end['oncurve_wrong'] += 1
            else:
                end['offcurve'] += 1
    res = dict(paths=paths, J=J, seed=seed, raw=raw, per_addition=per_add, garbage_chain=garb, end=end,
               corr_sum=corr_sum, corr_mean=corr_sum / paths, counts_DO_P=hist, secs=time.time() - t0)
    print(json.dumps(res), flush=True)
    json.dump(res, open(out_json, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
