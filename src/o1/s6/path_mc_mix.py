"""path-level Monte Carlo of the leader stream under the global-scalar run law (paper Sec. V.D and App. B; , Lemma G) for
a run whose a-register is a MIXED Eisenstein layout (o1/mixrad.py: hex windows hW and norm-7 x 4^w windows nw / mw),

  LAYOUT lists the run's additions lowest first, mixrad syntax without the sine top (the top belongs to the exact
  first lookup), e.g. the sigma 2.88 stack 'h9,n7,m7x10,h8x3'. Window i has modulus pi_i (2^W, or q 2^w with q = 3 + w
  for 'n', 2 - w for 'm') and sits at the Eisenstein position Pi_i = pi_0 ... pi_(i-1). Its addend scalar is
  t_i = phi_r(d_i Pi_i) = (Re + Im lam) mod r, where d_i = win.digit(z) is the window's digit (hex lift, or the
  rosette lift of mixrad.N7Window) of a uniform residue z of Z[w] / (pi_i): for a hex window z uniform in
  (Z / 2^W)^2; for a norm-7 window z = (x0 + 2^w t, x1) with x uniform in (Z / 2^w)^2 and t chosen so that the F_7 part
  phi_7(z) is a uniform c in F_7 (CRT: Z[w] / (q 2^w) = F_7 x (Z / 2^w)^2).

rule (executed = emitted CCX + CCZ - skipped > STREAM_CAP aborts the addition: a visible flag and a deviation),
for this layout (o1/s6/mixva6.py: W for a hex window, 100 + w for a norm-7 window, which is what the run's
cost['add_ws'] holds), 'labels' the expanded layout tokens (checked against the run by mixva6), and 'source' the
usage: python3 -I o1/s6/path_mc_mix.py OPS_RAW PATHS LAYOUT SEED OUT_JSON [STREAM_CAP]"""
import hashlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path[:0] = [os.path.join(ROOT, 'h1', 'hexcap'), os.path.join(ROOT, 'o1')]
import path_mc_cap as h1pmc
import mixrad as mr  # noqa: E402

R, LAM = h1pmc.R, h1pmc.LAM
simulate, batch_add, fixed_base_tables, fixed_base, on_curve = (h1pmc.simulate, h1pmc.batch_add,
                                                                h1pmc.fixed_base_tables, h1pmc.fixed_base,
                                                                h1pmc.on_curve)


def expand(layout):
    """'h9,n7,m7x10,h8x3' -> ['h9', 'n7', 'm7', ..., 'h8', ...] (a trailing sine top 'sW' is dropped)"""
    out = []
    for tok in layout.split(','):
        tok = tok.strip()
        cnt = 1
        if 'x' in tok:
            tok, c = tok.split('x')
            cnt = int(c)
        out += [tok] * cnt
    if out and out[-1][0] == 's':
        out = out[:-1]
    if any(t[0] not in 'hnm' for t in out):
        raise ValueError('layout %r: only hW / nw / mw windows below the top' % layout)
    return out


def code(tok):
    return int(tok[1:]) + (0 if tok[0] == 'h' else 100)


def uniform_residue(rng, win):
    """a uniform residue of Z[w] / (pi) as a representative z"""
    if win.kind == 'hex':
        M = 1 << win.W
        return (rng.randrange(M), rng.randrange(M))
    M = 1 << win.w
    x0, x1 = rng.randrange(M), rng.randrange(M)
    c = rng.randrange(7)
    t = (c - win.phi7((x0, x1))) * win.inv2w % 7          # phi7(x + 2^w t) = phi7(x) + 2^w t = c (mod 7)
    z = (x0 + (t << win.w), x1)
    assert win.phi7(z) == c and z[0] % M == x0 and z[1] % M == x1
    return z


def main(raw, paths, layout, seed, out_json, stream_cap):
    import numpy as np
    t0 = time.time()
    ops = np.memmap(raw, dtype=h1pmc.offdep.DT, mode='r')
    kinds = np.array(ops['kind'])
    n_tof = int(((kinds == 13) | (kinds == 14)).sum())       # the stream's emitted CCX + CCZ (761,147)
    del kinds
    labels = expand(layout)
    wins = mr.cached(','.join(labels))
    windows = [code(tk) for tk in labels]
    pos = mr.positions(wins)                                 # pos[i] = Pi_i (Eisenstein), lowest window at 1
    rng = random.Random(seed)
    tabs = fixed_base_tables()
    J = len(wins)
    order = list(range(J - 1, -1, -1))                       # top window first, the terminal (lowest) last
    cs = [rng.randrange(1, R) for _ in range(paths)]
    t = [[0] * paths for _ in range(J)]
    for w in range(J):
        win, P = wins[w], pos[w]
        memo = {}
        for k in range(paths):
            z = uniform_residue(rng, win)
            key = (z[0], z[1])
            if key not in memo:
                memo[key] = mr.phi_r(mr.emul(win.digit(z), P), R, LAM)
            t[w][k] = memo[key]
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
        xo, yo = h1pmc.lanes_value(q, 0, L), h1pmc.lanes_value(q, 256, L)
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
        print('window %d/%d (%s, code %d): %d lanes, %d deviated, %d aborted, executed mean %.1f max %d, %.0f s'
              % (step + 1, J, labels[w], windows[w], L, len(dev), int(ab.sum()), float(ex[:paths].mean()),
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
    src = os.path.join(ROOT, 'h1', 'hexcap', 'path_mc_cap.py')
    res = dict(paths=paths, J=J, windows=windows, labels=labels, layout=layout, seed=seed, raw=os.path.basename(raw),
               stream_cap=stream_cap, stream_toffoli=n_tof, per_addition=per_add, garbage_chain=garb, end=end,
               corr_sum=corr_sum, corr_mean=corr_sum / paths, counts_DO_P=hist,
               executed=dict(lanes=ex_n, mean=mean, std=(ex_sq / ex_n - mean * mean) ** 0.5, min=ex_min, max=ex_max),
               secs=time.time() - t0,
               source='h1/hexcap/path_mc_cap.py simulate() sha256 %s' % hashlib.sha256(open(src, 'rb').read()).hexdigest()[:16])
    print(json.dumps(res), flush=True)
    json.dump(res, open(out_json, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5],
         int(sys.argv[6]) if len(sys.argv) > 6 else 726000)
