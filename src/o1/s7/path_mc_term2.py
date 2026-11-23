"""o1/s6/path_mc_term.py with the terminal's prefix replayed AS
loaded and X-measured as the port does), the x register decoded as the lazy pair of -x_J (hexterm.py's form), and the
per-path executed-Toffoli run totals kept (an .npz next to the JSON) for a run-level abort cap. Everything else is

(s4/term/hexterm.py, KIND hex_terminal_add), under the global-scalar run law (paper Sec. V.D and App. B; , Lemma G) with the abort caps.

It is o1/s6/path_mc_mix.py (= h1/hexcap/path_mc_cap.py with the mixed-layout addend law) with the last addition replaced:

* Windows. LAYOUT lists the additions lowest first (mixrad syntax, sine top dropped); the lowest one is the terminal and
  takes its digit like any window of its type (hW: W = 9 or 8). Additions run top window first, the terminal last.
  STREAM_CAP = 726,000 (unchanged).
  TERM_TOF - skipped with TERM_TOF = 392,944 (the V-B circuit's stream count; the raw prefix has 392,942 CCX + CCZ, the
  2 extra are counted as always executed, the conservative reading), abort above TERM_CAP = 379,100. Then the lane is
  classified the way s4 measured the terminal's rates (s4/term/clsim_ph_prefix.py: the y register holds lambda and the
  x register the lazy pair (v, f) of x_B - x_J, f on wires 518, 521, 687, 693) and the way the terminal's tail acts on
  the measured record (s4/term/hexterm.py steps 1 and 3), emulated per lane on the replay's own values:
    decode: the four f copies equal, v odd, r = v - (2^256 - p) f in (0, p), f = 1 iff r even; x_J = x_B - r mod p
            (the raw stream's form; hexterm.py reads 2 x_A at x3 so its register holds -x_J, same checks);
    root:   x_J^3 + 7 a nonzero square, Y = min(y, p - y);
    sign:   lambda_+- = (y_B +- Y) / (x_B - x_J); sg = [low 32 bits of lambda_reg = those of lambda_+],
            sm = [low 16 bits of lambda_reg = those of lambda_-]; valid iff exactly one holds; y_J = Y if sg else -Y.
  A lane that fails decode / root / sign is a visibly flagged end ('tflag': the classical record shows it). A live
  ancilla left nonzero (any wire >= 512 other than the f wires) is Z-measured and flagged by hexterm.py too, so it is a
  'tflag' as well; an R hit is an 'rflag', an abort an 'aflag'. Otherwise the terminal's output is the on-curve point
  (x_J, +-Y): equal to the ideal sum with lambda_reg equal to that root's lambda in all bits -> correct (phase flip:
  a P event); a right point whose lambda_reg differs above the payload bits would get a garbage phase at the lambda
  unload, so it counts as a deviation; else a deviation whose path ends 'oncurve_wrong'
  (or 'recovered' if it equals the ideal end point of a path that deviated earlier).
  Per addition: DO = any flag, abort, dirty or wrong output; P = right output with a flipped phase (as path_mc_cap).

JSON: path_mc_cap's fields (stream_cap / stream_toffoli = the regular windows' 726,000 / 761,147), plus term_stop,
term_stream_cap, term_stream_toffoli, term_raw_ccx, a per-addition 'term' block, end['tflag'] and per-addition
'stream_caps' / 'stream_toffolis' lists (lowest first, the run's add order of the harness), and 'layout' (labels).
usage: python3 -I o1/s7/path_mc_term2.py OPS_RAW PATHS LAYOUT SEED OUT_JSON"""
import hashlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 's6')]
import path_mc_mix as pmm
import port_replay as prp

R, LAM = pmm.R, pmm.LAM
h1pmc = pmm.h1pmc
mr = pmm.mr
PF = 2 ** 256 - 2 ** 32 - 977                                # secp256k1 field prime
STREAM_CAP, STREAM_TOF = 726000, 761147
STOP, TERM_CAP, TERM_TOF = 6486360, 379100, 392944
FW = (518, 521, 687, 693)
CC_ = (1 << 256) - PF
PAY, CHK = 32, 16


def terminal_out(v, fl, lam_reg, xB, yB):
    """the terminal's classical tail on one lane's record: ('tflag', None) or ('out', (x_J, y_J))"""
    if len(set(fl)) != 1 or (v & 1) != 1:
        return 'tflag', None
    f = fl[0]
    r = v - CC_ * f
    if not (0 < r < PF) or f != (1 if r % 2 == 0 else 0):
        return 'tflag', None
    xJ = (PF - r) % PF                     # hexterm.py: x2 holds -x_J (2 xA read at x3)
    rhs = (xJ * xJ * xJ + 7) % PF
    y = pow(rhs, (PF + 1) // 4, PF)
    if y * y % PF != rhs or y == 0:
        return 'tflag', None
    Y = min(y, PF - y)
    den = (xB - xJ) % PF
    if den == 0:
        return 'tflag', None
    inv = pow(den, -1, PF)
    lp, lm = (yB + Y) * inv % PF, (yB - Y) * inv % PF
    sg = (lam_reg & ((1 << PAY) - 1)) == (lp & ((1 << PAY) - 1))
    sm = (lam_reg & ((1 << CHK) - 1)) == (lm & ((1 << CHK) - 1))
    if sg == sm:
        return 'tflag', None
    if lam_reg != (lp if sg else lm):
        return 'garb', (xJ, Y if sg else PF - Y)     # low bits matched, lambda wrong above them: the unload's phase
    return 'out', (xJ, Y if sg else PF - Y)           # fix-up is computed from the candidate, so a garbage phase


def main(raw, paths, layout, seed, out_json):
    import numpy as np
    t0 = time.time()
    ops = np.memmap(raw, dtype=h1pmc.offdep.DT, mode='r')
    kinds = np.array(ops['kind'])
    n_tof = int(((kinds == 13) | (kinds == 14)).sum())
    raw_pre = int(((kinds[:STOP] == 13) | (kinds[:STOP] == 14)).sum())
    del kinds
    assert n_tof == STREAM_TOF and raw_pre == 392942, (n_tof, raw_pre)
    ops_pre = ops[:STOP]
    labels = pmm.expand(layout)
    wins = mr.cached(','.join(labels))
    windows = [pmm.code(tk) for tk in labels]
    assert labels[0][0] == 'h', 'the terminal (lowest addition) must be a hex window'
    pos = mr.positions(wins)
    rng = random.Random(seed)
    tabs = h1pmc.fixed_base_tables()
    st = prp.site_tables()
    J = len(wins)
    order = list(range(J - 1, -1, -1))                       # top window first, the terminal (index 0) last
    cs = [rng.randrange(1, R) for _ in range(paths)]
    t = [[0] * paths for _ in range(J)]
    for w in range(J):
        win, P = wins[w], pos[w]
        memo = {}
        for k in range(paths):
            z = pmm.uniform_residue(rng, win)
            key = (z[0], z[1])
            if key not in memo:
                memo[key] = mr.phi_r(mr.emul(win.digit(z), P), R, LAM)
            t[w][k] = memo[key]
    A = h1pmc.fixed_base(tabs, [rng.randrange(1, R) for _ in range(paths)])
    print('setup %.0f s' % (time.time() - t0), file=sys.stderr, flush=True)
    nDO = [0] * paths
    run_tot = [0] * paths            # executed CCX + CCZ of the run (all additions, correct-operand lane)
    nP = [0] * paths
    real = [None] * paths            # None: on the correct path; else the deviated accumulator (x, y)
    state = ['ok'] * paths           # ok / dev / rflag / aflag / dirty / tflag
    per_add = dict(lanes=0, DO=0, P=0, wrong=0, rhit=0, dirty=0, aborted=0, wrong_oncurve=0, phase_any=0)
    term = dict(lanes=0, DO=0, P=0, wrong=0, rhit=0, dirty=0, aborted=0, tflag=0, wrong_pass=0, lam_garbage=0, phase_any=0,
                garbage_lanes=0, garbage_pass=0, garbage_tflag=0, garbage_rhit=0, garbage_aborted=0, ygarb=0)
    garb = dict(lanes=0, rhit=0, dirty=0, aborted=0, out_oncurve=0)
    ex_n, ex_sum, ex_sq, ex_min, ex_max = 0, 0, 0, None, None
    tx = dict(n=0, sum=0, sq=0, min=None, max=None)
    for step, w in enumerate(order):
        is_term = (w == 0)
        B = h1pmc.fixed_base(tabs, [cs[k] * t[w][k] % R for k in range(paths)])
        Anext = h1pmc.batch_add(A, B)
        dev = [k for k in range(paths) if state[k] == 'dev']
        P_in = A + [real[k] for k in dev]
        A_in = B + [B[k] for k in dev]
        L = len(P_in)
        if is_term:
            q, ph, rhit, sk, yg = prp.simulate_port(ops_pre, L, P_in, A_in, seed * 1000 + step + 1, 2, st)
        else:
            q, ph, rhit, sk = h1pmc.simulate(ops, L, P_in, A_in, seed * 1000 + step + 1)
            yg = 0
        if is_term:
            ex = TERM_TOF - sk
            ab = ex > TERM_CAP
            e_ = ex[:paths]
            tx['n'] += paths
            tx['sum'] += int(e_.sum())
            tx['sq'] += int((e_ * e_).sum())
            tx['min'] = int(e_.min())
            tx['max'] = int(e_.max())
        else:
            ex = n_tof - sk
            ab = ex > STREAM_CAP
            ex_n += paths
            ex_sum += int(ex[:paths].sum())
            ex_sq += int((ex[:paths] * ex[:paths]).sum())
            ex_min = int(ex[:paths].min()) if ex_min is None else min(ex_min, int(ex[:paths].min()))
            ex_max = int(ex[:paths].max()) if ex_max is None else max(ex_max, int(ex[:paths].max()))
        for k in range(paths):
            run_tot[k] += int(ex[k])
        xo, yo = h1pmc.lanes_value(q, 0, L), h1pmc.lanes_value(q, 256, L)
        dirty = 0
        for tq in range(512, 1227):
            if not (is_term and tq in FW):
                dirty |= q[tq]
        if is_term:
            fws = [q[wf] for wf in FW]
            outs = []
            for lane in range(L):
                fr, fd, fa = (rhit >> lane) & 1, (dirty >> lane) & 1, bool(ab[lane])
                if fa:
                    outs.append(('aflag', None))
                elif fr:
                    outs.append(('rflag', None))
                elif fd:
                    outs.append(('tflag', None))       # hexterm.py Z-measures the live extra wires: a detected failure
                else:
                    xB, yB = A_in[lane]
                    o_ = terminal_out(xo[lane], [(m >> lane) & 1 for m in fws], yo[lane], xB, yB)
                    if (yg >> lane) & 1:
                        term['ygarb'] += 1
                        if o_[0] == 'out':
                            o_ = ('garb', o_[1])
                    outs.append(o_)
            for k in range(paths):
                cls, out = outs[k]
                fp = (ph >> k) & 1
                term['lanes'] += 1
                term['phase_any'] += fp
                ok_out = cls == 'out' and Anext[k] is not None and out == Anext[k]
                term['lam_garbage'] += cls == 'garb'
                if cls == 'aflag':
                    term['aborted'] += 1
                elif cls == 'rflag':
                    term['rhit'] += 1
                elif cls == 'tflag':
                    term['tflag'] += 1
                    term['dirty'] += (dirty >> k) & 1
                elif not ok_out:
                    term['wrong_pass'] += 1         # an unflagged valid-looking end that is not the ideal sum
                do = not ok_out
                term['wrong'] += do and cls in ('out', 'garb')
                per_add['lanes'] += 1
                if do:
                    term['DO'] += 1
                    per_add['DO'] += 1
                    nDO[k] += 1
                elif fp:
                    term['P'] += 1
                    per_add['P'] += 1
                    nP[k] += 1
                if state[k] == 'ok' and do:
                    if cls in ('aflag', 'rflag', 'tflag'):
                        state[k] = cls
                    else:                                   # a valid-looking but wrong point: unflagged
                        state[k] = 'dev'
                        real[k] = out
            for i, k in enumerate(dev):
                lane = paths + i
                cls, out = outs[lane]
                term['garbage_lanes'] += 1
                garb['lanes'] += 1
                if cls in ('out', 'garb'):
                    term['garbage_pass'] += 1
                    garb['out_oncurve'] += 1
                    real[k] = out
                else:
                    term['garbage_' + ('tflag' if cls == 'tflag' else 'rhit' if cls == 'rflag' else 'aborted')] += 1
                    state[k] = cls
        else:
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
                if not ok_out and h1pmc.on_curve((xo[k], yo[k])):
                    per_add['wrong_oncurve'] += 1
                if do:
                    per_add['DO'] += 1
                    nDO[k] += 1
                elif fp:
                    per_add['P'] += 1
                    nP[k] += 1
                if state[k] == 'ok' and do:
                    if fa:
                        state[k] = 'aflag'
                    elif fr:
                        state[k] = 'rflag'
                    elif fd:
                        state[k] = 'dirty'
                    else:
                        state[k] = 'dev'
                        real[k] = (xo[k], yo[k])
            for i, k in enumerate(dev):
                lane = paths + i
                garb['lanes'] += 1
                fr, fd, fa = (rhit >> lane) & 1, (dirty >> lane) & 1, bool(ab[lane])
                out = (xo[lane], yo[lane])
                garb['rhit'] += fr
                garb['dirty'] += fd
                garb['aborted'] += fa
                garb['out_oncurve'] += h1pmc.on_curve(out)
                if fa:
                    state[k] = 'aflag'
                elif fr:
                    state[k] = 'rflag'
                elif fd:
                    state[k] = 'dirty'
                else:
                    real[k] = out
        A = Anext
        print('addition %d/%d (%s%s): %d lanes, %d deviated, %d aborted, executed mean %.1f max %d, %.0f s'
              % (step + 1, J, labels[w], ' TERMINAL' if is_term else '', L, len(dev), int(ab.sum()),
                 float(ex[:paths].mean()), int(ex[:paths].max()), time.time() - t0), file=sys.stderr, flush=True)
    end = dict(correct_plus=0, correct_minus=0, rflag=0, aflag=0, tflag=0, dirty=0, offcurve=0, oncurve_wrong=0,
               recovered=0)
    corr_sum = 0
    hist = {}
    ECODE = dict(correct_plus=0, correct_minus=1, rflag=2, aflag=3, tflag=4, dirty=5, offcurve=6, oncurve_wrong=7,
                 recovered=8)
    ecls = [0] * paths
    for k in range(paths):
        n = nDO[k] + nP[k]
        hist['%d,%d' % (nDO[k], nP[k])] = hist.get('%d,%d' % (nDO[k], nP[k]), 0) + 1
        if n >= 2:
            corr_sum += nDO[k] + 2 * nP[k] - 1 + ((-1) ** nP[k] if nDO[k] == 0 else 0)
        s = state[k]
        if s == 'ok':
            e_ = 'correct_plus' if nP[k] % 2 == 0 else 'correct_minus'
        elif s in ('rflag', 'aflag', 'dirty', 'tflag'):
            e_ = s
        else:
            X = real[k]
            if X == A[k]:
                e_ = 'recovered'
            elif h1pmc.on_curve(X):
                e_ = 'oncurve_wrong'
            else:
                e_ = 'offcurve'
        end[e_] += 1
        ecls[k] = ECODE[e_]
    mean = ex_sum / ex_n
    tmean = tx['sum'] / tx['n']
    import numpy as np
    rt = np.array(run_tot, dtype=np.int64)
    npz = out_json[:-5] + '_paths.npz' if out_json.endswith('.json') else out_json + '_paths.npz'
    np.savez_compressed(npz, run_total=rt, nDO=np.array(nDO, dtype=np.int16), nP=np.array(nP, dtype=np.int16),
                        end=np.array(ecls, dtype=np.int8))
    rmean, rsd = float(rt.mean()), float(rt.std())
    run_total = dict(mean=rmean, std=rsd, min=int(rt.min()), max=int(rt.max()),
                     skew=float(((rt - rmean) ** 3).mean() / rsd ** 3), npz=os.path.basename(npz), end_codes=ECODE)
    src = os.path.join(HERE, '..', '..', 'h1', 'hexcap', 'path_mc_cap.py')
    res = dict(paths=paths, J=J, windows=windows, labels=labels, layout=layout, seed=seed, raw=os.path.basename(raw),
               stream_cap=STREAM_CAP, stream_toffoli=n_tof, term_kind='hex_terminal_add', term_stop=STOP,
               term_stream_cap=TERM_CAP, term_stream_toffoli=TERM_TOF, term_raw_ccx=raw_pre,
               stream_caps=[TERM_CAP] + [STREAM_CAP] * (J - 1), stream_toffolis=[TERM_TOF] + [n_tof] * (J - 1),
               per_addition=per_add, term=term, garbage_chain=garb, end=end, corr_sum=corr_sum,
               corr_mean=corr_sum / paths, counts_DO_P=hist,
               executed=dict(lanes=ex_n, mean=mean, std=(ex_sq / ex_n - mean * mean) ** 0.5, min=ex_min, max=ex_max),
               executed_term=dict(lanes=tx['n'], mean=tmean, std=(tx['sq'] / tx['n'] - tmean * tmean) ** 0.5,
                                  min=tx['min'], max=tx['max']),
               run_total=run_total, term_replay='as built: x3 = 2 xA + OFF, y0 port load / X-measured unload (o1/s7/port_replay.py)',
               secs=time.time() - t0,
               source='h1/hexcap/path_mc_cap.py simulate() sha256 %s; o1/s7/path_mc_term2.py sha256 %s; o1/s7/port_replay.py sha256 %s' % (
                   hashlib.sha256(open(src, 'rb').read()).hexdigest()[:16],
                   hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:16],
                   hashlib.sha256(open(os.path.join(HERE, 'port_replay.py'), 'rb').read()).hexdigest()[:16]))
    print(json.dumps(res), flush=True)
    json.dump(res, open(out_json, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])
