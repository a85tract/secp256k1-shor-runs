#!/usr/bin/env python3
"""V-B with one change, for approximate circuits whose failing inputs leave garbage (s2/port/vbx.py, adapted by o2

harness/vb.py fails an approximate circuit on ANY residual garbage or -1 phase, so a circuit whose failure mode is
"the walk did not converge: wrong output AND leftover garbage / phase on that input" can never pass, although the
failure is confined to the failing inputs. Here a lane (one input) is FAILED if its output is wrong, or if it
leaves garbage or a -1 phase (with random outcomes); failed lanes are counted as failed additions and reported as
random_failures / exceptional_failures (a failure rate above MAXFAIL = 0.5 % of a random batch is a hard FAIL, and so is
any freed nonzero qubit). Failures are counted once, in the batch with random measurement outcomes; the forced
all-0 / all-1 batches are run as in harness/vb.py but only their dirty lanes are tallied. Everything else (builder, spec,
lanes, JSON) is harness/vb.py, which this script imports and patches. VBX_DUMP=path appends every failed input.
The stream's expected executed Toffoli (classically controlled CCX weighted by the fraction of lanes that execute them,
random-outcome batch) is added to the counts as toffoli_expected_executed_stream.
usage: python3 s2/port/vbx.py CIRCUIT.py [same options as harness/vb.py]
"""
import os
import sys

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'harness')  # o2/port -> repo/harness
sys.path.insert(0, HARNESS)
import vb


MAXFAIL = 0.005     # a random batch with more failing lanes than this fraction is a bug, not an approximation


def check_batch(mod, sp, n, assigns, rng, force=None):
    L = len(assigns)
    c = vb.Circuit(L, rng, force)
    regs = {name: c.load([a[name] for a in assigns], bits) for name, bits in sp.inputs()}
    outs = mod.build(c, n, regs) if sp.ctx is None else mod.build(c, n, regs, sp.ctx)
    fm = getattr(c, 'freed_mask', None)
    flm = getattr(c, 'flag_mask', 0)
    bad = dict(output=0, garbage=0, phase=0, flagged=0, wrong=0, dirty_correct=0,
               freed_nonzero=bin(fm & ~flm).count('1') if fm is not None else c.freed_nonzero)
    out_vals = {name: c.read(qs) for name, qs in outs.items()}
    out_qubits = {q for qs in outs.values() for q in qs}
    kept = {name: c.read(qs) for name, qs in regs.items()
            if name not in outs and all(q in c.s for q in qs) and not (set(qs) & out_qubits)}
    resid = 0
    for q, w in c.s.items():
        if q not in out_qubits and not any(q in regs[nm] for nm in kept):
            resid |= w
    fails, dump = [], []
    for lane, a in enumerate(assigns):
        if (flm >> lane) & 1:
            if force is None:
                bad['flagged'] += 1         # detected failure: excluded from the output / garbage checks (as vb.py)
            continue
        exp = sp.spec(a)
        ok_out = all(out_vals[k][lane] == v for k, v in exp.items() if k in out_vals) and \
            all(k in out_vals or kept.get(k, [None] * L)[lane] == v for k, v in exp.items())
        ok_kept = all(kept[k][lane] == a[k] for k in kept)
        dirty = bool(((resid >> lane) & 1) or ((c.phase >> lane) & 1))
        wrong = not (ok_out and ok_kept)
        bad['wrong'] += wrong
        bad['dirty_correct'] += (dirty and not wrong)
        if wrong or dirty:
            fails.append(a)
            dump.append((lane, wrong, dirty))
    # a failed lane = wrong output, or residual garbage / phase on a correct output (a failed addition: the
    # approximate circuit left that input's branch dirty). Counted once, from the batch with random outcomes;
    # forced-outcome batches only add their dirty lanes to the side tally (their wrong lanes are the same ones).
    if force is not None and bad['dirty_correct']:
        print('  [vbx] forced outcomes %s: %d lanes with a correct output are dirty' % (force, bad['dirty_correct']),
              file=sys.stderr, flush=True)
    if force is None:
        bad['output'] = len(dump)
        if L >= 1000 and len(dump) > MAXFAIL * L:
            bad['garbage'] = len(dump)              # too many to be an approximation error: hard failure
    dumpf = os.environ.get('VBX_DUMP')
    if dumpf and dump:
        import json
        with open(dumpf, 'a') as f:
            for lane, wrong, dirty in dump:
                f.write(json.dumps(dict(force=force, wrong=wrong, dirty=dirty,
                                        **{k: hex(v) for k, v in assigns[lane].items()})) + '\n')
    counts = dict(toffoli=c.toffoli, cnot=c.cnot, x=c.nx, measurements=c.meas, peak_qubits=c.peak,
                  live_at_end=c.live)
    st = getattr(mod, 'STATS', None)
    if st and force is None:
        counts['toffoli_expected_executed_stream'] = st.get('toff_exec_stream')
    if force is None:
        counts.update(_executed(c))
    return bad, counts, fails


# where the condition holds. Count, per lane, the conditional Toffolis it skips (bit-sliced counter on the circuit);
# check_batch reports mean / min / max of c.toffoli - skipped over the lanes of the random-outcome batch.
_orig_ccx, _orig_ccz = vb.Circuit.ccx, vb.Circuit.ccz


def _skip(c, m):
    planes = c.__dict__.setdefault('_o2_skip', [])
    i = 0
    while m:
        if i == len(planes):
            planes.append(0)
        planes[i], m = planes[i] ^ m, planes[i] & m
        i += 1


def _ccx(self, a, b, t, cond=None):
    if cond is not None:
        _skip(self, self.ALL & ~self._m(cond))
    return _orig_ccx(self, a, b, t, cond)


def _ccz(self, a, b, t, cond=None):
    if cond is not None:
        _skip(self, self.ALL & ~self._m(cond))
    return _orig_ccz(self, a, b, t, cond)


vb.Circuit.ccx, vb.Circuit.ccz = _ccx, _ccz


def _executed(c):
    planes = c.__dict__.get('_o2_skip', [])
    ex = [c.toffoli - sum(((pl >> lane) & 1) << i for i, pl in enumerate(planes)) for lane in range(c.L)]
    return dict(toffoli_executed_mean=sum(ex) / len(ex), toffoli_executed_min=min(ex), toffoli_executed_max=max(ex))


_orig_run_checks = vb.run_checks


def run_checks(mod, n, lanes, batches, exhaustive_max, seed):
    mode, lanes_done, tot, exc_tot, counts, examples, nexc = _orig_run_checks(mod, n, lanes, batches,
                                                                               exhaustive_max, seed)
    if mode.startswith('random'):
        lanes_done = lanes * batches            # forced-outcome repeats of batch 0 are not extra samples
    return mode, lanes_done, tot, exc_tot, counts, examples, nexc


vb.run_checks = run_checks
vb.check_batch = check_batch

if __name__ == '__main__':
    vb.main()
