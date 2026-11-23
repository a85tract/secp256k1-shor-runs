#!/usr/bin/env python3
"""V-B: reversible arithmetic checker.

A circuit file defines
  inputs(n)            -> list of (register name, bits)
  build(c, n, regs)    -> dict of output register name -> qubit list; emits gates on the builder `c`
  spec(n, vals)        -> dict of output name -> expected int, for one input assignment vals (name -> int)
  sample(n, rng)       -> one random valid input assignment
  exceptional(n)       -> list of exceptional input assignments (may be empty)
and optionally EXACT = False for approximate circuits (failures are then counted, not fatal), with
  declared_failure(n) -> the circuit's own bound on its failure probability per operation on the inputs a run
                         feeds it (failures too rare for the sampled lanes; stored in --json, priced by run.py)
  abort_cap(n)        -> hard cap on the Toffoli one operation may EXECUTE (gates with cond=m count only where m
                         its count would pass the cap; the abort is a detected failure, so on every random-outcome
                         lane whose executed count passes the cap V-B sets the flag (needs EXACT = False). The cost
                         model then charges the cap instead of the emitted count (harness/cost.py)

Builder `c` (qubits are ints; registers are little-endian lists):
  c.alloc(k) / c.free(qs)            fresh |0> qubits / release (checked to be |0> on every input)
  c.x(q) c.cx(a, t) c.ccx(a, b, t)   X, CNOT, Toffoli          (c.toffoli counts CCX, CCZ, AND)
  c.z(q) c.cz(a, b) c.ccz(a, b, t)   phase gates
  c.and_(a, b) -> t                  t = a AND b into a fresh qubit (one Toffoli; Gidney's 4-T AND)
  m = c.measure_x(q)                 X-basis measurement of q (q is released); returns the outcome m
  any gate with cond=m               applied only when outcome m is 1 (classically controlled fix-up)
  c.global_phase(cond=m)             a -1 phase on outcome m (to cancel outcome-dependent global phases)

Simulation: bit-sliced over many inputs at once (one Python int per qubit, one bit per input lane), with a
per-lane phase bit. Every gate above maps a basis state to a basis state times +-1, so the per-lane
simulation is exact for superpositions too. An X measurement of a qubit holding f(x) gives a uniformly random
outcome m whatever x is and multiplies the state by (-1)^(m f(x)); each lane draws its own outcomes, and two
extra batches force all outcomes to 0 and to 1, so both outcomes of every measurement are checked on every
input of those batches. Checks per lane: outputs equal spec; every other live qubit is |0> at the end (inputs
not declared as outputs must be restored, see below); no -1 phase. Counts: Toffoli, CNOT, measurements, peak
live qubits (inputs + ancillas).
Registers not listed in build()'s outputs must come back unchanged (they are checked against their inputs).

usage: python3 harness/vb.py CIRCUIT.py [-p KEY=VALUE] [--n 4,8,256] [--lanes 1024] [--batches 4] [--exhaustive-max 16]
"""
import argparse, ast, importlib.util, itertools, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Fail(Exception):
    pass


class Circuit:
    def __init__(self, lanes, rng, force=None):
        self.L, self.ALL = lanes, (1 << lanes) - 1
        self.rng, self.force = rng, force
        self.s = {}
        self.next = 0
        self.live = self.peak = 0
        self.toffoli = self.cnot = self.nx = self.meas = 0
        self.phase = 0
        self.freed_nonzero = 0
        self.freed_mask = 0                 # lanes that released a nonzero qubit (hard failure unless flagged)
        self.flag_mask = 0                  # lanes with a flag measurement outcome 1 (declared, detected failure)
        self.skip = []                      # bit planes of a per-lane count of conditional Toffolis not run

    def _new(self):
        q = self.next
        self.next += 1
        self.s[q] = 0
        self.live += 1
        self.peak = max(self.peak, self.live)
        return q

    def load(self, values, bits):
        qs = [self._new() for _ in range(bits)]
        for lane, v in enumerate(values):
            for i in range(bits):
                if (v >> i) & 1:
                    self.s[qs[i]] |= 1 << lane
        return qs

    def read(self, qs):
        out = [0] * self.L
        for i, q in enumerate(qs):
            w = self.s[q]
            while w:
                low = w & -w
                out[low.bit_length() - 1] |= 1 << i
                w ^= low
        return out

    def alloc(self, k):
        return [self._new() for _ in range(k)]

    def free(self, qs):
        for q in qs:
            w = self.s.pop(q)
            if w:
                self.freed_nonzero += 1
                self.freed_mask |= w
            self.live -= 1

    def flag(self, qs, cond=None):
        """Z-measure and discard qubits that are 0 on every input the circuit handles; an outcome 1 marks the lane
        (the run) as a detected failure, which the circuit must declare (EXACT = False) and the run discards.
        it stays allocated (live) for the others, so no borrowed ancilla is needed."""
        self.flag_used = True
        for q in qs:
            self.meas += 1
            if cond is None:
                self.flag_mask |= self.s.pop(q)
                self.live -= 1
            else:
                self.flag_mask |= self.s[q] & cond
                self.s[q] &= ~cond

    def _m(self, cond):
        return self.ALL if cond is None else cond

    def x(self, q, cond=None):
        self.nx += 1
        self.s[q] ^= self._m(cond)

    def cx(self, a, t, cond=None):
        self.cnot += 1
        self.s[t] ^= self.s[a] & self._m(cond)

    def _skipped(self, cond):
        """add 1 to the skip count of every lane outside cond (bit-sliced ripple counter, after vbx2)"""
        m, i, planes = self.ALL & ~cond, 0, self.skip
        while m:
            if i == len(planes):
                planes.append(0)
            planes[i], m = planes[i] ^ m, planes[i] & m
            i += 1

    def executed(self):
        """Toffoli executed per lane: every emitted one minus the conditional ones whose condition was 0 there"""
        return [self.toffoli - sum(((p >> lane) & 1) << i for i, p in enumerate(self.skip)) for lane in range(self.L)]

    def ccx(self, a, b, t, cond=None):
        self.toffoli += 1
        if cond is not None:
            self._skipped(cond)
        self.s[t] ^= self.s[a] & self.s[b] & self._m(cond)

    def z(self, q, cond=None):
        self.phase ^= self.s[q] & self._m(cond)

    def cz(self, a, b, cond=None):
        self.phase ^= self.s[a] & self.s[b] & self._m(cond)

    def ccz(self, a, b, t, cond=None):
        self.toffoli += 1
        if cond is not None:
            self._skipped(cond)
        self.phase ^= self.s[a] & self.s[b] & self.s[t] & self._m(cond)

    def swap(self, a, b):
        self.s[a], self.s[b] = self.s[b], self.s[a]

    def and_(self, a, b):
        t = self._new()
        self.toffoli += 1
        self.s[t] = self.s[a] & self.s[b]
        return t

    def measure_x(self, q, cond=None):
        """X-measure q (measurement-based uncomputation: outcome m, phase (-1)^(m q)) and discard it. With cond (lane
        resets q to 0 there and keeps it live elsewhere; same outcomes, phases and lane states as the borrowed-ancilla
        emulation (alloc anc, cx(q, anc, cond), cx(anc, q, cond), measure_x(anc)) without its extra qubit."""
        self.meas += 1
        m = {None: None, 0: 0, 1: self.ALL}[self.force]
        if m is None:
            m = self.rng.getrandbits(self.L)
        if cond is None:
            self.phase ^= m & self.s.pop(q)
            self.live -= 1
        else:
            self.phase ^= m & self.s[q] & cond
            self.s[q] &= ~cond
        return m

    def global_phase(self, cond=None):
        self.phase ^= self._m(cond)


def load(path, params=()):
    spec = importlib.util.spec_from_file_location('vbcircuit_' + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for kv in params:
        k, v = kv.split('=', 1)
        if not hasattr(mod, k):
            raise SystemExit('circuit has no parameter %s' % k)
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            pass
        setattr(mod, k, v)
    return mod


class _ModuleSpec:
    """The circuit file's own spec (for building blocks without a harness KIND)."""

    def __init__(self, mod, n):
        self.mod, self.n, self.ctx = mod, n, None

    def inputs(self):
        return self.mod.inputs(self.n)

    def spec(self, v):
        return self.mod.spec(self.n, v)

    def sample(self, rng):
        return self.mod.sample(self.n, rng)

    def exceptional(self, rng):
        return self.mod.exceptional(self.n) if hasattr(self.mod, 'exceptional') else []


def make_spec(mod, n, seed):
    kind = getattr(mod, 'KIND', None)
    if kind is None:
        return _ModuleSpec(mod, n)
    import vbspecs
    if kind not in vbspecs.KINDS:
        raise SystemExit('unknown KIND %r (known: %s)' % (kind, ', '.join(vbspecs.KINDS)))
    W = mod.window(n) if hasattr(mod, 'window') else mod.W
    return vbspecs.KINDS[kind](n, W, seed)


def check_batch(mod, sp, n, assigns, rng, force=None):
    L = len(assigns)
    c = Circuit(L, rng, force)
    regs = {name: c.load([a[name] for a in assigns], bits) for name, bits in sp.inputs()}
    outs = mod.build(c, n, regs) if sp.ctx is None else mod.build(c, n, regs, sp.ctx)
    bad = dict(output=0, garbage=0, phase=0, freed_nonzero=0, flagged=0, aborted=0)
    ex = c.executed()
    cap = mod.abort_cap(n) if force is None and hasattr(mod, 'abort_cap') else None
    if cap is not None:
        # abort cap: lanes whose executed count passes it are aborted, i.e. flagged. Forced-outcome batches skip
        # this, so both outcomes of every measurement are still checked on every input there
        over = sum(1 << lane for lane, v in enumerate(ex[:L]) if v > cap)
        bad['aborted'] = bin(over).count('1')
        c.flag_mask |= over
    out_vals = {name: c.read(qs) for name, qs in outs.items()}
    out_qubits = {q for qs in outs.values() for q in qs}
    kept = {name: c.read(qs) for name, qs in regs.items()
            if name not in outs and all(q in c.s for q in qs) and not (set(qs) & out_qubits)}
    resid = 0
    for q, w in c.s.items():
        if q not in out_qubits and not any(q in regs[nm] for nm in kept):
            resid |= w
    fails = []
    for lane, a in enumerate(assigns):
        if (c.flag_mask >> lane) & 1:
            bad['flagged'] += 1             # detected failure: excluded from the output / garbage checks
            fails.append(a)
            continue
        bad['freed_nonzero'] += (c.freed_mask >> lane) & 1
        exp = sp.spec(a)
        ok_out = all(out_vals[k][lane] == v for k, v in exp.items() if k in out_vals) and \
            all(k in out_vals or kept.get(k, [None] * L)[lane] == v for k, v in exp.items())
        ok_kept = all(kept[k][lane] == a[k] for k in kept)
        g = (resid >> lane) & 1
        ph = (c.phase >> lane) & 1
        bad['output'] += not (ok_out and ok_kept)
        bad['garbage'] += g
        bad['phase'] += ph
        if not (ok_out and ok_kept) or g or ph:
            fails.append(a)
    counts = dict(toffoli=c.toffoli, cnot=c.cnot, x=c.nx, measurements=c.meas, peak_qubits=c.peak,
                  live_at_end=c.live, flag_used=getattr(c, 'flag_used', False), _executed=ex)
    return bad, counts, fails


def run_checks(mod, n, lanes, batches, exhaustive_max, seed):
    rng = random.Random('%s/%d' % (seed, n))
    sp = make_spec(mod, n, seed)
    names = sp.inputs()
    total_bits = sum(b for _, b in names)
    sets = []
    if total_bits <= exhaustive_max and getattr(mod, 'ALL_INPUTS_VALID', False) and sp.ctx is None:
        allv = [dict(zip([nm for nm, _ in names], vals))
                for vals in itertools.product(*[range(1 << b) for _, b in names])]
        sets += [allv[i:i + lanes] for i in range(0, len(allv), lanes)]
        mode = 'exhaustive (%d inputs)' % len(allv)
    else:
        sets += [[sp.sample(rng) for _ in range(lanes)] for _ in range(batches)]
        mode = 'random (%d x %d inputs)' % (batches, lanes)
    exc = sp.exceptional(rng)
    tot = dict(output=0, garbage=0, phase=0, freed_nonzero=0, flagged=0, aborted=0)
    exc_tot = dict(output=0, garbage=0, phase=0, freed_nonzero=0, flagged=0, aborted=0)
    lanes_done, counts, examples, executed = 0, None, [], []      # executed: per random-outcome lane
    batches_all = [(a, False) for a in sets] + ([(exc, True)] if exc else [])
    for i, (assigns, is_exc) in enumerate(batches_all):
        for force in ((None, 0, 1) if i == 0 or is_exc else (None,)):
            bad, cnt, fails = check_batch(mod, sp, n, assigns, rng, force)
            ex = cnt.pop('_executed', None)         # absent when another checker replaces check_batch (o2/s2 vbx)
            if ex is not None and force is None and not is_exc:
                executed += ex[:len(assigns)]
            for k in tot:
                (exc_tot if is_exc else tot)[k] += bad.get(k, 0)    # other checkers may not report 'aborted'
            if not is_exc:
                lanes_done += len(assigns)
            examples += fails[:2]
            if counts is None:
                counts = cnt
            elif cnt['toffoli'] != counts['toffoli'] or cnt['peak_qubits'] != counts['peak_qubits']:
                counts['data_dependent_counts'] = True
    if executed:
        # Toffoli with a classical condition (e.g. measurement-outcome fix-up blocks) run only where it holds; the
        # cost model charges every emitted Toffoli unless the circuit declares abort_cap(n) (then the cap)
        counts.update(toffoli_executed_mean=sum(executed) / len(executed), toffoli_executed_max=max(executed),
                      outcome_lanes=len(executed))
    return mode, lanes_done, tot, exc_tot, counts, examples[:3], len(exc)


def _first_offset(mod, n):
    """counts of the random-offset first lookup (unsigned 2^W table), if the circuit file provides it"""
    import inspect
    if not hasattr(mod, 'first_lookup') or 'offset' not in inspect.signature(mod.first_lookup).parameters:
        return None
    return mod.first_lookup(n, offset=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('circuit')
    ap.add_argument('--n', default='4,8,16,64,256')
    ap.add_argument('--lanes', type=int, default=1024)
    ap.add_argument('--batches', type=int, default=4)
    ap.add_argument('--exhaustive-max', type=int, default=16, help='enumerate all inputs up to this many bits')
    ap.add_argument('--seed', default='0')
    ap.add_argument('-p', '--param', action='append', default=[], help='override a circuit constant KEY=VALUE')
    ap.add_argument('--json', help='write counts and verdicts (input to run.py --measured)')
    args = ap.parse_args()
    mod = load(args.circuit, args.param)
    exact = getattr(mod, 'EXACT', True)
    kind = getattr(mod, 'KIND', None)
    if exact and hasattr(mod, 'abort_cap'):
        raise SystemExit('abort_cap(n) makes aborts detected failures: the circuit needs EXACT = False')
    print('circuit %s %s(%s; spec: %s)' % (args.circuit, ' '.join(args.param) + ' ' if args.param else '',
                                           'exact' if exact else 'approximate: failures counted',
                                           'harness KIND ' + kind if kind else "the file's own spec()"))
    print('%5s %-26s %8s %9s %8s %8s %8s %6s %7s %8s  %s' % ('n', 'inputs', 'lanes', 'Toffoli', 'CNOT', 'meas.',
                                                              'peak q', 'outerr', 'garb/ph', 'exc.err', 'verdict'))
    all_ok, report = True, dict(circuit=args.circuit, params=args.param, kind=kind, exact=exact, sizes=[])
    for n in [int(x) for x in args.n.split(',')]:
        mode, lanes, tot, etot, cnt, ex, nexc = run_checks(mod, n, args.lanes, args.batches, args.exhaustive_max,
                                                           args.seed)
        hard = tot['garbage'] + tot['phase'] + tot['freed_nonzero'] + etot['garbage'] + etot['phase'] + \
            etot['freed_nonzero']
        ok = hard == 0 and (not exact or tot['output'] + etot['output'] + tot['flagged'] + etot['flagged'] == 0)
        all_ok &= ok
        print('%5d %-26s %8d %9d %8d %8d %8d %6d %3d/%-3d %8s  %s%s' % (
            n, mode, lanes, cnt['toffoli'], cnt['cnot'], cnt['measurements'], cnt['peak_qubits'],
            tot['output'], tot['garbage'] + etot['garbage'], tot['phase'] + etot['phase'],
            '%d/%d' % (etot['output'], nexc * 3), 'PASS' if ok else 'FAIL',
            ' (data-dependent counts)' if cnt.get('data_dependent_counts') else ''))
        if tot['freed_nonzero'] + etot['freed_nonzero']:
            print('      %d lanes freed a nonzero qubit' % (tot['freed_nonzero'] + etot['freed_nonzero']))
        if not exact and tot['output']:
            print('      approximate: %d of %d random lanes wrong (%.3g)' % (tot['output'], lanes,
                                                                          tot['output'] / lanes))
        if tot['flagged'] + etot['flagged']:
            print('      flagged (detected, declared failures): %d of %d random lanes, %d of %d exceptional lanes'
                  % (tot['flagged'], lanes, etot['flagged'], nexc * 3))
        cap = mod.abort_cap(n) if hasattr(mod, 'abort_cap') else None
        if cap is not None and (cnt.get('toffoli_executed_max') is None or
                                (cnt['toffoli_executed_max'] > cap and not tot['aborted'])):
            # a checker that replaces check_batch (o2/s2 vbx) must flag the over-cap lanes and report 'aborted'
            raise SystemExit('abort cap %d: executed counts missing, or lanes over the cap were not aborted '
                             '(executed max %s, aborted %d)' % (cap, cnt.get('toffoli_executed_max'), tot['aborted']))
        if cap is not None:
            print('      abort cap %d executed Toffoli (charged instead of %d emitted): %d of %d random-outcome lanes '
                  'over it (flagged); executed mean %.1f, max %d' % (cap, cnt['toffoli'], tot['aborted'],
                                                                     cnt.get('outcome_lanes', 0),
                                                            cnt.get('toffoli_executed_mean', -1),
                                                            cnt.get('toffoli_executed_max', -1)))
        for e in ex:
            print('      failing input: %s' % {k: hex(v) for k, v in e.items()})
        report['sizes'].append(dict(n=n, verdict='PASS' if ok else 'FAIL', mode=mode, lanes=lanes,
                                    random_failures=tot['output'] + tot['flagged'],
                                    random_flagged=tot['flagged'], exceptional_flagged=etot['flagged'],
                                    first_lookup=mod.first_lookup(n) if hasattr(mod, 'first_lookup') else None,
                                    first_lookup_offset=_first_offset(mod, n),
                                    exceptional_failures=etot['output'],
                                    exceptional_lanes=nexc * 3, counts=cnt, abort_cap=cap, aborted=tot['aborted'],
                                    window=getattr(mod, 'W', None) if not hasattr(mod, 'window') else mod.window(n),
                                    declared_failure=mod.declared_failure(n) if hasattr(mod, 'declared_failure')
                                    else None))
        if hasattr(mod, 'declared_failure'):
            print('      declared failure probability per operation (circuit file, analytic / sampled): %.3g'
                  % mod.declared_failure(n))
    if args.json:
        import json
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=1)
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
