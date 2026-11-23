#!/usr/bin/env python3
"""V-B for h1/hexcap/hexlead_cap.py: o2/hex/vbx2hex.py (harness/vb.py + 'hex_window_add' spec + relaxed
checker o2/port/vbx2.py) with the abort cap applied the way harness/vb.py's own check_batch applies it: after the build,
every random-outcome lane whose executed Toffoli passes mod.abort_cap(n) is flagged (a detected failure, so it leaves
the wrong-output / garbage checks), the count goes to bad['aborted'], and the per-lane executed counts go back to
run_checks (counts['_executed']) for toffoli_executed_mean / max and outcome_lanes. Forced-outcome batches skip the cap.
usage: python3 -B h1/hexcap/vbx2cap.py CIRCUIT.py [vb options]   (O2_OPS_RAW = the leader's raw stream)"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'harness'), os.path.join(REPO, 'o1'), os.path.join(REPO, 'o2', 'port')]
import vbhexw      # noqa: F401  (registers KIND hex_window_add)
import vbx2        # noqa: F401  (patches vb.check_batch / run_checks: relaxed checker)
import vb

_relaxed = vb.check_batch


def check_batch(mod, sp, n, assigns, rng, force=None):
    if force is not None or not hasattr(mod, 'abort_cap'):
        return _relaxed(mod, sp, n, assigns, rng, force)
    box, build = {}, mod.build

    def capped(c, *a):
        outs = build(c, *a)
        ex = c.executed()[:c.L]
        cap = mod.abort_cap(n)
        over = sum(1 << lane for lane, v in enumerate(ex) if v > cap)
        c.flag_mask |= over                      # vbx2 reads flag_mask after the build: aborted = flagged
        box.update(ex=ex, aborted=bin(over).count('1'))
        return outs
    mod.build = capped
    try:
        bad, counts, fails = _relaxed(mod, sp, n, assigns, rng, force)
    finally:
        mod.build = build
    bad['aborted'] = box['aborted']
    counts['_executed'] = box['ex']
    return bad, counts, fails


vb.check_batch = check_batch

if __name__ == '__main__':
    vb.main()
