#!/usr/bin/env python3
"""Capped V-B of o1/n7lead.py (norm-7 x 4^w window on leader port) for both norm-7 window types, .
KIND n7_window_add is o1/vbn7w.py's spec (pi = (3 + w) 2^w); this file also registers KIND m7_window_add, the same spec
with BAR = True (pi = (2 - w) 2^w: o1/mixrad.py N7Window(w, True), its table and the circuit's ctx['bar']), so the
V-B JSON's KIND names the window it checked and harness/cost.py labels it 'n7' / 'm7'. The checker is
usage: O2_OPS_RAW=RAW python3 -B h1/n7/vbx2capn7m7.py o1/n7lead.py -p KIND=m7_window_add -p W=7 -p HOLD=peak \
           --n 256 --lanes 1024 --batches 1 --json F"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'harness'), os.path.join(REPO, 'o1'), os.path.join(REPO, 'o2', 'port'),
                os.path.join(REPO, 'h1', 'hexcap')]
import vbn7w       # noqa: E402  (registers KIND n7_window_add)
import vbspecs     # noqa: E402


class M7WindowAdd(vbn7w.N7WindowAdd):
    """o1/vbn7w.py's spec for the conjugate window (2 - w) 2^w: BAR = True while the spec builds its window, table and
    ctx (the only places that read BAR)"""

    def __init__(self, n, w, seed='0'):
        old, vbn7w.BAR = vbn7w.BAR, True
        try:
            super().__init__(n, w, seed)
        finally:
            vbn7w.BAR = old
        assert self.win.bar and self.ctx['bar']


vbspecs.KINDS['m7_window_add'] = M7WindowAdd
import vbx2cap     # noqa: E402,F401  (h1: patches vb.check_batch with the cap)
import vb          # noqa: E402

if __name__ == '__main__':
    vb.main()
