#!/usr/bin/env python3
"""V-B for s4/term/hexterm.py: harness/vb.py + relaxed checker (o2/port/vbx2.py) with the spec KIND
'hex_terminal_add': the terminal window of a run, whose output point is MEASURED.

canonical x-coordinate of A + [digit] P (the measured x_J; in the circuit a classical record), 'sg' = 1 iff the
output's y equals Y_J = min(y, p - y) (the measured sign bit s), and xr unchanged. Every other qubit must return to
|0> and no phase may remain, for every input and every measurement outcome: with these checks the post-measurement
state of the digit register is the ideal coset state (the preimage of the measured point), for any superposition of
(accumulator, digit) inputs.
usage: python3 s4/term/vbxterm.py s4/term/hexterm.py [vb options]
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'harness'), os.path.join(REPO, 'o1'), os.path.join(REPO, 'o2', 'port'), HERE]
import vbspecs
import vbhexw      # noqa: F401  (HexWindowAdd)
import vbx2        # noqa: F401  (patches vb.check_batch / run_checks)
import vb


class HexTerminalAdd(vbhexw.HexWindowAdd):
    def __init__(self, n, W, seed='0'):
        super().__init__(n, W, seed)
        self.ctx['point'] = self.point

    def spec(self, v):
        A = None if (v['x'], v['y']) == (0, 0) else (v['x'], v['y'])
        R = self.E.add(A, self.point(v['xr']))
        p = self.E.p
        if R is None:
            return {'xJ': 0, 'sg': 0, 'xr': v['xr']}
        Y = min(R[1], p - R[1])
        return {'xJ': R[0], 'sg': 1 if R[1] == Y else 0, 'xr': v['xr']}


vbspecs.KINDS['hex_terminal_add'] = HexTerminalAdd

if __name__ == '__main__':
    vb.main()
