"""V-B check (own spec, n = w) of o1/n7digit.py: from the residue form, conversion to the orbit form (o1/n7win.py),
the correction-phase digit delta (two L-bit two's-complement registers, L = w + 3) against o1/mixrad.py
N7Window.digit, then everything undone. MODE 'value' outputs delta; MODE 'identity' gives the Toffoli count
(conversion + digit register + undo + inverse conversion; the digit register alone = identity - conversion).
usage: python3 harness/vb.py o1/s5/n7digit_check.py --n 2,3,4,5,7 [-p BAR=True] [-p MODE=identity]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..')]
import n7win as nw
import n7digit as nd
import mixrad as mr

MODE = 'value'
BAR = False


def inputs(n):
    return [('xr', 3 + 2 * n)]


def build(c, n, regs):
    xr = regs['xr']
    cr, xa, xb = xr[:3], xr[3:3 + n], xr[3 + n:]
    s, j0, j1, z = nw.conv_fwd(c, cr, xa, xb, n, BAR)
    out = {'xr': xr}
    if MODE != 'conv':
        D1, D2, undo = nd.n7_phase_digit(c, cr, xa, xb, s, n, BAR)
        if MODE == 'value':
            T1, T2 = c.alloc(n + 3), c.alloc(n + 3)
            for i in range(n + 3):
                c.cx(D1[i], T1[i])
                c.cx(D2[i], T2[i])
            out.update(d1=T1, d2=T2)
        undo()
    nw.conv_inv(c, cr, xa, xb, s, n, BAR)
    return out


def spec(n, v):
    win = mr.N7Window(n, BAR)
    M, L = 1 << n, n + 3
    cc, a, b = v['xr'] & 7, (v['xr'] >> 3) & (M - 1), v['xr'] >> (3 + n)
    k = (cc - win.phi7((a, b))) * win.inv2w % 7
    d1, d2 = win.digit((a + (k << n), b))
    out = {'xr': v['xr']}
    if MODE == 'value':
        out.update(d1=d1 % (1 << L), d2=d2 % (1 << L))
    return out


def sample(n, rng):
    M = 1 << n
    return {'xr': rng.randrange(7) | (rng.randrange(M) << 3) | (rng.randrange(M) << (3 + n))}


def exceptional(n):
    M = 1 << n
    return [{'xr': cc | (a << 3) | (b << (3 + n))} for cc in range(7) for a, b in ((0, 0), (M >> 1, 0), (0, M >> 1),
                                                                                  (M - 1, M - 1), (1, 0), (0, 1))]
