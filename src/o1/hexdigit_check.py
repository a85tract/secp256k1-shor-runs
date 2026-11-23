"""V-B check (own spec, n = W) of the window register pieces of o1/hexwin_add.py: conversion x -> (x'', s, j) and
the correction-phase digit register delta = (-1)^s w^j lift(x''). MODE 'value': outputs delta (two L-bit two's
complement registers, L = W + 3) next to the restored residue, checked against hexdigits.digit for every input
(exhaustive up to W = 8). MODE 'identity': conversion, delta computed and uncomputed, inverse conversion (what a
window spends around its addition, minus the addition): output = input, no garbage; gives the Toffoli count."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hexwin_add as hw
from hexdigits import digit

MODE = 'value'
ALL_INPUTS_VALID = True


def inputs(n):
    return [('xr', 2 * n)]


def build(c, n, regs):
    xr = regs['xr']
    xa, xb = xr[:n], xr[n:]
    s, j0, j1 = hw.conv_fwd(c, xa, xb, n)
    if MODE == 'conv':
        hw.conv_inv(c, xa, xb, s, j0, j1, n)
        return {'xr': xr}
    D1, D2, undo = hw.phase_digit(c, xa, xb, s, j0, j1, n)
    if MODE == 'value':
        T1, T2 = c.alloc(n + 3), c.alloc(n + 3)
        for i in range(n + 3):
            c.cx(D1[i], T1[i])
            c.cx(D2[i], T2[i])
        undo()
        hw.conv_inv(c, xa, xb, s, j0, j1, n)
        return {'xr': xr, 'd1': T1, 'd2': T2}
    undo()
    hw.conv_inv(c, xa, xb, s, j0, j1, n)
    return {'xr': xr}


def spec(n, v):
    M, L = 1 << n, n + 3
    a, b = v['xr'] & (M - 1), v['xr'] >> n
    d1, d2 = digit(a, b, n)
    out = {'xr': v['xr']}
    if MODE == 'value':
        out.update(d1=d1 % (1 << L), d2=d2 % (1 << L))
    return out


def sample(n, rng):
    return {'xr': rng.randrange(1 << (2 * n))}
