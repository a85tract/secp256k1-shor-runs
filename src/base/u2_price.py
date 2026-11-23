"""Baseline U2, pricing probe: U1's single unsplit run with the stacks' unit-orbit windows.

construction in harness/mixreg.py): a (on G) of log2 norm >= 256 and b (on -Q) of log2 norm >= 208, the register
lengths of U1 (base/u1_shor.py: a = 256, b = 208 bits, Haener et al.'s 48-bit-short b). Each register's top window is
a 4-bit sine box top ('s4'); the two tops (8 + 8 qubits) form the run's one 16-bit first lookup (o1/first_ss.py,
base/first_ss_a16_p512.json). The other windows are the kinds of the stack the variant is compared with, mixed to
the fewest windows that reach the two lengths; b's lowest window is the run's last addition (the terminal):
               bits); a = n7, m7 x 9, h8 x 5, s4 (256.07 bits); 27 additions
  VARIANT 'n'  (claim B' kinds): the same with a regular W = 9 window as terminal ('w9')
  VARIANT 'h'  (claim R3 kinds, hex only): b = h9 (regular 'w9' terminal), h9 x 3, h8 x 8, s4 (208 bits);
               a = h9 x 4, h8 x 11, s4 (256 bits); 27 additions
No decoding here (solve returns None, the verdict is FAIL; only the cost report matters). The single-run decoder
and its classical work are base/u2_decode.py. The tables of b are built from Q, so the harness would grade any
approximate run of this shape with its heuristic garbage-run model, as for U1.
"""
VARIANT = 'nt'
LAYOUTS = {
    'nt': ('n7,m7x9,h8x5,s4', ['n7'] + ['m7'] * 9 + ['w8'] * 5 + ['first'],
           'h9,n7,m7x7,h8x3,s4', ['term', 'n7'] + ['m7'] * 7 + ['w8'] * 3 + ['first']),
    'n': ('n7,m7x9,h8x5,s4', ['n7'] + ['m7'] * 9 + ['w8'] * 5 + ['first'],
          'h9,n7,m7x7,h8x3,s4', ['w9', 'n7'] + ['m7'] * 7 + ['w8'] * 3 + ['first']),
    'h': ('h9x4,h8x11,s4', ['w9'] * 4 + ['w8'] * 11 + ['first'],
          'h9,h9x3,h8x8,s4', ['w9'] * 4 + ['w8'] * 8 + ['first']),
}


def budget(m):
    return 1


def solve(h):
    if h.r.bit_length() < 256:
        return None                       # priced at n = 256 only
    la, ka, lb, kb = LAYOUTS[VARIANT]
    run = h.run()
    run.offset()
    a = run.mixreg(la)
    b = run.mixreg(lb)
    run.hexadd(a, h.G, ka)
    run.hexadd(b, -h.Q, kb)               # b's windows last: its lowest window is the run's terminal
    run.measure()
    return None
