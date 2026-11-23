"""Window-register pieces of the norm-7 x 4^w window.

Residue form (2w + 3 qubits): c = (c0, c1, c2) in [0, 7), x = (xa, xb) in (Z / 2^w)^2.
Orbit form (2w + 4 qubits): z = c0 (1 iff c = 0), the unit (s, j0, j1) (j0, j1 in the c1 / c2 qubits), x' in xa, xb:
  c != 0: u = (-1)^s w^j is the unit with phi7(u) = c and x' = u^-1 x (free, 4^w values);
c -> (z, j, s) for c != 0: the images of the units in F_7 are {1, 2, 4} (s = 0) and {6, 5, 3} (s = 1), so
s = MAJ(c0, c1, c2), complementing c when s = 1 leaves c one-hot, and c0 ^= NOR(c1, c2) turns c0 into z (3 Toffoli).
For q = 3 + w (w -> 4 in F_7) the one-hot is (c0: j = 0, c2: j = 1, c1: j = 2); for 2 - w (w -> 2) c1 and c2 swap.
The hex part runs controlled on z by starting the valuation chains of _j_bits / _s_bit at z instead of 1.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'h1'), HERE]
import vbarith as va           # noqa: E402
import hexwin_add as hw        # noqa: E402
import point_add as pa         # noqa: E402

_and_mbu = hw._and_mbu
_wrap_neg_undo = hw._wrap_neg_undo


def _j_bits_z(c, xa, xb, j0, j1, Wd, z0):
    """hexwin_add._j_bits with every XOR into (j0, j1) also controlled by z0"""
    undo = []
    z = z0
    for t in range(Wd):
        a, b = xa[t], xb[t]
        g, ug = _and_mbu(c, b, a, negb=True)
        h, uh = _and_mbu(c, a, b)
        undo += [ug, uh]
        c.ccx(z, g, j0)
        c.ccx(z, h, j1)
        if t < Wd - 1:
            c.x(a)
            c.x(b)
            nz, un = _and_mbu(c, a, b)
            c.x(a)
            c.x(b)
            undo.append(_wrap_neg_undo(c, un, (a, b)))
            z2, uz = _and_mbu(c, z, nz)
            undo.append(uz)
            z = z2
    for u in reversed(undo):
        u()


def _s_bit_z(c, xa, s, Wd, z0):
    """hexwin_add._s_bit with the XOR into s also controlled by z0"""
    undo = []
    z = z0
    for t in range(Wd - 1):
        k, uk = _and_mbu(c, xa[t], xa[t + 1])
        undo.append(uk)
        c.ccx(z, k, s)
        if t < Wd - 2:
            z2, uz = _and_mbu(c, z, xa[t], negb=True)
            undo.append(uz)
            z = z2
    for u in reversed(undo):
        u()


def jbits(cr, bar):
    """(z, j0, j1) qubits inside the c register"""
    return (cr[0], cr[1], cr[2]) if bar else (cr[0], cr[2], cr[1])


def _c_to_unit(c, cr, s):
    c.ccx(cr[0], cr[1], s)
    c.cx(cr[0], cr[1])
    c.ccx(cr[1], cr[2], s)
    c.cx(cr[0], cr[1])
    for q in cr:
        c.cx(s, q)
    c.x(cr[1])
    c.x(cr[2])
    c.ccx(cr[1], cr[2], cr[0])
    c.x(cr[1])
    c.x(cr[2])


def _c_to_unit_inv(c, cr, s):
    c.x(cr[1])
    c.x(cr[2])
    c.ccx(cr[1], cr[2], cr[0])
    c.x(cr[1])
    c.x(cr[2])
    for q in cr:
        c.cx(s, q)
    c.cx(cr[0], cr[1])
    c.ccx(cr[1], cr[2], s)
    c.cx(cr[0], cr[1])
    c.ccx(cr[0], cr[1], s)


def conv_fwd(c, cr, xa, xb, Wd, bar=False):
    """residue form -> orbit form; returns (s, j0, j1, z)"""
    s = c.alloc(1)[0]
    _c_to_unit(c, cr, s)
    z, j0, j1 = jbits(cr, bar)
    _j_bits_z(c, xa, xb, j0, j1, Wd, z)
    hw._csub(c, j0, xa, xb)
    va.cswap(c, j0, xa, xb)
    hw._cneg(c, j0, xb)
    hw._csub(c, j1, xb, xa)
    va.cswap(c, j1, xa, xb)
    hw._cneg(c, j1, xa)
    _s_bit_z(c, xa, s, Wd, z)
    hw._cneg(c, s, xa)
    hw._cneg(c, s, xb)
    return s, j0, j1, z


def conv_inv(c, cr, xa, xb, s, Wd, bar=False):
    z, j0, j1 = jbits(cr, bar)
    hw._cneg(c, s, xb)
    hw._cneg(c, s, xa)
    _s_bit_z(c, xa, s, Wd, z)
    hw._cneg(c, j1, xa)
    va.cswap(c, j1, xa, xb)
    hw._cadd(c, j1, xb, xa)
    hw._cneg(c, j0, xb)
    va.cswap(c, j0, xa, xb)
    hw._cadd(c, j0, xa, xb)
    _j_bits_z(c, xa, xb, j0, j1, Wd, z)
    _c_to_unit_inv(c, cr, s)
    c.free([s])


# ------------------------------------------------------------------ trees over orbit representatives
def _full(qs, key, val=0, i=0):
    """complete binary tree over the qubits qs (LSB first), leaves key(value)"""
    if i == len(qs):
        return ('leaf', key(val))
    return ('br', qs[i], _full(qs, key, val, i + 1), _full(qs, key, val | (1 << i), i + 1))


def n7_tree(z, xa, xb, Wd):
    """leaves (zf, (a, b)): z = 0: all (a, b); z = 1: caterpillar tree of hex-canonical residues"""
    qs = list(xa) + list(xb)
    M = 1 << Wd
    t0 = _full(qs, lambda v: (0, (v & (M - 1), v >> Wd)))

    def relabel(node):
        if node[0] == 'leaf':
            return ('leaf', (1, node[1]))
        return ('br', node[1], relabel(node[2]), relabel(node[3]))
    t1 = relabel(hw.cat_tree(Wd, xa, xb))
    return ('br', z, t0, t1)


def ss_tree_load(c, z, xa, xb, Wd, out2, pay):
    """two-way select-swap on xb[W-1] (_ss_tree_load extended to the z = 0 class): the unary iteration skips
    xb[W-1], every z = 0 leaf and every hex class v <= W - 2 loads both partners (slot = xb[W-1]); the hex (M/2, 0)
    and (0, 0) leaves load slot 0 only. Then one controlled swap on xb[W-1]."""
    M = 1 << Wd
    top = Wd - 1
    qs = list(xa) + list(xb[:top])
    t0 = _full(qs, lambda v: ('pair', 0, v & (M - 1), v >> Wd))

    def full(v):
        bits = [('a', t) for t in range(v + 2, Wd)] + [('b', t) for t in range(v + 1, Wd - 1)]

        def rec(i, a, b):
            if i == len(bits):
                return ('leaf', ('pair', 1, a, b))
            kd, t = bits[i]
            if kd == 'a':
                return ('br', xa[t], rec(i + 1, a, b), rec(i + 1, a | (1 << t), b))
            return ('br', xb[t], rec(i + 1, a, b), rec(i + 1, a, b | (1 << t)))
        return rec(0, 1 << v, 0)

    def spine(v):
        if v == Wd:
            return ('leaf', ('one', 1, 0, 0))
        if v == Wd - 1:
            return ('br', xa[Wd - 1], spine(Wd), ('leaf', ('one', 1, M >> 1, 0)))
        return ('br', xa[v], spine(v + 1), full(v))
    tree = ('br', z, t0, spine(0))
    s0, s1 = out2

    def put(f, reg, v):
        b = 0
        while v:
            if v & 1:
                c.cx(f, reg[b])
            v >>= 1
            b += 1

    def leaf(f, key):
        kind, zf, a, b = key
        put(f, s0, pay[(zf, (a, b))])
        if kind == 'pair':
            put(f, s1, pay[(zf, (a, b | (1 << top)))])
    hw.iter_tree(c, None, tree, leaf)
    va.cswap(c, xb[top], s0, s1)


def partner(kk, Wd, ys):
    """the value loaded into the junk slot of ss_tree_load for the window value kk"""
    zf, (a, b) = kk
    M = 1 << Wd
    if zf == 1 and b == 0 and (a == M >> 1 or a == 0):
        return 0
    return ys[(zf, (a, b ^ (1 << (Wd - 1))))]


def phase_oracle(c, z, xa, xb, s, j0, j1, Wd, F):
    """(-1)^F(kk, s, j) for every valid orbit form: one-hot over xb (2^W qubits), unary iteration over
    (z, s, j, xa) (all xa for z = 0; the hex classes of phase_oracle for z = 1)."""
    M = 1 << Wd
    onehot = c.alloc(M)
    pa._iterate(c, None, list(xb)[::-1], lambda f, i: c.cx(f, onehot[i]))

    def a0_tree(tag):
        return _full(list(xa), lambda v: (tag, v, None))

    def a_tree(tag, v):
        if v == Wd:
            return ('leaf', (tag, 0, Wd))
        if v == Wd - 1:
            return ('br', xa[Wd - 1], a_tree(tag, Wd), ('leaf', (tag, M >> 1, Wd - 1)))
        return ('br', xa[v], a_tree(tag, v + 1), a_full(tag, v))

    def a_full(tag, v):
        bits = list(range(v + 2, Wd))

        def rec(i, a):
            if i == len(bits):
                return ('leaf', (tag, a, v))
            t = bits[i]
            return ('br', xa[t], rec(i + 1, a), rec(i + 1, a | (1 << t)))
        return rec(0, 1 << v)

    def sj(zf, sv):
        mk = (lambda tag: a0_tree(tag)) if zf == 0 else (lambda tag: a_tree(tag, 0))
        return ('br', j0, ('br', j1, mk((zf, sv, 0)), mk((zf, sv, 2))), mk((zf, sv, 1)))
    tree = ('br', z, ('br', s, sj(0, 0), sj(0, 1)), ('br', s, sj(1, 0), sj(1, 1)))
    l = Wd - 1

    def leaf(f, key):
        (zf, sv, jj), a, v = key
        if zf == 0:
            cands = range(M)
        elif v == Wd:
            cands = [0] if (sv, jj) == (0, 0) else []
        elif v == Wd - 1:
            cands = [0] if sv == 0 else []
        else:
            cands = [2 * i for i in range(0, 1 << l, 1 << v)]
        for b in cands:
            g = F((zf, (a, b)), sv, jj)
            if g:
                c.cz(f, onehot[b], cond=g)
    hw.iter_tree(c, None, tree, leaf)
    pa._iterate(c, None, list(xb)[::-1], lambda f, i: c.cx(f, onehot[i]))
    c.free(onehot)
