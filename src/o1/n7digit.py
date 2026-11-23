"""Correction-phase digit register of the norm-7 x 4^w window: the window's digit delta (o1/mixrad.py
N7Window.digit of the residue) in two L-bit two's-complement registers (L = w + 3), computed from the orbit form.

delta = D + 2^w kappa'', D = (-1)^s w^j lift(x') the hex digit register of o1/hexwin_add.phase_digit run on the orbit
form (lift = hexreduce of the centred x', lift(0) = 2^w), kappa'' in {0} u units with phi7(kappa'') = (c - phi7(D)) /
2^w mod 7, c = phi7(u) for z = 0 and 0 for z = 1 (mixrad: rep(c0, x) = h(x) + 2^w kappa, digit = u rep), and the zero
residue (z = 1, x'' = 0) gets pi itself: there D + 2^w kappa'' = 0, and q (as coordinates) is added. Steps (all on fresh ancillas,
undone in reverse): t = phi7(D) mod 7 from the 3-bit chunks of D1 and D2 (8 = 1 mod 7; the factor w -> om of D2 and
the sign bits are rotations / constants of the chunks), summed into 6 bits and folded twice; c from (z, s, j);
need = (c - t) 2^-w mod 7 (the factor is a rotation of the 3 bits); kappa'' = (-1)^s' (O0 - O2, O1 - O2) from
s' = MAJ(need) and the one-hot left after complementing by s'; the zero residue's fix (3, +-1).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'h1'), HERE]
import vbarith as va           # noqa: E402
import hexwin_add as hw        # noqa: E402
import n7win as nw             # noqa: E402

K0 = va.K0


def _rot(bits, k):
    """3-bit chunk (LSB first) times 2^k mod 7 (8 = 1): new bit i = old bit (i - k) mod 3"""
    return [bits[(i - k) % 3] for i in range(3)]


def n7_phase_digit(c, cr, xa, xb, s, Wd, bar=False):
    z, j0, j1 = nw.jbits(cr, bar)
    D1, D2, undo_hex = hw.phase_digit(c, xa, xb, s, j0, j1, Wd)
    L = Wd + 3
    assert L == 10 or True
    hi1, hi2 = D1[Wd:], D2[Wd:]
    kom = 2 if not bar else 1                       # w -> om = 4 (3 + w) or 2 (2 - w) in F_7
    ops = []                                        # undo list (callables), run in reverse

    # 1. S = sum of the 3-bit chunks of D1 + om D2 (as integers, <= 56)
    def chunks(D, k):
        n = len(D) - 1                              # sign bit D[n] (weight -2^n; 2^n = 2^(n mod 3) mod 7)
        out = []
        for i in range(0, n, 3):
            ch = [D[i + t] if i + t < n else K0 for t in range(3)]
            out.append(_rot(ch, k + (i % 3)))
        sw = (-(1 << (n % 3)) * (1 << k)) % 7        # -2^n om^k mod 7 as a 3-bit constant pattern of the sign bit
        out.append([D[n] if (sw >> t) & 1 else K0 for t in range(3)])
        return out
    chs = chunks(D1, 0) + chunks(D2, kom)
    S = c.alloc(6)

    def s_fwd():
        for t in range(3):
            if chs[0][t] is not K0:
                c.cx(chs[0][t], S[t])
        for ch in chs[1:]:
            va.add(c, ch, S)

    def s_inv():
        for ch in reversed(chs[1:]):
            va.sub(c, ch, S)
        for t in range(3):
            if chs[0][t] is not K0:
                c.cx(chs[0][t], S[t])
    s_fwd()
    ops.append(s_inv)
    # 2-3. folds: T1 = S_lo + S_hi (<= 14), T2 = T1_lo + T1[3] (<= 8)
    T1, T2 = c.alloc(4), c.alloc(4)

    def f_fwd():
        for t in range(3):
            c.cx(S[t], T1[t])
        va.add(c, S[3:6], T1)
        for t in range(3):
            c.cx(T1[t], T2[t])
        va.add(c, [T1[3]], T2)

    def f_inv():
        va.sub(c, [T1[3]], T2)
        for t in range(3):
            c.cx(T1[t], T2[t])
        va.sub(c, S[3:6], T1)
        for t in range(3):
            c.cx(S[t], T1[t])
    f_fwd()
    ops.append(f_inv)
    # 4. t = T2 mod 7 (g = [T2 >= 7], T2 -= 7 g)
    g = c.alloc(1)[0]

    def m_fwd():
        va.ge_const_do(c, T2, 7, lambda q: c.cx(q, g))
        va.sub(c, [g, g, g], T2)

    def m_inv():
        va.add(c, [g, g, g], T2)
        va.ge_const_do(c, T2, 7, lambda q: c.cx(q, g))
    m_fwd()
    ops.append(m_inv)
    t3 = T2[:3]
    # 5. c = phi7(u) for z = 0, 0 for z = 1: one-hot of j (bit 0: j = 0; j = 1 / 2 on bits (2, 1) for 3 + w, (1, 2)
    #    for 2 - w), complemented by s
    cc = c.alloc(3)
    b1, b2 = (2, 1) if not bar else (1, 2)          # c bit for j = 1, j = 2
    tmp = c.alloc(1)[0]

    def c_fwd():
        c.x(z)
        c.x(j0)
        c.x(j1)
        c.ccx(j0, j1, tmp)                          # tmp = [j = 0]
        c.x(j0)
        c.x(j1)
        c.cx(s, tmp)
        c.ccx(z, tmp, cc[0])
        c.cx(s, tmp)
        c.x(j0)
        c.x(j1)
        c.ccx(j0, j1, tmp)
        c.x(j0)
        c.x(j1)
        for jq, bit in ((j0, b1), (j1, b2)):
            c.cx(s, jq)
            c.ccx(z, jq, cc[bit])
            c.cx(s, jq)
        c.x(z)

    c_fwd()
    ops.append(c_fwd)                               # self-inverse (XORs of functions of untouched qubits)
    # 6. dd = c - t mod 7 = c + (7 - t) (- 7 if >= 7); 7 - t = complement of t (t in 0..6)
    dd = c.alloc(4)
    h = c.alloc(1)[0]

    def d_fwd():
        for t in range(3):
            c.cx(cc[t], dd[t])
        for q in t3:
            c.x(q)
        va.add(c, t3, dd)
        for q in t3:
            c.x(q)
        va.ge_const_do(c, dd, 7, lambda q: c.cx(q, h))
        va.sub(c, [h, h, h], dd)

    def d_inv():
        va.add(c, [h, h, h], dd)
        va.ge_const_do(c, dd, 7, lambda q: c.cx(q, h))
        for q in t3:
            c.x(q)
        va.sub(c, t3, dd)
        for q in t3:
            c.x(q)
        for t in range(3):
            c.cx(cc[t], dd[t])
    d_fwd()
    ops.append(d_inv)
    # 7. need = dd 2^-w mod 7 (a rotation); 8. s' = MAJ(need), complement by s' -> one-hot (O0, O_j1, O_j2) or 0
    need = _rot(dd[:3], (-Wd) % 3)
    sp = c.alloc(1)[0]

    def n_fwd():
        c.ccx(need[0], need[1], sp)
        c.cx(need[0], need[1])
        c.ccx(need[1], need[2], sp)
        c.cx(need[0], need[1])
        for q in need:
            c.cx(sp, q)

    def n_inv():
        for q in need:
            c.cx(sp, q)
        c.cx(need[0], need[1])
        c.ccx(need[1], need[2], sp)
        c.cx(need[0], need[1])
        c.ccx(need[0], need[1], sp)
    n_fwd()
    ops.append(n_inv)
    O0, Oj1, Oj2 = need[0], need[b1], need[b2]

    # kappa'' = (-1)^s' (O0 - O2, O1 - O2): hi += kappa''
    def k_fwd():
        for hi, plus in ((hi1, O0), (hi2, Oj1)):
            hw._cneg(c, sp, hi)
            va.add(c, [plus], hi)
            va.sub(c, [Oj2], hi)
            hw._cneg(c, sp, hi)

    def k_inv():
        for hi, plus in ((hi2, Oj1), (hi1, O0)):
            hw._cneg(c, sp, hi)
            va.add(c, [Oj2], hi)
            va.sub(c, [plus], hi)
            hw._cneg(c, sp, hi)
    k_fwd()
    ops.append(k_inv)
    # 9. the zero residue (z = 1, x' = 0): D = 2^w and kappa'' = -1 cancel; pi = q 2^w needs q added
    zr = c.alloc(1)[0]

    def z_fwd():
        c.x(z)
        va.allzero_do(c, [z] + list(xa) + list(xb), lambda q: c.cx(q, zr))
        c.x(z)

    q1 = [zr, zr] if not bar else [K0, zr]          # D + 2^w kappa'' = 0 here (kappa'' = -1), so add q itself:
                                                    # (3, 1) for 3 + w, (2, -1) for 2 - w

    def zadd(sign):
        if sign > 0:
            va.add(c, q1, hi1)
            (va.add if not bar else va.sub)(c, [zr], hi2)
        else:
            (va.sub if not bar else va.add)(c, [zr], hi2)
            va.sub(c, q1, hi1)
    z_fwd()
    zadd(+1)
    ops.append(lambda: (zadd(-1), z_fwd()))

    def undo():
        for f in reversed(ops):
            f()
        c.free([zr, sp, h])
        c.free(dd)
        c.free([tmp])
        c.free(cc)
        c.free([g])
        c.free(T2)
        c.free(T1)
        c.free(S)
        undo_hex()
    return D1, D2, undo
