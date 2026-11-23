"""V-B circuit (KIND hex_window_add, spec o1/vbhexw.py): the ECDSA.fail board leader as a unit-orbit ("hex") window

Arithmetic: the leader's raw builder stream (Layr-Labs/ecdsafail-challenge main 21356d25, the promoted source of
(x'', s, j); the addend is (-1)^s w^j T[x''], i.e. xA = beta^j x_T, yA = (-1)^s y_T.

The stream needs the addend at five sites, and every addend-dependent op is conditioned on ONE bit of one value
(o2/hex/sitemap.py, exact on random and carry-forcing lanes):
  y0, y5   bits of yA        (a 256-bit load into the stream's own wires and its unload)
  x1       bits of xA        (start of the division)
  x3       bits of c3 = (3 xA + OFF) mod p, OFF the squaring offset (end of the division, 'x2 += 3 ox')
  x4       bits of xA + 1 and bits 236..254 of xA (end of the multiplication, 'x2 = ox - x2')
Reads ("hex-compatible sequential payload reads", sized to the stream's live count at each site):
  y sites: y_T read into the stream's wires (unary iteration over the caterpillar tree of canonical residues,
           YSEL = 1, or a two-way select-swap on b''[W-1], YSEL = 2), then negate_nonzero by s.
           (select_mcarry) leaves v(beta^j x_T) in one n-qubit register, the other slot is X-measured at once;
           x3 reads (3 x_T, 3 beta x_T) mod p and adds OFF mod p (forward-only adders, carries X-measured);
           x4 copies xA[236..254] (19 CNOTs) and increments xA in place (forward-only, carries X-measured).
  The stream's X(t)[g] becomes CX(S_b, t) (+ X(t) for a complemented bit), CX(c, t)[g] becomes CCX(S_b, c, t),
  Z(t)[g and m] becomes CZ(S_b, t)[m].
Every payload, slot and carry register is X-measured; ONE deferred phase oracle over (x'', s, j) at the end
as a flag (c.flag), as in o2/port/windowed_lead.py (O2_FLAGS=1).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'h1'), os.path.join(REPO, 'o1'), os.path.join(REPO, 'o2', 'port'), HERE]
import vbarith as va                 # h1
import hexwin_add as hw              # o1: conv_fwd / conv_inv, cat_tree, qload_tree, select_mcarry, phase_oracle
import windowed_lead as wl           # o2: stream access, op classes
import sitemap as smap               # o2: op -> (value, bit, const)

KIND = 'hex_window_add'
W = 9
EXACT = False
YSEL = 1            # y-site read: 1 = unary iteration over the tree, 2 = two-way select-swap on b''[W-1]
HOLD = 'orbit'      # 'residue': convert back to the 2W-qubit residue while the stream runs between sites;
OFF = smap.OFF
STATS = {}
_cache = {}


def window(n):
    return W


def declared_failure(n):
    return wl.declared_failure(n)


def _sites():
    if 'sites' in _cache:
        return _cache['sites']
    import json
    st = wl._stream()
    mp = os.environ.get('O2_SITEMAP', os.path.join(HERE, 'sitemap.json'))
    if os.path.exists(mp):
        raw = json.load(open(mp))
        res = {int(k): (tuple(v) if v else None) for k, v in raw.items()}
    else:
        res = smap.build_map()
    assert all(v is not None for v in res.values()), 'unmapped addend op'
    site_of = st['site_of']
    Tt = st['ops']['t']
    sites = []
    for k, (name, lo, hi, kind, _) in enumerate(wl.SITES):
        ids = sorted(i for i in res if site_of[i] == k)
        s = dict(name=name, kind=kind, ids=ids, first=ids[0], last=ids[-1])
        if kind == 'y':
            assert len(ids) == 512
            ld, ul = ids[:256], ids[256:]
            wlod = {res[i][1]: int(Tt[i]) for i in ld}
            wulo = {res[i][1]: int(Tt[i]) for i in ul}
            assert sorted(wlod) == list(range(256)) and wlod == wulo, 'site %s: not a load / unload pair' % name
            assert all(res[i][2] == 0 for i in ids)
            s.update(load0=ld[0], unload0=ul[0], wires=[wlod[b] for b in range(256)])
        sites.append(s)
    _cache['sites'] = (sites, res)
    return _cache['sites']


def _ss_tree_load(c, Wd, xa, xb, out2, pay):
    """two-way select-swap over the caterpillar tree: the unary iteration skips b''[W-1] (free in every class
    v <= W-2) and loads both partners into the two slots (slot = b''[W-1]); the (M/2, 0) and (0, 0) leaves load
    slot 0 only. Then one controlled swap on b''[W-1] moves the addressed entry into slot 0."""
    M = 1 << Wd
    top = Wd - 1

    def full(v):
        bits = [('a', t) for t in range(v + 2, Wd)] + [('b', t) for t in range(v + 1, Wd - 1)]

        def rec(i, a, b):
            if i == len(bits):
                return ('leaf', ('pair', a, b))
            kd, t = bits[i]
            if kd == 'a':
                return ('br', xa[t], rec(i + 1, a, b), rec(i + 1, a | (1 << t), b))
            return ('br', xb[t], rec(i + 1, a, b), rec(i + 1, a, b | (1 << t)))
        return rec(0, 1 << v, 0)

    def spine(v):
        if v == Wd:
            return ('leaf', ('one', 0, 0))
        if v == Wd - 1:
            return ('br', xa[Wd - 1], spine(Wd), ('leaf', ('one', M >> 1, 0)))
        return ('br', xa[v], spine(v + 1), full(v))
    tree = spine(0)
    s0, s1 = out2

    def put(f, reg, v):
        b = 0
        while v:
            if v & 1:
                c.cx(f, reg[b])
            v >>= 1
            b += 1

    def leaf(f, key):
        kind, a, b = key
        if kind == 'one':
            put(f, s0, pay[(a, b)])
        else:
            put(f, s0, pay[(a, b)])
            put(f, s1, pay[(a, b | (1 << top))])
    hw.iter_tree(c, None, tree, leaf)
    va.cswap(c, xb[top], s0, s1)


def build(c, n, regs, ctx):
    assert n == 256, 'the leader stream is a secp256k1 circuit'
    sites, smap_res = _sites()
    st = wl._stream()
    ops, nq, nb = st['ops'], st['nq'], st['nb']
    site_of = st['site_of']
    p, T, beta = ctx['p'], ctx['table'], ctx['beta']
    X, Y = list(regs['x']), list(regs['y'])
    xr = regs['xr']
    Wd = len(xr) // 2
    assert Wd == W
    xa, xb = xr[:Wd], xr[Wd:]
    s, j0, j1 = hw.conv_fwd(c, xa, xb, Wd)
    held = ['orbit']
    tree = hw.cat_tree(Wd, xa, xb)
    xs = {k: P[0] for k, P in T.items()}
    bx = {k: beta * x % p for k, x in xs.items()}
    b2x = {k: (p - x - bx[k]) % p for k, x in xs.items()}
    ys = {k: P[1] for k, P in T.items()}
    M1 = (1 << n)
    pair1 = {k: xs[k] | (bx[k] << n) for k in xs}
    pair3 = {k: (3 * xs[k] % p) | ((3 * bx[k] % p) << n) for k in xs}
    ypay = dict(ys)
    recs = []

    def sel_vals(kk, jj, mult):
        x, b, b2 = mult * xs[kk] % p, mult * bx[kk] % p, mult * b2x[kk] % p
        return [(x, b2), (b, b2), (b2, x)][jj]

    def xA(kk, jj):
        return sel_vals(kk, jj, 1)[0]

    def a0(kk, jj):
        return sel_vals(kk, jj, 3)[0]

    def to_orbit():
        nonlocal s, j0, j1
        if held[0] == 'residue':
            s, j0, j1 = hw.conv_fwd(c, xa, xb, Wd)
            held[0] = 'orbit'

    def to_residue(after=None):
        # x1 -> x3, and square + multiply, x3 -> x4); the y sites and x4 -> y5 sit below the peak and stay in orbit form.
        if HOLD == 'peak' and after not in ('x1', 'x3'):
            return
        if HOLD in ('residue', 'peak') and held[0] == 'orbit':
            hw.conv_inv(c, xa, xb, s, j0, j1, Wd)
            held[0] = 'residue'

    def read_pair(pay, mult):
        D = c.alloc(2 * n)
        hw.qload_tree(c, tree, D, pay)
        Dx, Db = D[:n], D[n:]
        hw.select_mcarry(c, Dx, Db, j0, j1, p, n, xs, bx, recs, mult)
        recs.append((hw._Dot([c.measure_x(q) for q in Db]), lambda kk, sv, jj, m=mult: sel_vals(kk, jj, m)[1]))
        return Dx

    def cmodadd(D, K, val):
        """D <- (D + K) mod p for D < p (forward-only adders, carries X-measured; val(kk, jj) = D before)"""
        e = c.alloc(1)[0]
        f = c.alloc(1)[0]
        S = list(D) + [e]
        L = n + 1
        hw._fwd_add(c, va.cbits(K, L), S, recs, lambda kk, sv, jj: hw._carries(K, val(kk, jj), L))
        kneg = (1 << L) - p
        hw._fwd_carry_out(c, va.cbits(kneg, L), S, f, recs,
                          lambda kk, sv, jj: hw._carries(kneg, val(kk, jj) + K, L))
        hw._fwd_add(c, [f if (kneg >> i) & 1 else va.K0 for i in range(L)], S, recs,
                    lambda kk, sv, jj: hw._carries(kneg if val(kk, jj) + K >= p else 0, val(kk, jj) + K, L))
        recs.append((hw._Dot([c.measure_x(f)]), lambda kk, sv, jj: 1 if val(kk, jj) + K >= p else 0))
        c.free([e])

    wm = [None] * nq
    for i in range(256):
        wm[i] = X[i]
        wm[256 + i] = Y[i]
    ALL = c.ALL
    L = c.L
    bits = [0] * nb
    stack = []
    base = ALL
    toff_exec = 0.0
    toff_skip = 0.0      # emitted CCX/CCZ of the stream minus their lane-average executed count
    S = None
    pk = [0, -1]
    NO = wl.NO
    MIXED = wl.MIXED
    FLAGS = wl.FLAGS
    toff_sites = {}

    def w(q):
        r = wm[q]
        if r is None:
            r = wm[q] = c.alloc(1)[0]
        return r

    nops = len(ops)
    CH = 1 << 19
    to_residue()
    for a in range(0, nops, CH):
        b = min(nops, a + CH)
        kinds = ops['kind'][a:b].tolist()
        C2 = ops['c2'][a:b].tolist()
        C1 = ops['c1'][a:b].tolist()
        TT = ops['t'][a:b].tolist()
        CT = ops['ct'][a:b].tolist()
        CC = ops['cc'][a:b].tolist()
        for j in range(b - a):
            kd = kinds[j]
            if kd == 1 or kd == 2 or kd == 17:
                continue
            i = a + j
            if c.peak > pk[0]:
                pk[0] = c.peak
                pk[1] = i
            if i in site_of:
                k = site_of[i]
                sd = sites[k]
                t0 = c.toffoli
                if sd['kind'] == 'y':
                    if i == sd['load0']:
                        to_orbit()
                        reg = [w(q) for q in sd['wires']]
                        if YSEL == 2:
                            junk = c.alloc(n)
                            _ss_tree_load(c, Wd, xa, xb, (reg, junk), ypay)
                            M = 1 << Wd

                            def partner(kk):
                                a_, b_ = kk
                                if b_ == 0 and (a_ == M >> 1 or a_ == 0):
                                    return 0
                                return ys[(a_, b_ ^ (1 << (Wd - 1)))]
                            recs.append((hw._Dot([c.measure_x(q) for q in junk]),
                                         lambda kk, sv, jj: partner(kk)))
                        else:
                            hw.qload_tree(c, tree, reg, ypay)
                        va.negate_nonzero(c, reg, p, s)        # yA = (-1)^s y_T
                        to_residue()
                    elif i == sd['unload0']:
                        to_orbit()
                        reg = [wm[q] for q in sd['wires']]
                        va.negate_nonzero(c, reg, p, s)
                        recs.append((hw._Dot([c.measure_x(q) for q in reg]), lambda kk, sv, jj: ys[kk]))
                        for q in sd['wires']:
                            wm[q] = None
                        to_residue()
                    toff_sites[sd['name']] = toff_sites.get(sd['name'], 0) + c.toffoli - t0
                    continue
                if i == sd['first']:
                    to_orbit()
                    nm = sd['name']
                    if nm == 'x1':
                        Dx = read_pair(pair1, 1)
                        S = {'x': Dx}
                    elif nm == 'x3':
                        Dx = read_pair(pair3, 3)
                        cmodadd(Dx, OFF, a0)
                        S = {'c3': Dx}
                    elif nm == 'x4':
                        Dx = read_pair(pair1, 1)
                        hi = c.alloc(19)
                        for t_ in range(19):
                            c.cx(Dx[236 + t_], hi[t_])
                        hw._fwd_add(c, va.cbits(1, n), Dx, recs, lambda kk, sv, jj: hw._carries(1, xA(kk, jj), n))
                        S = {'x+1': Dx, 'x': dict((236 + t_, hi[t_]) for t_ in range(19)), 'hi': hi}
                    else:
                        raise RuntimeError(nm)
                    to_residue(nm)
                vn, bit, cst = smap_res[i]
                g = S[vn][bit]
                if kd == 6:
                    c.cx(g, w(TT[j]))
                    if cst:
                        c.x(w(TT[j]))
                elif kd == 8:
                    c.ccx(g, w(C1[j]), w(TT[j]))
                    toff_exec += 1.0
                    if cst:
                        c.cx(w(C1[j]), w(TT[j]))
                elif kd == 7 and i in MIXED:
                    c.cz(g, w(TT[j]), bits[MIXED[i]] & base)
                    if cst:
                        c.z(w(TT[j]), bits[MIXED[i]] & base)
                else:
                    raise RuntimeError('unexpected addend op kind %d at %d' % (kd, i))
                if i == sd['last']:
                    nm = sd['name']
                    if nm == 'x1':
                        recs.append((hw._Dot([c.measure_x(q) for q in S['x']]), lambda kk, sv, jj: xA(kk, jj)))
                    elif nm == 'x3':
                        recs.append((hw._Dot([c.measure_x(q) for q in S['c3']]),
                                     lambda kk, sv, jj: (a0(kk, jj) + OFF) % p))
                    else:
                        recs.append((hw._Dot([c.measure_x(q) for q in S['x+1']]), lambda kk, sv, jj: xA(kk, jj) + 1))
                        recs.append((hw._Dot([c.measure_x(q) for q in S['hi']]),
                                     lambda kk, sv, jj: (xA(kk, jj) >> 236) & ((1 << 19) - 1)))
                    S = None
                toff_sites[sd['name']] = toff_sites.get(sd['name'], 0) + c.toffoli - t0
                continue
            cc = CC[j]
            cond = base if cc == NO else base & bits[cc]
            if kd == 8:
                c.cx(w(C1[j]), w(TT[j]), cond)
            elif kd == 13:
                c.ccx(w(C1[j]), w(C2[j]), w(TT[j]), cond)
                toff_exec += 1.0 if cond == ALL else bin(cond).count('1') / L
                if cond != ALL:
                    toff_skip += 1.0 - bin(cond).count('1') / L
            elif kd == 6:
                c.x(w(TT[j]), cond)
            elif kd == 9:
                c.cz(w(TT[j]), w(C1[j]), cond)
            elif kd == 15:
                stack.append(base)
                base = cond
            elif kd == 16:
                base = stack.pop()
            elif kd == 10:
                qt, q1 = w(TT[j]), w(C1[j])
                if cond == ALL:
                    c.swap(qt, q1)
                else:
                    c.cx(qt, q1, cond)
                    c.cx(q1, qt, cond)
                    c.cx(qt, q1, cond)
            elif kd == 12 or kd == 11:
                t = TT[j]
                qt = wm[t]
                if qt is None:
                    qt = c.alloc(1)[0]
                fl = kd == 11 and FLAGS
                if cond == ALL:
                    if fl:
                        c.flag([qt])
                    else:
                        m = c.measure_x(qt)
                    wm[t] = None
                else:
                    anc = c.alloc(1)[0]
                    c.cx(qt, anc, cond)
                    c.cx(anc, qt, cond)
                    if fl:
                        c.flag([anc])
                    else:
                        m = c.measure_x(anc)
                    STATS['cond_meas'] = STATS.get('cond_meas', 0) + 1
                    wm[t] = qt
                if kd == 12:
                    ct = CT[j]
                    bits[ct] = (bits[ct] & ~cond & ALL) | (m & cond)
            elif kd == 7:
                c.z(w(TT[j]), cond)
            elif kd == 14:
                c.ccz(w(TT[j]), w(C1[j]), w(C2[j]), cond)
                toff_exec += 1.0 if cond == ALL else bin(cond).count('1') / L
                if cond != ALL:
                    toff_skip += 1.0 - bin(cond).count('1') / L
            elif kd == 3:
                bits[CT[j]] ^= cond
            elif kd == 4:
                bits[CT[j]] &= ~cond & ALL
            elif kd == 5:
                bits[CT[j]] |= cond
            elif kd == 0:
                c.global_phase(cond)
            else:
                raise RuntimeError('unknown op kind %d' % kd)
    assert not stack and S is None
    to_orbit()
    t0 = c.toffoli

    def F(kk, sv, jj):
        g = 0
        for dot, fn in recs:
            g ^= dot(fn(kk, sv, jj))
        return g
    hw.phase_oracle(c, xa, xb, s, j0, j1, Wd, F)
    toff_sites['phase_oracle'] = c.toffoli - t0
    t0 = c.toffoli
    hw.conv_inv(c, xa, xb, s, j0, j1, Wd)
    toff_sites['conv_inv'] = c.toffoli - t0
    STATS['toff_exec_stream'] = toff_exec
    STATS['toffoli_executed_mean'] = c.toffoli - toff_skip
    STATS['peak_at'] = (pk[0], pk[1], c.peak)
    STATS['toff_sites'] = toff_sites
    print('O2HEX peak before op', pk, 'final', c.peak, 'toffoli', c.toffoli, 'toff_exec_stream', round(toff_exec, 1), 'toffoli_executed_mean', round(c.toffoli - toff_skip, 1), 'sites', toff_sites,
          'cond_meas', STATS.get('cond_meas'), file=sys.stderr)
    return {'x': [wm[i] for i in range(256)], 'y': [wm[256 + i] for i in range(256)]}


def first_lookup(n, offset=False):
    """the run's first window (FIRST_L-bit direct lookup of the b register), counted with gadgets as in
    o1/hexwin_add.py; recorded in the V-B JSON for o1/hexva.py"""
    return hw.first_lookup(n, offset)
