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
O2HEX = os.path.join(REPO, 'o2', 'hex')
sys.path[:0] = [os.path.join(REPO, 'o2', 'vb'), os.path.join(REPO, 'h1'), os.path.join(REPO, 'o1'), os.path.join(REPO, 'o2', 'port'), O2HEX, HERE]
import vbarith as va                 # h1
import hexwin_add as hw              # o1: conv_fwd / conv_inv, cat_tree, qload_tree, select_mcarry, phase_oracle
import qroam                    # o2: unary iteration
import windowed_lead as wl           # o2: stream access, op classes
import sitemap as smap               # o2: op -> (value, bit, const)

KIND = 'hex_terminal_add'
W = 9
EXACT = False
YSEL = 2
END = None          # probe: stop after this stream op, no tail
STOP = 6486360      # end of the stream's 'square' phase (HEO_PHASE ops=6486360): the terminal replays [0, STOP)
PAY = 32            # payload bits of the lambda_+ read / equality test (false match of the other root: 2^-PAY)
CHK = 16            # extra payload bits: low CHK bits of lambda_- (validity check: exactly one root must match)
DENSE = False       # test only: the read's table entries are evaluated for every lane at every leaf (not only at the lane's own digit)
TAMPER = None       # test only: 'x' flips a bit of x2 before its measurement, 'y' a low bit of lambda (garbage injection)
T_SS = 4            # select-swap: lambda = 2^T_SS slots
POINT = None
HOLD = 'peak'
OFF = smap.OFF
STATS = {}
_cache = {}


def window(n):
    return W


def declared_failure(n):
    """failure rate per terminal on random inputs, flagged or caught by the root test (terminal part of the stream only:
    the multiplication's walk is not run). Classical replays of the prefix [0, STOP) on 6 x 65,536 random lanes
    (s4/results/term/clsim_prefix_s1*.json, clph_prefix_s2*.json): 429 wrong outputs (lambda or x2 wrong at the end of the
    square; 1.09e-3, 95 % upper 1.36e-3), 22 of 196,608 phase-only failures (right output, flipped phase; 1.1e-4, 95 %
    upper 1.6e-4), root-test false flag 2^-16 = 1.5e-5: 1.36e-3 + 1.6e-4 + 1.5e-5 -> 1.6e-3 (whole leader: 2.75e-3)."""
    return 1.6e-3


STREAM_CAP = 379100    # executed CCX / CCZ of the stream prefix [0, STOP): mean 377,226, sd 404, max 378,849 over 25,600
                       # sampled lanes (s4/results/term/execcount_prefix_s400.json): cap = mean + 4.6 sd, 0 of 25,600 over


def abort_cap(n):
    """hard cap on the executed Toffoli of the terminal: every non-stream Toffoli (all unconditional) plus STREAM_CAP"""
    return int(STATS['toff_nonstream']) + STREAM_CAP


def _sites():
    if 'sites' in _cache:
        return _cache['sites']
    import json
    st = wl._stream()
    mp = os.environ.get('O2_SITEMAP', os.path.join(O2HEX, 'sitemap.json'))
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
    xr_vals = c.read(xr)          # lane inputs (simulator access; used only to evaluate classical tables at each lane's own digit)
    s, j0, j1 = hw.conv_fwd(c, xa, xb, Wd)
    held = ['orbit']
    tree = hw.cat_tree(Wd, xa, xb)
    xs = {k: P[0] for k, P in T.items()}
    bx = {k: beta * x % p for k, x in xs.items()}
    b2x = {k: (p - x - bx[k]) % p for k, x in xs.items()}
    ys = {k: P[1] for k, P in T.items()}
    M1 = (1 << n)
    pair1 = {k: xs[k] | (bx[k] << n) for k in xs}
    pair3 = {k: (2 * xs[k] % p) | ((2 * bx[k] % p) << n) for k in xs}   # terminal: 2 x_A (not 3 x_A): x2 ends as -x3
    ypay = dict(ys)
    recs = []

    def sel_vals(kk, jj, mult):
        x, b, b2 = mult * xs[kk] % p, mult * bx[kk] % p, mult * b2x[kk] % p
        return [(x, b2), (b, b2), (b2, x)][jj]

    def xA(kk, jj):
        return sel_vals(kk, jj, 1)[0]

    def a0(kk, jj):
        return sel_vals(kk, jj, 2)[0]

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
    toff_emit = 0         # emitted CCX / CCZ of the replayed stream (the abort cap bounds only their executed count)
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

    nops = STOP if END is None else END
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
                        Dx = read_pair(pair3, 2)
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
                    toff_emit += 1
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
                toff_emit += 1
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
                toff_emit += 1
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
    if END is not None:
        STATS['wm'] = wm
        STATS['held'] = held[0]
        STATS['stack'] = len(stack)
        STATS['S'] = S
        return {'x': [wm[i] for i in range(256)], 'y': [wm[256 + i] for i in range(256)]}
    assert not stack and S is None
    # ---------------------------------------------------------------- terminal tail
    # State at STOP (end of the stream's square phase): x2 register v (256 bits, always odd) with 4 identical copies
    # of a lazy-sign wire f: x2 = v - f 2^256 = r - f p, r = (-x3) mod p, and f = 1 iff r is even (the stream keeps x2
    # odd for the multiplication's binary GCD). (v, f) is therefore a bijective relabelling of x3 = x_J: measuring
    # both is the ideal measurement of x_J. lambda is in the y register, canonical.
    import collections
    point_of = ctx['point']
    MASK = (1 << PAY) - 1
    MASKC = (1 << CHK) - 1
    PW = PAY + CHK
    Lc = c.L
    Xq = [wm[i] for i in range(256)]
    Yq = [wm[256 + i] for i in range(256)]
    FW = (518, 521, 687, 693)
    extra = [i for i in range(512, nq) if wm[i] is not None]
    assert all(i in extra for i in FW), 'f wires not live at STOP'
    xr_in = xr_vals
    if TAMPER == 'x':
        c.x(Xq[100], cond=int('10' * (Lc // 2 + 1), 2) & c.ALL)
    elif TAMPER == 'y':
        c.x(Yq[5], cond=int('10' * (Lc // 2 + 1), 2) & c.ALL)
    # 1. Z-measure x2 and f; (v, f) -> x_J, a classical record per lane; invalid decodes are detected failures
    vals = c.read(Xq)
    fread = [c.read([wm[i]]) for i in FW]
    cc_ = (1 << 256) - p
    xJ, YJ, active, bad = [0] * Lc, [0] * Lc, [], 0
    for l in range(Lc):
        v, fl = vals[l], [fr[l] for fr in fread]
        f = fl[0]
        r = v - cc_ * f
        if len(set(fl)) == 1 and (v & 1) == 1 and 0 < r < p and f == (1 if r % 2 == 0 else 0):
            x3 = (p - r) % p
            rhs = (x3 * x3 * x3 + 7) % p
            y = pow(rhs, (p + 1) // 4, p)
            if y * y % p == rhs and y != 0:
                xJ[l], YJ[l] = x3, min(y, p - y)
                active.append(l)
                continue
        bad |= 1 << l
    c.flag_mask |= bad
    for i, q in enumerate(Xq):
        c.x(q, cond=sum(1 << l for l in range(Lc) if (vals[l] >> i) & 1))
    c.free(Xq)
    for w_, fr in zip(FW, fread):
        c.x(wm[w_], cond=sum(1 << l for l in range(Lc) if fr[l]))
        c.free([wm[w_]])
        wm[w_] = None
    for i in extra:
        if i not in FW:
            c.flag([wm[i]])             # zero in every valid run: a nonzero outcome is a detected failure
            wm[i] = None
    for i in range(256):
        wm[i] = None

    def lam_s(l, k, sgn):
        """lambda for lane l's measured point (xJ, sgn * YJ) and the digit k: (yA + y_J) / (xA - x_J)"""
        T_ = point_of(k)
        if T_ is None:
            return 0
        d = (T_[0] - xJ[l]) % p
        return 0 if d == 0 else (T_[1] + sgn * YJ[l]) * pow(d, -1, p) % p

    def entry(l, k):
        """payload of the read: low PAY bits of lambda_+(k), then low CHK bits of lambda_-(k)"""
        T_ = point_of(k)
        if T_ is None:
            return 0
        d = (T_[0] - xJ[l]) % p
        if d == 0:
            return 0
        di = pow(d, -1, p)
        return ((T_[1] + YJ[l]) * di % p & MASK) | (((T_[1] - YJ[l]) * di % p & MASKC) << PAY)

    # 2. select-swap read of the low PAY bits of lambda_+(k) over the 2W-bit digit, 2^T_SS slots
    LAM = 1 << T_SS
    addr = xa + xb
    low, high = addr[:T_SS], addr[T_SS:]
    slots = [c.alloc(PW) for _ in range(LAM)]
    byq = collections.defaultdict(list)
    for l in active:
        byq[xr_in[l] >> T_SS].append(l)

    def rleaf(f, q):
        masks = {}
        for l in (active if DENSE else byq.get(q, ())):
            for j in range(LAM):
                v, b = entry(l, (q << T_SS) | j), 0
                while v:
                    if v & 1:
                        masks[(j, b)] = masks.get((j, b), 0) | (1 << l)
                    v >>= 1
                    b += 1
        for (j, b), m in masks.items():
            c.cx(f, slots[j][b], cond=m)
    t0 = c.toffoli
    qroam.iterate(c, None, high[::-1], rleaf)
    toff_sites['lam_read_iter'] = c.toffoli - t0
    for u in range(T_SS - 1, -1, -1):
        for j in range(1 << u):
            va.cswap(c, low[u], slots[j], slots[j + (1 << u)])
    toff_sites['lam_read'] = c.toffoli - t0
    perms = {}
    for l in active:
        lo_, lab = xr_in[l] & (LAM - 1), list(range(LAM))
        for u in range(T_SS - 1, -1, -1):
            if (lo_ >> u) & 1:
                for j in range(1 << u):
                    lab[j], lab[j + (1 << u)] = lab[j + (1 << u)], lab[j]
        perms[l] = lab
    # 3. sign test: sg = [low PAY bits of lambda == low PAY bits of lambda_+(k)], sm = [low CHK bits of lambda ==
    #    low CHK bits of lambda_-(k)]; exactly one must hold (else garbage or a double match): a detected failure
    t0 = c.toffoli
    sg, sm = c.alloc(1)[0], c.alloc(1)[0]
    R = slots[0]
    for i in range(PAY):
        c.cx(Yq[i], R[i])
    va.allzero_do(c, R[:PAY], lambda z: c.cx(z, sg))
    for i in range(PAY):
        c.cx(Yq[i], R[i])
    for i in range(CHK):
        c.cx(Yq[i], R[PAY + i])
    va.allzero_do(c, R[PAY:], lambda z: c.cx(z, sm))
    for i in range(CHK):
        c.cx(Yq[i], R[PAY + i])
    toff_sites['sign_test'] = c.toffoli - t0
    sg_vals, sm_vals = c.read([sg]), c.read([sm])
    inval = sum(1 << l for l in range(Lc) if sg_vals[l] == sm_vals[l])
    c.flag_mask |= inval
    active = [l for l in active if not (inval >> l) & 1]
    STATS['sign_counts'] = (sum(sg_vals[l] for l in active), len(active), bin(inval).count('1'))
    c.x(sm, cond=sum(1 << l for l in range(Lc) if sm_vals[l]))     # Z-measured and reset: a classical bit
    c.free([sm])
    # 4. unload the read buffers and lambda by X-measurement; their phases are functions of the digit once
    #    (x_J, s) and the outcomes are known: folded into the single deferred phase oracle below
    msl = [[c.measure_x(q) for q in sl] for sl in slots]
    mY = [c.measure_x(q) for q in Yq]
    for i in range(256):
        wm[256 + i] = None
    gl = {}
    for l in active:
        k = xr_in[l]
        g = 0
        for j in range(LAM):
            v, b = entry(l, ((k >> T_SS) << T_SS) | perms[l][j]), 0
            while v:
                if v & 1:
                    g ^= (msl[j][b] >> l) & 1
                v >>= 1
                b += 1
        v, b = lam_s(l, k, 1 if sg_vals[l] else -1), 0
        while v:
            if v & 1:
                g ^= (mY[b] >> l) & 1
            v >>= 1
            b += 1
        gl[l] = g
    to_orbit()
    kx, kb, ks, kj0, kj1 = c.read(xa), c.read(xb), c.read([s]), c.read([j0]), c.read([j1])
    bucket = {}
    for l in active:
        if gl[l]:
            key = ((kx[l], kb[l]), ks[l], kj0[l] + 2 * kj1[l])
            bucket[key] = bucket.get(key, 0) ^ (1 << l)
    t0 = c.toffoli

    def F(kk, sv, jj):
        g = bucket.get((kk, sv, jj), 0)
        for dot, fn in recs:
            g ^= dot(fn(kk, sv, jj))
        return g
    hw.phase_oracle(c, xa, xb, s, j0, j1, Wd, F)
    toff_sites['phase_oracle'] = c.toffoli - t0
    t0 = c.toffoli
    hw.conv_inv(c, xa, xb, s, j0, j1, Wd)
    toff_sites['conv_inv'] = c.toffoli - t0
    STATS['toff_exec_stream'] = toff_exec
    STATS['toff_nonstream'] = c.toffoli - toff_emit
    STATS['toffoli_executed_mean'] = c.toffoli - toff_skip
    STATS['peak_at'] = (pk[0], pk[1], c.peak)
    STATS['toff_sites'] = toff_sites
    print('S4TERM peak before op', pk, 'final', c.peak, 'toffoli', c.toffoli, 'toff_exec_stream', round(toff_exec, 1),
          'sites', toff_sites, 'cond_meas', STATS.get('cond_meas'), 'active', len(active), 'of', Lc, file=sys.stderr)
    # the classical record of x_J (a measured value) as an output register for the checker: allocated last, when few
    # qubits are live, so it does not touch the peak
    rec_x = c.alloc(256)
    for i, q in enumerate(rec_x):
        c.x(q, cond=sum(1 << l for l in active if (xJ[l] >> i) & 1))
    return {'xJ': rec_x, 'sg': [sg]}


def first_lookup(n, offset=False):
    """the run's first window (FIRST_L-bit direct lookup of the b register), counted with gadgets as in
    o1/hexwin_add.py; recorded in the V-B JSON for o1/hexva.py"""
    return hw.first_lookup(n, offset)
