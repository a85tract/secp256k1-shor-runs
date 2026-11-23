"""V-B circuit (KIND n7_window_add, spec o1/vbn7w.py): the ECDSA.fail board leader as a norm-7 x 4^W window
with the window register of o1/n7win.py: residue (c, x) in F_7 x (Z / 2^W)^2 (2W + 3 qubits, held over the stream's
peak stretches with HOLD = 'peak'), orbit form (z, s, j, x') at the five read sites (2W + 4 qubits), reads by unary
iteration over the 4^W + (4^W - 4) / 6 + 2 orbit representatives (o1/n7win.n7_tree; y sites: two-way select-swap on
Run: python3 -B o1/vbx2n7.py o1/n7lead.py -p W=7 -p HOLD=peak --n 256 --lanes 1024 --batches 1 --json F
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path[:0] = [os.path.join(REPO, 'h1'), HERE, os.path.join(REPO, 'o2', 'port'), os.path.join(REPO, 'o2', 'hex')]
import vbarith as va                 # noqa: E402  h1
import hexwin_add as hw              # noqa: E402  o1
import n7win as nw
import windowed_lead as wl           # noqa: E402  o2
import hexlead as hl                 # noqa: E402  o2: _sites, sitemap

KIND = 'n7_window_add'
W = 7
EXACT = False
YSEL = 2
HOLD = 'peak'
MASKED = False
                    # classically conditioned measurements, instead of the borrowed-ancilla emulation (peak - 1)
OFF = hl.OFF
STATS = {}
_sites = hl._sites
smap = hl.smap


def window(n):
    return W


STREAM_CAP = 726000


def abort_cap(n):
    """h1/hexcap/hexlead_cap.py's cap: the window's non-stream Toffoli (unconditional) + STREAM_CAP on the stream's
    executed CCX + CCZ. Used only by o1/vbx2capn7.py (capped checker)."""
    if STATS.get('n') != n:
        raise RuntimeError('abort_cap(%d) before a build at that width' % n)
    return STATS['toffoli'] - STATS['stream_toffoli'] + STREAM_CAP


def declared_failure(n):
    return wl.declared_failure(n)


def build(c, n, regs, ctx):
    assert n == 256, 'the leader stream is a secp256k1 circuit'
    sites, smap_res = _sites()
    st = wl._stream()
    ops, nq, nb = st['ops'], st['nq'], st['nb']
    site_of = st['site_of']
    p, T, beta = ctx['p'], ctx['table'], ctx['beta']
    X, Y = list(regs['x']), list(regs['y'])
    xr = regs['xr']
    Wd = (len(xr) - 3) // 2
    assert Wd == W
    bar = ctx.get('bar', False)
    cr, xa, xb = xr[:3], xr[3:3 + Wd], xr[3 + Wd:]
    s, j0, j1, z = nw.conv_fwd(c, cr, xa, xb, Wd, bar)
    held = ['orbit']
    tree = nw.n7_tree(z, xa, xb, Wd)
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
        nonlocal s, j0, j1, z
        if held[0] == 'residue':
            s, j0, j1, z = nw.conv_fwd(c, cr, xa, xb, Wd, bar)
            held[0] = 'orbit'

    def to_residue(after=None):
        # x1 -> x3, and square + multiply, x3 -> x4); the y sites and x4 -> y5 sit below the peak and stay in orbit form.
        if HOLD == 'peak' and after not in ('x1', 'x3'):
            return
        if HOLD in ('residue', 'peak') and held[0] == 'orbit':
            nw.conv_inv(c, cr, xa, xb, s, Wd, bar)
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
    toff_stream = 0
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
                            nw.ss_tree_load(c, z, xa, xb, Wd, (reg, junk), ypay)
                            recs.append((hw._Dot([c.measure_x(q) for q in junk]),
                                         lambda kk, sv, jj: nw.partner(kk, Wd, ys)))
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
                toff_stream += 1
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
                elif MASKED:            # o1 s7: a classically controlled measurement, no borrowed qubit
                    if fl:
                        c.flag([qt], cond)
                    else:
                        m = c.measure_x(qt, cond)
                    STATS['cond_meas'] = STATS.get('cond_meas', 0) + 1
                    wm[t] = qt
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
                toff_stream += 1
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
    nw.phase_oracle(c, z, xa, xb, s, j0, j1, Wd, F)
    toff_sites['phase_oracle'] = c.toffoli - t0
    t0 = c.toffoli
    nw.conv_inv(c, cr, xa, xb, s, Wd, bar)
    toff_sites['conv_inv'] = c.toffoli - t0
    STATS['toff_exec_stream'] = toff_exec
    STATS.update(n=n, toffoli=c.toffoli, stream_toffoli=toff_stream)
    STATS['toffoli_executed_mean'] = c.toffoli - toff_skip
    STATS['peak_at'] = (pk[0], pk[1], c.peak)
    STATS['toff_sites'] = toff_sites
    print('O1N7 peak before op', pk, 'final', c.peak, 'toffoli', c.toffoli, 'toff_exec_stream', round(toff_exec, 1), 'toffoli_executed_mean', round(c.toffoli - toff_skip, 1), 'sites', toff_sites,
          'cond_meas', STATS.get('cond_meas'), file=sys.stderr)
    return {'x': [wm[i] for i in range(256)], 'y': [wm[256 + i] for i in range(256)]}


def first_lookup(n, offset=False):
    return hw.first_lookup(n, offset)
