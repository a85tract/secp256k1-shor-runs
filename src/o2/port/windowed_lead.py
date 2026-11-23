"""V-B circuit (KIND signed_windowed_point_add): the ECDSA.fail board leader as a windowed point addition.

Source of the arithmetic: the public ECDSA.fail benchmark repository (Layr-Labs/ecdsafail-challenge), commit
21356d25cae60e32c5bb08b49fc2112c8556a9c6 = main on 2026-11-12, the promoted source of the main-track leader
(submission 7571412f, odinfree, 1,227 qubits x 722,917 average executed Toffoli, classical addend). Its Rust
generator emits ONE point addition with a CLASSICAL addend A = (xA, yA) (qubit registers target_x, target_y;
classical bits offset_x = 0..255, offset_y = 256..511). Nothing of that code is copied here: this file replays
the generator's raw builder stream (`bind81-builder.raw`, written by build_generated() before its stream-level
post-passed ops.bin is not used) on the V-B builder and changes only how the addend enters:

  * o2/port/offdep.py finds every op whose condition depends on the addend: 4,968 X gates, 3 CX and 3 Z, in five
    sites (y load/unload at the start, x at the start of the division, 3x-derived bits at its end, x at the end of
    the multiplication, y at the end). Each X / CX is conditioned on one classical function g(xA, yA); each Z on
    g AND a measurement outcome (an MBU phase fix-up).
  * Here each site gets a scratch register S holding the values of the site's distinct functions g (<= 258) for
    the table point at the quantum address: one QROM read (select-swap, lambda = 2, or plain unary iteration) at
    the site's first addend op, a measurement-based unload after its last one. The g values per table entry come
    from evaluating the stream's own classical bit logic on all 2^(W-1) table points at once (measurement outcomes
    set to 1, which leaves every 'off' condition unchanged and turns the Z conditions into g).
    X(t) [g]  ->  CX(S_g, t);   CX(c, t) [g]  ->  CCX(S_g, c, t);   Z(t) [g and m]  ->  CZ(S_g, t) [m].
  * y sites: the functions are the 256 bits of yA; S is negated (negate_nonzero, controlled by the digit sign)
    after the read and before the unload, as in s2/port/windowed3616.py. x sites must be sign-independent
    (checked on the table).
Everything else of the stream is replayed one to one (classical bit logic, PUSH/POP, HMR / R with their fix-ups).
Approximate (EXACT = False): the stream has fixed walk budgets and truncated comparisons; use s2/port/vbx.py.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'h1'), os.path.join(REPO, 's2', 'vb'), os.path.join(REPO, 's2', 'port'), HERE]
import vbarith as va            # h1: negate_nonzero
import windowed3616 as w3       # s2: _qload, _qload_ss, _qunload (QROM read / MBU unload)
import offdep                   # o2: op classes

KIND = 'signed_windowed_point_add'
W = 16
EXACT = False
OPS = os.environ.get('O2_OPS_RAW', os.path.join(HERE, '..', 'ext', 'run', 'bind81-builder.raw'))
# (name, first op, end op, kind, select-swap read)
SITES = [('y0', 0, 5496, 'y', 1), ('x1', 5496, 1000000, 'x', 1), ('x3', 5000000, 5536370, 'x', 1),
         ('x4', 12000000, 12361729, 'x', 1), ('y5', 12361729, 1 << 62, 'y', 1)]
FLAGS = os.environ.get('O2_FLAGS', '1') == '1'   # Z-measure (c.flag) every R instead of X-measuring it
MIXED = {62240: 1574, 5534113: 1350, 12359496: 627}     # Z op -> classical bit of its measurement outcome
SELECT = os.environ.get('O2_SELECT')                     # e.g. '11011' overrides the per-site select-swap flags
if os.environ.get('O2_SITEFILE'):
    import json as _json
    _sf = _json.load(open(os.environ['O2_SITEFILE']))
    SITES = [tuple(x) for x in _sf['SITES']]
    MIXED = {int(k): int(v) for k, v in _sf['MIXED'].items()}
STATS = {}
NO = offdep.NO
_cache = {}


def window(n):
    return W


def _stream():
    if 'ops' in _cache:
        return _cache
    ops = np.memmap(OPS, dtype=offdep.DT, mode='r')
    nq = nb = 0
    for f in ('c1', 'c2', 't'):
        a = np.array(ops[f])
        nq = max(nq, int(a[a != NO].max()) + 1)
    for f in ('ct', 'cc'):
        a = np.array(ops[f])
        nb = max(nb, int(a[a != NO].max()) + 1)
    dep = np.load(OPS + '.offdep.npy')
    site_of = {}
    for i, kd, t in dep.tolist():
        s = [k for k, (_, lo, hi, _, _) in enumerate(SITES) if lo <= i < hi]
        assert len(s) == 1, 'addend op %d outside the sites' % i
        site_of[i] = s[0]
    _cache.update(ops=ops, nq=nq, nb=nb, dep=dep, site_of=site_of)
    return _cache


def _classical_conds(ops, want, offsets, hmr_value):
    """conditions (Python ints over len(offsets) lanes) of the ops in `want`, evaluating the stream's classical
    logic with addend lane l = offsets[l] and every measurement outcome = hmr_value"""
    L = len(offsets)
    ALL = (1 << L) - 1
    K = np.array(ops['kind'])
    CC = np.array(ops['cc'])
    push, pop = (K == 15), (K == 16)
    depth = np.cumsum(push.astype(np.int32) - pop.astype(np.int32))
    dprev = np.concatenate([[0], depth[:-1]])
    quantum = np.isin(K, list(offdep.QUANTUM))
    sel = np.nonzero(np.isin(K, [3, 4, 5, 15, 16]) | (K == 12) | (quantum & ((dprev > 0) | (CC != NO))))[0]
    Ks, CCs, CTs = K[sel].tolist(), CC[sel].tolist(), np.array(ops['ct'])[sel].tolist()
    nb = int(max(max(c for c in CCs if c != NO), max(c for c in CTs if c != NO))) + 1
    bits = [0] * max(nb, 512)
    for i in range(256):
        bits[i] = sum(((x >> i) & 1) << l for l, (x, y) in enumerate(offsets))
        bits[256 + i] = sum(((y >> i) & 1) << l for l, (x, y) in enumerate(offsets))
    hv = ALL if hmr_value else 0
    out = {}
    stack, base = [], ALL
    for j, i in enumerate(sel.tolist()):
        kd, cc = Ks[j], CCs[j]
        cond = base if cc == NO else base & bits[cc]
        if kd == 15:
            stack.append(base)
            base = cond
            continue
        if kd == 16:
            base = stack.pop()
            continue
        if kd == 3:
            bits[CTs[j]] ^= cond
        elif kd == 4:
            bits[CTs[j]] &= ~cond & ALL
        elif kd == 5:
            bits[CTs[j]] |= cond
        elif kd == 12 and CTs[j] != NO:
            b = CTs[j]
            bits[b] = (bits[b] & ~cond & ALL) | (hv & cond)
        if i in want:
            out[i] = cond
    return out


def _site_tables(p, T):
    """per site: function index of every addend op, and the read column (bit lists per table entry)"""
    st = _stream()
    ops, dep, site_of = st['ops'], st['dep'], st['site_of']
    want = set(int(i) for i in dep[:, 0])
    conds = _classical_conds(ops, want, [(P[0], P[1]) for P in T], 1)
    conds_neg = _classical_conds(ops, want, [(P[0], (p - P[1]) % p) for P in T], 1)
    L = len(T)
    sites = []
    for k, (name, lo, hi, kind, _) in enumerate(SITES):
        ids = [i for i in sorted(want) if site_of[i] == k]
        funcs, fidx = {}, {}
        for i in ids:
            v = conds[i]
            if kind == 'x':
                assert conds_neg[i] == v, 'site %s op %d depends on the digit sign' % (name, i)
            fidx[i] = funcs.setdefault(v, len(funcs))
        fl = sorted(funcs, key=funcs.get)
        if kind == 'y':
            # the functions must be the bits of yA: map function f -> bit position
            ybit = {}
            for b in range(256):
                ybit[sum(((T[l][1] >> b) & 1) << l for l in range(L))] = b
            assert all(v in ybit for v in fl), 'site %s: a function is not a bit of yA' % name
            fidx = {i: ybit[conds[i]] for i in ids}
            col = [[b for b in range(256) if (P[1] >> b) & 1] for P in T]
            width = 256
        else:
            nbytes = (L + 7) // 8
            M = np.stack([np.unpackbits(np.frombuffer(v.to_bytes(nbytes, 'little'), dtype=np.uint8),
                                        bitorder='little')[:L] for v in fl])
            col = [np.nonzero(M[:, l])[0].tolist() for l in range(L)]
            width = len(fl)
        site = dict(name=name, kind=kind, ids=ids, first=ids[0], last=ids[-1], fidx=fidx, col=col, width=width)
        if kind == 'y':
            # direct form: the site is one 256-bit load of yA into a zero stream register and one unload of it
            Tt = np.array(ops['t'])
            assert len(ids) == 512, 'site %s: %d addend ops' % (name, len(ids))
            ld, ul = ids[:256], ids[256:]
            wl = {fidx[i]: int(Tt[i]) for i in ld}
            wu = {fidx[i]: int(Tt[i]) for i in ul}
            assert sorted(wl) == list(range(256)) and wl == wu, 'site %s is not a load/unload pair' % name
            site.update(load0=ld[0], unload0=ul[0], wires=[wl[b] for b in range(256)])
        sites.append(site)
    return sites


def build(c, n, regs, ctx):
    assert n == 256, 'the leader stream is a secp256k1 circuit'
    st = _stream()
    ops, nq, nb = st['ops'], st['nq'], st['nb']
    p, T = ctx['p'], ctx['table']
    # content key: an id() can be reused by a later table once the earlier one is freed (uniform-pair MC with a
    # fresh random table per batch returned stale site tables on every other batch)
    key = (len(T), tuple(T[i] for i in (1, len(T) // 3, len(T) // 2, len(T) - 1)))
    if _cache.get('tkey') != key:
        _cache['sites'] = _site_tables(p, T)
        _cache['tkey'] = key
    sites = _cache['sites']
    sel_flags = [int(ch) for ch in SELECT] if SELECT else [s[4] for s in SITES]
    first = {s['first']: k for k, s in enumerate(sites)}
    last = {s['last']: k for k, s in enumerate(sites)}
    site_of = st['site_of']
    X, Y, kreg = list(regs['x']), list(regs['y']), regs['k']
    sgn, idx = kreg[-1], kreg[:-1]
    kv0 = c.read(kreg) if os.environ.get('O2_CLASSICAL') == '1' else None
    for q in idx:
        c.cx(sgn, q)
        c.x(q)                                  # address j = k' when the sign bit is 1, ~k' when it is 0

    ALL = c.ALL
    L = c.L
    wm = [None] * nq
    for i in range(256):
        wm[i] = X[i]
        wm[256 + i] = Y[i]
    bits = [0] * nb
    stack = []
    base = ALL
    toff_exec = 0.0
    S = {}
    pk = [0, -1]
    classical = os.environ.get('O2_CLASSICAL') == '1'
    if classical:
        # DIAGNOSTIC ONLY (not a valid windowed circuit): the leader's own classical-addend stream, with each lane's
        # addend read from the simulator, to separate replay problems from port problems
        Wb = len(kreg)
        kv = kv0
        pts = []
        for kk in kv:
            d = 2 * kk - (1 << Wb) + 1
            P_ = T[abs(d) >> 1]
            pts.append((P_[0], P_[1] if d > 0 else (p - P_[1]) % p))
        for i in range(256):
            bits[i] = sum(((pt[0] >> i) & 1) << l for l, pt in enumerate(pts))
            bits[256 + i] = sum(((pt[1] >> i) & 1) << l for l, pt in enumerate(pts))
        site_of = {}

    def w(q):
        r = wm[q]
        if r is None:
            r = wm[q] = c.alloc(1)[0]
        return r

    nops = len(ops)
    CH = 1 << 19
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
                s = sites[k]
                if s['kind'] == 'y':
                    if i == s['load0']:
                        reg = [w(q) for q in s['wires']]
                        (w3._qload_ss if sel_flags[k] else w3._qload)(c, idx, reg, s['col'])
                        c.x(sgn)
                        va.negate_nonzero(c, reg, p, sgn)      # yA -> p - yA for a negative digit
                        c.x(sgn)
                    elif i == s['unload0']:
                        reg = [wm[q] for q in s['wires']]
                        c.x(sgn)
                        va.negate_nonzero(c, reg, p, sgn)
                        c.x(sgn)
                        w3._qunload(c, idx, reg, s['col'])     # measures (releases) the register
                        for q in s['wires']:
                            wm[q] = None
                    continue
                if i == s['first']:
                    reg = c.alloc(s['width'])
                    (w3._qload_ss if sel_flags[k] else w3._qload)(c, idx, reg, s['col'])
                    if s['kind'] == 'y':
                        c.x(sgn)
                        va.negate_nonzero(c, reg, p, sgn)      # yA -> p - yA for a negative digit
                        c.x(sgn)
                    S[k] = reg
                g = S[k][s['fidx'][i]]
                if kd == 6:
                    c.cx(g, w(TT[j]))
                elif kd == 8:
                    c.ccx(g, w(C1[j]), w(TT[j]))
                    toff_exec += 1.0
                elif kd == 7 and i in MIXED:
                    c.cz(g, w(TT[j]), bits[MIXED[i]] & base)
                else:
                    raise RuntimeError('unexpected addend op kind %d at %d' % (kd, i))
                if i == s['last']:
                    reg = S.pop(k)
                    if s['kind'] == 'y':
                        c.x(sgn)
                        va.negate_nonzero(c, reg, p, sgn)
                        c.x(sgn)
                    w3._qunload(c, idx, reg, s['col'])
                continue
            cc = CC[j]
            cond = base if cc == NO else base & bits[cc]
            if kd == 8:
                c.cx(w(C1[j]), w(TT[j]), cond)
            elif kd == 13:
                c.ccx(w(C1[j]), w(C2[j]), w(TT[j]), cond)
                toff_exec += 1.0 if cond == ALL else bin(cond).count('1') / L
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
                fl = kd == 11 and FLAGS     # R meets a nonzero qubit only in a failing lane (o2/port/clsim.py)
                if cond == ALL:
                    if fl:
                        c.flag([qt])
                    else:
                        m = c.measure_x(qt)
                    wm[t] = None
                else:
                    # classically controlled X measurement + reset (the stream's semantics: only where cond holds):
                    # move the value into a fresh ancilla under cond, measure the ancilla (no Toffoli)
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
    assert not stack and not S
    STATS['toff_exec_stream'] = toff_exec
    STATS['peak_at'] = (pk[0], pk[1], c.peak)
    print('O2 peak before op', pk, 'final', c.peak, 'cond_meas', STATS.get('cond_meas'), file=sys.stderr)
    for q in idx:
        c.x(q)
        c.cx(sgn, q)
    return {'x': [wm[i] for i in range(256)], 'y': [wm[256 + i] for i in range(256)]}


def first_lookup(n, offset=False):
    """the run's first window as a direct lookup into the zero accumulator, counted with builder
    (h1/point_add.first_lookup: signed index transform + unary-iteration load + conditional y negation; offset=True:
    the unsigned 2^W-entry random-offset table of the harness's kappa model)"""
    import point_add as _pa
    old = (_pa.SIGNED, _pa.W, _pa.OFFSET)
    _pa.SIGNED, _pa.W, _pa.OFFSET = True, W, offset
    try:
        return _pa.first_lookup(n)
    finally:
        _pa.SIGNED, _pa.W, _pa.OFFSET = old


def declared_failure(n):
    """failure rate per addition on random inputs: 95 % upper bound of the classical replay's 158 wrong outputs in
    65,536 lanes (o2/port/clsim.py, o2/results4/clsim_L65536_s2); the leader's own eval_circuit gives 17 / 9,024 =
    1.9e-3 and V-B 5 / 4,096"""
    return 2.75e-3
