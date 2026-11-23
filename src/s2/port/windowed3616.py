"""V-B circuit (KIND signed_windowed_point_add): ECDSA.fail submission 3616dbf as a windowed point addition.

Source of the arithmetic: the public ECDSA.fail benchmark (Layr-Labs/ecdsafail-challenge, commit
897dda2b0cf267151ecd973252d2a5078cbf1b63, "3616dbf", arXiv 2609.09582 Sect. 5.2.3 / Table 3). Its Rust generator
emits a fixed reversible op stream `ops.bin` for ONE point addition with a CLASSICAL addend A = (xA, yA)
(register layout: target_x, target_y qubits; offset_x, offset_y classical bits). This file does not contain or
copy that code: it replays the op stream that the generator emits (decompressed to ops.raw by port/README.md's
steps) on the V-B builder and changes only what the addend is:

  * The stream uses the addend in exactly 10 blocks of 256 X gates, each conditioned on a classical bit that is a
    function of (xA, yA) alone (port/extract_groups.py finds them): load a constant into a scratch register, use
    it as the addend of an adder, load it again to clear the register. The loaded values are the table columns
    x, y, 3x mod p, x + 1 (5 load/unload pairs: x, y, 3x, y, x+1).
  * Here every load block becomes a QROM lookup of that column at the quantum window address (unary iteration,
    2^(W-1) - 2 Toffoli) into the same scratch register, and every clear block becomes a measurement-based
    unlookup (X-measure the register, fix the phase with a two-level phase lookup). Nothing else of the stream
    changes: all other gates are replayed one to one (classical bit logic, PUSH/POP conditions, X-basis
    measurements HMR/R with their classically controlled fix-ups).
  * Signed odd digits (harness KIND): k = (s, k'), address j = k' xor (not s) over W-1 bits, y column negated
    (n Toffoli, an involution) around its use when s = 0. 2^(W-1) table points, W exponent bits per addition.

Approximation: the stream has a fixed 704-round ping-pong budget per division, so it fails (wrong output and
EXACT = False. Rust and the license: the benchmark repo's NOTICE only covers circuit.rs/sim.rs (CC BY 4.0); no
licence file covers src/point_add, so nothing of it is copied or redistributed here, only used as measured input.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'h1'), os.path.join(REPO, 's2', 'vb')]
import vbarith as va            # h1: negate_nonzero
import qrom                     # s2: iterate (unary iteration)

KIND = 'signed_windowed_point_add'
W = 16                          # exponent bits per addition; table 2^(W-1) points
EXACT = False
OPS = os.environ.get('S2_OPS_RAW', os.path.join(REPO, 's2', 'src', 'run361', 'ops.raw'))
GROUPS = os.environ.get('S2_GROUPS', os.path.join(REPO, 's2', 'src', 'run361', 'groups.json'))
SELECT_SWAP = 1                 # 1: lambda = 2 select-swap loads (256 extra qubits while loading), 0: plain unary iteration
STATS = {}

NO = 2**64 - 1
DT = np.dtype([('kind', '<u4'), ('pad', '<u4'), ('c2', '<u8'), ('c1', '<u8'), ('t', '<u8'), ('ct', '<u8'),
               ('cc', '<u8'), ('r', '<u8')])
_cache = {}


def window(n):
    return W


def _stream():
    if 'ops' not in _cache:
        ops = np.memmap(OPS, dtype=DT, mode='r')
        nq = 0
        for f in ('c1', 'c2', 't'):
            a = np.array(ops[f])
            nq = max(nq, int(a[a != NO].max()) + 1)
        nb = 0
        for f in ('ct', 'cc'):
            a = np.array(ops[f])
            nb = max(nb, int(a[a != NO].max()) + 1)
        _cache['ops'] = ops
        _cache['nq'] = nq
        _cache['nb'] = nb
        _cache['groups'] = json.load(open(GROUPS))['groups']
    return _cache


def _columns(p, T):
    return {'x': [P[0] for P in T], 'y': [P[1] for P in T], '3x': [3 * P[0] % p for P in T],
            'x1': [(P[0] + 1) % 2**256 for P in T]}


def _bitlists(col):
    return [[b for b in range(256) if (v >> b) & 1] for v in col]


def _qload(c, idx, out, bl):
    """out ^= column[k]: unary iteration, 2^(W-1) - 2 Toffoli (SELECT_SWAP = 0)"""
    def leaf(f, i):
        for b in bl[i]:
            c.cx(f, out[b])
    qrom.iterate(c, None, idx[::-1], leaf)


def _phase_lookup(c, idx, ms, bl, xor):
    """apply the phase (-1)^(sum_b ms[b] * column[a ^ xor] bit b) at address a = idx: two-level phase lookup
    (one-hot of the low half of the address x unary iteration over the high half; ~2 (2^l - 2) + (2^h - 2) Toffoli)"""
    Wk = len(idx)
    l = (Wk + 1) // 2
    h = Wk - l
    lo, hi = idx[:l][::-1], idx[l:][::-1]
    onehot = c.alloc(1 << l)
    qrom.iterate(c, None, lo, lambda f, j: c.cx(f, onehot[j]))

    def g_of(i):
        g = 0
        for b in bl[i ^ xor]:
            g ^= ms[b]
        return g

    def leaf(f, ih):
        for j in range(1 << l):
            g = g_of((ih << l) | j)
            if g:
                if f is None:
                    c.z(onehot[j], cond=g)
                else:
                    c.cz(f, onehot[j], cond=g)
    if h:
        qrom.iterate(c, None, hi, leaf)
    else:
        leaf(None, 0)
    qrom.iterate(c, None, lo, lambda f, j: c.cx(f, onehot[j]))
    c.free(onehot)


def _qunload(c, idx, out, bl):
    """measurement-based unlookup of out = column[k] (Gidney): X-measure, then fix (-1)^(m . column[k])"""
    ms = [c.measure_x(q) for q in out]
    _phase_lookup(c, idx, ms, bl, 0)


def _qload_ss(c, idx, out, bl):
    """select-swap lookup with lambda = 2 (Berry et al. 2019): unary iteration over the high address bits writes
    column[2v] into `out` and column[2v+1] into a second register R1 (CNOT fan-out only), a swap controlled by the
    low address bit (one Toffoli per bit) puts column[a] into `out`, and R1 = column[a ^ 1] is erased by an X
    measurement plus a phase lookup. 2^(W-2) - 2 + 256 + ~2 sqrt(2^(W-1)) Toffoli, 256 extra qubits while it runs."""
    lo, hi = idx[0], idx[1:]
    R1 = c.alloc(len(out))

    def leaf(f, v):
        for b in bl[2 * v]:
            c.cx(f, out[b])
        for b in bl[2 * v + 1]:
            c.cx(f, R1[b])
    qrom.iterate(c, None, hi[::-1], leaf)
    for b in range(len(out)):
        c.cx(R1[b], out[b])
        c.ccx(lo, out[b], R1[b])
        c.cx(R1[b], out[b])
    ms = [c.measure_x(q) for q in R1]
    _phase_lookup(c, idx, ms, bl, 1)


def build(c, n, regs, ctx):
    assert n == 256, '3616dbf is a secp256k1 circuit'
    st = _stream()
    ops, nq, nb, groups = st['ops'], st['nq'], st['nb'], st['groups']
    p, T = ctx['p'], ctx['table']
    key = (id(T), len(T))
    if _cache.get('colkey') != key:
        cols = _columns(p, T)
        _cache['cols'] = {k: _bitlists(v) for k, v in cols.items()}
        _cache['colkey'] = key
    bls = _cache['cols']
    X, Y, k = list(regs['x']), list(regs['y']), regs['k']
    s, idx = k[-1], k[:-1]
    for q in idx:
        c.cx(s, q)
        c.x(q)                                  # j = k' when s = 1, complemented when s = 0

    ALL = c.ALL
    L = c.L
    wm = [None] * nq
    for i in range(256):
        wm[i] = X[i]
        wm[256 + i] = Y[i]
    bits = [0] * nb
    skip = set()
    ev = {}
    for gi, g in enumerate(groups):
        skip.update(g['gates'])
        ev[g['first']] = gi
    stack = []
    base = ALL
    toff_exec = 0.0
    nops = len(ops)
    CH = 1 << 19
    NOQ = NO
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
            if i in skip:
                gi = ev.get(i)
                if gi is not None:
                    g = groups[gi]
                    out = []
                    for w in g['wires']:
                        q = wm[w]
                        if q is None:
                            q = wm[w] = c.alloc(1)[0]
                        out.append(q)
                    bl = bls[g['column']]
                    if g['role'] == 'load':
                        (_qload_ss if SELECT_SWAP else _qload)(c, idx, out, bl)
                        if g['column'] == 'y':
                            c.x(s)
                            va.negate_nonzero(c, out, p, s)     # y -> p - y for a negative digit
                            c.x(s)
                    else:
                        if g['column'] == 'y':
                            c.x(s)
                            va.negate_nonzero(c, out, p, s)
                            c.x(s)
                        _qunload(c, idx, out, bl)
                        for w in g['wires']:
                            wm[w] = None
                continue
            cc = CC[j]
            cond = base if cc == NOQ else base & bits[cc]
            if kd == 8:                                     # CX
                t = TT[j]
                qt = wm[t]
                if qt is None:
                    qt = wm[t] = c.alloc(1)[0]
                qc = wm[C1[j]]
                if qc is None:
                    qc = wm[C1[j]] = c.alloc(1)[0]
                c.cx(qc, qt, cond)
            elif kd == 13:                                  # CCX
                t = TT[j]
                qt = wm[t]
                if qt is None:
                    qt = wm[t] = c.alloc(1)[0]
                q1 = wm[C1[j]]
                if q1 is None:
                    q1 = wm[C1[j]] = c.alloc(1)[0]
                q2 = wm[C2[j]]
                if q2 is None:
                    q2 = wm[C2[j]] = c.alloc(1)[0]
                c.ccx(q1, q2, qt, cond)
                toff_exec += 1.0 if cond == ALL else bin(cond).count('1') / L
            elif kd == 6:                                   # X
                t = TT[j]
                qt = wm[t]
                if qt is None:
                    qt = wm[t] = c.alloc(1)[0]
                c.x(qt, cond)
            elif kd == 9:                                   # CZ
                qa = wm[TT[j]]
                if qa is None:
                    qa = wm[TT[j]] = c.alloc(1)[0]
                qb = wm[C1[j]]
                if qb is None:
                    qb = wm[C1[j]] = c.alloc(1)[0]
                c.cz(qa, qb, cond)
            elif kd == 15:                                  # PUSH_CONDITION
                stack.append(base)
                base = base & bits[cc]
            elif kd == 16:                                  # POP_CONDITION
                base = stack.pop()
            elif kd == 10:                                  # SWAP(control1, target)
                t, c1 = TT[j], C1[j]
                qt, q1 = wm[t], wm[c1]
                if qt is None:
                    qt = wm[t] = c.alloc(1)[0]
                if q1 is None:
                    q1 = wm[c1] = c.alloc(1)[0]
                if cond == ALL:
                    c.swap(qt, q1)
                else:
                    c.cx(qt, q1, cond)
                    c.cx(q1, qt, cond)
                    c.cx(qt, q1, cond)
            elif kd == 12 or kd == 11:                      # HMR / R (X-basis measure + reset)
                t = TT[j]
                qt = wm[t]
                if qt is None:
                    qt = c.alloc(1)[0]
                if cond != ALL and c.s[qt] & ~cond & ALL:
                    raise RuntimeError('conditional measurement of a qubit that is nonzero outside the condition (op %d)' % i)
                m = c.measure_x(qt)
                wm[t] = None
                if kd == 12:
                    ct = CT[j]
                    bits[ct] = (bits[ct] & ~cond & ALL) | (m & cond)
            elif kd == 7:                                   # Z
                qt = wm[TT[j]]
                if qt is None:
                    qt = wm[TT[j]] = c.alloc(1)[0]
                c.z(qt, cond)
            elif kd == 14:                                  # CCZ
                qs = []
                for w in (TT[j], C1[j], C2[j]):
                    q = wm[w]
                    if q is None:
                        q = wm[w] = c.alloc(1)[0]
                    qs.append(q)
                c.ccz(qs[0], qs[1], qs[2], cond)
                toff_exec += 1.0 if cond == ALL else bin(cond).count('1') / L
            elif kd == 3:                                   # BitInvert
                bits[CT[j]] ^= cond
            elif kd == 4:                                   # BitStore0
                bits[CT[j]] &= ~cond & ALL
            elif kd == 5:                                   # BitStore1
                bits[CT[j]] |= cond
            elif kd == 0:                                   # Neg
                c.global_phase(cond)
            else:
                raise RuntimeError('unknown op kind %d' % kd)
    assert not stack
    STATS['toff_exec_stream'] = toff_exec
    for q in idx:
        c.x(q)
        c.cx(s, q)
    return {'x': [wm[i] for i in range(256)], 'y': [wm[256 + i] for i in range(256)]}
