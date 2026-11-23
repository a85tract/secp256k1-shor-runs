#!/usr/bin/env python3
"""Bit-sliced classical simulation of an ECDSA.fail op stream: failing inputs and free flag points.

The computational-basis trajectory of L lanes at once (Python ints as lane masks). Lane l adds the classical addend
A_l (classical bits 0..255 = x, 256..511 = y) to the point P_l (qubits 0..255 = x, 256..511 = y); the result is read
from qubits 0..511 at the end. Phase ops (Z, CZ, CCZ, Neg) are ignored. Hmr writes a random outcome into its classical
bit; R and Hmr reset their qubit where their condition holds (the leader's semantics; o2/port/windowed_lead.py).

Pass 1 marks the failing lanes: output != P_l + A_l, or a qubit >= 512 left nonzero at the end.
Pass 2 (same seeds, so the same trajectories) looks for free flag points: an R or Hmr op whose qubit is nonzero in some
failing lane and zero in every good lane of the sample. Z-measuring that qubit there (instead of resetting it) flags
the failure without Toffoli, and in a good lane it measures |0>, so nothing changes. Also reported: qubits left
nonzero at the end only in failing lanes, and how often R hits a nonzero qubit in good lanes.
usage: python3 -I clsim.py OPS_RAW LANES SEED [OUT_JSON]
"""
import json
import random
import sys
import time

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import offdep

P = 2**256 - 2**32 - 977
NO = 2**64 - 1
CHUNK = 400000


def rand_point(r):
    while True:
        x = r.getrandbits(256)
        if x >= P:
            continue
        s = (x * x * x + 7) % P
        y = pow(s, (P + 1) // 4, P)
        if y * y % P == s:
            return x, (y if r.getrandbits(1) else (P - y) % P)


def add(Pt, Q):
    (x1, y1), (x2, y2) = Pt, Q
    if x1 == x2:
        return None
    lam = (y2 - y1) * pow(x2 - x1, -1, P) % P
    x3 = (lam * lam - x1 - x2) % P
    return x3, (lam * (x1 - x3) - y1) % P


def to_lanes(vals, nbits):
    L = len(vals)
    a = np.frombuffer(b''.join(v.to_bytes(nbits // 8, 'little') for v in vals), dtype=np.uint8).reshape(L, nbits // 8)
    bits = np.unpackbits(a, axis=1, bitorder='little')
    packed = np.packbits(np.ascontiguousarray(bits.T), axis=1, bitorder='little')
    return [int.from_bytes(packed[i].tobytes(), 'little') for i in range(nbits)]


CKPT = 5000
_SZ = {}


def _sizes(ops):
    """qubit and classical-bit counts of the stream (1,227 / 1,595 for the leader's raw stream; larger for the
    widened-walk experiments)"""
    key = id(ops)
    if key not in _SZ:
        nq = nb = 0
        for fld in ('t', 'c1', 'c2'):
            a = np.asarray(ops[fld])
            nq = max(nq, int(a[a != NO].max()) + 1)
        for fld in ('ct', 'cc'):
            a = np.asarray(ops[fld])
            nb = max(nb, int(a[a != NO].max()) + 1)
        _SZ[key] = (max(nq, 1227), max(nb, 1595))
    return _SZ[key]


def simulate(ops, L, P_in, A_in, seed, G=None, F=None, log=None):
    ALL = (1 << L) - 1
    nq, nb = _sizes(ops)
    q = [0] * nq
    bits = [0] * nb
    xs, ys = to_lanes([p[0] for p in P_in], 256), to_lanes([p[1] for p in P_in], 256)
    ax, ay = to_lanes([a[0] for a in A_in], 256), to_lanes([a[1] for a in A_in], 256)
    q[0:256], q[256:512] = xs, ys
    bits[0:256], bits[256:512] = ax, ay
    rh = random.Random(seed)
    base, stack = ALL, []
    flags, flagged, flagged_r, first_r = [], 0, 0, {}
    r_good_nonzero = 0
    flagged_ck, ck_events = 0, []
    N = len(ops)
    t0 = time.time()
    for c0 in range(0, N, CHUNK):
        ch = ops[c0:c0 + CHUNK]
        K, C1, C2, T = ch['kind'].tolist(), ch['c1'].tolist(), ch['c2'].tolist(), ch['t'].tolist()
        CT, CC = ch['ct'].tolist(), ch['cc'].tolist()
        for j in range(len(K)):
            if G is not None and (c0 + j) % CKPT == 0:
                # qubits with one value on every good lane and another on some failing lane: a CNOT copy + Z flag
                for t in range(1227):
                    v = q[t]
                    vg = v & G
                    if vg == 0:
                        c = v & F
                    elif vg == G:
                        c = F & ~v
                    else:
                        continue
                    if c & ~flagged_ck:
                        ck_events.append((c0 + j, t, 0 if vg == 0 else 1, bin(c & ~flagged_ck).count('1')))
                        flagged_ck |= c
            kd = K[j]
            cc = CC[j]
            if kd == 8:
                if cc == NO and base == ALL:
                    q[T[j]] ^= q[C1[j]]
                else:
                    q[T[j]] ^= q[C1[j]] & (base if cc == NO else base & bits[cc])
            elif kd == 13:
                if cc == NO and base == ALL:
                    q[T[j]] ^= q[C1[j]] & q[C2[j]]
                else:
                    q[T[j]] ^= q[C1[j]] & q[C2[j]] & (base if cc == NO else base & bits[cc])
            elif kd == 6:
                q[T[j]] ^= base if cc == NO else base & bits[cc]
            elif kd == 11 or kd == 12:
                cond = base if cc == NO else base & bits[cc]
                t = T[j]
                v = q[t] & cond
                if G is not None and v:
                    if v & G:
                        if kd == 11:
                            r_good_nonzero += 1
                    elif v & F:
                        flags.append((c0 + j, kd, t, bin(v & F).count('1')))
                        flagged |= v & F
                        if kd == 11:
                            new = v & F & ~flagged_r
                            flagged_r |= new
                            while new:
                                b = new & -new
                                first_r[b.bit_length() - 1] = c0 + j
                                new ^= b
                q[t] &= ~cond
                if kd == 12:
                    ct = CT[j]
                    bits[ct] = (bits[ct] & ~cond) | (rh.getrandbits(L) & cond)
            elif kd == 10:
                cond = base if cc == NO else base & bits[cc]
                a, b = T[j], C1[j]
                d = (q[a] ^ q[b]) & cond
                q[a] ^= d
                q[b] ^= d
            elif kd == 15:
                stack.append(base)
                base = base if cc == NO else base & bits[cc]
            elif kd == 16:
                base = stack.pop()
            elif kd == 3:
                bits[CT[j]] ^= base if cc == NO else base & bits[cc]
            elif kd == 4:
                bits[CT[j]] &= ~(base if cc == NO else base & bits[cc])
            elif kd == 5:
                bits[CT[j]] |= base if cc == NO else base & bits[cc]
            # 0 Neg, 7 Z, 9 CZ, 14 CCZ: phase only; 1, 2, 17: metadata
        if log:
            print('  ops %9d  %.0f s' % (c0 + len(K), time.time() - t0), file=log, flush=True)
    return q, flags, flagged, r_good_nonzero, flagged_r, first_r, flagged_ck, ck_events


def lanes_value(q, lo, L):
    """per-lane integers of the 256-bit register q[lo:lo+256]"""
    nb = (L + 7) // 8
    a = np.frombuffer(b''.join(q[lo + i].to_bytes(nb, 'little') for i in range(256)), dtype=np.uint8).reshape(256, nb)
    bits = np.unpackbits(a, axis=1, bitorder='little')[:, :L]
    packed = np.packbits(np.ascontiguousarray(bits.T), axis=1, bitorder='little')
    return [int.from_bytes(packed[l].tobytes(), 'little') for l in range(L)]


def main(raw, L, seed, out_json=None):
    ops = np.memmap(raw, dtype=offdep.DT, mode='r')
    r = random.Random(seed)
    P_in = [rand_point(r) for _ in range(L)]
    A_in = [rand_point(r) for _ in range(L)]
    exp = [add(p, a) for p, a in zip(P_in, A_in)]
    t0 = time.time()
    q, _, _, _, _, _, _, _ = simulate(ops, L, P_in, A_in, seed + 1, log=sys.stderr)
    ALL = (1 << L) - 1
    xo, yo = lanes_value(q, 0, L), lanes_value(q, 256, L)
    F_out = 0
    for l in range(L):
        if exp[l] is None or (xo[l], yo[l]) != exp[l]:
            F_out |= 1 << l
    dirty = 0
    for t in range(512, 1227):
        dirty |= q[t]
    F = F_out | dirty
    G = ALL & ~F
    print('pass 1: %d lanes, %.0f s: output wrong %d, ancilla dirty at end %d (both %d), failing %d (%.2e)' % (
        L, time.time() - t0, bin(F_out).count('1'), bin(dirty).count('1'), bin(F_out & dirty).count('1'),
        bin(F).count('1'), bin(F).count('1') / L), flush=True)
    res = dict(lanes=L, seed=seed, wrong_output=bin(F_out).count('1'), dirty_end=bin(dirty).count('1'),
               failing=bin(F).count('1'))
    res['failing_data'] = [(l, hex(P_in[l][0]), hex(P_in[l][1]), hex(A_in[l][0]), hex(A_in[l][1]), hex(xo[l]), hex(yo[l]))
                           for l in range(L) if (F >> l) & 1]
    if F:
        q2, flags, flagged, rgn, flagged_r, first_r, flagged_ck, ck_events = simulate(ops, L, P_in, A_in, seed + 1, G, F)
        assert lanes_value(q2, 0, L) == xo
        end_flags = [(t, bin(q2[t] & F).count('1')) for t in range(512, 1227) if q2[t] & F and not q2[t] & G]
        for t, _ in end_flags:
            flagged |= q2[t] & F
        silent = F & ~flagged
        silent_r = F & ~flagged_r
        nk = {k: sum(1 for f in flags if f[1] == k) for k in (11, 12)}
        first = {}
        for i, kd, t, nf in flags:
            first.setdefault(t, (i, kd, nf))
        print('pass 2: free flag points %d (on %d qubits), end-of-stream flag qubits %d; failing lanes flagged %d of %d,'
              ' silent %d; R ops hitting a nonzero qubit in good lanes: %d' % (
                  len(flags), len(first), len(end_flags), bin(flagged).count('1'), bin(F).count('1'),
                  bin(silent).count('1'), rgn), flush=True)
        print('        flag events by kind (R, Hmr): %s; failing lanes caught by R flags alone %d, silent with R flags only %d'
              % (nk, bin(flagged_r).count('1'), bin(silent_r).count('1')), flush=True)
        print('        checkpoint flags (every %d ops): %d events, failing lanes caught %d; R + checkpoints: silent %d;'
              ' R + Hmr-zero + checkpoints: silent %d' % (CKPT, len(ck_events), bin(flagged_ck).count('1'),
                                                          bin(F & ~(flagged_r | flagged_ck)).count('1'),
                                                          bin(F & ~(flagged | flagged_ck)).count('1')), flush=True)
        fr = sorted(first_r.values())
        if fr:
            print('        first R flag per failing lane, op index: min %d, median %d, max %d' % (fr[0], fr[len(fr) // 2], fr[-1]))
        res.update(flag_points=len(flags), flag_qubits=len(first), end_flags=end_flags[:50],
                   flagged=bin(flagged).count('1'), silent=bin(silent).count('1'), r_hits_good=rgn,
                   flag_events_by_kind=nk, flagged_by_r=bin(flagged_r).count('1'), silent_r_only=bin(silent_r).count('1'),
                   first_r_flag=sorted(first_r.items()), ck_events=ck_events[:400],
                   silent_r_ck=[l for l in range(L) if ((F & ~(flagged_r | flagged_ck)) >> l) & 1],
                   flags_head=flags[:200], silent_lanes=[l for l in range(L) if (silent >> l) & 1][:50])
    if out_json:
        json.dump(res, open(out_json, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else None)
