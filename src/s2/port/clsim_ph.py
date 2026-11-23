#!/usr/bin/env python3
"""Phase-tracking variant of o2/port/clsim.py. Same bit-sliced computational-basis replay of an ECDSA.fail op stream
(lane l adds the CLASSICAL addend A_l to P_l), plus the phase of every lane: Neg, Z, CZ, CCZ flip it under their condition,
and R / Hmr X-measure their qubit with a random outcome m (phase ^= m & qubit & cond), exactly as s2/port/windowed_head.py
does in V-B. Lane classes at the end: output wrong; phase flipped; an ancilla (qubit >= 512) left nonzero; some R op met a
nonzero qubit (the lane a Z-measurement flag at every R would catch, o2); the output's point on the curve or not.
Inputs: P_l uniform on the curve; A_l uniform ('uniform') or a signed odd digit times 2^(16 j) G, the entries of the
fixed windowed G tables of a run ('gtable', j uniform in 0..15, |digit| odd < 2^16).
usage: python3 -I clsim_ph.py OPS_RAW LANES SEED MODE OUT_JSON"""
import json, os, random, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', '..', 'o2', 'port')]
import offdep
from clsim import rand_point, add, to_lanes, lanes_value, P, NO, CHUNK

GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


def dbl(Pt):
    x, y = Pt
    lam = 3 * x * x * pow(2 * y, -1, P) % P
    x3 = (lam * lam - 2 * x) % P
    return x3, (lam * (x - x3) - y) % P


def smul(k, Pt):
    R = None
    for b in bin(k)[2:]:
        if R is not None:
            R = dbl(R)
        if b == '1':
            R = Pt if R is None else add(R, Pt)
    return R


def simulate(ops, L, P_in, A_in, seed):
    ALL = (1 << L) - 1
    q = [0] * 1227
    bits = [0] * 1595
    xs, ys = to_lanes([p[0] for p in P_in], 256), to_lanes([p[1] for p in P_in], 256)
    ax, ay = to_lanes([a[0] for a in A_in], 256), to_lanes([a[1] for a in A_in], 256)
    q[0:256], q[256:512] = xs, ys
    bits[0:256], bits[256:512] = ax, ay
    rh = random.Random(seed)
    FORCE = os.environ.get('CL_FORCE')   # '0' / '1': every X-measurement outcome forced (V-B's forced batches)
    base, stack = ALL, []
    ph = 0
    rhit = 0        # lanes in which some R op met a nonzero qubit
    hhit = 0        # lanes in which some Hmr op met a nonzero qubit that is not a data qubit (diagnostic only)
    N = len(ops)
    for c0 in range(0, N, CHUNK):
        ch = ops[c0:c0 + CHUNK]
        K, C1, C2, T = ch['kind'].tolist(), ch['c1'].tolist(), ch['c2'].tolist(), ch['t'].tolist()
        CT, CC = ch['ct'].tolist(), ch['cc'].tolist()
        for j in range(len(K)):
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
                if v:
                    m = rh.getrandbits(L) if FORCE is None else (ALL if FORCE == '1' else 0)
                    ph ^= m & v
                    if kd == 11:
                        rhit |= v
                    q[t] &= ~cond
                else:
                    m = (rh.getrandbits(L) if FORCE is None else (ALL if FORCE == '1' else 0)) if kd == 12 else 0
                if kd == 12:
                    ct = CT[j]
                    bits[ct] = (bits[ct] & ~cond) | ((m if v else (rh.getrandbits(L) if FORCE is None else (ALL if FORCE == '1' else 0))) & cond)
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
            elif kd == 7:
                ph ^= q[T[j]] & (base if cc == NO else base & bits[cc])
            elif kd == 9:
                ph ^= q[T[j]] & q[C1[j]] & (base if cc == NO else base & bits[cc])
            elif kd == 14:
                ph ^= q[T[j]] & q[C1[j]] & q[C2[j]] & (base if cc == NO else base & bits[cc])
            elif kd == 0:
                ph ^= base if cc == NO else base & bits[cc]
    return q, ph, rhit


def main(raw, L, seed, mode, out_json):
    ops = np.memmap(raw, dtype=offdep.DT, mode='r')
    r = random.Random(seed)
    t0 = time.time()
    P_in = [rand_point(r) for _ in range(L)]
    if mode == 'uniform':
        A_in = [rand_point(r) for _ in range(L)]
        meta = [None] * L
    else:
        Bj = [(GX, GY)]
        for j in range(1, 16):
            Z = Bj[-1]
            for _ in range(16):
                Z = dbl(Z)
            Bj.append(Z)
        A_in, meta = [], []
        for _ in range(L):
            j, k = r.randrange(16), r.randrange(1 << 15) * 2 + 1
            Pt = smul(k, Bj[j])
            if r.getrandbits(1):
                Pt = (Pt[0], (P - Pt[1]) % P)
            A_in.append(Pt); meta.append((j, k))
    exp = [add(p, a) for p, a in zip(P_in, A_in)]
    print('inputs %.0f s' % (time.time() - t0), file=sys.stderr, flush=True)
    q, ph, rhit = simulate(ops, L, P_in, A_in, seed + 1)
    xo, yo = lanes_value(q, 0, L), lanes_value(q, 256, L)
    ALL = (1 << L) - 1
    wrong = 0
    oncurve = 0
    for l in range(L):
        if exp[l] is None or (xo[l], yo[l]) != exp[l]:
            wrong |= 1 << l
        if (yo[l] * yo[l] - xo[l] ** 3 - 7) % P == 0:
            oncurve |= 1 << l
    dirty = 0
    for t in range(512, 1227):
        dirty |= q[t]
    cnt = lambda m: bin(m).count('1')
    cls = {}
    for l in range(L):
        w, p_, rh_, oc, d = [(m >> l) & 1 for m in (wrong, ph, rhit, oncurve, dirty)]
        if w or p_ or rh_ or d:
            key = ('wrong' if w else 'right') + ('+phase' if p_ else '') + ('+Rhit' if rh_ else '') + ('+dirtyend' if d else '') + \
                  ('' if oc else ' offcurve') if True else ''
            cls[key] = cls.get(key, 0) + 1
    res = dict(lanes=L, seed=seed, mode=mode, wrong=cnt(wrong), phase=cnt(ph), rhit=cnt(rhit), dirty_end=cnt(dirty),
               wrong_oncurve=cnt(wrong & oncurve), classes=cls, secs=time.time() - t0)
    res['failing_lanes'] = [(l, meta[l], (wrong >> l) & 1, (ph >> l) & 1, (rhit >> l) & 1, (oncurve >> l) & 1)
                            for l in range(L) if ((wrong | ph | rhit | dirty) >> l) & 1]
    print(json.dumps({k: v for k, v in res.items() if k != 'failing_lanes'}), flush=True)
    json.dump(res, open(out_json, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
