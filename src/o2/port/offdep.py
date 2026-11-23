#!/usr/bin/env python3
"""Which ops of an ECDSA.fail op stream depend on the CLASSICAL addend (o2, after s2/port/classical_logic.py).

ops.raw = decompressed ops.bin (56-byte records: kind u32, pad u32, q_control2, q_control1, q_target, c_target,
c_condition, r_target as u64; NO = 2^64 - 1 means unused). Classical bits 0..255 / 256..511 are offset_x / offset_y.
The classical bit logic (BitInvert/BitStore0/BitStore1, Hmr outcomes, the PushCondition/PopCondition stack and each
op's own c_condition) is evaluated bit-sliced on LANES random addends in three runs: base, a second Hmr-outcome seed,
other addends (first seed). Every quantum op (X, Z, CX, CZ, Swap, R, Hmr, CCX, CCZ, Neg) gets its effective
condition in each run and one class:
  always / never          the same in every lane and run
  hmr                     changes with the measurement seed only (classically controlled fix-ups)
  off                     changes with the addend only (the addend steers the circuit)
  mixed                   changes with both
Prints counts per (kind, class) and the Toffoli-weighted expected execution per class; saves the indices of the
'off' and 'mixed' ops to OPS_RAW.offdep.npy (columns: index, kind, q_target).
usage: python3 -I offdep.py OPS_RAW [lanes]
"""
import collections
import random
import sys

import numpy as np

DT = np.dtype([('kind', '<u4'), ('pad', '<u4'), ('c2', '<u8'), ('c1', '<u8'), ('t', '<u8'), ('ct', '<u8'),
               ('cc', '<u8'), ('r', '<u8')])
NO = 2**64 - 1
QUANTUM = {0: 'Neg', 6: 'X', 7: 'Z', 8: 'CX', 9: 'CZ', 10: 'Swap', 11: 'R', 12: 'Hmr', 13: 'CCX', 14: 'CCZ'}


def run(K, CC, CT, lanes, seed_off, seed_hmr):
    """effective condition (uint64 lane mask) of every op in K (Push/Pop and classical ops included)"""
    assert lanes == 64
    ALL = (1 << lanes) - 1
    r_off, r_hmr = random.Random(seed_off), random.Random(seed_hmr)
    nb = int(max(CC[CC != NO].max(initial=0), CT[CT != NO].max(initial=0))) + 1
    bits = [0] * max(nb, 512)
    for l in range(lanes):
        x, y = r_off.getrandbits(256), r_off.getrandbits(256)
        for i in range(256):
            if (x >> i) & 1:
                bits[i] |= 1 << l
            if (y >> i) & 1:
                bits[256 + i] |= 1 << l
    stack, base = [], ALL
    conds = np.zeros(len(K), dtype=np.uint64)
    Kl, CCl, CTl = K.tolist(), CC.tolist(), CT.tolist()
    for i in range(len(Kl)):
        kd = Kl[i]
        cc = CCl[i]
        cond = base if cc == NO else base & bits[cc]
        if kd == 15:
            stack.append(base)
            base = base & bits[cc]
            continue
        if kd == 16:
            base = stack.pop()
            continue
        if kd == 3:
            bits[CTl[i]] ^= cond
        elif kd == 4:
            bits[CTl[i]] &= ~cond & ALL
        elif kd == 5:
            bits[CTl[i]] |= cond
        elif kd == 12 and CTl[i] != NO:
            b = CTl[i]
            bits[b] = (bits[b] & ~cond & ALL) | (r_hmr.getrandbits(lanes) & cond)
        conds[i] = cond
    return conds, ALL


def main(path, lanes=64):
    ops = np.memmap(path, dtype=DT, mode='r')
    K = np.array(ops['kind'])
    CCa = np.array(ops['cc'])
    push, pop = (K == 15), (K == 16)
    depth = np.cumsum(push.astype(np.int32) - pop.astype(np.int32))
    dprev = np.concatenate([[0], depth[:-1]])
    quantum = np.isin(K, list(QUANTUM))
    # an op can depend on classical data only through the condition stack or its own c_condition
    sel = np.nonzero(np.isin(K, [3, 4, 5, 15, 16]) | (K == 12) | (quantum & ((dprev > 0) | (CCa != NO))))[0]
    print('ops', len(K), 'selected', len(sel), flush=True)
    un = quantum.copy()
    un[sel] = False
    cnt = collections.Counter()
    for kd, name in QUANTUM.items():
        m = int(np.count_nonzero(un & (K == kd)))
        if m:
            cnt[(name, 'always')] += m
    Ks, CCs, CTs = K[sel], CCa[sel], np.array(ops['ct'])[sel]
    del CCa
    a, ALL = run(Ks, CCs, CTs, lanes, 1000, 77)
    b, _ = run(Ks, CCs, CTs, lanes, 1000, 78)
    c, _ = run(Ks, CCs, CTs, lanes, 1001, 77)
    A = np.uint64(ALL)
    qs = np.isin(Ks, list(QUANTUM))
    cls = np.full(len(sel), 'mixed', dtype=object)
    same_ab, same_ac = (a == b), (a == c)
    cls[same_ab & same_ac & (a == A)] = 'always'
    cls[same_ab & same_ac & (a == 0)] = 'never'
    cls[same_ab & same_ac & (a != A) & (a != 0)] = 'const-lanes'
    cls[same_ac & ~same_ab] = 'hmr'
    cls[same_ab & ~same_ac] = 'off'
    pc = lambda v: np.array([bin(int(x)).count('1') for x in v], dtype=np.float64)
    expT = collections.Counter()
    for j in np.nonzero(qs)[0].tolist():
        cnt[(QUANTUM[int(Ks[j])], cls[j])] += 1
    tof = np.nonzero(qs & np.isin(Ks, [13, 14]))[0]
    pa, pb, pcc = pc(a[tof]), pc(b[tof]), pc(c[tof])
    for k in set(cls[tof].tolist()):
        m = cls[tof] == k
        expT[k] = float((pa[m] + pb[m] + pcc[m]).sum() / (3 * lanes))
    expT['always(unconditioned)'] = int(np.count_nonzero(un & np.isin(K, [13, 14])))
    for key in sorted(cnt):
        print('%-5s %-11s %9d' % (key[0], key[1], cnt[key]))
    print('Toffoli (CCX+CCZ) expected executed per class:', {k: round(v, 1) for k, v in expT.items()},
          'total', round(sum(expT.values()), 1))
    keep = np.nonzero(qs & np.isin(cls, ['off', 'mixed']))[0]
    T = np.array(ops['t'])
    np.save(path + '.offdep.npy', np.stack([sel[keep], Ks[keep].astype(np.int64), T[sel[keep]].astype(np.int64)], 1))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 64)
