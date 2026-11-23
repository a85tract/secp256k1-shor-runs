"""a bit-sliced replay of the leader stream with the leader PORT's
site semantics, so that the terminal's prefix is replayed as s4/term/hexterm.py builds it.

with the per-lane count of skipped conditional CCX / CCZ), copied op for op, with the addend sites of the prefix
handled as the port handles them:

* x3 site (1,317 ops, sitemap value 'c3', every cst = 0): the condition of each op is bit b of
      c3 = (x3mult * xA + OFF) mod p
  taken from the lane's addend, as the port's read S holds it (hexterm.py: pair3 = (2 x_T, 2 beta x_T), the unit
  selection with mult 2, then + OFF mod p), instead of the stream's own classical logic (which computes 3 xA + OFF).
  X(t) -> t ^= g; CX(c, t) -> t ^= c & g; the mixed Z op (MIXED[i] = its measurement bit) -> phase ^= t & g & m.
  The stream's condition stack is not applied to site ops (the port applies CX(S_g, t) unconditionally; every
  site op sits inside its own addend-condition Push at depth 1 or at depth 0, o1/s7/site_struct.py).
  x3mult = 2 is the terminal as built.
* y0 site (the port's two-step load / unload): at load0 all 256 bits of yA are XORed into the site's wires and the
  other 255 load ops are skipped; at unload0 the wires are X-measured (hexterm.py: negate_nonzero, measure_x, the
  phase fixed by the single deferred phase oracle from the IDEAL yA). A lane whose wires do not hold exactly yA there
  gets a garbage phase: it is returned in 'ygarb' (a deviation of that addition); the wires are then fresh (zero) on
  every lane, as the port frees them, and the other 255 unload ops are skipped. No other op touches the wires inside
  the load span or the unload span (site_struct.py), so on every lane whose wires hold yA at unload0 this is the raw
  stream's semantics exactly.
* x1 site: the port's S holds xA exactly and applies the ops one by one, which is the raw semantics (the stream's
  classical logic computes the same bits, checked bit for bit by check_port.py), so x1 runs as raw ops.
returns q, ph, rhit, sk, ygarb"""
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path[:0] = [os.path.join(ROOT, 'h1', 'hexcap'), os.path.join(ROOT, 's2', 'port'), os.path.join(ROOT, 'o2', 'port')]
from clsim import to_lanes, NO, CHUNK  # noqa: E402

PF = 2 ** 256 - 2 ** 32 - 977
OFF = 0x5fffff85c00000f4000000000000770be3c0018c4000ef0c077fffffe      # o2/hex/sitemap.py
MIXED = {62240: 1574, 5534113: 1350, 12359496: 627}                       # o2/port/windowed_lead.py
X3 = (5000000, 5536370)
Y0 = (0, 5496)


def site_tables():
    """op -> (site, bit) for the x3 and y0 sites from o2/hex/sitemap.json; y0 load / unload lists and wires"""
    sm = json.load(open(os.path.join(ROOT, 'o2', 'hex', 'sitemap.json')))
    ids = sorted(int(k) for k in sm)
    x3 = {i: sm[str(i)][1] for i in ids if X3[0] <= i < X3[1]}
    assert len(x3) == 1317 and all(sm[str(i)][0] == 'c3' and sm[str(i)][2] == 0 for i in x3)
    y0 = [i for i in ids if Y0[0] <= i < Y0[1]]
    assert len(y0) == 512 and all(sm[str(i)][0] == 'y' and sm[str(i)][2] == 0 for i in y0)
    ld, ul = y0[:256], y0[256:]
    return dict(x3=x3, ld=ld, ul=ul, ybit={i: sm[str(i)][1] for i in y0})


def simulate_port(ops, L, P_in, A_in, seed, x3mult, st):
    ALL = (1 << L) - 1
    q = [0] * 1227
    bits = [0] * 1595
    xs, ys = to_lanes([p[0] for p in P_in], 256), to_lanes([p[1] for p in P_in], 256)
    ax, ay = to_lanes([a[0] for a in A_in], 256), to_lanes([a[1] for a in A_in], 256)
    c3 = to_lanes([(x3mult * a[0] + OFF) % PF for a in A_in], 256)
    q[0:256], q[256:512] = xs, ys
    bits[0:256], bits[256:512] = ax, ay
    rh = random.Random(seed)
    base, stack = ALL, []
    ph = 0
    rhit = 0
    ygarb = 0
    planes = []
    x3, ld, ul, ybit = st['x3'], st['ld'], st['ul'], st['ybit']
    ld0, ul0 = ld[0], ul[0]
    ldset, ulset = set(ld), set(ul)
    Tall = ops['t']
    ywire = {ybit[i]: int(Tall[i]) for i in ld}
    assert ywire == {ybit[i]: int(Tall[i]) for i in ul}
    special = np.zeros(len(ops), dtype=bool)
    special[list(x3)] = True
    special[ld] = True
    special[ul] = True

    def skip(m):
        i = 0
        while m:
            if i == len(planes):
                planes.append(0)
            planes[i], m = planes[i] ^ m, planes[i] & m
            i += 1
    N = len(ops)
    for c0 in range(0, N, CHUNK):
        ch = ops[c0:c0 + CHUNK]
        K, C1, C2, T = ch['kind'].tolist(), ch['c1'].tolist(), ch['c2'].tolist(), ch['t'].tolist()
        CT, CC = ch['ct'].tolist(), ch['cc'].tolist()
        SP = special[c0:c0 + CHUNK].tolist()
        for j in range(len(K)):
            kd = K[j]
            if SP[j]:
                i = c0 + j
                if i in x3:
                    g = c3[x3[i]]
                    if kd == 6:
                        q[T[j]] ^= g
                    elif kd == 8:
                        q[T[j]] ^= q[C1[j]] & g
                    elif kd == 7:
                        assert i in MIXED
                        ph ^= q[T[j]] & g & bits[MIXED[i]]
                    else:
                        raise RuntimeError('x3 site op kind %d at %d' % (kd, i))
                    continue
                if i == ld0:
                    for b, w in ywire.items():
                        q[w] ^= ay[b]
                    continue
                if i == ul0:
                    for b, w in ywire.items():
                        ygarb |= q[w] ^ ay[b]
                        q[w] = 0
                    continue
                assert i in ldset or i in ulset
                continue
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
                    cond = base if cc == NO else base & bits[cc]
                    q[T[j]] ^= q[C1[j]] & q[C2[j]] & cond
                    if cond != ALL:
                        skip(ALL & ~cond)
            elif kd == 6:
                q[T[j]] ^= base if cc == NO else base & bits[cc]
            elif kd == 11 or kd == 12:
                cond = base if cc == NO else base & bits[cc]
                t = T[j]
                v = q[t] & cond
                if v:
                    m = rh.getrandbits(L)
                    ph ^= m & v
                    if kd == 11:
                        rhit |= v
                    q[t] &= ~cond
                else:
                    m = rh.getrandbits(L) if kd == 12 else 0
                if kd == 12:
                    ct = CT[j]
                    bits[ct] = (bits[ct] & ~cond) | ((m if v else rh.getrandbits(L)) & cond)
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
                cond = base if cc == NO else base & bits[cc]
                ph ^= q[T[j]] & q[C1[j]] & q[C2[j]] & cond
                if cond != ALL:
                    skip(ALL & ~cond)
            elif kd == 0:
                ph ^= base if cc == NO else base & bits[cc]
    nby = (L + 7) // 8
    sk = np.zeros(L, dtype=np.int64)
    for i, pl in enumerate(planes):
        b = np.unpackbits(np.frombuffer(pl.to_bytes(nby, 'little'), dtype=np.uint8), bitorder='little')[:L]
        sk += b.astype(np.int64) << i
    return q, ph, rhit, sk, ygarb
