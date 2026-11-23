#!/usr/bin/env python3
"""Exact map of every addend-dependent op of the leader stream to (site value, bit, constant).
Site values: y0, y5: yA; x1: xA; x3: c3 = (3 xA + OFF) mod p; x4: xA + 1 (bits) or xA (bits).
Lanes: random addends plus structured ones that force long carries (xA with its low t bits set, a0 near p - OFF).
Writes o2/hex/sitemap.json: {op: [value_name, bit, const]}."""
import os, sys, random, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'port')]
os.environ.setdefault('O2_OPS_RAW', sys.argv[1] if len(sys.argv) > 1 else '')
import windowed_lead as wl
p = 2**256 - 2**32 - 977
OFF = 0x5fffff85c00000f4000000000000770be3c0018c4000ef0c077fffffe
inv3 = pow(3, p - 2, p)
VALUES = {'y': lambda x, y: y, 'x': lambda x, y: x, 'c3': lambda x, y: (3 * x + OFF) % p,
          'x+1': lambda x, y: x + 1}
SITEVAL = {'y0': ['y'], 'y5': ['y'], 'x1': ['x'], 'x3': ['c3'], 'x4': ['x+1', 'x']}


def lanes(seed):
    rng = random.Random(seed)
    out = [(rng.randrange(p), rng.randrange(p)) for _ in range(160)]
    for t in range(1, 257, 3):                          # xA with its low t bits set (carry chains of x + 1)
        x = (rng.randrange(p) | ((1 << t) - 1)) % p
        out.append((x, rng.randrange(p)))
    for t in range(1, 257, 3):                          # a0 = 3 xA mod p close to p - OFF (the mod-p reduction)
        a0 = (p - OFF + rng.randrange(-(1 << t), 1 << t)) % p
        out.append((a0 * inv3 % p, rng.randrange(p)))
    for t in range(1, 257, 5):                          # carries of a0 + OFF through long runs
        a0 = ((~OFF) & ((1 << t) - 1)) | (rng.randrange(1 << 200) << t)
        out.append(((a0 % p) * inv3 % p, rng.randrange(p)))
    out += [(0, 0), (1, 1), (p - 1, p - 1), (2**255, 3)]
    return out


def build_map():
    st = wl._stream()
    ops, dep, site_of = st['ops'], st['dep'], st['site_of']
    want = set(int(i) for i in dep[:, 0])
    res = {}
    for seed in (1, 2):
        offs = lanes(seed)
        L = len(offs)
        ALL = (1 << L) - 1
        conds = wl._classical_conds(ops, want, offs, 1)
        masks = {}
        for vn, f in VALUES.items():
            vs = [f(x, y) for x, y in offs]
            for b in range(257):
                masks[(vn, b)] = sum(((v >> b) & 1) << l for l, v in enumerate(vs))
        for i in sorted(want):
            name = wl.SITES[site_of[i]][0]
            hit = None
            for vn in SITEVAL[name]:
                for b in range(257):
                    m = masks[(vn, b)]
                    if conds[i] == m:
                        hit = (vn, b, 0)
                    elif conds[i] == ALL ^ m:
                        hit = (vn, b, 1)
                    if hit:
                        break
                if hit:
                    break
            if seed == 1:
                res[i] = hit
            elif res.get(i) != hit:
                res[i] = None
    return res


if __name__ == '__main__':
    res = build_map()
    st = wl._stream()
    site_of = st['site_of']
    summ = {}
    for i, h in res.items():
        nm = wl.SITES[site_of[i]][0]
        key = (nm, None if h is None else h[0], None if h is None else h[2])
        summ[key] = summ.get(key, 0) + 1
    for k in sorted(summ, key=str):
        print(k, summ[k])
    bits = {}
    for i, h in res.items():
        if h:
            bits.setdefault((wl.SITES[site_of[i]][0], h[0]), set()).add(h[1])
    for k in sorted(bits):
        b = sorted(bits[k])
        print(k, len(b), 'bits, range', b[0], b[-1], 'missing', [x for x in range(b[0], b[-1] + 1) if x not in bits[k]][:12])
    json.dump({str(i): h for i, h in res.items()}, open(os.environ.get('O2_SITEMAP', os.path.join(HERE, 'sitemap.json')), 'w'))
