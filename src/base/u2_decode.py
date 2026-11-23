#!/usr/bin/env python3
"""Baseline U2: single-run decoding of base/u2_price.py's layouts, sampled at n = 256.

The run's two registers are independent given the accumulator's character u (harness/hcore.py Run.measure), so each
is sampled on its own with the harness's exact sampler (harness/mixreg.py sample) at a uniform u c (c = 1 for a on
G, c = -d for b on -Q; for uniform u the law of the error does not depend on c). Decoder, generalising Ekera's
one-run post-processing (harness/candidates/shor.py) to the two-dimensional registers:
  a (log2 norm >= 256): the candidates for u are the points of the lattice L = {(x, lam x) mod r} nearest to
    t_a = r Y_a / N_a in the metric |M^T e|^2 of h1/candidates/stack_n7_l9.py decode_mix (rank 0 = decode_mix's
    answer); the script records the rank of the true point (how many lattice points are nearer to t_a);
  b (log2 norm >= 208, 48 bits short of r as U1's b): v = u c_b lies near t_b = r Y_b / N_b; the decoder searches
    the lattice points within |M^T e|^2 <= QB, about pi QB / (r norm(Ptop_b)) points (eps_b = |M^T e|^2 /
    (r norm(Ptop_b)) of the true point is recorded), by a two-dimensional baby-step giant-step: baby steps once per
    run, giant steps per u candidate, then d = -v / u and one check.
Prints, per layout: P(rank_a = 0), the rank quantiles, eps_b quantiles, and the classical work of the decoder at the
usage: python3 base/u2_decode.py VARIANT SAMPLES SEED      (VARIANT nt / n / h as in base/u2_price.py)
"""
import json, math, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path[:0] = [os.path.join(REPO, 'harness'), os.path.join(REPO, 'h1', 'candidates'), HERE]
import mixreg                     # exact sampler of a mixed-radix Eisenstein register
import stack_n7_l9 as st
import u2_price

R = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141          # secp256k1 group order
LAM = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72        # eigenvalue of (x, y) -> (beta x, y)
assert (LAM * LAM + LAM + 1) % R == 0
RANK_EXACT = 50000


def frame(spec):
    pis = st.moduli(spec)
    Ptop = (1, 0)
    for z in pis[:-1]:
        Ptop = st.emul(Ptop, z)
    N = st.norm(st.emul(Ptop, pis[-1]))
    a, b = Ptop
    return mixreg.cached(spec), N, Ptop, ((a, b), (-b, a - b))


def qform(MT, v):
    x = MT[0][0] * v[0] + MT[0][1] * v[1]
    y = MT[1][0] * v[0] + MT[1][1] * v[1]
    return x * x + y * y


def reduced(MT):
    """Gauss-reduced basis of L = {(x, lam x mod r)} in the metric qform (as decode_mix)"""
    b1, b2 = (1, LAM), (0, R)
    while True:
        if qform(MT, b2) < qform(MT, b1):
            b1, b2 = b2, b1
        num = (qform(MT, (b1[0] + b2[0], b1[1] + b2[1])) - qform(MT, b1) - qform(MT, b2)) // 2
        mu = (2 * num + qform(MT, b1)) // (2 * qform(MT, b1))
        if mu == 0:
            return b1, b2
        b2 = (b2[0] - mu * b1[0], b2[1] - mu * b1[1])


def wrap(x, m):
    x %= m
    return x - m if 2 * x > m else x


def sample_reg(rng, wins, N, uc):
    eta = mixreg.sample(rng, wins, uc, R, LAM)
    Y = (eta[0] * N, eta[1] * N)
    assert Y[0].denominator == 1 and Y[1].denominator == 1
    return int(Y[0]), int(Y[1])


def err_q(MT, N, Y, p):
    """N^2 |M^T (t - p)|^2 with t = r Y / N, the difference taken mod r (integers: r Y - N p mod r N)"""
    e = (wrap(R * Y[0] - N * p[0], R * N), wrap(R * Y[1] - N * p[1], R * N))
    return qform(MT, e)


def rank(MT, N, Y, basis, p):
    """number of lattice points strictly nearer to t = r Y / N than p. Every such point v has |v - p| <= 2 |t - p|
    (metric qform); in the Gauss-reduced basis (angle in [60, 120] degrees) its coordinates are at most
    (2 / sqrt 3) 2 |t - p| / |b_i|, so the box below holds all of them (exact count). Far outliers (the continuous
    count pi |M^T (t - p)|^2 / (r norm(Ptop)) of points in the disk above RANK_EXACT) get that estimate instead:
    every candidate cap used below is far smaller, so only "beyond the cap" matters for them."""
    b1, b2 = basis
    qt = err_q(MT, N, Y, p)
    est = math.pi * (qt / (N * N * R * abs(MT[0][0] * MT[1][1] - MT[0][1] * MT[1][0])))
    if est > RANK_EXACT:
        return int(est)
    K1 = int(4 / math.sqrt(3) * math.sqrt(qt / (N * N * qform(MT, b1)))) + 1
    K2 = int(4 / math.sqrt(3) * math.sqrt(qt / (N * N * qform(MT, b2)))) + 1
    n = 0
    for i in range(-K1, K1 + 1):
        for j in range(-K2, K2 + 1):
            if i == 0 and j == 0:
                continue
            v = (p[0] + i * b1[0] + j * b2[0], p[1] + i * b1[1] + j * b2[1])
            if err_q(MT, N, Y, v) < qt:
                n += 1
    return n


def quant(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(math.ceil(q * len(s))) - 1)]


def main():
    variant, S, seed = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    la, _, lb, _ = u2_price.LAYOUTS[variant]
    wa, Na, Pa, MTa = frame(la)
    wb, Nb, Pb, MTb = frame(lb)
    basis_a = reduced(MTa)
    rng = random.Random('u2dec:%s:%s' % (variant, seed))
    print('variant %s: a = %s (log2 N %.3f), b = %s (log2 N %.3f); %d samples, seed %s'
          % (variant, la, math.log2(Na), lb, math.log2(Nb), S, seed))
    ranks, epsb, ok = [], [], 0
    for s in range(S):
        u = rng.randrange(1, R)
        Ya = sample_reg(rng, wa, Na, u)
        pa = (u, u * LAM % R)
        ok += st.decode_mix(R, LAM, Ya[0], Ya[1], Na, Pa) == u
        ranks.append(rank(MTa, Na, Ya, basis_a, pa))
        v = rng.randrange(1, R)
        Yb = sample_reg(rng, wb, Nb, v)
        qb = err_q(MTb, Nb, Yb, (v, v * LAM % R))
        epsb.append(qb / (Nb * Nb * R * st.norm(Pb)))
        if (s + 1) % 1000 == 0:
            print('  %d samples: decode_mix correct %d, max rank %d' % (s + 1, ok, max(ranks)), flush=True)
    print('a: decode_mix correct %d / %d = %.5f; rank quantiles 0.99 / 0.999 / max: %d / %d / %d'
          % (ok, S, ok / S, quant(ranks, 0.99), quant(ranks, 0.999), max(ranks)))
    print('b: eps_b quantiles 0.5 / 0.99 / 0.999 / max (log2): %.2f / %.2f / %.2f / %.2f'
          % tuple(math.log2(quant(epsb, q)) for q in (0.5, 0.99, 0.999, 1.0)))
    # decoder caps: the CA lattice points nearest to t_a, and the b region at the qe quantile of eps_b; per run
    # 2 sqrt(points) group operations of baby steps (once) and giant steps (per candidate), plus one check per
    # candidate (1.5 x 256 doublings and additions); the region is a reduced-basis parallelogram around the ellipse
    # (pi E lattice points), at most 4.62 E points, charged 5 E
    table = []
    print('%6s %8s %9s %10s %8s %9s %9s' % ('CA', 'P(a)', 'b quant.', 'log2 E', 'P(b)', 'P(a)P(b)', 'log2 work'))
    for ca in (1, 16, 64, 256, 1024, 4096):
        pa_ok = sum(1 for x in ranks if x < ca) / S
        for qe in (0.99, 0.995, 0.999):
            E = quant(epsb, qe)
            pb_ok = sum(1 for x in epsb if x <= E) / S
            work = 2 * math.sqrt(5 * E) * (1 + ca) + ca * 256 * 1.5
            table.append(dict(CA=ca, Pa=pa_ok, qb=qe, log2E=math.log2(E), Pb=pb_ok, P=pa_ok * pb_ok,
                              log2work=math.log2(work)))
            print('%6d %8.5f %9.3f %10.2f %8.5f %9.5f %9.2f' % (ca, pa_ok, qe, math.log2(E), pb_ok, pa_ok * pb_ok,
                                                                math.log2(work)))

    json.dump(dict(variant=variant, samples=S, seed=seed, layout_a=la, layout_b=lb, log2Na=math.log2(Na),
                   log2Nb=math.log2(Nb), decode_mix_ok=ok, rank_q99=quant(ranks, 0.99), caps=table),
              open(os.path.join(HERE, 'u2_decode_%s.json' % variant), 'w'), indent=1)


if __name__ == '__main__':
    main()
