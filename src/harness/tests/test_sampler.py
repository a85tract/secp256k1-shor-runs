"""Cross-checks of the exact run sampler (hcore.sample_register + uniform u) against brute force.

1. Full state-vector simulation of tiny runs (cyclic group Z_r, registers of a few qubits, arbitrary per-qubit
   labels, both QFT sign conventions): the exact joint outcome distribution is compared with
   (a) the distribution implied by the sampler's chain formula (must agree to ~1e-12) and
   (b) 400k samples drawn by the sampler itself (total-variation distance and a chi-square statistic).
2. Ekerå's exact probability (arXiv 1905.09084, App. B), computed on a supercomputer by summing over all e:
   r = 915725, m = l = 20, sigma = 0, d = 33979, (j, k) = (965620, 199053) -> 5.0487069192234575e-7.
Run: python3 harness/tests/test_sampler.py
"""
import math, os, random, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from hcore import sample_register, sample_register_amps, grid_dist, grid_sample, Context  # noqa: E402
from ecc import instances_for, Curve  # noqa: E402


def brute(r, regs, sign):
    """regs: list of per-qubit label lists. Exact P over the joint outcome grid (QFT kernel exp(sign 2 pi i x j / N))."""
    sizes = [1 << len(l) for l in regs]
    grids = np.meshgrid(*[np.arange(s) for s in sizes], indexing='ij')
    lab = np.zeros(sizes, dtype=np.int64)
    for g, labs in zip(grids, regs):
        for i, c in enumerate(labs):
            lab = (lab + ((g >> i) & 1) * c) % r
    P = np.zeros(sizes)
    norm = math.prod(sizes)
    for e in range(r):
        M = (lab == e).astype(complex)
        F = np.fft.ifftn(M) * norm if sign > 0 else np.fft.fftn(M)
        P += np.abs(F) ** 2
    return P / norm ** 2


def chain_prob(labs, u, r, j):
    """Probability of outcome j implied by the sampler's chain (same formula as sample_register)."""
    l, p, low = len(labs), 1.0, 0
    for t in range(l):
        phi = (u * labs[l - 1 - t] % r) / r - low / (1 << (t + 1))
        phi -= round(phi)
        p0 = math.cos(math.pi * phi) ** 2
        y = (j >> t) & 1
        p *= p0 if y == 0 else 1 - p0
        low |= y << t
    return p


def chain_dist(r, regs):
    sizes = [1 << len(l) for l in regs]
    P = np.zeros(sizes)
    for u in range(r):
        vecs = [np.array([chain_prob(labs, u, r, j) for j in range(1 << len(labs))]) for labs in regs]
        T = vecs[0]
        for v in vecs[1:]:
            T = np.multiply.outer(T, v)
        P += T
    return P / r


def sampled_dist(r, regs, N, seed):
    rng = random.Random(seed)
    sizes = [1 << len(l) for l in regs]
    C = np.zeros(sizes)
    for _ in range(N):
        u = rng.randrange(r)
        C[tuple(sample_register(rng, labs, u, r) for labs in regs)] += 1
    return C / N


def case(name, r, regs, N=400000):
    Pb = brute(r, regs, +1)
    Pb2 = brute(r, regs, -1)
    Pc = chain_dist(r, regs)
    Ps = sampled_dist(r, regs, N, name)
    sym = np.abs(Pb - Pb2).max()
    exact = np.abs(Pb - Pc).max()
    tv = 0.5 * np.abs(Pb - Ps).sum()
    mask = Pb > 0
    chi2 = (N * (Ps[mask] - Pb[mask]) ** 2 / Pb[mask]).sum()
    dof = mask.sum() - 1
    z = (chi2 - dof) / math.sqrt(2 * dof)
    leak = (Ps[~mask] > 0).sum()
    ok = sym < 1e-12 and exact < 1e-12 and abs(z) < 5 and leak == 0
    print('%-34s r=%-3d outcomes=%-5d |brute(+)-brute(-)|=%.1e |brute-chain|=%.1e TV(samples)=%.4f '
          'chi2 z=%+.2f %s' % (name, r, Pb.size, sym, exact, tv, z, 'OK' if ok else 'FAIL'))
    return ok


def ekera_appB():
    r, d, A, B, j, k = 915725, 33979, 2 ** 20, 2 ** 20, 965620, 199053
    u = np.arange(r, dtype=np.int64)

    def F(N, num):   # F_N(x) for x = num / (N r), num reduced to the symmetric range, extended precision
        num = (num + N * r // 2) % (N * r) - N * r // 2
        L = np.longdouble
        pi = L('3.14159265358979323846264338327950288')
        top = np.sin(pi * ((num % r).astype(L) / L(r)))
        bot = np.sin(pi * num.astype(L) / L(N * r))
        return np.where(num == 0, L(1), top ** 2 / (L(N) ** 2 * np.where(num == 0, L(1), bot) ** 2))
    # labels: a -> +1 (G), b -> -d (-Q); register outcome peaks at j/A = u c / r
    P = (F(A, j * r - u * A) * F(B, k * r - ((-u * d) % r) * B)).sum() / r
    ref = 5.0487069192234575e-07
    rel = abs(float(P) - ref) / ref
    print('Ekerå App. B exact probability: ours %.16e, Ekerå %.16e, rel. diff %.1e %s' % (
        float(P), ref, rel, 'OK' if rel < 1e-9 else 'FAIL'))
    return rel < 1e-9


def grid_cases(rng):
    """Hash runs: (a) fn = 'label' (the full point) must equal the linear-run distribution (chain formula);
    (b) fn = 'legendre_x' / 'x_low:2' must equal an independent brute force on a real 12-bit curve."""
    oks = []
    inst = instances_for(12)[0]
    r = inst['r']
    E = Curve(inst['p'], inst['a'], inst['b'])
    G = (inst['Gx'], inst['Gy'])
    h = Context(inst, rng.randrange(1, r), 5)
    for regs in ([[(1 << i) % r for i in range(5)], [rng.randrange(r) for _ in range(4)]],
                 [[rng.randrange(r) for _ in range(3)], [rng.randrange(r) for _ in range(3)],
                  [rng.randrange(r) for _ in range(2)]]):
        sizes, dist = grid_dist(h, regs, 'label')
        Pg = sum(dist.values())
        diff = np.abs(Pg - chain_dist(r, regs)).max()
        ok = diff < 1e-12
        print('%-34s r=%-5d outcomes=%-5d |grid(label)-chain|=%.1e %s' % ('hash run, fn=label', r, Pg.size, diff,
                                                                          'OK' if ok else 'FAIL'))
        oks.append(ok)
        for fn in ('legendre_x', 'x_low:2'):
            sizes, dist = grid_dist(h, regs, fn)
            grids = np.meshgrid(*[np.arange(z) for z in sizes], indexing='ij')
            lab = np.zeros(sizes, dtype=np.int64)
            for g, labs in zip(grids, regs):
                for i, c in enumerate(labs):
                    lab = (lab + ((g >> i) & 1) * c) % r
            vals = np.zeros(sizes, dtype=np.int64)
            for idx in np.ndindex(*sizes):
                P = E.mul(int(lab[idx]), G)
                if P is None:
                    v = 0
                elif fn == 'legendre_x':
                    v = 1 if pow(P[0], (inst['p'] - 1) // 2, inst['p']) == inst['p'] - 1 else 0
                else:
                    v = P[0] & 3
                vals[idx] = v
            N = vals.size
            worst = 0.0
            for v in np.unique(vals):
                ref = np.abs(np.fft.ifftn((vals == v).astype(complex)) * N) ** 2 / N ** 2
                worst = max(worst, np.abs(ref - dist.get(int(v), 0)).max())
            total = sum(d.sum() for d in dist.values())
            ok = worst < 1e-12 and abs(total - 1) < 1e-12 and set(dist) == set(int(v) for v in np.unique(vals))
            print('%-34s r=%-5d outcomes=%-5d |grid-brute|=%.1e total=%.15f %s' % (
                'hash run, fn=' + fn, r, N, worst, total, 'OK' if ok else 'FAIL'))
            oks.append(ok)
    # the one-FFT sampler draws from grid_dist's distribution
    regs = [[rng.randrange(r) for _ in range(3)], [rng.randrange(r) for _ in range(3)]]
    sizes, dist = grid_dist(h, regs, 'legendre_x')
    S = 100000
    cnt = {}
    qr = random.Random(9)
    for _ in range(S):
        o = tuple(grid_sample(h, regs, 'legendre_x', qr))
        cnt[o] = cnt.get(o, 0) + 1
    chi2, dof = 0.0, -1
    for v, P in dist.items():
        for idx in np.ndindex(*sizes):
            pe = P[idx]
            if pe > 0:
                chi2 += (cnt.get(idx + (v,), 0) - S * pe) ** 2 / (S * pe)
                dof += 1
    z = (chi2 - dof) / math.sqrt(2 * dof)
    ok = abs(z) < 5
    print('%-34s r=%-5d outcomes=%-5d chi2 z=%+.2f %s' % ('hash run sampler vs grid_dist', r, dof + 1, z,
                                                         'OK' if ok else 'FAIL'))
    oks.append(ok)
    return oks


def amp_case(rng):
    """Linear run with one non-uniform (random complex) register and one uniform register: brute-force state
    vector vs the sampler's per-u FFT formula, and sampled counts."""
    r = 23
    regs = [[rng.randrange(r) for _ in range(3)], [(1 << i) % r for i in range(3)]]
    amps = np.array([complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(8)])
    amps /= np.linalg.norm(amps)
    sizes = [8, 8]
    psi = np.multiply.outer(amps, np.full(8, 1 / math.sqrt(8)))
    g0, g1 = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
    lab = np.zeros(sizes, dtype=np.int64)
    for g, labs in ((g0, regs[0]), (g1, regs[1])):
        for i, c in enumerate(labs):
            lab = (lab + ((g >> i) & 1) * c) % r
    Pb = np.zeros(sizes)
    for e in range(r):
        Pb += np.abs(np.fft.fftn(psi * (lab == e)) / 8) ** 2     # unitary 2-D QFT, kernel exp(-2 pi i ...)
    Pc = np.zeros(sizes)
    x = np.arange(8)
    for u in range(r):
        ph = sum(((x >> i) & 1) * ((u * c % r) / r) for i, c in enumerate(regs[0]))
        p0 = np.abs(np.fft.fft(amps * np.exp(2j * np.pi * ph))) ** 2 / 8   # unitary normalisation
        p1 = np.array([chain_prob(regs[1], u, r, j) for j in range(8)])
        Pc += np.multiply.outer(p0, p1)
    Pc /= r
    diff = np.abs(Pb - Pc).max()
    N = 200000
    qr = random.Random(3)
    C = np.zeros(sizes)
    for _ in range(N):
        u = qr.randrange(r)
        C[sample_register_amps(qr, regs[0], u, r, amps), sample_register(qr, regs[1], u, r)] += 1
    mask = Pb > 1e-15
    chi2 = (N * (C[mask] / N - Pb[mask]) ** 2 / Pb[mask]).sum()
    dof = mask.sum() - 1
    z = (chi2 - dof) / math.sqrt(2 * dof)
    ok = diff < 1e-12 and abs(z) < 5
    print('%-34s r=%-5d outcomes=%-5d |brute-formula|=%.1e chi2 z=%+.2f %s' % (
        'amplitude register + uniform', r, Pb.size, diff, z, 'OK' if ok else 'FAIL'))
    return ok


def env_case(rng):
    """Envelope register (Run.reg with 2^k amplitudes, k < bits: top k qubits shaped, low qubits uniform) and a
    uniform register: brute-force state vector vs sampled counts, and the sampler's P(j | u) vs the FFT of the
    expanded 2^bits amplitude vector."""
    r, l, k = 29, 5, 2
    regs = [[rng.randrange(r) for _ in range(l)], [(1 << i) % r for i in range(3)]]
    phi = np.array([complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(1 << k)])
    phi /= np.linalg.norm(phi)
    full = np.repeat(phi, 1 << (l - k)) / math.sqrt(1 << (l - k))      # amplitude of x = y 2^(l-k) + z
    sizes = [1 << l, 8]
    psi = np.multiply.outer(full, np.full(8, 1 / math.sqrt(8)))
    g0, g1 = np.meshgrid(np.arange(sizes[0]), np.arange(8), indexing='ij')
    lab = np.zeros(sizes, dtype=np.int64)
    for g, labs in ((g0, regs[0]), (g1, regs[1])):
        for i, c in enumerate(labs):
            lab = (lab + ((g >> i) & 1) * c) % r
    Pb = np.zeros(sizes)
    for e in range(r):
        Pb += np.abs(np.fft.fftn(psi * (lab == e))) ** 2 / psi.size     # unitary 2-D QFT
    x = np.arange(sizes[0])
    pfull = []
    for u in range(r):
        ph = sum(((x >> i) & 1) * ((u * c % r) / r) for i, c in enumerate(regs[0]))
        pfull.append(np.abs(np.fft.fft(full * np.exp(2j * np.pi * ph))) ** 2 / sizes[0])
    N = 200000
    qr = random.Random(5)
    C = np.zeros(sizes)
    dp = 0.0
    for _ in range(N):
        u = qr.randrange(r)
        j, pj = sample_register_amps(qr, regs[0], u, r, phi, with_prob=True)
        dp = max(dp, abs(pj - pfull[u][j]))
        C[j, sample_register(qr, regs[1], u, r)] += 1
    mask = Pb > 1e-15
    chi2 = (N * (C[mask] / N - Pb[mask]) ** 2 / Pb[mask]).sum()
    dof = mask.sum() - 1
    z = (chi2 - dof) / math.sqrt(2 * dof)
    ok = dp < 1e-12 and abs(z) < 5 and abs(Pb.sum() - 1) < 1e-12
    print('%-34s r=%-5d outcomes=%-5d |P(j|u)-full FFT|=%.1e chi2 z=%+.2f %s' % (
        'envelope register (k=2 of 5) + unif.', r, Pb.size, dp, z, 'OK' if ok else 'FAIL'))
    return ok


if __name__ == '__main__':
    rng = random.Random(1)
    oks = []
    # Shor-type: a -> G (label 1), b -> -Q (label -d), r prime, registers covering and not covering r
    for r, la, lb in ((13, 4, 4), (29, 5, 4), (31, 3, 2), (61, 6, 3)):
        d = rng.randrange(1, r)
        regs = [[(1 << i) % r for i in range(la)], [(-d << i) % r for i in range(lb)]]
        oks.append(case('shor-type r=%d la=%d lb=%d' % (r, la, lb), r, regs))
    # arbitrary per-qubit labels (windowed adds of several points, endomorphism-style registers), 3 registers
    for r in (11, 37):
        regs = [[rng.randrange(r) for _ in range(3)], [rng.randrange(r) for _ in range(3)],
                [rng.randrange(r) for _ in range(2)]]
        oks.append(case('random labels, 3 registers r=%d' % r, r, regs))
    oks.append(ekera_appB())
    oks += grid_cases(rng)
    oks.append(amp_case(rng))
    oks.append(env_case(rng))
    print('ALL OK' if all(oks) else 'SOME CHECKS FAILED')
    sys.exit(0 if all(oks) else 1)
