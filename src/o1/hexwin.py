"""Unit-orbit (mu_6) windows for j = 0 curves: exact toy checks.

Window digit delta in Z[w] (w = primitive cube root of 1, acting as lam on points), taken from H_W, the minimal-norm
residue system of Z[w] / 2^W (mu_6-closed). Exponent a = sum_i delta_i(lam) 2^(W i) mod r: the coordinates
(a1, a2) of sigma = sum delta_i 2^(W i) = a1 + a2 w are in bijection with (Z / 2^(Wm))^2, so the window registers are
a balanced base-2^W encoding of (a1, a2) and the semiclassical QFT runs over (Z/2^(Wm))^2 as usual.
Table: one point per mu_6-orbit of digits; the unit is applied to the looked-up point ((x,y) -> (beta^j x, +-y)).

Checks here: (1) H_W is a complete residue system, mu_6-closed, orbit count; (2) unit application reproduces
delta*P on a toy curve for every digit; (3) exact QFT information per run of the tile domain vs the 1D signed box
and the GLV (a1, a2) box at equal |S| (labels raw, uniform d).
"""
import sys, math, json, itertools
import numpy as np
from toy import Curve, find_curve
from audit import H


def norm(u, v):
    return u * u - u * v + v * v


def mulw(u, v):
    # w * (u + v w) = u w + v w^2 = -v + (u - v) w
    return (-v, u - v)


def hexdigits(W):
    M = 1 << W
    R = M  # search window for representatives
    best = {}
    for u in range(-R, R + 1):
        for v in range(-R, R + 1):
            key = (u % M, v % M)
            nm = norm(u, v)
            if key not in best or nm < best[key][0]:
                best[key] = (nm, [(u, v)])
            elif nm == best[key][0]:
                best[key][1].append((u, v))
    # mu_6-consistent choice, orbit by orbit; unit (sg, j) means sg * w^j
    chosen = {}
    orbits = []
    for key in sorted(best):
        if key in chosen:
            continue
        rep = min(best[key][1])
        orb = []
        for sg in (1, -1):
            for j in range(3):
                d = rep
                for _ in range(j):
                    d = mulw(*d)
                d = (sg * d[0], sg * d[1])
                k2 = (d[0] % M, d[1] % M)
                if k2 not in chosen:
                    assert norm(*d) == best[k2][0]
                    chosen[k2] = (d, len(orbits), (sg, j))
                    orb.append(d)
        orbits.append((rep, orb))
    D = sorted(v[0] for v in chosen.values())
    assert len(D) == M * M and len(set((u % M, v % M) for u, v in D)) == M * M
    return D, chosen, orbits


def check_units(C, W, base_k=5):
    """delta * P == unit(T[orbit rep]) on the toy curve for every digit; w^j: (x, y) -> (beta^j x, y), -1: (x, -y).
    beta^2 x is formed as -x - beta x (one modular addition), as the circuit would."""
    D, chosen, orbits = hexdigits(W)
    p, r, lam, beta = C.p, C.r, C.lam, C.beta
    bad = 0
    for key, (d, oi, (sg, j)) in chosen.items():
        rep = orbits[oi][0]
        T = C.pts[(rep[0] + rep[1] * lam) * base_k % r]
        want = C.pts[(d[0] + d[1] * lam) * base_k % r]
        if T is None:
            got = None
        else:
            x, y = T
            bx = beta * x % p
            xs = [x, bx, (-x - bx) % p][j]
            got = (xs, y if sg == 1 else (-y) % p)
        bad += (got != want)
    return len(D), len(orbits), bad


if __name__ == "__main__" and sys.argv[1] == "digits":
    for W in (1, 2, 3, 4, 5):
        D, chosen, orbits = hexdigits(W)
        sizes = sorted(len(o[1]) for o in orbits)
        print(f"W={W}: |H|={len(D)} orbits={len(orbits)} sizes={dict((s, sizes.count(s)) for s in set(sizes))} "
              f"max norm={max(norm(*d) for d in D)} max|coord|={max(max(abs(u),abs(v)) for u,v in D)}")
    for pmin in (600, 1000):
        p, b, r = find_curve(pmin)
        C = Curve(p, b, r)
        for W in (1, 2, 3, 4):
            print(f"curve r={r} W={W}: digits, orbits, unit-application mismatches =", check_units(C, W))


# ------------------------------------------------------------------ exact QFT information of window domains
def tile(W, m):
    """All sigma = sum_i delta_i M^i, delta_i in H_W: integer coords (a1, a2) and residue labels mod M^m."""
    D, _, _ = hexdigits(W)
    M = 1 << W
    pts = np.zeros((1, 2), dtype=np.int64)
    for i in range(m):
        Dv = np.array(D, dtype=np.int64) * (M ** i)
        pts = (pts[:, None, :] + Dv[None, :, :]).reshape(-1, 2)
    Mm = M ** m
    lab = pts % Mm
    assert len(set(map(tuple, lab))) == len(pts) == Mm * Mm
    return pts, lab, (Mm, Mm)


def qft_info(r, lam, apts, alab, amods, B, ds=None):
    """Exponent register = (a-part label, b), QFT over Z_{amods[0]} x Z_{amods[1]} x Z_B; uniform amplitudes.
    a-part group value = a1 + a2 lam; b-part = b * d. Exact over all d (or the list ds)."""
    M1, M2 = amods
    Na = len(apts)
    assert Na == M1 * M2
    N = Na * B
    aval = (apts[:, 0] + apts[:, 1] * lam) % r
    bb = np.arange(B)
    l1 = np.repeat(alab[:, 0], B)
    l2 = np.repeat(alab[:, 1], B)
    l3 = np.tile(bb, Na)
    ds = range(r) if ds is None else ds
    Ptot = np.zeros(M1 * M2 * B)
    Hc = 0.0
    for d in ds:
        g = ((aval[:, None] + bb[None, :] * d) % r).ravel()
        o = np.argsort(g, kind='stable')
        gs = g[o]
        hist = np.zeros(M1 * M2 * B)
        hist[0] += N
        k = 1
        while True:
            eq = np.flatnonzero(gs[k:] == gs[:-k])
            if len(eq) == 0:
                break
            i, j = o[eq], o[eq + k]
            for (u, v) in ((i, j), (j, i)):
                idx = (((l1[u] - l1[v]) % M1) * M2 + (l2[u] - l2[v]) % M2) * B + (l3[u] - l3[v]) % B
                np.add.at(hist, idx, 1.0)
            k += 1
        P = np.real(np.fft.fftn(hist.reshape(M1, M2, B))).ravel() / N ** 2
        P = np.clip(P, 0, None)
        Ptot += P
        Hc += H(P)
    n = len(ds)
    return H(Ptot / n) - Hc / n


def box1d(A):
    a = np.arange(A, dtype=np.int64)
    pts = np.stack([a, np.zeros_like(a)], 1)
    return pts, pts.copy(), (A, 1)


def box2d(M):
    a1 = np.repeat(np.arange(M), M)
    a2 = np.tile(np.arange(M), M)
    pts = np.stack([a1, a2], 1).astype(np.int64)
    return pts, pts.copy(), (M, M)


if __name__ == "__main__" and sys.argv[1] == "info":
    p, b, r = find_curve(int(sys.argv[2]))
    C = Curve(p, b, r)
    out = []
    for spec in sys.argv[3].split(";"):
        W, m, B = map(int, spec.split(","))
        Mm = (1 << W) ** m
        shapes = {"1D box a": box1d(Mm * Mm), "GLV box (a1,a2)": box2d(Mm), f"hex tile W={W} m={m}": tile(W, m)}
        for name, (pts, lab, mods) in shapes.items():
            I = qft_info(r, C.lam, pts, lab, mods, B)
            c = Mm * Mm * B / r
            out.append(dict(r=r, shape=name, Na=Mm * Mm, B=B, c=c, I=I))
            print(f"r={r} |S|={Mm*Mm*B} c={c:.2f} log2c={math.log2(c):.2f} {name:20s} I_qft={I:.4f} "
                  f"(log2c - I = {math.log2(c) - I:.3f})", flush=True)
    with open(sys.argv[4], "w") as f:
        json.dump(out, f, indent=0)


def qft_info_fast(r, lam, apts, alab, amods, B):
    """Same quantity as qft_info, via the eigenstate decomposition of the point start:
    rho_d = (1/r) sum_t |alpha_t><alpha_t| (x) |beta_td><beta_td|, so P(j,k|d) = (1/r) sum_t A_t(j) K(k, t d)."""
    M1, M2 = amods
    Na = len(apts)
    aval = (apts[:, 0] + apts[:, 1] * lam) % r
    A = np.zeros((r, M1 * M2))
    grid = np.zeros((M1, M2), dtype=complex)
    for t in range(r):
        grid[:] = 0
        grid[alab[:, 0], alab[:, 1]] = np.exp(2j * np.pi * t * aval / r)
        A[t] = (np.abs(np.fft.fft2(grid)) ** 2).ravel() / Na ** 2
    b = np.arange(B)
    s = np.arange(r)
    k = np.arange(B)
    amp = np.exp(2j * np.pi * (s[:, None, None] / r - k[None, :, None] / B) * b[None, None, :]).sum(2)
    K = np.abs(amp) ** 2 / B ** 2          # K[s, k]
    Ptot = np.zeros((M1 * M2, B))
    Hc = 0.0
    for d in range(r):
        P = A.T @ K[(np.arange(r) * d) % r] / r
        P = np.clip(P, 0, None)
        Ptot += P
        Hc += H(P.ravel())
    return H((Ptot / r).ravel()) - Hc / r


if __name__ == "__main__" and sys.argv[1] == "fast":
    p, b, r = find_curve(int(sys.argv[2]))
    C = Curve(p, b, r)
    out = []
    for spec in sys.argv[3].split(";"):
        kind, W, m, B = spec.split(",")
        W, m, B = int(W), int(m), int(B)
        Mm = (1 << W) ** m
        shapes = {"1D box a": box1d(Mm * Mm), "GLV box (a1,a2)": box2d(Mm), f"hex tile W={W} m={m}": tile(W, m)}
        if kind == "check":
            for name, (pts, lab, mods) in shapes.items():
                I1 = qft_info(r, C.lam, pts, lab, mods, B)
                I2 = qft_info_fast(r, C.lam, pts, lab, mods, B)
                print(f"CHECK r={r} {name}: pairs {I1:.6f} eigen {I2:.6f}", flush=True)
            continue
        for name, (pts, lab, mods) in shapes.items():
            I = qft_info_fast(r, C.lam, pts, lab, mods, B)
            c = Mm * Mm * B / r
            out.append(dict(r=r, shape=name, Na=Mm * Mm, B=B, c=c, I=I, a_over_r=Mm * Mm / r))
            print(f"r={r} a/r={Mm*Mm/r:.1f} B={B} log2c={math.log2(c):.2f} {name:20s} I={I:.4f} "
                  f"log2c-I={math.log2(c) - I:.3f}", flush=True)
    if out:
        with open(sys.argv[4], "w") as f:
            json.dump(out, f, indent=0)


def t_posterior(r, lam, apts, alab, amods):
    """a-part alone: outcome law A_t(j) given eigenphase index t (uniform t). Returns H(t | j) and P(MAP t correct)."""
    M1, M2 = amods
    Na = len(apts)
    aval = (apts[:, 0] + apts[:, 1] * lam) % r
    A = np.zeros((r, M1 * M2))
    grid = np.zeros((M1, M2), dtype=complex)
    for t in range(r):
        grid[:] = 0
        grid[alab[:, 0], alab[:, 1]] = np.exp(2j * np.pi * t * aval / r)
        A[t] = (np.abs(np.fft.fft2(grid)) ** 2).ravel() / Na ** 2
    Pj = A.sum(0)                      # r * P(j)
    with np.errstate(divide='ignore', invalid='ignore'):
        post = np.where(A > 0, A / Pj[None, :], 1.0)
        Ht = float(-(A * np.log2(post)).sum() / r)
    psucc = float(A.max(0).sum() / r)
    return Ht, psucc


if __name__ == "__main__" and sys.argv[1] == "tpost":
    p, b, r = find_curve(int(sys.argv[2]))
    C = Curve(p, b, r)
    out = []
    for spec in sys.argv[3].split(";"):
        W, m = map(int, spec.split(","))
        Mm = (1 << W) ** m
        shapes = {"1D box a": box1d(Mm * Mm), "GLV box (a1,a2)": box2d(Mm), f"hex tile W={W} m={m}": tile(W, m)}
        for name, (pts, lab, mods) in shapes.items():
            Ht, ps = t_posterior(r, C.lam, pts, lab, mods)
            out.append(dict(r=r, shape=name, a_over_r=Mm * Mm / r, H_t=Ht, P_map=ps))
            print(f"r={r} log2(a/r)={math.log2(Mm*Mm/r):5.2f} {name:20s} H(t|j)={Ht:.4f} P_MAP={ps:.4f}", flush=True)
    with open(sys.argv[4], "w") as f:
        json.dump(out, f, indent=0)
