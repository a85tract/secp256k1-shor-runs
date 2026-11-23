"""Check of Theorem 1 on exact toy curves.

f_d(z) computed, group register measured or discarded) reveals at most log2(1 + (|S|-1)/r) bits about a uniform d.

Here: exact Holevo information chi = S(E_d rho_d) - E_d S(rho_d) of the exponent register (an upper bound on what
ANY measurement extracts), and the mutual information of the standard QFT measurement, for
  E1  raw labels (control; inside the theorem),
  E2  compute-hash-uncompute runs: a function h of the point is kept, the point itself uncomputed (outside the theorem),
  E3  non-generic initial group states sum_P g(P)|P> built from coordinates (outside the theorem).
"""
import sys, json, math
import numpy as np
from toy import Curve, find_curve


def H(pr):
    pr = pr[pr > 1e-300]
    return float(-(pr * np.log2(pr)).sum())


def vn(mat):
    ev = np.linalg.eigvalsh(mat)
    return H(np.clip(ev, 0, None))


def holevo_partition(labels, amp2):
    """labels: (r, N) int array, output label of each z for each d; amp2: (N,) |alpha_z|^2 (sum 1).
    rho_d = sum_v |phi_v><phi_v| is block diagonal with rank-1 blocks, so S(rho_d) = Shannon entropy of fibre weights."""
    r, N = labels.shape
    a = np.sqrt(amp2)
    acc = np.zeros((N, N))
    Sd = 0.0
    for d in range(r):
        L = labels[d]
        _, inv = np.unique(L, return_inverse=True)
        Sd += H(np.bincount(inv, weights=amp2))
        acc += (L[:, None] == L[None, :])
    sigma = (a[:, None] * a[None, :]) * acc / r
    return vn(sigma) - Sd / r


def qft_info_box(labels, A, B):
    """Standard measurement: QFT over Z_A x Z_B of the exponent register (uniform box), outcome (j, k).
    P(j,k|d) = (1/(AB)^2) sum_v |sum_{z in v} e(.)|^2."""
    r, N = labels.shape
    assert N == A * B
    Ptot = np.zeros((A, B))
    Hcond = 0.0
    for d in range(r):
        L = labels[d].reshape(A, B)
        vals, inv = np.unique(L, return_inverse=True)
        inv = inv.reshape(A, B)
        if len(vals) <= 64:
            P = np.zeros((A, B))
            for v in range(len(vals)):
                P += np.abs(np.fft.fft2(inv == v)) ** 2
        else:
            # raw labels: few non-singleton fibres; difference histogram of ordered colliding pairs
            Hd = np.zeros((A, B))
            Hd[0, 0] += N  # diagonal pairs z = z' (singletons included)
            order = np.argsort(inv, axis=None)
            flat = inv.ravel()[order]
            cuts = np.flatnonzero(np.diff(flat)) + 1
            groups = np.split(order, cuts)
            for g in groups:
                if len(g) > 1:
                    ga, gb = np.divmod(g, B)
                    da = (ga[:, None] - ga[None, :]) % A
                    db = (gb[:, None] - gb[None, :]) % B
                    np.add.at(Hd, (da.ravel(), db.ravel()), 1.0)
                    Hd[0, 0] -= len(g)  # diagonal already counted
            P = np.real(np.fft.fft2(Hd))
        P /= (A * B) ** 2
        Ptot += P
        Hcond += H(P.ravel())
    Ptot /= r
    return H(Ptot.ravel()) - Hcond / r


def box_labels(C, A, B, outfn):
    a = np.repeat(np.arange(A), B)
    b = np.tile(np.arange(B), A)
    lab = np.empty((C.r, A * B), dtype=np.int64)
    for d in range(C.r):
        k = (a + b * d) % C.r
        lab[d] = outfn[k]
    return lab


def hashes(C):
    p = C.p
    X, Y = C.X, C.Y
    O = X < 0
    leg = np.array([0 if x < 0 else (1 if pow(int(x), (p - 1) // 2, p) == 1 else 2) for x in X])
    out = {
        "raw": np.arange(C.r),
        "x_mod2": np.where(O, 2, X % 2),
        "legendre_x": leg,
        "x_mod4": np.where(O, 4, X % 4),
        "x_mod16": np.where(O, 16, X % 16),
        "mu6_class": np.array([-1 if x < 0 else (int(x) ** 3 % p) * p + (int(y) ** 2 % p) for x, y in zip(X, Y)]),
        "x_only": np.where(O, -1, X),
    }
    return out


def e1_e2(C, shapes, outnames, do_qft=True):
    outs = hashes(C)
    rows = []
    for (A, B) in shapes:
        N = A * B
        bound = math.log2(1 + (N - 1) / C.r)
        for name in outnames:
            lab = box_labels(C, A, B, outs[name])
            chi = holevo_partition(lab, np.full(N, 1.0 / N))
            iq = qft_info_box(lab, A, B) if do_qft else float("nan")
            rows.append(dict(r=C.r, A=A, B=B, c=N / C.r, out=name, chi=chi, I_qft=iq, bound=bound))
            print(f"r={C.r} box {A}x{B} c={N/C.r:.3f} out={name:20s} chi={chi:.4f} I_qft={iq:.4f} "
                  f"bound={bound:.4f} chi/bound={chi/bound:.2f}", flush=True)
    return rows


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "e12":
        p, b, r = find_curve(int(sys.argv[2]))
        C = Curve(p, b, r)
        print(f"curve p={p} b={b} r={r}")
        shapes = [tuple(map(int, s.split("x"))) for s in sys.argv[3].split(",")]
        names = sys.argv[4].split(",")
        rows = e1_e2(C, shapes, names)
        with open(sys.argv[5], "w") as f:
            json.dump(rows, f, indent=0)


# ---------------------------------------------------------------- E3: non-generic initial group states
def autocorr(g):
    """A(k) = <Phi|T_k|Phi> = sum_j conj(g[j+k]) g[j] (circular over Z_r, index = discrete log)."""
    G = np.fft.fft(g)
    return np.fft.ifft(np.conj(G) * G)[::-1].copy() if False else np.array(
        [np.vdot(np.roll(g, -k), g) for k in range(len(g))])


def initial_states(C):
    p, r = C.p, C.r
    X, Y = C.X, C.Y
    O = X < 0
    def char(x, m):
        return pow(int(x), (p - 1) // m, p)
    w3 = next(c for c in range(2, p) if pow(c, 3, p) == 1)  # a primitive cube root of 1 in F_p
    cube_idx = {1: 0, w3: 1, w3 * w3 % p: 2}
    gs = {}
    gs["x<p/2"] = np.where(~O & (X < p // 2), 1.0, 0.0)
    gs["QR(x)"] = np.array([0.0 if (x <= 0) else float(char(x, 2) == 1) for x in X])
    gs["cube(x)"] = np.array([0.0 if (x <= 0) else float(char(x, 3) == 1) for x in X])
    gs["(-1)^x"] = np.where(O, 0.0, (-1.0) ** (X % 2))
    gs["chi3(x)"] = np.array([0j if x <= 0 else np.exp(2j * np.pi * cube_idx[char(x, 3)] / 3) for x in X])
    gs["(-1)^y"] = np.where(O, 0.0, (-1.0) ** (Y % 2))
    gs["all_points"] = np.where(O, 0.0, 1.0)
    gs["CONTROL known t=1"] = np.exp(-2j * np.pi * np.arange(r) / r)
    gs["CONTROL segment K=8"] = (np.arange(r) < 8).astype(float)
    gs["CONTROL point O"] = (np.arange(r) == 0).astype(float)
    for k in gs:
        gs[k] = gs[k].astype(complex)
        gs[k] /= np.linalg.norm(gs[k])
    return gs


def chi_initial(C, g, A, B):
    r = C.r
    N = A * B
    a = np.repeat(np.arange(A), B)
    b = np.tile(np.arange(B), A)
    Ak = autocorr(g)
    alpha = np.full(N, 1.0 / math.sqrt(N))
    aa = alpha[:, None] * alpha[None, :]
    sig = np.zeros((N, N), dtype=complex)
    Sd = 0.0
    for d in range(r):
        f = (a + b * d) % r
        rho = aa * Ak[(f[:, None] - f[None, :]) % r]
        Sd += vn(rho)
        sig += rho
    chi = vn(sig / r) - Sd / r
    w = np.abs(np.fft.fft(g)) ** 2 / r
    P2r = float((w ** 2).sum() * r)
    amax = float(np.abs(Ak[1:]).max())
    return chi, P2r, amax, float(w[0])


def e3(C, shapes, names=None):
    gs = initial_states(C)
    rows = []
    for (A, B) in shapes:
        N = A * B
        bound = math.log2(1 + (N - 1) / C.r)
        for name, g in gs.items():
            if names and name not in names:
                continue
            chi, P2r, amax, w0 = chi_initial(C, g, A, B)
            rows.append(dict(r=C.r, A=A, B=B, c=N / C.r, state=name, chi=chi, rP2=P2r, maxA=amax, w0=w0, bound=bound))
            print(f"r={C.r} box {A}x{B} c={N/C.r:.3f} init={name:22s} chi={chi:.4f} bound={bound:.4f} "
                  f"r*sum w^2={P2r:7.3f} max|A(k!=0)|={amax:.3f} w0={w0:.3f}", flush=True)
    return rows


if __name__ == "__main__" and sys.argv[1] == "e3":
    p, b, r = find_curve(int(sys.argv[2]))
    C = Curve(p, b, r)
    print(f"curve p={p} b={b} r={r}")
    shapes = [tuple(map(int, s.split("x"))) for s in sys.argv[3].split(",")]
    names = sys.argv[5].split(",") if len(sys.argv) > 5 else None
    rows = e3(C, shapes, names)
    with open(sys.argv[4], "w") as f:
        json.dump(rows, f, indent=0)
